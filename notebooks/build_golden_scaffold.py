"""
build_golden_scaffold.py — samples ~200 messages stratified by intent keyword heuristics.
produces golden_set/golden_set.csv with pre-filled suggested_intent for human review.

the human must:
  - verify / correct 'intent' column (change suggested_intent if wrong)
  - fill 'should_escalate' (yes/no)
  - fill 'escalation_reason' (short phrase or blank)
  - fill 'notes' (anything unusual about the example)
"""

import json
import random
import csv
import yaml
import re
import os

random.seed(99)

# --- load intents + keywords ---
with open("configs/intents.yaml", encoding="utf-8") as f:
    taxonomy = yaml.safe_load(f)["intents"]

INTENT_KEYWORDS = {i["name"]: i["keywords"] for i in taxonomy}
INTENT_NAMES = [i["name"] for i in taxonomy]

# --- load cleaned threads ---
with open("data/cleaned_threads.json", encoding="utf-8") as f:
    threads = json.load(f)

# only substantive, non-dm-redirect
pool = [t for t in threads if not t["is_dm_redirect"] and len(t["customer_msg"]) > 20]
print(f"eligible pool: {len(pool):,} threads")


def suggest_intent(text: str) -> tuple[str, float]:
    """keyword-match to suggest an intent. returns (intent_name, confidence 0-1)."""
    text_lower = text.lower()
    scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        scores[intent] = hits

    best = max(scores, key=scores.get)
    best_score = scores[best]
    if best_score == 0:
        return "unknown", 0.0

    # simple confidence: hits / total keywords for that intent
    confidence = best_score / len(INTENT_KEYWORDS[best])
    return best, round(confidence, 2)


# --- stratified sampling: target ~17-18 per intent (12 intents * ~17 = ~200) ---
TARGET_PER_INTENT = 17
TARGET_UNKNOWN = 6  # a few hard/ambiguous ones
buckets = {name: [] for name in INTENT_NAMES}
unknowns = []

random.shuffle(pool)
for t in pool:
    intent, conf = suggest_intent(t["customer_msg"])
    t["_suggested_intent"] = intent
    t["_confidence"] = conf
    if intent == "unknown":
        if len(unknowns) < TARGET_UNKNOWN:
            unknowns.append(t)
    elif len(buckets[intent]) < TARGET_PER_INTENT:
        buckets[intent].append(t)

    # stop early if all buckets full
    if all(len(v) >= TARGET_PER_INTENT for v in buckets.values()) and len(unknowns) >= TARGET_UNKNOWN:
        break

# flatten + shuffle
samples = []
for name in INTENT_NAMES:
    samples.extend(buckets[name])
samples.extend(unknowns)
random.shuffle(samples)

print(f"\nsampled: {len(samples)} examples")
print("distribution:")
from collections import Counter
dist = Counter(t["_suggested_intent"] for t in samples)
for k, v in dist.most_common():
    print(f"  {k:<35s} {v}")

# --- write csv ---
os.makedirs("golden_set", exist_ok=True)
out_path = "golden_set/golden_set.csv"

fieldnames = [
    "id",
    "customer_msg",
    "brand_reply",
    "suggested_intent",   # pre-filled by keyword heuristic — verify this
    "intent",             # FILL: correct if suggested_intent is wrong, else copy it
    "should_escalate",    # FILL: yes / no
    "escalation_reason",  # FILL: short phrase why (or blank if no)
    "notes",              # FILL: anything unusual, ambiguous, multi-intent etc.
    "confidence",         # heuristic confidence — low = more likely to need correction
]

with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for idx, t in enumerate(samples):
        writer.writerow({
            "id": idx + 1,
            "customer_msg": t["customer_msg"],
            "brand_reply": t["brand_reply"],
            "suggested_intent": t["_suggested_intent"],
            "intent": t["_suggested_intent"],  # user overwrites if wrong
            "should_escalate": "",
            "escalation_reason": "",
            "notes": "",
            "confidence": t["_confidence"],
        })

print(f"\nsaved to: {out_path}")
print(f"total rows: {len(samples)}")
