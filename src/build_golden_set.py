"""
build_golden_set.py — samples ~250 messages stratified across intents,
pre-fills suggested intent + escalation labels using keyword heuristics.

output: golden_set/golden_set.csv
the user must review and correct every row before trusting the labels.
"""

import json
import csv
import os
import re
import random
import yaml

random.seed(99)

CLEANED_PATH = "data/cleaned_threads.json"
INTENTS_PATH = "configs/intents.yaml"
OUT_PATH = "golden_set/golden_set.csv"
TARGET_PER_INTENT = 20   # ~240 total across 12 intents
MAX_TOTAL = 250

# escalation heuristics — messages that likely need a human
ESCALATION_KEYWORDS = [
    r"\blegal\b", r"\blawyer\b", r"\bsue\b", r"\bfraud\b", r"\bscam\b",
    r"\brefund\b", r"\bunauthorized\b", r"\bstolen\b", r"\bhacked\b",
    r"\bidentity\b", r"\bcharged.*without\b", r"\bcharge.*wrong\b",
    r"\baccount.*locked\b", r"\blocked.*account\b",
    r"\bdata.*lost\b", r"\ball.*data.*gone\b", r"\bpermanent\b",
    r"\bvery angry\b", r"\bfurious\b", r"\bextremely\b.*\bfrustrat\b",
    r"\bprepare.*rude\b", r"\bdisastrous\b", r"\bshame\b",
    r"\burgent\b", r"\bemergency\b",
]
_ESC_RE = [re.compile(p, re.IGNORECASE) for p in ESCALATION_KEYWORDS]

# load intent taxonomy
with open(INTENTS_PATH, encoding="utf-8") as f:
    taxonomy = yaml.safe_load(f)["intents"]

intent_names = [i["name"] for i in taxonomy]

# build per-intent keyword regex
intent_patterns = {}
for intent in taxonomy:
    kws = intent["keywords"]
    pattern = re.compile("|".join(r"\b" + re.escape(k) + r"\b" for k in kws), re.IGNORECASE)
    intent_patterns[intent["name"]] = pattern


def suggest_intent(text: str) -> str:
    """score text against each intent's keywords; return best match."""
    scores = {}
    for name, pattern in intent_patterns.items():
        hits = len(pattern.findall(text))
        if hits:
            scores[name] = hits
    if not scores:
        return "product_and_feature_question"  # fallback
    return max(scores, key=scores.get)


def suggest_escalation(text: str) -> tuple:
    """return (should_escalate bool, reason str)."""
    for pat in _ESC_RE:
        m = pat.search(text)
        if m:
            return True, f"keyword match: '{m.group()}'"
    return False, ""


# load cleaned threads
print("[+] loading cleaned threads...")
with open(CLEANED_PATH, encoding="utf-8") as f:
    threads = json.load(f)

# only use non-dm-redirect, substantive customer messages
pool = [t for t in threads if not t["is_dm_redirect"] and len(t["customer_msg"]) > 20]
print(f"    usable threads: {len(pool):,}")

# stratified sampling: bucket threads by suggested intent, sample per bucket
buckets = {name: [] for name in intent_names}
for t in pool:
    intent = suggest_intent(t["customer_msg"])
    buckets[intent].append(t)

print("\n[+] bucket sizes (before sampling):")
for name, items in buckets.items():
    print(f"    {name:<35s}: {len(items):,}")

# sample per intent
sampled = []
for name in intent_names:
    bucket = buckets[name]
    n = min(TARGET_PER_INTENT, len(bucket))
    chosen = random.sample(bucket, n)
    for t in chosen:
        esc, esc_reason = suggest_escalation(t["customer_msg"])
        sampled.append({
            "id": "",                          # filled below
            "customer_msg": t["customer_msg"],
            "brand_reply_ref": t["brand_reply"],
            "suggested_intent": name,
            "intent": name,                    # user should correct this
            "should_escalate": "yes" if esc else "no",
            "escalation_reason": esc_reason,
            "notes": "",
        })

# shuffle to remove ordering bias, then assign ids
random.shuffle(sampled)
for i, row in enumerate(sampled):
    row["id"] = f"gs_{i+1:03d}"

print(f"\n[+] total sampled: {len(sampled)}")

# save csv
os.makedirs("golden_set", exist_ok=True)
cols = ["id", "customer_msg", "brand_reply_ref", "suggested_intent",
        "intent", "should_escalate", "escalation_reason", "notes"]

with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    writer.writerows(sampled)

print(f"[+] saved to {OUT_PATH}")
print()
print("[!] next step: open golden_set/golden_set.csv and review every row.")
print("    - correct 'intent' if the suggested one is wrong")
print("    - set 'should_escalate' to yes/no based on your judgment")
print("    - fill 'escalation_reason' when yes")
print("    - use 'notes' for anything noteworthy about the example")
print("    labels MUST be yours — the suggestions are just a starting point.")
