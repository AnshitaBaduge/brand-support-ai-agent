"""
sample_for_intents.py — samples 300 customer messages and saves to json for intent analysis.
run once during chunk 4.
"""
import json, random

random.seed(42)

with open("data/cleaned_threads.json", encoding="utf-8") as f:
    threads = json.load(f)

# only non-dm-redirect threads (we want substantive exchanges)
substantive = [t for t in threads if not t["is_dm_redirect"]]
print(f"substantive threads (non-dm-redirect): {len(substantive):,}")

sample = random.sample(substantive, min(300, len(substantive)))

out = []
for i, t in enumerate(sample):
    out.append({
        "id": i,
        "customer_msg": t["customer_msg"],
        "brand_reply": t["brand_reply"],
    })

with open("notebooks/intent_sample.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"saved {len(out)} samples to notebooks/intent_sample.json")
