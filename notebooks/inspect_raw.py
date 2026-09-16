"""
inspect_samples.py — saves 50 raw samples to a json for review without encoding issues.
"""
import json, re

with open("data/clean_threads.json", encoding="utf-8") as f:
    threads = json.load(f)

samples = []
for t in threads[:50]:
    samples.append({
        "c": t["customer_msg"],
        "b": t["brand_reply"],
    })

with open("notebooks/samples.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

print("saved 50 samples to notebooks/samples.json")

# also show text stats
c_lens = [len(t["customer_msg"]) for t in threads]
b_lens = [len(t["brand_reply"]) for t in threads]
urls = sum(1 for t in threads if "http" in t["customer_msg"])
mentions = sum(1 for t in threads if "@" in t["customer_msg"])
html_ents = sum(1 for t in threads if "&amp;" in t["brand_reply"] or "&gt;" in t["brand_reply"])
empty_c = sum(1 for t in threads if len(t["customer_msg"].strip()) < 10)

print(f"total threads:          {len(threads):,}")
print(f"avg customer msg len:   {sum(c_lens)/len(c_lens):.0f} chars")
print(f"avg brand reply len:    {sum(b_lens)/len(b_lens):.0f} chars")
print(f"customer msgs with url: {urls:,} ({100*urls/len(threads):.1f}%)")
print(f"customer msgs with @:   {mentions:,} ({100*mentions/len(threads):.1f}%)")
print(f"brand replies w/ html:  {html_ents:,} ({100*html_ents/len(threads):.1f}%)")
print(f"near-empty customer (<10 chars): {empty_c:,}")
