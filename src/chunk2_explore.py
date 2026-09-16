"""
chunk2_explore.py — brand selection and thread building for AppleSupport.
run once: python src/chunk2_explore.py
"""

import pandas as pd
import json
import os

BRAND = "AppleSupport"
RAW_PATH = os.path.join("data", "raw", "twcs.csv")
BRAND_CONV_PATH = os.path.join("data", "brand_conversations.csv")
THREADS_PATH = os.path.join("data", "clean_threads.json")

print(f"[+] loading raw data...")
df = pd.read_csv(RAW_PATH, dtype=str)
print(f"    total rows: {len(df):,}")

# --- filter brand ---
print(f"\n[+] filtering for brand: {BRAND}")
brand_replies = df[df["author_id"] == BRAND].copy()
replied_ids = set(brand_replies["in_response_to_tweet_id"].dropna().tolist())
inbound = df[df["tweet_id"].isin(replied_ids) & (df["inbound"] == "True")].copy()

brand_df = pd.concat([inbound, brand_replies]).drop_duplicates(subset="tweet_id")
brand_df.to_csv(BRAND_CONV_PATH, index=False)
print(f"    brand replies:           {len(brand_replies):,}")
print(f"    inbound customer tweets: {len(inbound):,}")
print(f"    total rows saved:        {len(brand_df):,}")
print(f"    saved to: {BRAND_CONV_PATH}")

# --- build threads ---
print("\n[+] building conversation threads...")
id2text = dict(zip(df["tweet_id"], df["text"]))

threads = []
for _, row in brand_replies.iterrows():
    cid = str(row.get("in_response_to_tweet_id") or "")
    if not cid or cid == "nan":
        continue
    ctext = id2text.get(cid, "")
    if not ctext:
        continue
    threads.append({
        "customer_tweet_id": cid,
        "brand_tweet_id": str(row["tweet_id"]),
        "customer_msg": str(ctext),
        "brand_reply": str(row["text"]),
        "created_at": str(row.get("created_at", "")),
    })

with open(THREADS_PATH, "w", encoding="utf-8") as f:
    json.dump(threads, f, indent=2, ensure_ascii=False)
print(f"    threads built: {len(threads):,}")
print(f"    saved to: {THREADS_PATH}")

# --- stats ---
customer_lens = [len(t["customer_msg"]) for t in threads]
brand_lens = [len(t["brand_reply"]) for t in threads]
print(f"\n[+] thread stats:")
print(f"    avg customer msg length: {sum(customer_lens)/len(customer_lens):.0f} chars")
print(f"    avg brand reply length:  {sum(brand_lens)/len(brand_lens):.0f} chars")

# --- sample threads ---
print("\n[+] sample threads (first 3):")
for i, t in enumerate(threads[:3]):
    print(f"\n  [{i+1}] customer: {t['customer_msg'][:120]}")
    print(f"       brand:    {t['brand_reply'][:120]}")

print("\n[done] chunk 2 complete.")
