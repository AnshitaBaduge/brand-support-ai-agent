"""
data_prep.py — load, clean, and filter the twitter customer-support dataset.
run this first to prepare data before any other scripts.

brand selected: AppleSupport (chunk 2 decision)
- 106,860 brand replies, 106,623 matched inbound threads
- avg reply length: 137 chars — detailed and informative
- ~99.8% thread completeness — best among top brands
"""

import os
import sys
import json
import pandas as pd

RAW_PATH = os.path.join("data", "raw", "twcs.csv")
BRAND_CONV_PATH = os.path.join("data", "brand_conversations.csv")
CLEAN_PATH = os.path.join("data", "clean_threads.json")

# chosen brand — locked after chunk 2 exploration
BRAND_ID = "AppleSupport"


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def check_raw_data():
    """check if raw data exists, print download instructions if not."""
    if not os.path.exists(RAW_PATH):
        print("\n[!] raw data not found at:", RAW_PATH)
        print("\nhow to download:")
        print("  1. go to: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
        print("  2. download 'twcs.csv' and place it at: data/raw/twcs.csv")
        print("  (or) set up kaggle api key and run:")
        print("       kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw --unzip")
        return False
    return True


def load_raw(sample_n=None):
    """load the raw csv, optional subsample for speed."""
    print(f"[+] loading raw data from {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH, dtype=str)  # keep all as str to avoid type issues
    print(f"    total rows: {len(df):,}")
    print(f"    columns: {list(df.columns)}")
    if sample_n:
        df = df.sample(n=min(sample_n, len(df)), random_state=42)
        print(f"    subsampled to: {len(df):,} rows")
    return df


def sanity_check(df):
    """print basic stats and a few sample rows."""
    print("\n[+] sanity check:")
    print(f"    shape: {df.shape}")
    print(f"    null counts:\n{df.isnull().sum().to_string()}")

    # inbound column tells us who sent the tweet
    if "inbound" in df.columns:
        vc = df["inbound"].value_counts()
        print(f"\n    inbound (customer) vs outbound (brand):\n{vc.to_string()}")

    # top brands by tweet count (outbound tweets, i.e. brand replies)
    if "author_id" in df.columns and "inbound" in df.columns:
        brands = df[df["inbound"] == "False"]["author_id"].value_counts().head(20)
        print(f"\n    top 20 brands by reply count:\n{brands.to_string()}")

    print("\n    sample rows:")
    print(df.head(3).to_string())


def filter_brand(df, brand_id):
    """filter dataset to a single brand's threads."""
    print(f"\n[+] filtering for brand: {brand_id}")

    # get all tweet ids from/to this brand
    brand_replies = df[df["author_id"] == brand_id]
    print(f"    brand reply tweets: {len(brand_replies):,}")

    # find inbound tweets that the brand replied to
    replied_to_ids = set(brand_replies["in_response_to_tweet_id"].dropna().tolist())
    inbound = df[df["tweet_id"].isin(replied_to_ids) & (df["inbound"] == "True")]
    print(f"    inbound customer tweets with brand replies: {len(inbound):,}")

    # combine and save
    brand_df = pd.concat([inbound, brand_replies]).drop_duplicates(subset="tweet_id")
    brand_df.to_csv(BRAND_CONV_PATH, index=False)
    print(f"    saved to: {BRAND_CONV_PATH}")
    return brand_df


def build_threads(df, brand_id):
    """reconstruct conversation threads as list of {customer_msg, brand_reply} pairs."""
    print("\n[+] building conversation threads ...")

    # brand replies with a clear response target
    brand_replies = df[df["author_id"] == brand_id].copy()
    brand_replies = brand_replies[brand_replies["in_response_to_tweet_id"].notna()]

    # build lookup: tweet_id -> text
    id2text = dict(zip(df["tweet_id"], df["text"]))

    threads = []
    for _, row in brand_replies.iterrows():
        customer_id = row.get("in_response_to_tweet_id")
        customer_text = id2text.get(str(customer_id), "")
        if not customer_text:
            continue
        threads.append({
            "customer_tweet_id": str(customer_id),
            "brand_tweet_id": str(row["tweet_id"]),
            "customer_msg": str(customer_text),
            "brand_reply": str(row["text"]),
            "created_at": str(row.get("created_at", "")),
        })

    print(f"    total threads built: {len(threads):,}")
    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    with open(CLEAN_PATH, "w", encoding="utf-8") as f:
        json.dump(threads, f, indent=2, ensure_ascii=False)
    print(f"    saved to: {CLEAN_PATH}")
    return threads


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)

    if not check_raw_data():
        sys.exit(1)

    df = load_raw()
    sanity_check(df)

    # brand will be set after chunk 2 exploration
    # for now just run the sanity check
    print("\n[!] brand not yet selected — run sanity_check to pick one, then set BRAND_ID below")
    print("    next step: open notebooks/explore.ipynb or re-run with BRAND_ID set")
