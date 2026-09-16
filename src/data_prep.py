"""
src/data_prep.py — load, clean, and filter the twitter customer-support dataset.
run this first to prepare data before any other scripts.

brand selected: AppleSupport (chunk 2 decision)
- 106,860 brand replies, 106,623 matched inbound threads
- avg reply length: 137 chars — detailed and informative
- ~99.8% thread completeness — best among top brands
"""

import os
import re
import sys
import json
import html
import pandas as pd

RAW_PATH = os.path.join("data", "raw", "twcs.csv")
BRAND_CONV_PATH = os.path.join("data", "brand_conversations.csv")
CLEAN_PATH = os.path.join("data", "clean_threads.json")
CLEANED_PATH = os.path.join("data", "cleaned_threads.json")

# chosen brand — locked after chunk 2 exploration
BRAND_ID = "AppleSupport"

# dm-redirect pattern — brand replies that just say "send us a dm"
_DM_RE = re.compile(r"\bdm\b|\bdirect message\b", re.IGNORECASE)


# ------------------------------------------------------------------
# text cleaning
# ------------------------------------------------------------------

def clean_text(text: str, is_brand: bool = False) -> str:
    """
    clean a single tweet:
    - decode html entities (&amp; &gt; &lt;)
    - strip urls
    - strip @mentions
    - strip hashtag symbols (keep the word)
    - collapse whitespace
    """
    if not isinstance(text, str):
        return ""

    # decode html entities first
    text = html.unescape(text)

    # remove urls
    text = re.sub(r"https?://\S+", "", text)

    # remove @mentions (twitter user refs and anonymised ids like @115854)
    text = re.sub(r"@\w+", "", text)

    # strip hashtag symbol but keep word
    text = re.sub(r"#(\w+)", r"\1", text)

    # collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_valid_thread(customer_clean: str, brand_clean: str) -> bool:
    """discard threads where either side is too short after cleaning."""
    if len(customer_clean) < 8:
        return False
    if len(brand_clean) < 8:
        return False
    return True


def is_dm_redirect(brand_clean: str) -> bool:
    """flag brand replies that are purely a dm redirect (no real content)."""
    # if the cleaned reply is short and mentions dm
    return len(brand_clean) < 80 and bool(_DM_RE.search(brand_clean))


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
    df = pd.read_csv(RAW_PATH, dtype=str)
    print(f"    total rows: {len(df):,}")
    if sample_n:
        df = df.sample(n=min(sample_n, len(df)), random_state=42)
        print(f"    subsampled to: {len(df):,} rows")
    return df


def sanity_check(df):
    """print basic stats and top brands."""
    print("\n[+] sanity check:")
    print(f"    shape: {df.shape}")
    print(f"    null counts:\n{df.isnull().sum().to_string()}")
    if "inbound" in df.columns and "author_id" in df.columns:
        brands = df[df["inbound"] == "False"]["author_id"].value_counts().head(20)
        print(f"\n    top 20 brands by reply count:\n{brands.to_string()}")


def filter_brand(df, brand_id):
    """filter dataset to a single brand's threads."""
    print(f"\n[+] filtering for brand: {brand_id}")
    brand_replies = df[df["author_id"] == brand_id]
    replied_to_ids = set(brand_replies["in_response_to_tweet_id"].dropna().tolist())
    inbound = df[df["tweet_id"].isin(replied_to_ids) & (df["inbound"] == "True")]
    brand_df = pd.concat([inbound, brand_replies]).drop_duplicates(subset="tweet_id")
    brand_df.to_csv(BRAND_CONV_PATH, index=False)
    print(f"    brand replies: {len(brand_replies):,}")
    print(f"    inbound tweets: {len(inbound):,}")
    print(f"    saved to: {BRAND_CONV_PATH}")
    return brand_df


def build_threads(df, brand_id):
    """reconstruct conversation threads as list of dicts."""
    print("\n[+] building conversation threads ...")
    brand_replies = df[df["author_id"] == brand_id].copy()
    brand_replies = brand_replies[brand_replies["in_response_to_tweet_id"].notna()]
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


def run_cleaning(threads):
    """
    apply text cleaning to all threads.
    returns list of cleaned thread dicts, drops invalid ones.
    """
    print("\n[+] cleaning threads ...")
    cleaned = []
    dropped = 0
    dm_flagged = 0

    for t in threads:
        c_clean = clean_text(t["customer_msg"])
        b_clean = clean_text(t["brand_reply"], is_brand=True)

        if not is_valid_thread(c_clean, b_clean):
            dropped += 1
            continue

        cleaned.append({
            "customer_tweet_id": t["customer_tweet_id"],
            "brand_tweet_id": t["brand_tweet_id"],
            "customer_msg_raw": t["customer_msg"],
            "brand_reply_raw": t["brand_reply"],
            "customer_msg": c_clean,
            "brand_reply": b_clean,
            "is_dm_redirect": is_dm_redirect(b_clean),
            "created_at": t.get("created_at", ""),
        })
        if cleaned[-1]["is_dm_redirect"]:
            dm_flagged += 1

    print(f"    threads before cleaning: {len(threads):,}")
    print(f"    dropped (too short):     {dropped:,}")
    print(f"    kept:                    {len(cleaned):,}")
    print(f"    dm-redirect replies:     {dm_flagged:,} ({100*dm_flagged/len(cleaned):.1f}%)")

    with open(CLEANED_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    print(f"    saved to: {CLEANED_PATH}")
    return cleaned


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)

    if not check_raw_data():
        sys.exit(1)

    # if clean_threads.json already exists, skip rebuild and go straight to cleaning
    if os.path.exists(CLEAN_PATH):
        print(f"[+] {CLEAN_PATH} already exists — loading for cleaning step")
        with open(CLEAN_PATH, encoding="utf-8") as f:
            threads = json.load(f)
        print(f"    loaded {len(threads):,} threads")
    else:
        df = load_raw()
        sanity_check(df)
        filter_brand(df, BRAND_ID)
        threads = build_threads(df, BRAND_ID)

    cleaned = run_cleaning(threads)

    # print a few cleaned examples
    print("\n[+] sample cleaned threads:")
    for i, t in enumerate(cleaned[:3]):
        print(f"\n  [{i+1}] customer: {t['customer_msg'][:120]}")
        print(f"       brand:    {t['brand_reply'][:120]}")
        print(f"       dm_flag:  {t['is_dm_redirect']}")

    print("\n[done] data_prep complete.")
