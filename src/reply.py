"""
src/reply.py — reply generation pipeline.

architecture:
  1. tfidf retriever — indexes all cleaned threads, finds top-k similar
     to any incoming customer message (cosine similarity)
  2. template replier — returns the single best-match historical reply
     (retrieval-only baseline, no generation)
  3. llm replier — drafts a new reply using retrieved context as few-shot
     grounding; falls back to template if no api key

run:
  python src/reply.py              # generates replies for golden set (subsample)
  python src/reply.py --full       # full golden set
"""

import os
import csv
import json
import pickle
import sys
import time
import random
from typing import List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

random.seed(42)

# --- paths ---
CLEANED_PATH    = "data/cleaned_threads.json"
GOLDEN_PATH     = "golden_set/golden_set.csv"
INDEX_PATH      = "outputs/reply_index.pkl"
OUT_PATH        = "outputs/reply_outputs.json"

TOP_K           = 5        # retrieved examples per query
SUBSAMPLE_N     = 30       # how many golden-set rows to run on by default
SLEEP_BETWEEN   = 0.3      # api call spacing


# ------------------------------------------------------------------
# retriever
# ------------------------------------------------------------------

class TfidfRetriever:
    """
    tfidf cosine-similarity retriever over the cleaned thread pool.
    indexes customer_msg; returns top-k (customer_msg, brand_reply) pairs.
    """

    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k
        self.vec   = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2),
            sublinear_tf=True, min_df=2,
        )
        self.threads: List[Dict] = []
        self.matrix = None

    def build(self, threads: List[Dict]):
        """index all threads — call once, then save."""
        # only substantive non-dm threads
        self.threads = [
            t for t in threads
            if not t.get("is_dm_redirect", False)
            and len(t.get("customer_msg", "")) > 10
        ]
        corpus = [t["customer_msg"] for t in self.threads]
        print(f"    building tfidf index over {len(corpus):,} threads ...")
        self.matrix = self.vec.fit_transform(corpus)
        print(f"    index shape: {self.matrix.shape}")
        return self

    def query(self, text: str, exclude_exact: bool = True) -> List[Dict]:
        """return top-k most similar threads for a given customer message.
        exclude_exact=True skips results that are identical to the query (self-match guard)."""
        q_vec = self.vec.transform([text])
        sims  = cosine_similarity(q_vec, self.matrix).flatten()
        # sort descending, take enough candidates to fill top_k after exclusion
        top_idx = np.argsort(sims)[::-1][: self.top_k * 3]
        results = []
        for idx in top_idx:
            t = self.threads[idx]
            # skip self-match
            if exclude_exact and t["customer_msg"].strip() == text.strip():
                continue
            results.append({
                "score":        float(sims[idx]),
                "customer_msg": t["customer_msg"],
                "brand_reply":  t["brand_reply"],
            })
            if len(results) == self.top_k:
                break
        return results

    def save(self, path: str = INDEX_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"    index saved to {path}")

    @staticmethod
    def load(path: str = INDEX_PATH) -> "TfidfRetriever":
        with open(path, "rb") as f:
            return pickle.load(f)


# ------------------------------------------------------------------
# reply generators
# ------------------------------------------------------------------

class TemplateReplier:
    """
    retrieval-only baseline — returns the single best-match historical reply.
    no llm needed; honest lower bound for reply quality.
    """

    def __init__(self, retriever: TfidfRetriever):
        self.retriever = retriever

    def draft(self, customer_msg: str, intent: str = "") -> Dict:
        hits = self.retriever.query(customer_msg)
        best = hits[0] if hits else {"brand_reply": "We'd be happy to help. Please DM us.", "score": 0.0}
        return {
            "draft":       best["brand_reply"],
            "method":      "retrieval_template",
            "top_match_score": best["score"],
            "retrieved":   hits,
        }


class LlmReplier:
    """
    llm-based reply drafter — uses retrieved context as grounding.
    falls back to template if no api key.
    """

    def __init__(self, retriever: TfidfRetriever, model: str = "gpt-4o-mini"):
        self.retriever = retriever
        self.model     = model
        api_key        = os.environ.get("OPENAI_API_KEY", "")
        self.use_mock  = not api_key
        if not self.use_mock:
            import openai
            self.client = openai.OpenAI(api_key=api_key)

    def _build_prompt(self, customer_msg: str, intent: str, hits: List[Dict]) -> str:
        examples = ""
        for i, h in enumerate(hits[:3], 1):
            examples += (
                f"example {i}:\n"
                f"  customer: {h['customer_msg'][:160]}\n"
                f"  reply:    {h['brand_reply'][:200]}\n\n"
            )
        return (
            f"you are an apple support twitter agent. draft a helpful, concise reply "
            f"to the customer message below.\n\n"
            f"intent: {intent}\n\n"
            f"similar past interactions (use these to ground your reply):\n{examples}"
            f"customer message:\n{customer_msg}\n\n"
            f"rules:\n"
            f"- reply in apple support's tone: professional, warm, action-oriented\n"
            f"- keep it under 240 characters where possible\n"
            f"- do not make up technical steps not supported by the examples\n"
            f"- do not start with 'Hi' or 'Hello'\n\n"
            f"draft reply:"
        )

    def draft(self, customer_msg: str, intent: str = "") -> Dict:
        hits = self.retriever.query(customer_msg)

        if self.use_mock:
            # mock: stitch together elements from top retrieved replies
            if hits and hits[0]["score"] > 0.05:
                base = hits[0]["brand_reply"]
                # light personalisation: prepend a contextual opener
                opener_map = {
                    "battery_issue":          "Battery life is important to us.",
                    "software_bug":           "We'd like to help with this update issue.",
                    "connectivity_issue":     "We can look into this connectivity issue with you.",
                    "account_and_password":   "Account access is critical.",
                    "payment_and_billing":    "We'd be happy to assist with your billing concern.",
                    "hardware_and_accessories": "We want to help with your hardware issue.",
                    "app_and_store_issue":    "We'd be glad to help with this app issue.",
                    "device_performance":     "We'd like to get your device running smoothly.",
                    "data_and_privacy":       "Data loss is a priority for us.",
                    "order_and_delivery":     "We want to help with your order.",
                    "feedback_and_complaint": "We always want to improve your experience.",
                    "product_and_feature_question": "Great question!",
                }
                opener = opener_map.get(intent, "We're here to help.")
                draft  = f"{opener} {base}"
            else:
                draft = "We're here to help. Please send us a DM with more details."
            method = "mock_llm"
        else:
            prompt = self._build_prompt(customer_msg, intent, hits)
            resp   = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=120,
            )
            draft  = resp.choices[0].message.content.strip()
            method = f"llm ({self.model})"
            time.sleep(SLEEP_BETWEEN)

        return {
            "draft":            draft,
            "method":           method,
            "top_match_score":  hits[0]["score"] if hits else 0.0,
            "retrieved":        hits,
        }


# ------------------------------------------------------------------
# main — build index + generate replies for golden set
# ------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    full = "--full" in sys.argv

    # 1. build or load retrieval index
    if os.path.exists(INDEX_PATH):
        print("[+] loading existing retrieval index ...")
        retriever = TfidfRetriever.load(INDEX_PATH)
        print(f"    loaded {len(retriever.threads):,} threads")
    else:
        print("[+] building retrieval index ...")
        with open(CLEANED_PATH, encoding="utf-8") as f:
            threads = json.load(f)
        retriever = TfidfRetriever(top_k=TOP_K).build(threads)
        retriever.save(INDEX_PATH)

    # 2. load golden set
    golden = list(csv.DictReader(open(GOLDEN_PATH, encoding="utf-8")))
    if not full:
        golden = random.sample(golden, min(SUBSAMPLE_N, len(golden)))
        print(f"[+] golden set subsampled to {len(golden)} rows")
    else:
        print(f"[+] golden set: {len(golden)} rows (full run)")

    # 3. init both repliers
    template_replier = TemplateReplier(retriever)
    llm_replier      = LlmReplier(retriever)
    mode = "mock_llm" if llm_replier.use_mock else "llm"
    print(f"[+] llm replier mode: {mode}")

    # 4. generate replies
    outputs = []
    print(f"\n[+] generating replies ...")
    for i, row in enumerate(golden):
        msg    = row["customer_msg"]
        intent = row["intent"].strip()

        t_out  = template_replier.draft(msg, intent)
        l_out  = llm_replier.draft(msg, intent)

        outputs.append({
            "id":           row["id"],
            "customer_msg": msg,
            "true_intent":  intent,
            "retrieval": {
                "top_k":    TOP_K,
                "top_score": t_out["top_match_score"],
                "examples": [
                    {"score": r["score"], "customer": r["customer_msg"][:100], "reply": r["brand_reply"][:150]}
                    for r in t_out["retrieved"]
                ],
            },
            "draft_template": t_out["draft"],
            "draft_llm":      l_out["draft"],
            "draft_method":   l_out["method"],
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/{len(golden)}] done")

    # 5. save
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    print(f"\n[+] saved {len(outputs)} reply pairs to {OUT_PATH}")

    # 6. show samples
    print("\n[+] sample outputs (3 examples):")
    for ex in outputs[:3]:
        print(f"\n  id: {ex['id']} | intent: {ex['true_intent']}")
        print(f"  customer:  {ex['customer_msg'][:100]}")
        print(f"  top-match: {ex['retrieval']['top_score']:.3f}")
        print(f"  template:  {ex['draft_template'][:120]}")
        print(f"  llm({mode}):{ex['draft_llm'][:120]}")

    print("\n[done] chunk 8 complete.")
