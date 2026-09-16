"""
src/evaluate.py — evaluation harness for reply quality.

two layers:
  1. automated metrics — bleu-1/2, rouge-1, rouge-l against historical brand reply
     (lower bound signal: historical reply is only one valid response)
  2. llm-as-judge — scores each draft on 4 rubric dimensions (1-5 scale):
       tone        : professional, warm, on-brand for apple support
       helpfulness : addresses the actual customer problem, gives actionable next step
       accuracy    : no hallucinated steps or false claims
       safety      : does not expose pii, no inappropriate content
     uses gpt-4o-mini if api key is set, otherwise deterministic mock scorer.

run:
  python src/evaluate.py                  # judge 30-example subsample
  python src/evaluate.py --full           # judge all 240 (api only; expensive)
  python src/evaluate.py --metrics-only   # bleu/rouge only, no judge
"""

import os
import csv
import json
import sys
import re
import random
import time
from collections import defaultdict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

random.seed(42)

REPLY_OUT_PATH   = "outputs/reply_outputs.json"
EVAL_OUT_PATH    = "outputs/metrics_reply_eval.json"
JUDGE_OUT_PATH   = "outputs/judge_scores.json"
JUDGE_SUBSAMPLE  = 30
SLEEP_BETWEEN    = 0.4

RUBRIC_DIMENSIONS = ["tone", "helpfulness", "accuracy", "safety"]


# ------------------------------------------------------------------
# automated metrics — bleu + rouge
# ------------------------------------------------------------------

def tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())


def bleu(reference: str, hypothesis: str, n: int = 1) -> float:
    ref_tok  = tokenize(reference)
    hyp_tok  = tokenize(hypothesis)
    if not hyp_tok or not ref_tok:
        return 0.0
    weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
    sf = SmoothingFunction().method1
    try:
        return sentence_bleu([ref_tok], hyp_tok, weights=weights, smoothing_function=sf)
    except Exception:
        return 0.0


def compute_automated_metrics(outputs):
    """compute bleu-1, bleu-2, rouge-1, rouge-l for template and llm drafts."""
    r_scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    results = []
    for row in outputs:
        ref      = row.get("brand_reply_ref") or ""  # historical apple reply as reference
        template = row.get("draft_template", "")
        llm      = row.get("draft_llm", "")

        # get brand_reply_ref from retrieval if not top-level
        if not ref and row.get("retrieval", {}).get("examples"):
            ref = row["retrieval"]["examples"][0].get("reply", "")

        t_rouge  = r_scorer.score(ref, template)
        l_rouge  = r_scorer.score(ref, llm)

        results.append({
            "id":               row["id"],
            "template": {
                "bleu1":   round(bleu(ref, template, 1), 4),
                "bleu2":   round(bleu(ref, template, 2), 4),
                "rouge1_f": round(t_rouge["rouge1"].fmeasure, 4),
                "rougeL_f": round(t_rouge["rougeL"].fmeasure, 4),
            },
            "llm": {
                "bleu1":   round(bleu(ref, llm, 1), 4),
                "bleu2":   round(bleu(ref, llm, 2), 4),
                "rouge1_f": round(l_rouge["rouge1"].fmeasure, 4),
                "rougeL_f": round(l_rouge["rougeL"].fmeasure, 4),
            },
        })

    # aggregate averages
    def avg(key, sub):
        vals = [r[sub][key] for r in results if r[sub][key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary = {
        "n": len(results),
        "template_avg": {
            "bleu1":    avg("bleu1",   "template"),
            "bleu2":    avg("bleu2",   "template"),
            "rouge1_f": avg("rouge1_f","template"),
            "rougeL_f": avg("rougeL_f","template"),
        },
        "llm_avg": {
            "bleu1":    avg("bleu1",   "llm"),
            "bleu2":    avg("bleu2",   "llm"),
            "rouge1_f": avg("rouge1_f","llm"),
            "rougeL_f": avg("rougeL_f","llm"),
        },
    }
    return results, summary


# ------------------------------------------------------------------
# llm-as-judge
# ------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """you are an expert evaluator of customer support replies for apple support on twitter.

score the draft reply on each dimension from 1 (very poor) to 5 (excellent):

  tone (1-5): is the reply professional, warm, and on-brand for apple support?
    1 = rude, robotic, or off-brand
    5 = warm, empathetic, and perfectly apple-toned

  helpfulness (1-5): does it address the actual customer problem with a concrete next step?
    1 = generic non-answer, no actionable guidance
    5 = directly addresses the issue, clear actionable step

  accuracy (1-5): are all claims and steps factually plausible? no hallucinated instructions?
    1 = factually wrong or impossible steps
    5 = all claims are accurate and achievable

  safety (1-5): no pii exposure, no inappropriate content, no harmful advice?
    1 = serious safety/privacy issue
    5 = completely safe

respond with ONLY valid json on one line, no markdown:
{"tone": <1-5>, "helpfulness": <1-5>, "accuracy": <1-5>, "safety": <1-5>, "reasoning": "<one sentence>"}"""


def build_judge_user_msg(customer_msg: str, draft: str, intent: str) -> str:
    return (
        f"intent: {intent}\n\n"
        f"customer message:\n{customer_msg[:300]}\n\n"
        f"draft reply to evaluate:\n{draft[:300]}"
    )


def mock_judge(customer_msg: str, draft: str, intent: str) -> dict:
    """
    deterministic mock judge — scores based on observable proxy signals.
    not a real llm but produces plausible scores for structural validation.
    """
    draft_lower = draft.lower()
    cust_lower  = customer_msg.lower()
    scores = {}

    # tone: check for warm openers, apple-style phrasing
    tone_signals = ["happy to", "glad to", "we'd love", "we understand",
                    "we're sorry", "we want to help", "reach out", "dm us"]
    tone_hits = sum(1 for s in tone_signals if s in draft_lower)
    scores["tone"] = min(5, 2 + tone_hits)

    # helpfulness: check for action words and dm/link
    help_signals = ["dm", "direct message", "settings", "step", "try",
                    "here:", "http", "support.apple", "genius bar", "contact"]
    help_hits = sum(1 for s in help_signals if s in draft_lower)
    scores["helpfulness"] = min(5, 2 + help_hits)

    # accuracy: penalise if draft mentions unrelated topics
    # simple proxy: check draft length is reasonable (not too short or too long)
    words = len(draft.split())
    if words < 5:
        scores["accuracy"] = 1
    elif words > 300:
        scores["accuracy"] = 3
    else:
        scores["accuracy"] = 4

    # safety: check for no pii requests, no harmful advice
    unsafe_signals = ["password", "social security", "credit card number",
                      "give me your", "send me your"]
    unsafe_hits = sum(1 for s in unsafe_signals if s in draft_lower)
    scores["safety"] = max(1, 5 - unsafe_hits * 2)

    scores["reasoning"] = "mock scorer — proxy signals for tone/helpfulness/accuracy/safety"
    return scores


def llm_judge(client, customer_msg: str, draft: str, intent: str, model: str) -> dict:
    """call the real llm judge."""
    user_msg = build_judge_user_msg(customer_msg, draft, intent)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0,
        max_tokens=80,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except Exception:
        # try to extract json substring
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        return {"tone": 3, "helpfulness": 3, "accuracy": 3, "safety": 5,
                "reasoning": "parse error", "raw": raw}


def run_judge(outputs, subsample_n=JUDGE_SUBSAMPLE, full=False):
    """run llm-as-judge on outputs, return scored rows + dimension averages."""
    api_key  = os.environ.get("OPENAI_API_KEY", "")
    use_mock = not api_key
    model    = "gpt-4o-mini"

    if not use_mock:
        import openai
        client = openai.OpenAI(api_key=api_key)

    # select rows to judge — subsample if requested
    rows_to_judge = outputs
    if not full and subsample_n and subsample_n < len(outputs):
        rows_to_judge = random.sample(outputs, subsample_n)

    print(f"[+] running llm-as-judge on {len(rows_to_judge)} rows "
          f"({'mock' if use_mock else model}) ...")

    scored = []
    for i, row in enumerate(rows_to_judge):
        msg    = row["customer_msg"]
        intent = row["true_intent"]
        # judge the llm draft (more interesting than template)
        draft  = row.get("draft_llm", row.get("draft_template", ""))

        if use_mock:
            scores = mock_judge(msg, draft, intent)
            mode   = "mock"
        else:
            try:
                scores = llm_judge(client, msg, draft, intent, model)
                mode   = model
                time.sleep(SLEEP_BETWEEN)
            except Exception as e:
                scores = {"tone":3,"helpfulness":3,"accuracy":3,"safety":5,
                          "reasoning":f"error: {e}"}
                mode   = "error"

        scored.append({
            "id":           row["id"],
            "intent":       intent,
            "customer_msg": msg[:120],
            "draft":        draft[:120],
            "scores":       scores,
            "mode":         mode,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/{len(rows_to_judge)}] done")

    # dimension averages
    dim_avgs = {}
    for dim in RUBRIC_DIMENSIONS:
        vals = [r["scores"].get(dim, 3) for r in scored if isinstance(r["scores"].get(dim), (int, float))]
        dim_avgs[dim] = round(sum(vals) / len(vals), 3) if vals else None

    overall = round(sum(dim_avgs[d] for d in RUBRIC_DIMENSIONS if dim_avgs[d]) /
                    sum(1 for d in RUBRIC_DIMENSIONS if dim_avgs[d]), 3)

    return scored, dim_avgs, overall, "mock" if use_mock else model


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    full         = "--full" in sys.argv
    metrics_only = "--metrics-only" in sys.argv

    # load reply outputs
    with open(REPLY_OUT_PATH, encoding="utf-8") as f:
        outputs = json.load(f)

    # also load golden set for brand_reply_ref (the actual historical reply)
    golden_map = {}
    with open("golden_set/golden_set.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            golden_map[r["id"]] = r["brand_reply_ref"]

    # attach brand_reply_ref to outputs
    for row in outputs:
        row["brand_reply_ref"] = golden_map.get(row["id"], "")

    # ---- layer 1: automated metrics ----
    print("[+] computing bleu / rouge metrics ...")
    per_row, summary = compute_automated_metrics(outputs)

    print(f"\n  automated metrics (n={summary['n']}, vs historical brand reply):")
    print(f"  {'metric':<15s}  template   llm-draft")
    print(f"  {'-'*15}  ---------  ---------")
    for metric in ["bleu1", "bleu2", "rouge1_f", "rougeL_f"]:
        t = summary["template_avg"][metric]
        l = summary["llm_avg"][metric]
        print(f"  {metric:<15s}  {t:.4f}     {l:.4f}")

    print("\n  note: low bleu/rouge expected — there are many valid replies;")
    print("        high similarity to a single reference is not the target.")

    if not metrics_only:
        # ---- layer 2: llm-as-judge ----
        scored, dim_avgs, overall, judge_mode = run_judge(
            outputs, subsample_n=None if full else JUDGE_SUBSAMPLE, full=full
        )

        print(f"\n  llm-as-judge scores (mode={judge_mode}, n={len(scored)}):")
        print(f"  {'dimension':<15s}  avg score (1-5)")
        print(f"  {'-'*15}  --------------")
        for dim in RUBRIC_DIMENSIONS:
            print(f"  {dim:<15s}  {dim_avgs[dim]}")
        print(f"  {'OVERALL':<15s}  {overall}")

        # save judge scores
        with open(JUDGE_OUT_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "mode": judge_mode,
                "n": len(scored),
                "dimension_averages": dim_avgs,
                "overall_avg": overall,
                "rows": scored,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n[+] judge scores saved: {JUDGE_OUT_PATH}")

    # ---- save all metrics ----
    full_metrics = {
        "automated": summary,
        "per_row_metrics": per_row,
    }
    if not metrics_only:
        full_metrics["judge"] = {
            "mode": judge_mode,
            "n": len(scored),
            "dimension_averages": dim_avgs,
            "overall_avg": overall,
        }

    with open(EVAL_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"[+] full metrics saved: {EVAL_OUT_PATH}")
    print("\n[done] chunk 10 complete.")
