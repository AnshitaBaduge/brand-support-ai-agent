"""
src/human_judge_agreement.py — computes human vs judge agreement.

reads golden_set/human_eval.csv (filled in by user),
computes cohen's kappa + % agreement per dimension and overall,
saves outputs/metrics_human_judge_agreement.json.

run after filling in human_eval.csv:
  python src/human_judge_agreement.py
"""

import csv
import json
import os
import sys
from collections import defaultdict

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    print("run: pip install scikit-learn")
    sys.exit(1)

HUMAN_EVAL_PATH = "golden_set/human_eval.csv"
OUT_PATH        = "outputs/metrics_human_judge_agreement.json"
DIMENSIONS      = ["tone", "helpfulness", "accuracy", "safety"]


def load_scores(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    # validate no blanks
    blanks = []
    for r in rows:
        for dim in DIMENSIONS:
            if not r[f"human_{dim}"].strip():
                blanks.append((r["id"], dim))
    if blanks:
        print(f"[!] {len(blanks)} missing human scores:")
        for id_, dim in blanks[:10]:
            print(f"    {id_}: human_{dim} is blank")
        sys.exit(1)
    return rows


def compute_agreement(rows):
    results = {}
    human_all, judge_all = [], []

    for dim in DIMENSIONS:
        human = [int(r[f"human_{dim}"]) for r in rows]
        judge = [int(r[f"judge_{dim}"]) for r in rows]

        # % exact agreement
        exact = sum(h == j for h, j in zip(human, judge)) / len(human)
        # % within-1 agreement
        within1 = sum(abs(h - j) <= 1 for h, j in zip(human, judge)) / len(human)

        # cohen's kappa (treats scores as categories)
        try:
            kappa = cohen_kappa_score(human, judge)
        except Exception:
            kappa = None

        results[dim] = {
            "kappa":        round(kappa, 4) if kappa is not None else None,
            "pct_exact":    round(exact, 4),
            "pct_within1":  round(within1, 4),
            "human_avg":    round(sum(human) / len(human), 3),
            "judge_avg":    round(sum(judge) / len(judge), 3),
            "n":            len(human),
        }

        human_all.extend(human)
        judge_all.extend(judge)

        kappa_str = f"{kappa:.3f}" if kappa is not None else "n/a"
        print(f"  {dim:<15s}  kappa={kappa_str:<7}  "
              f"exact={exact:.1%}  within1={within1:.1%}  "
              f"human_avg={sum(human)/len(human):.2f}  judge_avg={sum(judge)/len(judge):.2f}")

    # overall (treat all dims pooled)
    try:
        overall_kappa = cohen_kappa_score(human_all, judge_all)
    except Exception:
        overall_kappa = None

    overall_exact   = sum(h == j for h, j in zip(human_all, judge_all)) / len(human_all)
    overall_within1 = sum(abs(h-j) <= 1 for h, j in zip(human_all, judge_all)) / len(human_all)

    results["overall"] = {
        "kappa":       round(overall_kappa, 4) if overall_kappa is not None else None,
        "pct_exact":   round(overall_exact, 4),
        "pct_within1": round(overall_within1, 4),
        "n":           len(human_all),
    }

    overall_kappa_str = f"{overall_kappa:.3f}" if overall_kappa is not None else "n/a"
    print(f"\n  {'OVERALL':<15s}  kappa={overall_kappa_str:<7}  "
          f"exact={overall_exact:.1%}  within1={overall_within1:.1%}")

    return results


def kappa_interpretation(k):
    if k is None:
        return "n/a"
    if k < 0:
        return "worse than chance"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


if __name__ == "__main__":
    rows = load_scores(HUMAN_EVAL_PATH)
    print(f"[+] loaded {len(rows)} human-scored rows")
    print(f"\n  {'dimension':<15s}  {'kappa':<10}  exact     within1   h_avg  j_avg")
    print(f"  {'-'*15}  {'-'*10}  --------  --------  -----  -----")

    results = compute_agreement(rows)

    # interpret kappa
    print("\n  kappa interpretation:")
    for dim in DIMENSIONS + ["overall"]:
        k = results[dim]["kappa"]
        print(f"    {dim:<15s}: {k}  ({kappa_interpretation(k)})")

    os.makedirs("outputs", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] saved to {OUT_PATH}")
    print("[done] chunk 11 complete.")
