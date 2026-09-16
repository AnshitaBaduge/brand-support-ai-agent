"""
src/escalation.py — escalation router.

decides whether a customer message should be auto-handled or escalated to a human.

two approaches:
  1. rule-based — keyword pattern matching (interpretable, no training needed)
  2. tfidf + logreg — trained on golden set labels with class-weight balancing
     (5-fold stratified cv to handle the 37 yes / 203 no imbalance)

run:
  python src/escalation.py
"""

import os
import re
import csv
import json
import pickle
import numpy as np
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score
)

GOLDEN_PATH   = "golden_set/golden_set.csv"
MODEL_PATH    = "outputs/escalation_model.pkl"
METRICS_PATH  = "outputs/metrics_escalation.json"

# escalation signal patterns (human-readable, interpretable)
ESCALATION_PATTERNS = [
    # billing / fraud
    (r"\bunauthorized\b",           "unauthorized charge"),
    (r"\bfraud\b",                  "fraud"),
    (r"\bscam\b",                   "scam"),
    (r"\brefund\b",                 "refund request"),
    (r"\bcharg\w+\s+without\b",     "charged without consent"),
    (r"\bwrong(ful)?\s+charg\b",    "wrongful charge"),
    # account security
    (r"\bhack(ed)?\b",              "account hacked"),
    (r"\bstolen\b",                 "stolen device/account"),
    (r"\bicloud.*lock\b",           "icloud locked"),
    (r"\baccount.*lock\b",          "account locked"),
    # data loss
    (r"\ball.*data.*gone\b",        "all data lost"),
    (r"\bpermanent.*loss\b",        "permanent data loss"),
    (r"\b\d{3,}\s*photos.*lost\b",  "large photo loss"),
    (r"\bdata.*delete\b",           "data deleted"),
    # physical safety
    (r"\bsmok(e|ing)\b",            "device smoking"),
    (r"\bswollen\b",                "swollen battery"),
    (r"\bmelting\b",                "device melting"),
    (r"\bbatt.*hot\b",              "battery overheating"),
    (r"\bexplo\w+\b",               "explosion risk"),
    (r"\bliquid.*damage\b",         "liquid damage"),
    # legal / severity
    (r"\blawyer\b",                 "legal threat"),
    (r"\bsue\b",                    "sue threat"),
    (r"\blegal\b",                  "legal mention"),
    (r"\bcompensation\b",           "compensation request"),
    # repeated failure + high frustration
    (r"\b(multiple|several|many)\s+(times?|attempts?|calls?|resets?)\b",
                                    "repeated failed attempts"),
    (r"\bno.*(respon|help|solution)\b.*\b(week|month|day)s?\b",
                                    "prolonged unresolved"),
    # store / advisor complaint
    (r"\bgenius\s+bar\b.*\b(worse|broken|fail)\b",
                                    "genius bar made it worse"),
    (r"\badvisor\b.*\b(rude|awful|terrible|shame)\b",
                                    "advisor complaint"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), label) for p, label in ESCALATION_PATTERNS]


# ------------------------------------------------------------------
# rule-based escalator
# ------------------------------------------------------------------

class RuleBasedEscalator:
    """keyword-pattern escalation — fully interpretable."""

    def predict(self, texts):
        return [1 if self._match(t) else 0 for t in texts]

    def predict_proba_pos(self, texts):
        """binary confidence: 1.0 if any rule fires, 0.0 otherwise."""
        return [1.0 if self._match(t) else 0.0 for t in texts]

    def explain(self, text):
        """return list of fired rules for a single message."""
        fired = []
        for pattern, label in _COMPILED:
            if pattern.search(text):
                fired.append(label)
        return fired or ["no rule matched — auto-handle"]

    def _match(self, text):
        return any(p.search(text) for p, _ in _COMPILED)


# ------------------------------------------------------------------
# tfidf + logreg escalator
# ------------------------------------------------------------------

class TfidfEscalator:
    """tfidf + balanced logistic regression for escalation binary classification.
    uses a tuned probability threshold (default 0.391) instead of 0.5 —
    necessary because class imbalance (15% escalations) shifts optimal decision boundary."""

    def __init__(self, C=1.0, threshold=0.391):
        self.threshold = threshold
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=10000, ngram_range=(1, 2),
                sublinear_tf=True, min_df=1,
            )),
            ("clf", LogisticRegression(
                C=C, class_weight="balanced",
                max_iter=1000, random_state=42,
            )),
        ])

    def fit(self, texts, labels):
        self.pipe.fit(texts, labels)
        return self

    def predict(self, texts):
        proba = self.pipe.predict_proba(texts)[:, 1]
        return (proba >= self.threshold).astype(int).tolist()

    def predict_proba(self, texts):
        return self.pipe.predict_proba(texts)[:, 1].tolist()

    def save(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path=MODEL_PATH):
        with open(path, "rb") as f:
            return pickle.load(f)


# ------------------------------------------------------------------
# evaluation helpers
# ------------------------------------------------------------------

def evaluate_binary(labels_true, labels_pred, name, labels_score=None):
    acc  = accuracy_score(labels_true, labels_pred)
    prec = precision_score(labels_true, labels_pred, zero_division=0)
    rec  = recall_score(labels_true, labels_pred, zero_division=0)
    f1   = f1_score(labels_true, labels_pred, zero_division=0)
    auc  = roc_auc_score(labels_true, labels_score) if labels_score is not None else None

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  accuracy          : {acc:.4f}")
    print(f"  precision (esc)   : {prec:.4f}")
    print(f"  recall    (esc)   : {rec:.4f}   <- most important")
    print(f"  f1        (esc)   : {f1:.4f}")
    if auc is not None:
        print(f"  roc-auc           : {auc:.4f}")
    print()
    print(classification_report(
        labels_true, labels_pred,
        target_names=["auto-handle", "escalate"], zero_division=0
    ))

    result = {
        "model": name, "accuracy": round(acc, 4),
        "precision_escalate": round(prec, 4),
        "recall_escalate": round(rec, 4),
        "f1_escalate": round(f1, 4),
    }
    if auc is not None:
        result["roc_auc"] = round(auc, 4)
    return result


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    # load golden set
    rows = list(csv.DictReader(open(GOLDEN_PATH, encoding="utf-8")))
    texts  = [r["customer_msg"] for r in rows]
    labels = [1 if r["should_escalate"].strip().lower() == "yes" else 0 for r in rows]

    print(f"[+] golden set: {len(texts)} rows")
    print(f"    escalate=yes: {sum(labels)}  escalate=no: {len(labels)-sum(labels)}")
    print(f"    class balance: {sum(labels)/len(labels)*100:.1f}% escalation rate")

    all_results = {}

    # ---- baseline 1: rule-based ----
    print("\n[+] rule-based escalator ...")
    rule_clf = RuleBasedEscalator()
    preds_rule  = rule_clf.predict(texts)
    scores_rule = rule_clf.predict_proba_pos(texts)
    res1 = evaluate_binary(labels, preds_rule, "rule-based (keyword patterns)", scores_rule)
    all_results["rule_based"] = res1

    # show a few escalation explanations
    print("  sample explanations (escalate=yes):")
    esc_idx = [i for i, l in enumerate(labels) if l == 1][:4]
    for i in esc_idx:
        fired = rule_clf.explain(texts[i])
        print(f"    [{rows[i]['id']}] {texts[i][:70]}")
        print(f"           rules fired: {fired}")

    # ---- baseline 2: tfidf + logreg (5-fold cv, tuned threshold) ----
    THRESHOLD = 0.391  # tuned via precision-recall curve (best f1 on cv probabilities)
    print(f"\n[+] tfidf + logreg (5-fold stratified cv, threshold={THRESHOLD}) ...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000, ngram_range=(1, 2),
            sublinear_tf=True, min_df=1,
        )),
        ("clf", LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=1000, random_state=42,
        )),
    ])
    # get oof probabilities, apply tuned threshold
    scores_lr = cross_val_predict(pipe, texts, labels, cv=skf, method="predict_proba")[:, 1]
    preds_lr  = (scores_lr >= THRESHOLD).astype(int)
    res2 = evaluate_binary(labels, preds_lr, f"tfidf + logreg (threshold={THRESHOLD})", scores_lr)
    all_results["tfidf_logreg"] = res2

    # ---- train final model on full set ----
    print("\n[+] training final model on full golden set ...")
    final = TfidfEscalator(C=1.0).fit(texts, labels)
    final.save()
    print(f"    model saved to {MODEL_PATH}")

    # ---- summary ----
    print("\n[+] summary:")
    print(f"  {'model':<45s}  prec   rec    f1     auc")
    print(f"  {'-'*45}  -----  -----  -----  -----")
    for r in all_results.values():
        auc_str = f"{r.get('roc_auc', 0):.4f}" if "roc_auc" in r else "  —  "
        print(f"  {r['model']:<45s}  {r['precision_escalate']:.4f}  {r['recall_escalate']:.4f}  {r['f1_escalate']:.4f}  {auc_str}")

    # save metrics
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] metrics saved to {METRICS_PATH}")
    print("[done] chunk 9 complete.")


def should_escalate(text: str, method: str = "rule") -> tuple:
    """
    classify a single message for escalation.
    returns (bool, reason_str).
    method: 'rule' or 'tfidf'
    """
    if method == "rule":
        clf = RuleBasedEscalator()
        fired = clf.explain(text)
        escalate = any("no rule" not in r for r in fired)
        reason = "; ".join(fired) if escalate else ""
        return escalate, reason
    else:
        clf = TfidfEscalator.load()
        pred = clf.predict([text])[0]
        return bool(pred), "model prediction" if pred else ""
