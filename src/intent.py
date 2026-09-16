"""
src/intent.py — intent classification.
implements two baselines and later the llm-based classifier.

baselines:
  1. trivial — always predicts the majority class
  2. simple  — tfidf + logistic regression (5-fold cross-val on golden set)
"""

import json
import os
import csv
import pickle
import numpy as np
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import yaml

GOLDEN_PATH = "golden_set/golden_set.csv"
INTENTS_PATH = "configs/intents.yaml"
MODEL_PATH = "outputs/tfidf_logreg.pkl"


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def load_golden():
    """load annotated golden set, return (texts, labels)."""
    rows = list(csv.DictReader(open(GOLDEN_PATH, encoding="utf-8")))
    texts  = [r["customer_msg"] for r in rows]
    labels = [r["intent"].strip() for r in rows]
    return texts, labels, rows


def load_intent_names():
    with open(INTENTS_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [i["name"] for i in data["intents"]]


# ------------------------------------------------------------------
# classifiers
# ------------------------------------------------------------------

class TrivialClassifier:
    """always predicts majority class — lower-bound baseline."""

    def fit(self, texts, labels):
        self.majority = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, texts):
        return [self.majority] * len(texts)


class TfidfLogregClassifier:
    """tfidf features + logistic regression."""

    def __init__(self, max_features=10000, C=1.0):
        self.vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
        self.clf = LogisticRegression(
            C=C, max_iter=1000, solver="lbfgs",
            multi_class="multinomial", random_state=42,
        )

    def fit(self, texts, labels):
        X = self.vec.fit_transform(texts)
        self.clf.fit(X, labels)
        return self

    def predict(self, texts):
        X = self.vec.transform(texts)
        return self.clf.predict(X).tolist()

    def save(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"    model saved to {path}")

    @staticmethod
    def load(path=MODEL_PATH):
        with open(path, "rb") as f:
            return pickle.load(f)


# ------------------------------------------------------------------
# evaluation
# ------------------------------------------------------------------

def evaluate(labels_true, labels_pred, name, intent_names):
    """compute and print all metrics, return dict."""
    acc  = accuracy_score(labels_true, labels_pred)
    mf1  = f1_score(labels_true, labels_pred, average="macro", zero_division=0)
    wf1  = f1_score(labels_true, labels_pred, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  accuracy      : {acc:.4f}")
    print(f"  macro-f1      : {mf1:.4f}")
    print(f"  weighted-f1   : {wf1:.4f}")
    print()
    print(classification_report(
        labels_true, labels_pred,
        labels=intent_names, zero_division=0
    ))

    return {
        "model": name,
        "accuracy": round(acc, 4),
        "macro_f1": round(mf1, 4),
        "weighted_f1": round(wf1, 4),
        "per_class": classification_report(
            labels_true, labels_pred,
            labels=intent_names, zero_division=0, output_dict=True
        ),
    }


# ------------------------------------------------------------------
# main — run baselines
# ------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    intent_names = load_intent_names()
    texts, labels, rows = load_golden()

    print(f"[+] golden set loaded: {len(texts)} examples, {len(set(labels))} intents")
    print(f"    class distribution: {Counter(labels).most_common()}")

    all_results = {}

    # ---- baseline 1: trivial (majority class) ----
    print("\n[+] running trivial baseline (majority class)...")
    trivial = TrivialClassifier().fit(texts, labels)
    preds_trivial = trivial.predict(texts)
    res1 = evaluate(labels, preds_trivial, "trivial (majority class)", intent_names)
    all_results["trivial"] = res1

    # ---- baseline 2: tfidf + logreg (5-fold cv) ----
    print("\n[+] running tfidf + logreg (5-fold stratified cv)...")
    model = TfidfLogregClassifier(max_features=15000, C=1.0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # cross_val_predict gives oof predictions for the whole set
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer as TV
    from sklearn.linear_model import LogisticRegression as LR

    pipe = Pipeline([
        ("tfidf", TV(max_features=15000, ngram_range=(1,2),
                     sublinear_tf=True, min_df=2)),
        ("clf",   LR(C=1.0, max_iter=1000, solver="lbfgs",
                     multi_class="multinomial", random_state=42)),
    ])

    preds_logreg = cross_val_predict(pipe, texts, labels, cv=skf)
    res2 = evaluate(labels, preds_logreg.tolist(), "tfidf + logreg (5-fold cv)", intent_names)
    all_results["tfidf_logreg"] = res2

    # also train on full set and save for later use in chunk 7+
    print("\n[+] training final tfidf+logreg on full golden set...")
    model.fit(texts, labels)
    model.save()

    # ---- save all results ----
    out_path = "outputs/metrics_intent_baselines.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] metrics saved to {out_path}")

    # ---- summary table ----
    print("\n[+] summary:")
    print(f"  {'model':<35s}  accuracy  macro-f1  weighted-f1")
    print(f"  {'-'*35}  --------  --------  -----------")
    for name, r in all_results.items():
        print(f"  {r['model']:<35s}  {r['accuracy']:.4f}    {r['macro_f1']:.4f}    {r['weighted_f1']:.4f}")

    print("\n[done] chunk 6 complete.")


def classify_intent(text, method="tfidf"):
    """
    classify a single customer message.
    method: 'tfidf' (loads saved model) or 'trivial'
    """
    if method == "trivial":
        return "software_bug"  # majority class
    model = TfidfLogregClassifier.load()
    return model.predict([text])[0]
