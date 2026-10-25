"""
src/intent_llm.py — llm-based intent classifier using few-shot prompting.

production use:
  set OPENAI_API_KEY=sk-...
  python src/intent_llm.py           # runs 50-example subsample
  python src/intent_llm.py --full    # runs full eval set

chunk 7 note:
  api key not available during development — mock mode used for structural
  validation. mock uses keyword scoring identical to the tfidf baseline but
  with intent descriptions added. real llm eval to be run before submission.
"""

import os
import csv
import json
import random
import re
import sys
import time
from collections import Counter

import yaml
from sklearn.metrics import accuracy_score, f1_score, classification_report

random.seed(42)


def _load_env():
    """load .env file into os.environ if present."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(root, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_env()

# --- config ---
MODEL             = "qwen/qwen3.8-27b"  # groq model; falls back to gpt-4o-mini for openai
GOLDEN_PATH       = "golden_set/golden_set.csv"
INTENTS_PATH      = "configs/intents.yaml"
OUT_PREDS         = "outputs/llm_intent_predictions.json"
OUT_METRICS       = "outputs/metrics_intent_llm.json"
SHOTS_PER_INTENT  = 1        # 1 shot keeps system prompt small (groq tpm limit)
SUBSAMPLE_N       = 50
SLEEP_BETWEEN     = 7.0      # safe spacing for groq free tier (~6k tpm)


def load_golden():
    return list(csv.DictReader(open(GOLDEN_PATH, encoding="utf-8")))


def load_taxonomy():
    with open(INTENTS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["intents"]


def load_intent_names(taxonomy):
    return [i["name"] for i in taxonomy]


def build_few_shot_pool(rows, shots_per_intent):
    """split golden set into demo pool and eval set."""
    by_intent = {}
    for r in rows:
        by_intent.setdefault(r["intent"].strip(), []).append(r)
    demos, eval_rows = [], []
    for intent, items in by_intent.items():
        random.shuffle(items)
        n = min(shots_per_intent, len(items))
        demos.extend(items[:n])
        eval_rows.extend(items[n:])
    return demos, eval_rows


# ------------------------------------------------------------------
# mock llm — used when no api key is present
# simulates a few-shot llm with richer keyword + description scoring
# than the pure tfidf baseline, but without real llm inference
# ------------------------------------------------------------------

def build_mock_classifier(taxonomy, demos):
    """
    builds a rule-based scorer that mimics few-shot llm behavior:
    - uses intent keyword lists from taxonomy
    - uses description keywords
    - uses demo examples for simple nearest-neighbor boost
    all to give a more realistic approximation than pure keyword matching.
    """
    intent_names = [i["name"] for i in taxonomy]

    # combine keywords + words from description per intent
    patterns = {}
    for intent in taxonomy:
        kws = set(intent["keywords"])
        # extract significant words from description
        desc_words = re.findall(r"[a-z]{4,}", intent["description"].lower())
        all_terms = kws | set(desc_words)
        patterns[intent["name"]] = re.compile(
            "|".join(r"\b" + re.escape(t) + r"\b" for t in all_terms if len(t) > 3),
            re.IGNORECASE,
        )

    # demo index: map (intent, keywords from msg) for nearest-neighbor
    demo_index = {}
    for d in demos:
        intent = d["intent"].strip()
        words = set(re.findall(r"\b[a-z]{4,}\b", d["customer_msg"].lower()))
        demo_index.setdefault(intent, []).append(words)

    def predict(text):
        text_words = set(re.findall(r"\b[a-z]{4,}\b", text.lower()))

        # score 1: keyword pattern hits
        scores = {name: len(pat.findall(text)) for name, pat in patterns.items()}

        # score 2: jaccard similarity to demo examples (simulates few-shot retrieval)
        for intent, demo_word_sets in demo_index.items():
            best_jac = max(
                (len(text_words & dws) / max(len(text_words | dws), 1))
                for dws in demo_word_sets
            )
            scores[intent] = scores.get(intent, 0) + best_jac * 3  # weight demos higher

        if max(scores.values()) == 0:
            return "software_bug"
        return max(scores, key=scores.get)

    return predict


def build_system_prompt(intent_names, demos):
    intent_list = "\n".join(f"  - {n}" for n in intent_names)
    shot_block = ""
    for d in demos:
        shot_block += (
            f'message: "{d["customer_msg"][:200]}"\n'
            f'intent: {d["intent"].strip()}\n\n'
        )
    return f"""you are an intent classifier for an apple support twitter account.
classify each customer message into exactly one of these intents:
{intent_list}

few-shot examples:
{shot_block}
respond with ONLY the intent label, nothing else."""


def run_llm_eval(subsample_n=SUBSAMPLE_N, full=False):
    taxonomy     = load_taxonomy()
    intent_names = load_intent_names(taxonomy)
    rows         = load_golden()
    demos, eval_rows = build_few_shot_pool(rows, SHOTS_PER_INTENT)

    print(f"[+] few-shot demos: {len(demos)} ({SHOTS_PER_INTENT} per intent)")
    print(f"[+] eval set size:  {len(eval_rows)}")

    if not full and subsample_n and subsample_n < len(eval_rows):
        eval_rows = random.sample(eval_rows, subsample_n)
        print(f"[+] subsampled to:  {len(eval_rows)}")

    # detect provider: groq > openai > mock
    groq_key   = os.environ.get("GROQ_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    use_mock   = not groq_key and not openai_key
    model_name = MODEL  # groq default

    if use_mock:
        print("\n[!] no api key found — running mock llm (structural validation only)")
        print("[!] results will be labelled as mock in outputs")
        predict_fn = build_mock_classifier(taxonomy, demos)
    elif groq_key:
        from groq import Groq
        client = Groq(api_key=groq_key)
        system_prompt = build_system_prompt(intent_names, demos)
        print(f"[+] provider: groq  model: {model_name}")
    else:
        import openai
        client = openai.OpenAI(api_key=openai_key)
        model_name = "gpt-4o-mini"
        system_prompt = build_system_prompt(intent_names, demos)
        print(f"[+] provider: openai  model: {model_name}")

    predictions = []
    labels_true = []
    labels_pred = []
    errors = 0

    print(f"\n[+] classifying {len(eval_rows)} messages...")
    for i, row in enumerate(eval_rows):
        msg   = row["customer_msg"]
        label = row["intent"].strip()

        try:
            if use_mock:
                pred = predict_fn(msg)
            else:
                # retry up to 3 times on rate limit errors
                for attempt in range(3):
                    try:
                        resp = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user",   "content": f'message: "{msg[:300]}"'},
                            ],
                            temperature=0,
                            max_tokens=20,
                        )
                        break
                    except Exception as re:
                        if attempt < 2 and ("429" in str(re) or "rate" in str(re).lower()):
                            wait = (attempt + 1) * 15
                            print(f"  rate limit, retrying in {wait}s...")
                            time.sleep(wait)
                        else:
                            raise
                raw = resp.choices[0].message.content.strip().lower()
                raw = raw.replace('"','').replace("'","")
                matched = [n for n in intent_names if n == raw or n in raw or raw in n]
                pred = matched[0] if matched else "software_bug"
                time.sleep(SLEEP_BETWEEN)
        except Exception as e:
            pred   = "software_bug"
            status = f"error: {e}"
            errors += 1
        else:
            status = "mock" if use_mock else "ok"

        predictions.append({
            "id":          row["id"],
            "customer_msg": msg[:120],
            "true_intent": label,
            "pred_intent": pred,
            "correct":     label == pred,
            "status":      status,
        })
        labels_true.append(label)
        labels_pred.append(pred)

        if (i + 1) % 10 == 0:
            running_acc = sum(p["correct"] for p in predictions) / len(predictions)
            print(f"  [{i+1:3d}/{len(eval_rows)}]  running acc: {running_acc:.3f}")

    # metrics
    acc = accuracy_score(labels_true, labels_pred)
    mf1 = f1_score(labels_true, labels_pred, average="macro",    zero_division=0)
    wf1 = f1_score(labels_true, labels_pred, average="weighted", zero_division=0)
    provider_tag = "mock" if use_mock else ("groq/" + model_name if groq_key else "openai/" + model_name)
    label_str = "mock few-shot (no api)" if use_mock else f"llm few-shot ({provider_tag})"

    print(f"\n{'='*60}")
    print(f"  {label_str}  — n={len(eval_rows)}")
    print(f"{'='*60}")
    print(f"  accuracy    : {acc:.4f}")
    print(f"  macro-f1    : {mf1:.4f}")
    print(f"  weighted-f1 : {wf1:.4f}")
    print(f"  api errors  : {errors}")
    print()
    print(classification_report(labels_true, labels_pred, zero_division=0))

    os.makedirs("outputs", exist_ok=True)
    with open(OUT_PREDS, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    results = {
        "model":          label_str,        "mode":           "mock" if use_mock else "api",
        "n_eval":         len(eval_rows),
        "n_demos":        len(demos),
        "shots_per_intent": SHOTS_PER_INTENT,
        "subsample":      subsample_n,
        "accuracy":       round(acc, 4),
        "macro_f1":       round(mf1, 4),
        "weighted_f1":    round(wf1, 4),
        "api_errors":     errors,
        "per_class":      classification_report(
            labels_true, labels_pred, zero_division=0, output_dict=True
        ),
    }
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[+] predictions: {OUT_PREDS}")
    print(f"[+] metrics:     {OUT_METRICS}")
    return results


if __name__ == "__main__":
    full = "--full" in sys.argv
    run_llm_eval(subsample_n=None if full else SUBSAMPLE_N, full=full)
