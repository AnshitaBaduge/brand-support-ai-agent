# brand-support-ai-agent

an ai-powered customer support agent built for **AppleSupport** using real twitter support conversations (~3m tweets). classifies customer intents, drafts replies grounded in historical brand replies, and routes messages to escalation when needed.

> **hiver sde intern take-home assignment**

---

## reproduce headline results in under 15 minutes

```bash
# 1. clone
git clone https://github.com/AnshitaBaduge/brand-support-ai-agent.git
cd brand-support-ai-agent

# 2. install dependencies (python 3.9+)
pip install -r requirements.txt

# 3. download dataset — place twcs.csv at data/raw/twcs.csv
#    kaggle: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
#    (file is ~516mb; gitignored — must be downloaded manually)

# 4. rebuild cleaned data (takes ~2 min on full file)
python src/data_prep.py

# 5. run intent classifier baselines
python src/intent.py
# outputs: outputs/metrics_intent_baselines.json

# 6. run reply generation (builds tfidf index + generates 240 replies)
python src/reply.py --full
# outputs: outputs/reply_outputs.json, outputs/reply_index.pkl

# 7. run escalation router
python src/escalation.py
# outputs: outputs/metrics_escalation.json

# 8. run evaluation harness (bleu/rouge + llm-as-judge)
python src/evaluate.py
# outputs: outputs/metrics_reply_eval.json, outputs/judge_scores.json

# 9. (optional) run llm intent classifier — requires OPENAI_API_KEY
set OPENAI_API_KEY=sk-...          # windows
# export OPENAI_API_KEY=sk-...     # linux/mac
python src/intent_llm.py

# 10. (optional) human vs judge agreement — after filling golden_set/human_eval.csv
python src/human_judge_agreement.py
```

> steps 4–8 run fully without an api key (mock mode). all results in `outputs/` are precomputed and committed.

---

## headline results

### intent classification (golden set, n=240)

| model | accuracy | macro-f1 | notes |
|-------|----------|----------|-------|
| trivial (majority class) | 0.2083 | 0.0287 | always predicts `software_bug` |
| tfidf + logreg (5-fold cv) | 0.3292 | 0.1872 | strong on battery/app; fails thin classes |
| llm few-shot (mock, n=50) | 0.5800 | 0.5105 | mock — real gpt-4o-mini will score higher |

### reply quality

| metric | template replier | llm draft |
|--------|-----------------|-----------|
| rouge-1 | 0.2459 | 0.2663 |
| bleu-1 | 0.0000 | 0.0000 |
| judge overall (mock, n=30) | — | 3.63 / 5 |
| human-judge kappa | — | 0.402 (moderate) |

### escalation routing

| model | precision | recall | f1 | auc |
|-------|-----------|--------|----|-----|
| rule-based (27 patterns) | 0.857 | 0.162 | 0.273 | 0.579 |
| tfidf + logreg (t=0.391) | 0.277 | 0.757 | 0.406 | 0.723 |

> recall is the primary metric for escalation — missing a safety case is worse than a false alarm.

---

## what this builds

1. **intent classifier** — classifies customer messages into 12 intents defined from the data.
   two baselines (trivial, tfidf+logreg) + llm few-shot path with gpt-4o-mini.

2. **reply generator** — tfidf retriever finds top-5 most similar historical threads from 91k
   apple support replies. drafts a response using the retrieved context. two modes: template
   (retrieval-only baseline) and llm (gpt-4o-mini with retrieved context as grounding).

3. **escalation router** — classifies whether to auto-handle or escalate. rule-based baseline
   (27 keyword patterns) + tfidf+logreg with threshold tuning (0.391).

4. **evaluation harness** — automated metrics (bleu, rouge) + llm-as-judge rubric
   (tone/helpfulness/accuracy/safety, 1–5) + cohen's kappa for human-judge agreement.

---

## project structure

```
brand-support-ai-agent/
├── AGENTS.md              — agent handoff + full work log
├── BLUEPRINT.pdf          — original assignment spec (read-only)
├── PLAN.html              — execution plan
├── PROJECT.html           — project info, tools, dataset
├── README.md              — this file
├── RESULTS.md             — full results, failure analysis, misleading headline
├── decision_log.md        — 12 non-obvious decisions with reasoning
├── requirements.txt       — all dependencies
├── .env.example           — api key template
├── src/
│   ├── data_prep.py       — cleaning pipeline (html, urls, mentions, dm-redirect)
│   ├── intent.py          — tfidf+logreg intent classifier baselines
│   ├── intent_llm.py      — few-shot llm intent classifier
│   ├── reply.py           — tfidf retriever + template/llm reply generation
│   ├── escalation.py      — rule-based + tfidf logreg escalation router
│   ├── evaluate.py        — bleu/rouge + llm-as-judge evaluation harness
│   └── human_judge_agreement.py — cohen's kappa computation
├── configs/
│   └── intents.yaml       — 12 intents with descriptions, keywords, examples
├── golden_set/
│   ├── golden_set.csv     — 240 hand-labelled examples (intent + escalation)
│   ├── human_eval.csv     — 30 human-scored examples for judge agreement
│   └── annotation_notes.md — annotation guide and escalation criteria
└── outputs/               — all metrics and predictions (precomputed, committed)
    ├── metrics_intent_baselines.json
    ├── metrics_intent_llm.json
    ├── reply_outputs.json
    ├── metrics_escalation.json
    ├── metrics_reply_eval.json
    ├── judge_scores.json
    └── metrics_human_judge_agreement.json
```

---

## dataset

- **primary:** [customer support on twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) — ~3m tweets, 2.8m rows, multi-turn threads, dozens of brands
- **brand:** AppleSupport — 106,648 complete threads (largest with >99% thread completeness)
- `data/raw/twcs.csv` — gitignored (516mb); must be downloaded from kaggle

---

## problem framing

**what "good" means for AppleSupport:**
- replies are short (<240 chars), action-oriented, and always offer a concrete next step (dm link, settings path, or phone number)
- never generic ("sorry for the inconvenience") — always specific
- escalation is for safety risks, fraud, unresolved multi-attempt failures, and store/advisor complaints

**what we chose not to build:**
- multi-turn conversation management (single-turn only — see failure mode 2)
- real-time inference api or ui
- fine-tuned language model (cost/infra constraint; retrieval+prompting is the right approach at this scale)
- dense embedding retrieval (torch dll error in environment; tfidf is the principled fallback)

---

## failure analysis

see [RESULTS.md](RESULTS.md) for the full failure analysis with real examples. summary:

1. **software_bug class dominance** — tfidf collapses 9 intents into the majority class
2. **mid-thread fragments** — single-turn system fails on partial conversation turns
3. **intent-agnostic retrieval** — wrong-topic context retrieved when vocabulary overlaps
4. **escalation misses nuanced signals** — 86% of escalation cases have no keyword match
5. **mock llm draft incoherence** — opener + retrieved reply combination can be contradictory

---

## what i'd do with one more week

1. **intent-conditioned retrieval** — retrieve only from same-intent threads; use sentence-transformers once torch is fixed
2. **real gpt-4o-mini runs** — run actual api calls for intent classification, reply generation, and judge scoring
3. **conversation history** — pass prior turn context to both classifier and reply generator
4. **threshold calibration** — use platt scaling or isotonic regression on escalation model to calibrate probabilities
5. **larger golden set** — 500+ examples with balanced classes; focus on thin intents (product_and_feature_question)
6. **reply diversity** — retrieve top-5 and pick best with a re-ranker; filter out dm-redirect replies from candidate pool

---

## decision log

see [decision_log.md](decision_log.md) for all 12 non-obvious decisions with full reasoning.

---

## credits & citations

- dataset: [thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (kaggle)
- evaluation: `nltk` bleu, `rouge_score` rouge, `scikit-learn` kappa
- ai coding assistants were used during development (as permitted by assignment rules)
- all code is original; no external implementations were copied

---

## author

**anshita baduge** — [github.com/AnshitaBaduge](https://github.com/AnshitaBaduge)
