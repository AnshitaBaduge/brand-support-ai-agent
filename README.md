# brand-support-ai-agent

an ai-powered customer support agent built for a single brand using real twitter support conversations. classifies intents, drafts replies grounded in historical brand behavior, and decides whether to auto-handle or escalate to a human.

> **hiver sde intern take-home assignment**

---

## quick start

```bash
# clone
git clone https://github.com/AnshitaBaduge/brand-support-ai-agent.git
cd brand-support-ai-agent

# install dependencies
pip install -r requirements.txt

# set up api key (if using an llm api)
export OPENAI_API_KEY="your-key-here"  # or whichever llm you configure

# run the full pipeline
python src/evaluate.py

# results will be in outputs/
```

> reproducing headline results should take under 15 minutes.

---

## what this does

1. **intent classification** — categorizes incoming customer messages into a defined set of intents derived from the data
2. **reply drafting** — generates responses grounded in how the brand has historically resolved similar issues
3. **escalation routing** — decides whether a message should be auto-handled or escalated to a human, with a stated reason

---

## project structure

```
├── AGENTS.md              — agent handoff document
├── BLUEPRINT.pdf          — original assignment spec
├── PLAN.html              — execution plan (chunked for easy follow-along)
├── PROJECT.html           — project info, tools, dataset details
├── RESULTS.md             — results and insights
├── decision_log.md        — 10-15 non-obvious decisions
├── requirements.txt       — dependencies
├── src/                   — source code
├── data/                  — raw and processed data
├── golden_set/            — 150-250 hand-labelled evaluation examples
├── configs/               — intent definitions, model configs
├── notebooks/             — exploratory analysis
└── outputs/               — metrics, plots, model outputs
```

---

## dataset

- **primary:** [customer support on twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) — ~3m tweets, multi-turn threads, dozens of brands
- **optional:** [banking77](https://huggingface.co/datasets/PolyAI/banking77) — 13k queries, 77 labelled intents (for intent work only)

---

## evaluation

- **golden set:** 150-250 hand-labelled examples with annotation notes
- **automated metrics:** accuracy, f1, bleu, rouge, precision, recall
- **llm-as-judge:** rubric-based reply quality scoring with human agreement evidence (cohen's kappa)
- **baselines:** results compared against a trivial baseline and a simple baseline

---

## report

_this section will be populated as the project progresses._

### problem framing
<!-- what "good" means for this brand, what we chose not to build -->

### results vs baselines
<!-- comparison table -->

### failure analysis
<!-- top 5 failure modes with examples and hypotheses -->

### what is misleading about my headline number?
<!-- honest assessment of metric limitations -->

### what i'd do next with one more week
<!-- improvements and extensions -->

---

## decision log

see [decision_log.md](decision_log.md) for the full list of 10-15 non-obvious decisions made during this project.

---

## credits & citations

- dataset: [thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- ai coding assistants were used during development (as permitted by assignment rules)
- all borrowed code/ideas are cited inline

---

## author

**anshita baduge** — [github.com/AnshitaBaduge](https://github.com/AnshitaBaduge)
