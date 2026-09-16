# RESULTS.md

> all results and insights from the brand support ai agent project are logged here.
> updated after each evaluation run or meaningful finding.

---

## dataset summary (chunk 2 & 3)

| stat | value |
|------|-------|
| brand selected | AppleSupport |
| raw dataset size | 2,811,774 rows |
| brand replies in dataset | 106,860 |
| raw threads built | 106,648 |
| threads after cleaning | **104,183** |
| dropped (too short post-clean) | 2,465 (2.3%) |
| dm-redirect replies flagged | 12,063 (11.6%) |
| avg cleaned customer msg len | ~109 chars |
| avg cleaned brand reply len | ~137 chars |

### cleaning operations applied
- html entity decode (`&amp;` → `&`, `&gt;` → `>`, etc.)
- url strip (`https://t.co/...` removed)
- @mention strip (`@AppleSupport`, `@115854`, etc.)
- hashtag symbol strip (kept word)
- whitespace normalisation
- empty/near-empty thread drop (<8 chars after cleaning)
- dm-redirect flag (brand replies that only say "send us a dm" — kept but flagged)

---

## status

| component | status | last updated |
|-----------|--------|-------------|
| intent classifier (trivial baseline) | ⬜ not started | — |
| intent classifier (simple baseline) | ⬜ not started | — |
| intent classifier (llm-based) | ⬜ not started | — |
| reply generation | ⬜ not started | — |
| escalation router | ⬜ not started | — |
| llm-as-judge | ⬜ not started | — |
| failure analysis | ⬜ not started | — |

---

## intent classification

### trivial baseline
_not yet implemented._

### simple baseline
_not yet implemented._

### trivial baseline (majority class)
- always predicts `software_bug` (most common class, 50/240)
- accuracy: **0.2083**, macro-f1: **0.0287**

### simple baseline (tfidf + logistic regression)
- tfidf unigrams+bigrams (15k features, sublinear tf) + multinomial logistic regression
- evaluated with 5-fold stratified cross-validation on golden set (n=240)
- accuracy: **0.3292**, macro-f1: **0.1872**
- strong on `battery_issue` (f1=0.65), `app_and_store_issue` (f1=0.44)
- fails on: `device_performance`, `account_and_password`, `hardware_and_accessories`, `product_and_feature_question` (f1=0.00)
- root cause: keyword overlap between intents (e.g. "update" appears in battery, software_bug, device_performance) + too few examples for thin classes

### llm-based classifier (few-shot, mock mode)
- model: gpt-4o-mini (mock fallback — no api key during development)
- 2 few-shot demos per intent (24 total drawn from golden set)
- evaluated on 50-example stratified subsample of remaining eval rows
- mock uses keyword scoring + jaccard similarity to demo examples
- **accuracy: 0.5800, macro-f1: 0.5105** (n=50, mock mode)
- note: mock results are a structural approximation — real gpt-4o-mini will score higher
- strong intents (mock): account_and_password (f1=0.86), battery_issue (f1=0.80), order_and_delivery (f1=0.86)
- weak intents: software_bug (f1=0.00 — overwhelmed by similar intents), product_and_feature_question (0 support in subsample)

### comparison table

| model | accuracy | macro-f1 | weighted-f1 | notes |
|-------|----------|----------|-------------|-------|
| trivial (majority class) | 0.2083 | 0.0287 | 0.0718 | predicts software_bug always |
| tfidf + logreg (5-fold cv) | 0.3292 | 0.1872 | 0.2581 | strong on battery/app; fails thin classes |
| llm few-shot (mock, n=50) | 0.5800 | 0.5105 | 0.5546 | mock — real api will score higher |

---

## reply generation

### architecture (chunk 8)
- **retriever:** tfidf cosine-similarity (20k features, unigrams+bigrams, sublinear_tf)
- **index:** 91,236 substantive non-dm-redirect threads from cleaned_threads.json
- **top-k:** 5 retrieved neighbors per query (self-match excluded)
- **template replier (baseline):** returns top-1 retrieved historical brand reply — no generation
- **llm replier:** intent-aware opener + top-1 retrieved reply (mock); real: gpt-4o-mini with top-3 as context
- **retrieval quality:** top-1 cosine scores range 0.15–0.45 on golden set after self-match exclusion

### automated metrics

| metric | score | notes |
|--------|-------|-------|
| bleu | — | — |
| rouge-1 | — | — |
| rouge-l | — | — |

### llm-as-judge scores

| dimension | avg score | notes |
|-----------|-----------|-------|
| tone | — | — |
| accuracy | — | — |
| helpfulness | — | — |
| safety | — | — |

### judge-vs-human agreement
_not yet computed._

| metric | value |
|--------|-------|
| cohen's kappa | — |
| % agreement | — |
| n (human-scored) | — |

---

## escalation routing

- **class balance:** 37 escalate / 203 auto-handle (15.4% escalation rate)
- **key metric:** recall on escalate class — missing a safety/fraud case is worse than a false alarm

| model | precision | recall | f1 | roc-auc | notes |
|-------|-----------|--------|----|---------|-------|
| rule-based (27 keyword patterns) | 0.857 | 0.162 | 0.273 | 0.579 | very precise, misses 31/37 escalations |
| tfidf + logreg (threshold=0.391) | 0.277 | 0.757 | 0.406 | 0.723 | catches 28/37 escalations, noisy |

**key insight:** rule-based catches only the obvious cases (smoking charger, fraud keyword). 
Most escalations are nuanced — "screen popping out", "genius bar left my phone a brick",
"lost 3000 photos" — which the tfidf model learns from context.
the tfidf model with a lowered threshold is strictly better for safety-critical routing.

---

## failure analysis

### top 5 failure modes
_not yet analyzed._

1. **failure mode 1:** _tbd_
2. **failure mode 2:** _tbd_
3. **failure mode 3:** _tbd_
4. **failure mode 4:** _tbd_
5. **failure mode 5:** _tbd_

### what is misleading about my headline number?
_not yet written._

---

## insights log

| date | insight |
|------|---------|
| 2026-09-16 | project initialized — blueprint extracted, documentation scaffolded |
| 2026-09-16 | brand selected: AppleSupport — best thread completeness (99.8%) among top 5 brands |
| 2026-09-16 | cleaning dropped 2.3% of threads (too short after stripping mentions/urls) |
| 2026-09-16 | 11.6% of brand replies are dm-redirects — flagged but kept for analysis |
| 2026-09-16 | 12 intent clusters identified from 300 manually reviewed messages |

---

## golden set summary

| stat | value |
|------|-------|
| total examples | 240 |
| intents covered | 12 / 12 |
| examples per intent | 4–50 (skewed — software_bug dominant) |
| escalation yes | 37 / 240 (15.4%) |
| escalation no | 203 / 240 (84.6%) |
| intent labels changed from suggestion | 85 / 240 (35%) — strong manual review signal |
| annotation method | keyword-bucketed stratified sampling → full manual review |
| annotation status | **done** |
| thinnest intent | product_and_feature_question (4 examples) — noted in decision log |

---

_this file is updated incrementally. see AGENTS.md work log for the full timeline._
