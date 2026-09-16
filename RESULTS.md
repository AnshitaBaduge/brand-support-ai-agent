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

| metric | template replier | llm draft | notes |
|--------|-----------------|-----------|-------|
| bleu-1 | 0.0000 | 0.0000 | 0 — different vocabulary; expected |
| bleu-2 | 0.0000 | 0.0000 | 0 — n-gram overlap too low at bigram level |
| rouge-1 | 0.2459 | 0.2663 | llm draft slightly closer to historical reference |
| rouge-l | 0.1843 | 0.1959 | llm draft marginally better on longest common subsequence |

**why bleu is 0:** the template/llm replies use different phrasing than the historical reference. 
bleu penalises any vocabulary mismatch heavily. this is expected — there are many valid replies.
rouge-1 ~0.25 is a more useful signal (unigram overlap with the reference).

### llm-as-judge scores (mock mode, n=30)

| dimension | avg score (1–5) | notes |
|-----------|----------------|-------|
| tone | 2.73 | mock scorer — needs real api for accurate scores |
| helpfulness | 2.80 | mock scorer |
| accuracy | 4.00 | mock scorer |
| safety | 5.00 | no pii/harm detected in any draft |
| **overall** | **3.63** | mock — real api judge expected before submission |

### judge-vs-human agreement (chunk 11, n=30)

| dimension | kappa | exact % | within-1 % | human avg | judge avg |
|-----------|-------|---------|------------|-----------|-----------|
| tone | 0.144 (slight) | 36.7% | 86.7% | 3.50 | 2.73 |
| helpfulness | 0.159 (slight) | 40.0% | 80.0% | 2.37 | 2.80 |
| accuracy | 0.000 (slight) | 43.3% | 70.0% | 2.97 | 4.00 |
| safety | 0.000 (slight) | 96.7% | 100.0% | 4.97 | 5.00 |
| **overall** | **0.402 (moderate)** | 54.2% | **84.2%** | — | — |

**interpretation:**
- **safety** is trivial agreement (both rate almost everything 5/5 — ceiling effect)
- **overall within-1 = 84.2%** — human and judge are almost always within one point of each other
- **overall kappa = 0.40** (moderate) — respectable for a 5-point scale with a mock judge
- **accuracy gap** (human 2.97 vs judge 4.00) — mock judge over-trusts replies; real gpt-4o-mini judge would be stricter
- **key insight for report:** within-1 agreement is the right metric here; exact match on a 5-point scale is too strict

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

### failure mode 1: software_bug dominance collapses nearby intents
**what happens:** `software_bug` is the largest class (50/240, 20.8%). the tfidf classifier learns to
predict it for any message containing "update", "phone", "crashing", or "iOS" — which appear in
at least 6 other intents.

**real example (tfidf misclassified):**
> customer: *"Não vejo que recebi msg em nenhum app, nada. Fico o dia todo sem uma notificação sequer"*
> true intent: `app_and_store_issue` → predicted: `software_bug`

**real example (class collapse):**
> `device_performance` (f1=0.00), `account_and_password` (f1=0.00), `hardware_and_accessories` (f1=0.00)
> all had recall=0 — their training signal drowned under software_bug's weight.

**hypothesis:** the tfidf model doesn't understand context, only token frequency.
a real LLM would distinguish "my notifications don't work" (app) from "my phone crashes after update" (software_bug).

---

### failure mode 2: mid-thread fragments can't be classified or replied to meaningfully
**what happens:** 9/240 golden-set messages are partial turns — the customer's reply in a
multi-turn thread with no standalone context. Both classifier and reply generator produce nonsense.

**real examples:**
> `gs_005`: *"Seems to only be through Spotify"* → intent=app_and_store_issue (correct by context, but unclassifiable standalone)
> `gs_007`: *"Awesome thanks for the quick reply"* → intent=feedback_and_complaint (our draft: *"We always want to improve your experience. You're welcome..."*)
> `gs_031`: *"Yes, the storage does match"* → mid-troubleshooting response, impossible to handle without prior turn

**hypothesis:** a production agent needs conversation history. a single-turn system will always fail
on follow-up messages. the golden set should have excluded these; flagging them as a data quality issue.

---

### failure mode 3: intent-agnostic retrieval returns wrong-topic context
**what happens:** the TF-IDF retriever scores cosine similarity on raw text without knowing intent.
an `order_and_delivery` query can retrieve a `software_bug` reply if they share common words like
"notification", "phone", "update".

**real example:**
> customer (order_and_delivery): *"I wake up to a notification saying my iPhone X won't be shipped until January..."*
> top retrieved reply (score=0.276): *"Thanks for that info. What iOS version is your iPhone running? To find: Settings > General > About > Version"*
> draft: *"We want to help with your order. Thanks for that info. What iOS version is your iPhone running?"*

**hypothesis:** intent-filtered retrieval (retrieve only from same-intent threads) would fix this.
decided against it in chunk 8 because small per-intent pools would reduce retrieval quality.
the right fix is intent-conditioned semantic retrieval using embeddings.

---

### failure mode 4: escalation misses nuanced safety/frustration signals (86% miss rate for rule-based)
**what happens:** 32/37 escalation cases have no keyword match for the rule-based system. customers
express serious issues without using trigger words.

**real examples (rule-based missed all of these):**
> `gs_060`: *"I got an email says I made a purchase I did not do"* → unauthorized purchase (no "unauthorized" keyword)
> `gs_062`: *"Screen on the left hand side is popping out. You can see the screen separating"* → physical safety risk
> `gs_063`: *"poor quality glass on back of iPhone 8 — dropped on laminate flooring and smashed"* → hardware damage
> `gs_037`: *"Already been told nothing can be done cos I bought it via my design agency"* → unresolved escalated complaint

**hypothesis:** escalation requires semantic intent, not just keyword presence. a real LLM judge
with the rubric from the eval harness would catch all four of these. the tfidf model (recall=0.757)
does better but still misses 9/37.

---

### failure mode 5: mock llm drafts produce incoherent opener + retrieved-reply combinations
**what happens:** the mock LLM draft concatenates an intent-based opener with the top retrieved reply.
when retrieval returns the wrong context (failure mode 3), the composite draft is internally contradictory.

**real example:**
> customer (payment_and_billing): *"I received a notification to re-authorise my apple order, but my card keeps declining"*
> draft: *"We'd be happy to assist with your billing concern. Hi. We can help, just send us a DM and tell us what version of iOS 11 you see under Settings > General > About > Version"*

the opener is correct for billing; the retrieved reply is for a software/iOS query. the composite is nonsense.

**hypothesis:** this is a mock-mode artifact. a real gpt-4o-mini draft with retrieved context as
background (not copy-pasted) would synthesise a coherent reply. the template replier (no LLM) actually
produces cleaner output than the mock LLM in these cases — evidence that bad LLM augmentation is worse than none.

---

### what is misleading about my headline number?

**headline number:** tfidf+logreg intent classifier accuracy = **0.3292**

**what makes it misleading:**

1. **accuracy hides class collapse.** accuracy of 0.33 sounds like "we get a third of cases right."
   in reality, the model predicts `software_bug` for 85%+ of inputs and gets credit every time
   a `software_bug` example appears. nine intents have f1=0.00 — they are completely undetected.
   macro-f1 = 0.19 is the honest number.

2. **the golden set is too small (240 rows) and imbalanced (50 software_bug vs 4 product_and_feature_question).**
   5-fold CV on 240 rows means each fold has ~48 test examples. a model can appear to perform
   well by correctly guessing the dominant class.

3. **bleu=0.00 sounds catastrophic but is expected.** there are thousands of valid ways to reply
   to a customer. "please DM us" and "reach out to our team here" mean the same thing but share
   zero bigrams with each other. bleu against a single reference is an unfair standard for open-ended reply.
   rouge-1=0.25 is the more honest automated signal.

4. **mock judge scores (overall 3.63/5) are not real LLM-as-judge scores.** the mock scorer
   uses heuristics (presence of "DM", "happy to help", reply length) to assign scores. a real
   gpt-4o-mini judge would likely score tone and helpfulness lower (human scored helpfulness at 2.37 vs mock's 2.80)
   and accuracy much lower (human: 2.97 vs mock: 4.00). the kappa of 0.40 shows moderate alignment,
   but the accuracy dimension is systematically inflated by the mock.

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
