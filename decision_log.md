# decision log

non-obvious decisions made during the project, with reasoning.
updated as decisions are made.

---

## 1. brand selection — AppleSupport
- **decision:** use `AppleSupport` as the target brand
- **options considered:** AmazonHelp (169k replies), AppleSupport (106k), Uber_Support (56k), SpotifyCares (43k), Delta (42k)
- **reasoning:**
  - 99.8% thread completeness (106,623 matched inbound/reply pairs) — best ratio in top 5
  - avg brand reply length 137 chars — more informative than amazon (very short, dm-heavy) or delta
  - english-dominant, globally recognised support voice — easy to define "good" tone
  - rich intent diversity: device, software, account, billing, hardware, connectivity all appear
  - amazon had heavy multilingual content and many non-english threads
- **outcome:** 106,648 complete threads; 213,483 rows in filtered csv

---

## 2. subsampling strategy — brand filter then label subsample
- **decision:** filter to one brand first; then subsample to a golden set of 240 for evaluation
- **alternatives:** random sample across all brands; use a fixed 10k sample from the full dataset
- **reasoning:** the assignment explicitly permits subsampling. filtering to one brand is cleaner
  for evaluation — we define "good" once (apple support tone) rather than for dozens of brands.
  240 examples is the minimum for 5-fold stratified CV across 12 classes (20 per class).
- **tradeoff accepted:** models trained on 240 labels have high variance; cv scores are noisy.
  this is disclosed in the "misleading headline" section.

---

## 3. intent granularity — 12 intents
- **decision:** 12 intents (see `configs/intents.yaml`)
- **taxonomy:**
  1. `software_bug` — ios/macos bugs triggered by a specific update/version
  2. `device_performance` — general slowness, freezing, crashing (not update-specific)
  3. `battery_issue` — drain, inaccurate %, charging problems
  4. `connectivity_issue` — wifi, bluetooth, cellular, airdrop
  5. `account_and_password` — apple id, icloud, 2fa, account recovery
  6. `payment_and_billing` — charges, refunds, gift cards, billing errors
  7. `hardware_and_accessories` — speaker, earphones, charging port, screen, buttons
  8. `app_and_store_issue` — app store, itunes, apple music, siri, facetime
  9. `product_and_feature_question` — how-to, compatibility, feature availability
  10. `order_and_delivery` — orders, pre-orders, shipping, reservations
  11. `feedback_and_complaint` — general dissatisfaction, design opinions
  12. `data_and_privacy` — backup/restore failures, photo/message loss, privacy
- **reasoning:** manually reviewed 300 real messages; 12 was the natural cluster count.
  fewer than 8 merges very different issue types; more than 12 creates intents with <5 examples.
- **note:** `software_bug` and `device_performance` look similar but split on whether the issue
  is update-triggered (version-specific) or general.

---

## 4. llm model choice — gpt-4o-mini
- **decision:** target model is `gpt-4o-mini` for both classification and generation/judge
- **alternatives considered:** gpt-4o (too expensive for bulk runs), claude haiku (no key),
  google gemini flash (no key), local llama via ollama (torch dll error in this environment)
- **reasoning:** gpt-4o-mini is the cheapest production openai model with strong instruction
  following. at ~$0.15/1m input tokens, running 240 classification + 30 judge calls costs <$0.02.
  for generation, quality > cost since only 30 examples are judge-scored.
- **mock fallback decision:** api key unavailable during development; built a deterministic mock
  (keyword scoring + jaccard similarity) to validate the full pipeline structure without incurring
  costs. mock results are clearly labelled throughout. the pipeline auto-switches to real mode
  when `OPENAI_API_KEY` is set.

---

## 5. escalation criteria definition — human-labelled on golden set
- **decision:** define escalation by manual annotation on each of the 240 golden set examples,
  rather than using a pre-defined rule set
- **alternatives:** use a fixed keyword list from the start; ask the llm to decide
- **reasoning:** escalation is brand-specific and context-dependent. a pre-defined list would
  miss nuanced cases (e.g. "screen popping out" — physical safety risk with no obvious trigger word).
  having a human annotate 240 examples gives the cleanest ground truth for training and evaluation.
- **outcome:** 37/240 escalations (15.4%); the annotation revealed that 86% of escalation cases
  have no simple keyword match — validating the decision not to rely on rules alone.

---

## 6. dm-redirect threads — flag, not drop
- **decision:** flag threads where the brand reply is a "dm redirect" (asking user to DM privately)
  rather than dropping them
- **alternatives:** drop them entirely; include them normally
- **reasoning:** dm-redirect replies (12,063 threads, 11.6%) are real apple support responses but
  carry no resolution information. including them would pollute retrieval (a reply of "please DM us"
  ranks highly for cosine similarity but helps no one). flagging allows exclusion from the retrieval
  index while preserving them for analysis.
- **outcome:** retrieval index uses 91,236 non-dm-redirect threads; dm-redirects excluded.

---

## 7. retrieval architecture — tfidf over sentence-transformers
- **decision:** use tfidf cosine similarity for retrieval, not sentence-transformers / dense embeddings
- **alternatives:** `all-MiniLM-L6-v2` via sentence-transformers; bm25 (sparse lexical)
- **reasoning:** sentence-transformers requires pytorch. the environment has a torch dll error
  (`c10.dll` missing) that was not resolvable without a full reinstall. tfidf is fast, interpretable,
  and well-suited for short twitter text (avg 15 words). bm25 was not materially better than tfidf
  on short texts in literature.
- **tradeoff accepted:** tfidf retrieval is intent-agnostic — it can retrieve a software_bug reply
  for an order_and_delivery query if vocabulary overlaps. this is documented as failure mode 3.
  the fix (intent-conditioned dense retrieval) is listed in "what I'd do next."
- **logged in:** decision_log #7

---

## 8. escalation threshold tuning — 0.391 instead of default 0.5
- **decision:** use probability threshold = 0.391 for the tfidf+logreg escalation classifier
  instead of sklearn's default 0.5
- **reasoning:** with only 15.4% positive class rate, the model's calibrated probabilities for
  true positives cluster between 0.2–0.5. the default threshold of 0.5 gave 0.00 recall (caught
  0 of 37 escalations). precision-recall curve analysis found 0.391 maximises f1 (0.406) while
  maintaining meaningful precision (0.277). in a safety-critical routing context, recall matters
  more than precision — missing an escalation is worse than a false alarm.
- **outcome:** recall improved from 0.000 → 0.757; f1 from 0.000 → 0.406

---

## 9. golden set sampling — stratified by keyword-suggested intent
- **decision:** stratified sampling across 12 intents (20 per intent target) using keyword
  heuristics to pre-bucket messages before manual annotation
- **alternatives:** purely random sample; sample by thread length; sample by date
- **reasoning:** random sampling from 104k threads would massively oversample `software_bug`
  (the largest cluster) and give zero or near-zero examples of rare intents. stratified sampling
  ensures the evaluation set covers all 12 intents for meaningful per-class metrics.
  keyword pre-bucketing is a suggestion, not a label — annotators corrected 85/240 (35%).
- **tradeoff:** `product_and_feature_question` ended up with only 4 examples post-correction —
  too thin for CV. this is flagged in results and does not affect other intents.

---

## 10. evaluation metric priority — macro-f1 over accuracy for intent; recall over precision for escalation
- **decision:** report macro-f1 as the headline metric for intent classification;
  report recall as the headline metric for escalation routing
- **reasoning for macro-f1:** with 12 classes and severe imbalance, accuracy is dominated by
  `software_bug`. macro-f1 treats all classes equally — a model that ignores 9 intents gets
  penalised, not rewarded. this is honest.
- **reasoning for recall (escalation):** missing a safety/fraud escalation has higher real-world
  cost than a false alarm. a human reviewer can triage false alarms in seconds; a missed swollen
  battery report could be a liability. recall is the right optimisation target.

---

## 11. reply evaluation — rouge over bleu as primary automated metric
- **decision:** report rouge-1 as the primary automated reply quality metric, not bleu
- **reasoning:** bleu requires exact n-gram matches and smoothing for short texts. on twitter-length
  support replies (avg 20–30 words), bleu-1 was 0.00 for every draft because apple support
  uses highly variable phrasing ("please DM us", "reach out here", "we'd be happy to help").
  rouge-1 unigram recall (0.25–0.27) captures partial vocabulary overlap and is more interpretable
  for open-ended generation. bleu is still reported for completeness and to support the "misleading
  headline" argument.
- **note:** even rouge-1 = 0.25 is a weak signal. the human+judge eval is the meaningful quality measure.

---

## 12. human eval agreement — within-1 % as primary agreement metric, not exact kappa
- **decision:** report "within-1 % agreement" (84.2%) alongside cohen's kappa (0.40) when
  presenting judge vs human alignment
- **reasoning:** on a 1–5 integer scale, exact agreement kappa penalises small, arguably meaningless
  differences (e.g. human scores 3, judge scores 4). in practice, both raters are making the same
  categorical judgement. within-1 agreement (84.2%) is the operationally relevant number —
  it tells evaluators whether the judge is "close enough" for use as a scalable proxy.
- **note:** safety dimension has kappa=0.00 due to ceiling effect (both parties scored 5/5 for
  almost every example). this is not a failure; it means both parties agree safety is not an issue.

---

_13–15 entries to be added if further non-obvious decisions arise during readme and submission prep._
