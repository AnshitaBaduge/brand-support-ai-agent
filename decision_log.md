# decision log

non-obvious decisions made during the project, with reasoning.
updated as decisions are made.

---

## 1. brand selection
- **decision:** `AppleSupport`
- **options considered:** AmazonHelp (169k), AppleSupport (106k), Uber_Support (56k), SpotifyCares (43k), Delta (42k)
- **reasoning:**
  - 99.8% thread completeness (106,623 matched inbound out of 106,860 replies) — best ratio
  - avg brand reply length 137 chars — most informative replies
  - english-dominant, well-known support voice
  - rich intent diversity: device bugs, account issues, payment, software, connectivity
  - amazon had multilingual tweets; delta had lower completeness ratio
- **data:** 106,648 complete threads saved to `data/clean_threads.json`

## 2. subsampling strategy
- **decision:** work on a subsample of the full dataset (~3m tweets)
- **reason:** full dataset would be impractical for iteration; assignment explicitly permits subsampling
- **approach:** filter to one brand first, then subsample if still too large

## 3. intent granularity
- **decision:** _tbd — will define after reading ~200 customer messages_
- **principle:** intents should be actionable and distinct, not overlapping; 8-12 is the target range

## 4. llm choice for classification and generation
- **decision:** _tbd — will be chosen based on cost, api availability, and quality_
- **options:** openai gpt-4o-mini, google gemini flash, anthropic claude haiku, local (ollama)
- **constraint:** prefer a cheap/fast model for bulk classification; stronger model for generation/judge

## 5. escalation criteria definition
- **decision:** _tbd — will be defined after reviewing golden set_
- **initial candidates:** high negative sentiment, account/billing issues, legal mentions, multi-topic complexity

---

_more decisions will be added as the project progresses_
