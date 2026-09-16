# decision log

non-obvious decisions made during the project, with reasoning.
updated as decisions are made.

---

## 1. brand selection
- **decision:** _tbd — will be chosen in chunk 2 after inspecting tweet volume per brand_
- **options considered:** amazon, apple support, spotify, delta airlines, ask lyft
- **criterion:** min ~500 inbound threads; variety of intent types; brand with clear "voice"

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
