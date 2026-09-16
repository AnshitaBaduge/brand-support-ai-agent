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
- **decision:** 12 intents (see configs/intents.yaml)
- **taxonomy:**
  1. `software_bug` — ios/macos bugs, glitches, autocorrect issues
  2. `device_performance` — slow, freezing, crashing (not update-specific)
  3. `battery_issue` — drain, inaccurate %, charging problems
  4. `connectivity_issue` — wifi, bluetooth, cellular, airdrop
  5. `account_and_password` — apple id, icloud, 2fa, recovery
  6. `payment_and_billing` — unexpected charges, refunds, gift cards
  7. `hardware_and_accessories` — speaker, earphones, port, screen, buttons
  8. `app_and_store_issue` — app store, itunes, apple music, siri, facetime
  9. `product_and_feature_question` — how-to, compatibility, feature queries
  10. `order_and_delivery` — orders, pre-orders, shipping, reservations
  11. `feedback_and_complaint` — general dissatisfaction, design opinions
  12. `data_and_privacy` — backup, restore, photo loss, message deletion
- **reasoning:** reviewed 300 real customer messages manually; 12 was the natural cluster count
  that is both actionable and non-overlapping; fewer than 8 would merge very different issues;
  more than 12 would create intents with only a handful of examples each
- **note:** `software_bug` and `device_performance` look similar but split cleanly —
  software_bug is update-triggered and usually version-specific; device_performance is general


## 4. llm choice for classification and generation
- **decision:** _tbd — will be chosen based on cost, api availability, and quality_
- **options:** openai gpt-4o-mini, google gemini flash, anthropic claude haiku, local (ollama)
- **constraint:** prefer a cheap/fast model for bulk classification; stronger model for generation/judge

## 5. escalation criteria definition
- **decision:** _tbd — will be defined after reviewing golden set_
- **initial candidates:** high negative sentiment, account/billing issues, legal mentions, multi-topic complexity

---

_more decisions will be added as the project progresses_
