# AGENTS.md

> **this file is the authoritative handoff document for all automated agents working in this repository.**
> every agent must read this file in full before making any change.

---

## project overview

**repo:** [brand-support-ai-agent](https://github.com/AnshitaBaduge/brand-support-ai-agent)
**owner:** AnshitaBaduge
**scope:** hiver sde intern take-home assignment — build an ai support agent for a brand using the twitter customer-support dataset, with evaluation, golden-set annotation, and reproducible results.

---

## current scope

- experimental software (intent classifier, reply drafter, escalation router)
- reproducible data and evaluation artifacts
- golden-set annotation (150–250 hand-labelled examples)
- results and failure logging
- decision logging (10–15 non-obvious decisions)
- repository documentation (README, PLAN, PROJECT, RESULTS)

---

## what agents must not do

- do not fabricate labels, results, human-evaluation evidence, or citations
- do not draft submission-report content unless explicitly instructed
- do not modify BLUEPRINT.pdf
- do not commit secrets, api keys, or credentials
- do not run code on the full dataset without explicit instruction (subsampling expected)
- do not introduce large dependencies without logging a decision

---

## conventions

### code style
- small, to-the-point comments wherever required — all lowercase, no caps
- no verbose docstrings unless requested
- python preferred; notebooks for exploration only

### git
- commit messages: lowercase, concise, no caps
- commit after each meaningful set of changes
- branch: `main`

### file structure (planned)
```
brand-support-ai-agent/
├── AGENTS.md              # this file
├── BLUEPRINT.pdf          # original assignment spec (read-only)
├── PLAN.html              # execution plan for the project
├── PROJECT.html           # project info, tools, dataset details
├── README.md              # github repo readme
├── RESULTS.md             # results and insights log
├── data/                  # raw and processed data (gitignored if large)
├── src/                   # source code
│   ├── data_prep.py       # data loading and cleaning
│   ├── intent.py          # intent classification
│   ├── reply.py           # reply generation
│   ├── escalation.py      # escalation logic
│   └── evaluate.py        # evaluation harness
├── golden_set/            # hand-labelled evaluation examples
├── notebooks/             # exploratory notebooks
├── configs/               # config files
├── outputs/               # model outputs, metrics
└── decision_log.md        # 10-15 non-obvious decisions
```

### work log update rules
after each meaningful action — code edit, dependency change, dataset preparation, command execution, result recorded, decision made, or blocker encountered — add a dated, factual entry to the work log below. keep entries concise enough for a subsequent agent to resume safely.

---

## agent preferences (from user)

- credits are limited — work in small, self-contained chunks
- each chunk should produce a committable result
- do not leave work half-done if a session might end
- prioritize getting something working end-to-end over perfection
- update AGENTS.md work log and RESULTS.md after each meaningful step

---

## work log

### 2026-09-16
- `[init]` read BLUEPRINT.pdf — extracted full assignment spec
- `[init]` initialized git repo, added remote origin (github.com/AnshitaBaduge/brand-support-ai-agent)
- `[docs]` created AGENTS.md, PLAN.html, PROJECT.html, README.md, RESULTS.md
- `[docs]` initial commit with project scaffolding and documentation
- `[chunk-1]` created full project skeleton: src/, data/, golden_set/, notebooks/, configs/, outputs/
- `[chunk-1]` created requirements.txt, .env.example, .gitignore, decision_log.md
- `[chunk-1]` created src/data_prep.py with sanity_check, filter_brand, build_threads
- `[chunk-1]` created stub files: src/intent.py, src/reply.py, src/escalation.py, src/evaluate.py
- `[chunk-1]` fixed aiohttp version (upgraded to 3.14.3) — resolves openai import error
- `[chunk-1]` raw data not yet downloaded — kaggle credentials missing; see data_prep.py for instructions
- `[chunk-1]` committed and pushed: "chunk 1: project skeleton, src stubs, requirements, decision log"
- `[blocker]` dataset download pending: need kaggle api credentials or manual download of twcs.csv → data/raw/twcs.csv
- `[chunk-2]` twcs.csv placed in data/raw/ by user (516 MB, 2,811,774 rows)
- `[chunk-2]` explored top 30 brands by reply count; analyzed top 5 candidates in detail
- `[chunk-2]` selected brand: AppleSupport — 99.8% thread completeness, 106,648 threads, avg 137 char replies
- `[chunk-2]` saved data/brand_conversations.csv (213,483 rows, 39 MB) and data/clean_threads.json (106,648 threads, 44 MB)
- `[chunk-2]` updated decision_log.md #1 with brand selection rationale
- `[chunk-2]` created src/chunk2_explore.py (reusable brand filter + thread builder)
- `[chunk-2]` committed and pushed: "chunk 2: brand selection (applesupport), filter data, build threads"
- `[chunk-3]` inspected raw threads — found: @mentions (99.9%), urls (14.8%), html entities (5.5%), near-empty (<10 chars): 1
- `[chunk-3]` implemented clean_text() in src/data_prep.py: html decode, url strip, @mention strip, hashtag norm, whitespace collapse
- `[chunk-3]` added is_valid_thread() and is_dm_redirect() filters
- `[chunk-3]` cleaning result: 106,648 → 104,183 kept, 2,465 dropped, 12,063 dm-redirect flagged (11.6%)
- `[chunk-3]` output saved to data/cleaned_threads.json (72 MB) with raw + cleaned fields preserved
- `[chunk-3]` updated RESULTS.md with cleaning stats
- `[chunk-3]` committed and pushed: "chunk 3: data cleaning pipeline"
- `[chunk-4]` sampled 300 substantive (non-dm-redirect) customer messages from cleaned_threads.json
- `[chunk-4]` manually reviewed all 300 samples — identified 12 natural intent clusters
- `[chunk-4]` created configs/intents.yaml with 12 intents, each with description, keywords, real examples
- `[chunk-4]` intents: software_bug, device_performance, battery_issue, connectivity_issue, account_and_password, payment_and_billing, hardware_and_accessories, app_and_store_issue, product_and_feature_question, order_and_delivery, feedback_and_complaint, data_and_privacy
- `[chunk-4]` updated decision_log.md #3 with taxonomy rationale
- `[chunk-4]` committed and pushed: "chunk 4: intent taxonomy — 12 intents defined in configs/intents.yaml"
- `[next]` chunk 5: golden set creation — hand-label 150-250 examples with intent + escalation decision
