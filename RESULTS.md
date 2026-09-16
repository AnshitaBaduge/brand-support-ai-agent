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

### llm-based classifier
_not yet implemented._

### comparison table

| model | accuracy | macro-f1 | notes |
|-------|----------|----------|-------|
| trivial (random/majority) | — | — | — |
| simple (tfidf + logreg) | — | — | — |
| llm (few-shot) | — | — | — |

---

## reply generation

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

| metric | value | notes |
|--------|-------|-------|
| precision | — | — |
| recall | — | — |
| f1 | — | — |

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

---

## golden set summary

| stat | value |
|------|-------|
| total examples | — |
| intents covered | — |
| escalation cases | — |
| annotation method | — |

---

_this file is updated incrementally. see AGENTS.md work log for the full timeline._
