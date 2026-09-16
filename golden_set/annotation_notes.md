# golden set annotation notes

**file:** `golden_set/golden_set.csv`
**total examples:** 240 (20 per intent × 12 intents)
**labelled by:** anshita baduge
**sampling date:** 2026-09-16

---

## how examples were sampled

1. loaded `data/cleaned_threads.json` (104,183 threads, all non-dm-redirect)
2. each thread was bucketed into one of 12 intents using keyword matching against `configs/intents.yaml`
3. 20 examples were randomly sampled from each intent bucket (stratified)
4. the sample was shuffled and ids assigned (gs_001 … gs_240)

---

## column descriptions

| column | what to put |
|--------|-------------|
| `id` | do not change — unique row id (gs_001 … gs_240) |
| `customer_msg` | the cleaned customer tweet — do not edit |
| `brand_reply_ref` | apple's actual historical reply — reference only, do not edit |
| `suggested_intent` | machine-suggested label — **do not change this column** (used for comparison later) |
| `intent` | **your label** — correct this if the suggested intent is wrong |
| `should_escalate` | `yes` or `no` — your judgment on whether a human agent should handle this |
| `escalation_reason` | if yes: brief reason (1 line). if no: leave blank |
| `notes` | optional: anything worth noting about this example (ambiguous, multi-intent, etc.) |

---

## the 12 valid intent labels

```
software_bug
device_performance
battery_issue
connectivity_issue
account_and_password
payment_and_billing
hardware_and_accessories
app_and_store_issue
product_and_feature_question
order_and_delivery
feedback_and_complaint
data_and_privacy
```

---

## escalation criteria — when to say `yes`

escalate when the message involves any of:

- **billing disputes** — unexpected charge, refund request, card fraud ("I didn't make this purchase")
- **account security** — hacked account, unauthorized access, stolen device
- **data loss** — permanent data deletion, failed backup that lost everything
- **legal / threat** — mentions lawyer, lawsuit, regulator, fraud complaint
- **device safety** — overheating, battery swelling, physical damage risk
- **extreme frustration with no self-service path** — multiple failed attempts, repeated contacts
- **sensitive personal info** — user shared private details publicly that shouldn't be
- **store / advisor complaint** — complaint about a named apple staff member or store visit

---

## what stays `no` (auto-handled)

- standard software troubleshooting (slow phone, update issues, connectivity)
- general how-to / feature questions
- battery tips
- order status queries (not disputes)
- feedback / opinion messages

---

## annotation tips

- if a message could fit two intents, pick the **primary** one (what does the person most want resolved?)
- `software_bug` vs `device_performance`: bug = triggered by a specific update/version; performance = general sluggishness not tied to one update
- `feedback_and_complaint` is for general dissatisfaction with no actionable specific issue — if there's a real technical problem, use the technical intent
- short/ambiguous messages (e.g. "that didn't work") — label based on context in `brand_reply_ref` which usually clarifies what was being discussed

---

## review checklist

- [ ] gone through all 240 rows
- [ ] corrected `intent` wherever suggested label was wrong
- [ ] set all `should_escalate` values (no blanks)
- [ ] filled `escalation_reason` for every `yes`
- [ ] added `notes` for ambiguous examples
