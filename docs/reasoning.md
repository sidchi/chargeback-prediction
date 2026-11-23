# LLM reasoning layer

<!-- Examples and the backtest table were generated with `cbp reason` / `cbp playbook` on the
default synthetic dataset (`cbp generate`). Synthetic data only. -->

When a dispute lands, an analyst has a fixed response window to decide whether to fight it and, if so, with
what evidence. `cbp reason` (or `chargeback_prediction.reasoning`) gives them a case note: a
short rationale grounded in the model's score and SHAP factors, a recommended strategy
(**represent**, **accept** or **review**), and the evidence to assemble for that reason code.

## Who decides what

| Concern | Owner | Why |
|---|---|---|
| Dispute probability | Calibrated model (items 4–5) | Measurable, backtestable, reproducible |
| Which strategies are on the table | Rule-based playbook (`reasoning/playbook.py`) | A money and network-compliance decision must be reviewable and deterministic |
| The words in the case note | LLM, behind `LLMClient` | Turning 30 features and SHAP values into two readable sentences is what LLMs are good at |
| Evidence list | Playbook, never the LLM | A hallucinated evidence item in a representment packet is worse than none |

```
features + SHAP ─┐
reason code ─────┼─▶ playbook ─▶ recommended + allowed strategies, signals, evidence
                 │                         │
                 └─────────────▶ prompt (structured JSON context only)
                                           ▼
                                 LLMClient.complete()  ──error──┐
                                           ▼                     │
                                 validate reply ──rejected──▶ retry once with the error
                                           │                     │
                                       accepted              still failing
                                           ▼                     ▼
                                 RiskRationale(source=llm)   RiskRationale(source=fallback,
                                                              deterministic template)
```

## The playbook

Features are read as **signals** that lean one of three ways:

- **true fraud**: account under 7 days old, disposable email, IP or billing country ≠ the
  card's issuing country, ≥4 authorization attempts or ≥2 declines in 24 h;
- **cardholder** (friendly fraud): account ≥120 days old with no IP/issuer mismatch, ≥2 prior
  approved renewals, earlier disputes by the same customer;
- **merchant error**: the customer asked to cancel before the charge.

Then, per reason-code category:

| Situation | Recommended | LLM may choose |
|---|---|---|
| Charged after a cancellation request (any code) | accept | accept only |
| Fraud code, ≥2 stolen-card signals, no cardholder signals | accept | accept, review |
| Fraud code, ≥2 cardholder signals, no stolen-card signals | represent | represent, review |
| Fraud code, anything else | review | any |
| Consumer code, ≥2 stolen-card signals | review | any |
| Consumer code, otherwise | represent | represent, review |

The LLM can always soften a decision to *review* but can never turn an *accept* into a fight
on its own, and a merchant-error dispute is never argued.

### Backtest against the synthetic ground truth

The generator labels every dispute with whether the real cardholder filed it
(`is_friendly_fraud`), so the rules can be scored (`cbp playbook`):

| Category | Playbook strategy | Disputes | Friendly fraud (ground truth) |
|---|---|---|---|
| consumer | represent | 269 | 176 (65%) |
| consumer | accept | 55 | 0 (0%) |
| consumer | review | 1 | 0 (0%) |
| fraud | represent | 59 | 59 (100%) |
| fraud | accept | 150 | 0 (0%) |
| fraud | review | 90 | 67 (74%) |

- **Fraud codes are clean at the extremes.** Every *accept* was a stolen card and every
  *represent* was friendly fraud, i.e. a representment the merchant should win.
- **Review is where the volume hides.** 90 fraud-coded disputes have mixed or single signals,
  and three quarters of them are friendly fraud. The common case is an established account
  with only one cardholder signal; using the pre-dispute score as a tie-breaker (a low score
  on a long-tenured account) would move most of them to *represent*. That needs win/loss
  outcomes to tune, so it waits for the evaluation item.
- **Consumer codes are represented by default**, and about a third of those are genuine
  complaints. The features say nothing about service quality, so evidence of usage is the only
  lever; real win rates would decide whether a cheaper "refund and keep the customer" policy
  beats fighting.

These numbers come from synthetic data whose patterns the rules were written against, so they
show the rules do what they say, not that they would hold up in production.

## Guardrails on every LLM reply

`parse_reply` accepts a reply only if it is a JSON object (a Markdown code fence is tolerated)
with:

- `strategy` in the playbook's allowed set;
- `confidence` of `low`, `medium` or `high`;
- a non-empty `rationale` of at most 1,200 characters (whitespace is normalised);
- 1–6 `key_factors`, each a feature name the model was actually shown (top SHAP factors and
  playbook signals). This catches the most common hallucination: citing a plausible-sounding
  feature such as `device_fingerprint_risk` that the pipeline does not have.

A rejected reply is sent back with the validation error and retried once. Provider errors
(network, HTTP status, malformed JSON) are retried the same way. If both attempts fail, the
result is a deterministic template rationale with `source="fallback"` and the playbook's
strategy, and the warnings say why. The decision path never depends on the LLM being up.

The prompt context is built only from our own features, SHAP values and reason-code catalog,
so there is no cardholder- or issuer-written text to inject instructions through. When item 7
adds dispute narratives, they must go in a separately delimited, explicitly untrusted field.

## Providers

`LLMClient` is a one-method protocol, `complete(CompletionRequest) -> str`:

- `MockLLM` (default) reads the JSON context out of the prompt and writes a valid reply. It
  keeps the CLI and tests offline and deterministic, and proves the prompt carries everything
  a model needs.
- `ScriptedLLM` replays canned replies and exceptions for testing the guardrails.
- `OpenAICompatibleClient` speaks `POST /chat/completions`, which most hosted APIs and local
  model servers implement, using only the standard library. Temperature defaults to 0 so case
  notes are reproducible. The API key comes from `$CBP_LLM_API_KEY` and is kept out of `repr`.

```bash
uv run cbp reason --transaction-id txn_00025755                     # mock, offline
CBP_LLM_API_KEY=... uv run cbp reason --transaction-id txn_00025755 \
    --provider openai-compatible --llm-base-url http://localhost:11434/v1 --llm-model llama3.1
```

## Example output (mock provider)

```
txn_00038102 · mastercard:4837 No Cardholder Authorization
  strategy:    accept (playbook: accept, confidence: high, llm via mock)
  rationale:   Filed under mastercard:4837 (No Cardholder Authorization). Before the dispute, the model scored this charge at 40.8%, 22.0× a typical transaction. The score was driven by attempts_24h = 2 (×7.23 odds), is_one_off = yes (×4.74 odds), account_age_days = 0.6 days (×4.07 odds). Playbook signals: account created 0.6 days before the charge; customer IP country differs from the card's issuing country. Recommended action: accept. 2 stolen-card signals and none pointing to the real cardholder.
  key factors: attempts_24h, is_one_off, account_age_days, ip_bin_mismatch
  evidence:
    - refund confirmation, or accept the chargeback without a response

txn_00025755 · visa:10.4 Other Fraud - Card Absent Environment
  strategy:    represent (playbook: represent, confidence: high, llm via mock)
  rationale:   Filed under visa:10.4 (Other Fraud - Card Absent Environment). Before the dispute, the model scored this charge at 4.0%, 2.2× a typical transaction. The score was driven by is_one_off = yes (×4.74 odds), prior_approved_renewals = 0 (×0.60 odds), amount_usd = $39.10 (×1.48 odds). Playbook signals: established account (1,041.3 days old) with no IP/issuer country mismatch; 1 earlier dispute(s) filed by this customer (possible serial disputer). Recommended action: represent. 2 signals that the real cardholder made the charge and none of a stolen card: likely friendly fraud.
  key factors: is_one_off, prior_approved_renewals, amount_usd, account_age_days, prior_disputes
  evidence:
    - Visa Compelling Evidence 3.0: two or more prior undisputed transactions on the same card, 120–365 days before the dispute, sharing at least two identifiers (IP address, device ID, user account, shipping address) with the disputed charge
    - account logins or service usage after the disputed charge
    - AVS and CVV results from the original authorization

txn_00030225 · visa:13.2 Cancelled Recurring Transaction
  strategy:    accept (playbook: accept, confidence: medium, llm via mock)
  rationale:   Filed under visa:13.2 (Cancelled Recurring Transaction). Before the dispute, the model scored this charge at 99.1%, 53.6× a typical transaction. The score was driven by charged_after_cancel_request = yes (×1,668 odds), au_updates_90d = 1 (×3.92 odds), days_since_card_update = 47.2 days (×1.69 odds). Playbook signals: the customer had already asked to cancel before this charge. Recommended action: accept. The customer asked to cancel before this charge, so the merchant is at fault.
  key factors: charged_after_cancel_request, au_updates_90d, days_since_card_update
  evidence:
    - refund confirmation, or accept the chargeback without a response

txn_00009329 · mastercard:4837 No Cardholder Authorization
  strategy:    review (playbook: review, confidence: low, llm via mock)
  rationale:   Filed under mastercard:4837 (No Cardholder Authorization). Before the dispute, the model scored this charge at 6.1%, 3.3× a typical transaction. The score was driven by is_one_off = yes (×4.74 odds), prior_approved_renewals = 0 (×0.60 odds), amount_usd = $39.95 (×1.50 odds). Playbook signals: established account (306.1 days old) with no IP/issuer country mismatch. Recommended action: review. Mixed or weak signals (0 stolen-card, 1 cardholder).
  key factors: is_one_off, prior_approved_renewals, amount_usd, account_age_days
  evidence:
    - proof the cardholder participated: prior undisputed transactions on the same card and account
    - 3-D Secure authentication result, if the charge was authenticated
    - account logins or service usage after the disputed charge
```

`txn_00025755` is a one-off purchase disputed as fraud on a three-year-old account by a
customer who has disputed before: the model scored it low (4.0%), the playbook sees friendly
fraud, and the evidence list points at Visa CE 3.0. `txn_00030225` is the opposite lesson:
the model was right to score it at 99% because the customer had already asked to cancel, and
the only correct response is to refund.

Evidence lists summarise public network guidance in illustrative form; check current Visa and
Mastercard rules before relying on them.
