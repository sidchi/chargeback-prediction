# chargeback-prediction

[![CI](https://github.com/sidchi/chargeback-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/sidchi/chargeback-prediction/actions/workflows/ci.yml)

An ML + LLM pipeline that scores card-not-present transactions for **chargeback (dispute) risk**,
explains each score, and drafts a representment strategy when a dispute does land.

> **All data in this repository is synthetic.** No real cardholder, merchant or employer data is
> used anywhere — not in code, fixtures, notebooks or docs.

## The problem

For subscription and other card-not-present merchants, a chargeback costs far more than the
refunded amount: a network fee ($15–$100), operational time to fight it, and — above roughly a
1% dispute ratio — enrollment in network monitoring programs (Visa VDMP/VFMP, Mastercard ECM)
with escalating fines. Much of the volume is **friendly fraud**: the real cardholder forgot a
renewal or didn't recognise a descriptor and filed "fraud" (Visa 10.4) or "cancelled recurring"
(Visa 13.2) instead of contacting the merchant.

Two decisions benefit from a good risk score:

1. **Before the charge** — should a renewal be retried, sent a pre-billing reminder, routed for
   step-up authentication, or skipped?
2. **After the dispute** — is it worth representing, and what evidence wins under this reason code?

A score alone isn't actionable for an ops analyst, so every prediction comes with the factors
that drove it and a plain-language rationale.

## Architecture

```
                 ┌────────────────────┐
                 │ Synthetic generator│  customers, subscriptions, transactions,
                 │   (item 2)         │  disputes with Visa/MC reason codes
                 └─────────┬──────────┘
                           ▼
                 ┌────────────────────┐
                 │ Feature pipeline   │  velocity, account age, card-on-file updates,
                 │   (item 3)         │  retry history, geo/BIN mismatch, prior disputes
                 └─────────┬──────────┘
                           ▼
      ┌────────────────────┴───────────────────┐
      ▼                                        ▼
┌──────────────┐   time-split eval    ┌──────────────────┐
│ Models       │ ───────────────────▶ │ Evaluation       │  PR-AUC, precision@k,
│ LR + GBM (4) │                      │ report (9)       │  calibration, cost-weighted
└──────┬───────┘                      └──────────────────┘
       ▼
┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│ SHAP per     │ ──▶ │ LLM reasoning    │ ──▶ │ Representment        │
│ prediction(5)│     │ layer (6)        │     │ evidence packet (7)  │
└──────────────┘     └──────────────────┘     └──────────────────────┘
       │                      │
       └──────────┬───────────┘
                  ▼
        ┌────────────────────┐
        │ FastAPI /score,    │
        │ /explain  (8)      │
        └────────────────────┘
```

Numbers in parentheses map to items in [ROADMAP.md](ROADMAP.md).

### Design choices so far

- **Reason codes are a typed catalog** (`chargeback_prediction.reason_codes`), not strings
  scattered across modules. The generator, model features and evidence assembler all key off the
  same `ReasonCode` objects, normalised into `fraud` / `consumer` / `processing` categories so
  models can generalise across networks.
- **Time-based splits only** (from item 4): disputes arrive 30–120 days after the charge, so a
  random split leaks the future and overstates performance.
- **LLM is an explainer, not the scorer**: the calibrated model owns the probability; the LLM
  turns features + SHAP values into prose behind a provider-agnostic interface that is mocked in
  tests.

## Getting started

Requires [uv](https://docs.astral.sh/uv/) (it will fetch Python 3.12 if needed).

```bash
uv sync                 # create .venv and install dev dependencies
uv run cbp reason-codes # list the dispute reason codes the pipeline models
uv run cbp generate     # write a synthetic dataset to data/synthetic/*.csv
uv run cbp features     # build the point-in-time feature table at data/features.csv
```

## Synthetic data

`cbp generate` (or `chargeback_prediction.synthetic.generate()`) simulates a card-not-present
subscription merchant. It is seeded and pure-stdlib, so the same `--seed` always produces the
same dataset. The default run (5,000 customers over 540 days) yields about 40k authorizations
and a ~1.7% dispute rate, roughly half of it friendly fraud.

| Table | One row per | Notable columns |
|---|---|---|
| `customers` | account | `created_at`, `email_domain_type`, `card_bin`, `bin_country`, `persona`* |
| `subscriptions` | subscription | `plan` (monthly/annual), `price_cents`, `cancel_requested_at` |
| `transactions` | authorization attempt | `kind` (initial/renewal/one_off), `attempt`, `approved`, `decline_code`, `ip_country` |
| `card_updates` | card-on-file change | `source` (account_updater/customer) |
| `disputes` | chargeback | `reason_code`, `category`, `filed_at`, `is_friendly_fraud`* |

\* Ground-truth labels for slicing evaluation results. **Never use them as features.**

Four hidden personas generate the patterns the model has to learn:

| Persona | Behaviour | Typical codes |
|---|---|---|
| good | Pays; rare genuine "not received" disputes | 13.1 / 4853 |
| forgetful | Stops using the product, card gets reissued via account updater, disputes the renewal. 25% actually cancelled and were billed anyway (a *legitimate* dispute) | 13.2 / 4853, some 10.4 / 4837 |
| serial disputer | Repeatedly claims fraud or non-receipt on their own one-off orders | 10.4 / 4837, 13.1 / 4853 |
| fraudster | Stolen card: new account, disposable email, card-testing declines, IP vs. BIN country mismatch, purchase burst within 72h | 10.4 / 4837 |

Renewals retry soft declines (`51`, `05`) with fixed spacing and stop on a hard decline (`54`).
Disputes are only emitted if **filed** inside the window, mirroring production label
censoring: the newest transactions have not had time to be disputed yet.

## Features

`cbp features` (or `chargeback_prediction.features.build_features()`) reads the CSV tables
back with `read_dataset` and emits **one row per approved transaction** with 30 numeric
features plus labels. The full documented table is in [docs/features.md](docs/features.md);
it is rendered from the catalog in `features/spec.py`, and a test fails if the two drift.

| Group | Examples |
|---|---|
| transaction | `amount_usd`, `is_renewal`, `attempt` (retry number) |
| account | `account_age_days`, `email_disposable` |
| geo / BIN | `ip_bin_mismatch`, `billing_bin_mismatch`, `ip_missing` |
| velocity | `attempts_1h/24h/7d`, `declines_24h/30d`, `approved_usd_24h` |
| card on file | `au_updates_90d` (account-updater reissues), `days_since_card_update` |
| subscription / retry | `prior_subscription_declines`, `charged_after_cancel_request` |
| prior disputes | `prior_disputes`, `prior_fraud_disputes`, `days_since_last_dispute` |

**Point-in-time correctness is the main design constraint.** Each customer's and
subscription's events are indexed into sorted timelines with prefix sums, and every query asks
for events in `[t - window, t)`: nothing at or after the charge is visible. Prior-dispute
features count disputes by **filing** date — a dispute on last month's charge that is filed
next week is not something the merchant knew at authorization, and counting it by charge date
is the most common leak in chargeback models. The default dataset (~37k rows) builds in under
a second in pure Python.

**Label censoring is explicit.** `label_mature` is true only once the 120-day dispute window
has fully elapsed before the snapshot (`observed_until`). Recent rows with `disputed = false`
are "not yet", not "no", and the time-based split in item 4 trains on mature rows only.

Univariate lift on the default dataset (mature rows, base dispute rate 1.9%) confirms each
planted pattern survives the pipeline as signal:

| Feature | Share of rows | Dispute rate | Lift |
|---|---|---|---|
| `charged_after_cancel_request` > 0 | 0.2% | 86.4% | 45× |
| `declines_24h` > 0 | 0.3% | 68.8% | 36× |
| `email_disposable` > 0 | 0.5% | 66.9% | 35× |
| `prior_fraud_disputes` > 0 | 0.5% | 31.4% | 17× |
| `ip_bin_mismatch` > 0 | 1.5% | 24.8% | 13× |
| `au_updates_90d` > 0 | 2.5% | 7.4% | 4× |

## Development

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy                  # strict type-check
uv run pytest                # tests + coverage
```

CI runs the same four steps on every push and pull request.

## Project layout

```
src/chargeback_prediction/   package (src layout, typed: py.typed)
  reason_codes.py            Visa / Mastercard dispute reason-code catalog
  synthetic/                 seeded generator: schema, config, personas, CSV reader/writer
  features/                  feature catalog (spec.py) and point-in-time pipeline
  cli.py                     `cbp` command-line entry point
tests/                       pytest suite
docs/features.md             generated feature table
.github/workflows/ci.yml     lint, format, type-check, test
```

## License

MIT
