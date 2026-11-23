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
```

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
  cli.py                     `cbp` command-line entry point
tests/                       pytest suite
.github/workflows/ci.yml     lint, format, type-check, test
```

## License

MIT
