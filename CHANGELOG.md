# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/). The public API is the `cbp` command line, the HTTP
service contract (`/v1/*`) and the documented Python modules.

## [Unreleased]

### Added

- **`cbp monitor`**: production monitoring on two clocks. Score and per-feature PSI against
  the training window, with a noise floor, are available as soon as traffic arrives. An
  observed-vs-expected dispute z-test runs on matured labels only. `--fail-on` gives scheduled
  jobs an exit code. See `docs/monitoring.md` and the generated `docs/monitoring/report.md`.
- Roadmap Phase 2.

## [1.0.0] - 2026-11-25

First stable release: a complete, synthetic-data pipeline from raw billing events to a scored,
explained and argued (or conceded) dispute.

### Added

- **`cbp demo`**: runs the whole pipeline on a fresh dataset and narrates each step; `--record`
  writes an asciinema cast, a looping animated SVG and a transcript (`docs/demo/`). A test
  regenerates the recording so it can't drift from the code.
- **Architecture doc** (`docs/architecture.md`): component and request-path diagrams, what each
  stage owns, the cross-module invariants and the tests that enforce them.
- **Release workflow**: a `v*` tag is checked against the package version, the full suite runs,
  and a GitHub release is published with the wheel, sdist and this changelog's notes.
- README walkthrough from quick start to dispute response, and a documentation index.
- Scoring service: FastAPI `/v1/score` and `/v1/explain` with strict, catalog-pinned schemas,
  model ids on every response, OpenAPI docs and a self-contained Docker image (#8).
- Evaluation report: cost-weighted decisions (chargebacks vs. false declines), stratified and
  paired bootstrap intervals, a model card and a notebook with charts (#9).
- Representment evidence assembler: per-reason-code exhibits built point in time, readiness
  verdicts and draft response packets (#7).
- LLM reasoning layer: playbook-bounded strategy, validated LLM rationale with retry and a
  deterministic fallback, provider-agnostic client with an offline mock (#6).
- Explainability: exact interventional SHAP for the linear and tree models, top-factor
  summaries and single-feature dominance warnings (#5).
- Baseline models: logistic regression and gradient boosting on a label-aware time split,
  PR-AUC, precision@k and calibration, metrics implemented in NumPy (#4).
- Point-in-time feature pipeline with 30 documented features and explicit label censoring (#3).
- Seeded synthetic data generator: customers, subscriptions, retries, account-updater events
  and disputes under Visa 10.4/13.1/13.2 and Mastercard 4837/4853 (#2).
- Project scaffold: uv, ruff, strict mypy, pytest with coverage, GitHub Actions CI (#1).

[1.0.0]: https://github.com/sidchi/chargeback-prediction/releases/tag/v1.0.0
