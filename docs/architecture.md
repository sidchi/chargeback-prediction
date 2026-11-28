# Architecture

How the pieces of `chargeback-prediction` fit together, what each one owns, and the invariants
that hold across them. Every component is a module under `src/chargeback_prediction/` and a
`cbp` subcommand; `cbp demo` runs them all in order (see [the recording](demo/transcript.txt)).

> All data is synthetic. Nothing here reads real cardholder, merchant or employer data.

## Components and data flow

```mermaid
flowchart LR
    subgraph offline["Offline: batch, reproducible from a seed"]
        direction TB
        gen["Synthetic generator<br/><code>synthetic/</code>"] -->|"5 CSV tables"| feat["Point-in-time features<br/><code>features/</code>"]
        feat -->|"feature table<br/>(30 features + labels)"| train["Time-split training<br/>LR + GBM<br/><code>modeling/</code>"]
        feat --> evalr["Cost-weighted evaluation<br/>bootstrap CIs, model card<br/><code>evaluation/</code>"]
        train -. "same split (retrained)" .-> evalr
    end

    subgraph online["Online: stateless service"]
        direction TB
        api["FastAPI<br/>/v1/score · /v1/explain<br/><code>service/</code>"]
        scorer["Scorer<br/>calibrated probability"]
        shap["Exact SHAP<br/>linear + tree<br/><code>explain/</code>"]
        api --> scorer --> shap
    end

    subgraph dispute["When a dispute lands"]
        direction TB
        playbook["Representment playbook<br/>rules pick allowed strategies"]
        llm["LLM client (mock / OpenAI-compatible)<br/>writes the rationale, validated"]
        evidence["Evidence assembler<br/>per-reason-code exhibits<br/><code>representment/</code>"]
        playbook --> llm
        playbook -. "strategy" .-> evidence
    end

    train -->|"model artifact"| scorer
    feat -. "feature vectors<br/>(feature store in prod)" .-> api
    shap -->|"factors"| playbook
    gen -. "raw history" .-> evidence
```

Solid arrows carry data the next stage reads; dotted arrows are looser couplings (the evaluator
retrains on the same split rather than loading the pickled models, and the service receives
feature vectors from the caller rather than computing them).

| Stage | Module | Input | Output | Owns |
|---|---|---|---|---|
| Generate | `synthetic/` | seed, size, window | `customers`, `subscriptions`, `transactions`, `card_updates`, `disputes` CSVs | personas and the patterns the model must find; label censoring |
| Features | `features/` | the five tables | one row per approved charge: 30 features, `disputed`, `label_mature` | point-in-time correctness |
| Train | `modeling/` | feature table | `model_{logreg,gbm}.joblib`, report, calibration plot | the time split with a 120-day label gap |
| Evaluate | `evaluation/` | feature table | cost report, model card, bootstrap intervals | dollar costs of each decision |
| Explain | `explain/` | model + feature rows | exact SHAP values, ranked factors | additivity: `base + Σ φ = log-odds` |
| Reason | `reasoning/` | score, factors, reason code | rationale, strategy, evidence list | which strategies are *allowed* (playbook) |
| Evidence | `representment/` | raw tables + dispute | response packet, readiness verdict | never claiming evidence it does not hold |
| Serve | `service/` | feature vectors (JSON) | probabilities, factors, rationale | the request contract |

## The `/v1/explain` request path

```mermaid
sequenceDiagram
    autonumber
    participant C as Caller (dispute ops tool)
    participant A as FastAPI app
    participant S as Scorer
    participant X as SHAP explainer
    participant P as Playbook
    participant L as LLM client

    C->>A: POST /v1/explain {features, reason_code}
    A->>A: validate: every feature present, typed, bounded
    A->>S: score(row)
    S->>X: shap(row, background)
    X-->>S: φ per feature (log-odds)
    S-->>A: probability, top factors
    opt reason_code given
        A->>P: recommend(features, reason_code)
        P-->>A: allowed strategies + evidence list
        A->>L: prompt(score, factors, allowed)
        L-->>A: JSON reply
        A->>A: validate reply, retry once, else template (source=fallback)
    end
    A-->>C: model_id, probability, factors, summary, rationale
```

The model and the SHAP background load once at startup, so a bad artifact fails the deploy
rather than the first request. The LLM is the only network call, sits only on the explain
path, and always has a deterministic fallback.

## Invariants

These hold across module boundaries, and each has a test that fails if it breaks.

1. **Nothing from the future.** Features at time `t` only see events in `[t − window, t)`, and
   prior disputes count by *filing* date, not charge date (`tests/test_features.py`).
2. **Training labels are final.** Train rows end 120 days before the test window starts, so
   every training dispute was filed before the first test charge (`tests/test_modeling.py`).
3. **Decisions are priced, not thresholded.** The cost rule
   `p · C_cb > (1 − p) · C_fd + C_action` needs calibrated `p`, so models are trained without
   class re-weighting and every report carries ECE; the cost curve is checked against brute
   force (`tests/test_metrics.py`, `tests/test_evaluation.py`).
4. **Explanations add up.** SHAP values reproduce each row's log-odds to ~1e-14 and match a
   brute-force enumeration of coalitions (`tests/test_explain.py`).
5. **Rules decide, the LLM explains.** The LLM can soften a strategy to *review* but never
   turn an *accept* into a fight, and cannot cite features it was not shown
   (`tests/test_reasoning.py`).
6. **One contract.** The service schema, the feature catalog and `docs/features.md` are
   generated from, or checked against, the same spec (`tests/test_service.py`,
   `tests/test_features.py`).
7. **Docs don't drift.** The feature table and the demo recording are regenerated in tests
   and compared with the committed copies, and the notebook is executed end to end
   (`tests/test_features.py`, `tests/test_demo.py`, `tests/test_notebook.py`).

## What production would change

| Here | In production |
|---|---|
| Synthetic generator | Event streams from the payment gateway, CRM and dispute feeds from the acquirer |
| Feature table built in batch | Feature store with the same point-in-time definitions, online and offline |
| Two baselines, fixed hyper-parameters | Cross-validated tuning, champion/challenger |
| `cbp monitor` batch report ([monitoring.md](monitoring.md)) | The same PSI and calibration checks on a schedule, per segment, with alert routing |
| Mock LLM by default | Hosted or self-hosted model behind the same `LLMClient` interface, with prompt and reply logging |
| `attach` items in packets | Connectors to 3DS, AVS/CVV, login logs and signed terms |
| One container | Service behind the authorization path's latency budget; explain/reason on the async ops path |
