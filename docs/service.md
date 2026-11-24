# Scoring service

`cbp serve` runs a FastAPI app over a model trained with `cbp train`. It answers two questions
over HTTP:

- **`POST /v1/score`**: how likely is each of these transactions to be disputed? Batch of up to
  1,000, one vectorised `predict_proba` call.
- **`POST /v1/explain`**: why did this one transaction score that way? Exact SHAP factors, a
  one-paragraph summary, and, if you pass the dispute's reason code, a representment
  rationale and strategy from the reasoning layer.

Plus `GET /v1/model` (what is loaded), `GET /health`, and interactive OpenAPI docs at `/docs`.

> All data is synthetic. The Docker image trains on a seeded synthetic dataset at build time.

## Run it

```bash
# local, after `cbp generate && cbp features && cbp train`
uv run cbp serve                      # http://127.0.0.1:8000/docs
uv run cbp serve --model gbm --port 9000

# Docker: self-contained, trains on synthetic data at build time
docker build -t chargeback-prediction .
docker run --rm -p 8000:8000 chargeback-prediction
```

```bash
curl -s localhost:8000/v1/score   -H 'Content-Type: application/json' -d @examples/score_request.json
curl -s localhost:8000/v1/explain -H 'Content-Type: application/json' -d @examples/explain_request.json
```

### Configuration

Command-line flags win; anything unset falls back to the environment.

| Variable | Flag | Default | Meaning |
|---|---|---|---|
| `CBP_MODELS_DIR` | `--models` | `artifacts/baseline` | `cbp train` output directory |
| `CBP_FEATURES` | `--features` | `data/features.csv` | feature table the SHAP background is sampled from |
| `CBP_MODEL` | `--model` | `logreg` | `logreg` or `gbm` |
| `CBP_BACKGROUND_SIZE` | | `300` | training rows in the SHAP background |
| `CBP_LLM_BASE_URL`, `CBP_LLM_MODEL` | | unset | OpenAI-compatible server for rationales; unset means the offline mock |
| `CBP_LLM_API_KEY` | | unset | bearer token for that server |

## Example

Request (`examples/explain_request.json`, features abbreviated):

```json
{
  "transaction": {
    "transaction_id": "txn_demo_renewal_after_cancel",
    "features": { "amount_usd": 9.99, "is_renewal": 1, "charged_after_cancel_request": 1, "...": "..." }
  },
  "top_k": 3,
  "reason_code": "visa:13.2"
}
```

Response (abbreviated):

```json
{
  "model_id": "logreg@4218dc11467c",
  "transaction_id": "txn_demo_renewal_after_cancel",
  "probability": 0.9971,
  "baseline_probability": 0.0185,
  "relative_risk": 53.93,
  "factors": [
    { "feature": "charged_after_cancel_request", "display_value": "yes", "odds_multiplier": 1668.4, "direction": "raises" },
    { "feature": "au_updates_90d", "display_value": "1", "odds_multiplier": 3.92, "direction": "raises" },
    { "feature": "prior_approved_renewals", "display_value": "12", "odds_multiplier": 3.43, "direction": "raises" }
  ],
  "summary": "txn_demo_renewal_after_cancel: dispute risk 99.7% (53.9× a typical transaction at 1.8%). Raised by charged_after_cancel_request = yes (×1,668 odds); ...",
  "rationale": {
    "reason_code": "visa:13.2",
    "strategy": "accept",
    "playbook_strategy": "accept",
    "source": "llm",
    "provider": "mock",
    "evidence": ["..."]
  }
}
```

A renewal charged after the customer asked to cancel is the one 13.2 dispute the merchant
should not fight, and the playbook says so.

## Design notes

**The API scores feature vectors, not raw transactions.** Every feature is a point-in-time
aggregate over the customer's history (velocity, prior disputes, account-updater reissues).
Computing those at request time would mean the service owns that history. In production it
belongs to a feature store fed by the same code as `cbp features`; the service stays stateless
and horizontally scalable. The cost is a wide request body (30 fields), which is why the
schema is strict.

**Strict, explicit contract.** `TransactionFeatures` lists every field with its type and bounds
(flags 0/1, counts non-negative integers, numerics finite and non-negative) and forbids unknown
fields. A misspelt or missing feature is a 422, never a silent zero that quietly shifts a
score. The fields are written out rather than generated from the catalog so the contract reads
cleanly in review and in `/docs`; a test fails if they drift from `features/spec.py`.

**Probabilities, not decisions.** The service returns a calibrated probability and the ratio to
a typical transaction. Whether 5% means "send a pre-billing reminder" or "skip the retry" is
business policy that changes faster than the model, so thresholds stay with the caller.

**Every response names its model.** `model_id` is the model kind plus a SHA-256 prefix of the
artifact, so a score in an audit log can be traced to the exact model even after a retrain
overwrites the artifact directory.

**Explanations agree with the offline reports.** The SHAP background is sampled from the
training window with the same split and seed as `cbp explain`, so `/v1/explain` and the
committed reports give identical factors for the same transaction (tested).

**Load once, fail at deploy time.** The model, its background and the LLM client are loaded in
the app's lifespan hook, and `Scorer.build` runs a one-row SHAP call so an unsupported model
fails startup instead of the first request. The container health check therefore only goes
green once the model is usable.

**Sync endpoints on purpose.** Scoring and SHAP are CPU-bound numpy work. FastAPI runs plain
`def` endpoints on a worker thread pool; `async def` would run them on the event loop and stall
every other request. The `Scorer` is immutable after construction, so sharing it across threads
is safe.

**The LLM never blocks a score.** `/v1/score` does not touch it. `/v1/explain` only calls it when
a reason code is given, and the reasoning layer falls back to a deterministic template
(`"source": "fallback"`) if the provider errors or keeps failing validation.

## Limits and next steps

- **Latency with a real LLM.** The OpenAI-compatible client allows 30 s per attempt and the
  reasoner makes up to two attempts, so a slow provider can hold an `/v1/explain` call for up to
  a minute. A production deployment would make rationales asynchronous (job + callback) or put
  a tighter deadline on them.
- **Background from the feature CSV.** The service reads the full feature table at startup just
  to sample 300 background rows. Persisting the background next to the model in `cbp train`
  would make the model artifact self-contained.
- **No auth or rate limiting.** It is an internal service; put it behind the platform's gateway.
- **Pickled models.** joblib artifacts execute code on load. Only serve artifacts you built.
