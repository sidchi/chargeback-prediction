# Roadmap — chargeback-prediction
ML + LLM pipeline that scores card transactions for dispute risk and explains each score. All data is synthetic. No real cardholder or employer data, ever. Each item is one PR. The daily pipeline takes the first unchecked item.
- [x] 1. Scaffold: Python 3.12, uv/poetry, ruff, pytest, GitHub Actions CI, README with problem statement and architecture sketch
- [x] 2. Synthetic data generator: transactions, subscriptions, customers, and disputes with realistic reason codes (Visa 10.4, 13.1, 13.2; Mastercard 4837, 4853) and fraud/friendly-fraud patterns
- [x] 3. Feature pipeline: velocity, account age, card-on-file updates, retry history, geo/BIN mismatch, prior disputes; documented feature table
- [x] 4. Baseline model: logistic regression + gradient boosting, time-based train/test split, PR-AUC, precision@k, calibration plot
- [x] 5. Explainability: SHAP values per prediction, top-factor summary per transaction
- [x] 6. LLM reasoning layer: turn features + SHAP + reason code into a human-readable risk rationale and a suggested representment strategy (provider-agnostic interface, mock for tests)
- [x] 7. Representment evidence assembler: given a dispute, gather the relevant evidence fields per reason code and produce a draft response packet
- [x] 8. FastAPI service: /score and /explain endpoints, Pydantic schemas, OpenAPI docs, Dockerfile
- [x] 9. Evaluation report: model card, cost-weighted metrics (chargeback fees vs. false declines), notebook with charts
- [x] 10. Polish: architecture diagram, README walkthrough, demo GIF or recorded CLI run, tagged v1.0 release

## Phase 2

v1.0 scores, explains and argues disputes on one static snapshot. Phase 2 makes the pipeline behave like a production system: it notices when the world moves, retrains safely, serves the same features it trained on, and turns scores into decisions with a measured payoff.
- [x] 11. Monitoring: score and per-feature PSI against the training window (with a sampling-noise floor), label-delayed calibration as an observed-vs-expected dispute z-test on matured rows only, `cbp monitor` report and `--fail-on` exit code for scheduled jobs
- [ ] 12. Walk-forward backtest: rolling-origin retraining with the label gap at each origin, metric and calibration stability over time, a data-backed retraining cadence (monthly vs. quarterly) priced with the cost model
- [ ] 13. Model registry + champion/challenger: versioned artifacts with content hash and feature-spec fingerprint, a promotion gate on paired-bootstrap savings, `/v1/score` serving the champion while shadow-scoring the challenger
- [ ] 14. Online feature computation: incremental per-customer state that computes the 30 features event by event, a parity test against the batch pipeline (training/serving skew), and a `/v1/score` variant that accepts a raw authorization instead of a feature vector
- [ ] 15. Pre-billing intervention policy: choose retry, reminder email, step-up (3DS) or skip per renewal from the score and per-action costs and effects; evaluate the policy offline on synthetic counterfactual outcomes
- [ ] 16. Dispute-ratio early warning: monthly forecast of the network dispute ratio with intervals from the scores of charges still inside their dispute window, alerts before the Visa VDMP / Mastercard ECM thresholds are crossed
- [ ] 17. Representment win-rate model: synthetic representment outcomes in the generator, a model of win probability from reason code and packet evidence, and fight/accept decisions by expected recovery net of fees
- [ ] 18. Segment robustness: PR-AUC, calibration and cost by network, BIN country, plan and tenure with bootstrap intervals; worst-segment section in the model card and per-segment checks in `cbp monitor`
- [ ] 19. Scoring performance: vectorised batch feature pipeline and a `/v1/score:batch` endpoint, latency benchmark (p50/p99) with a budget enforced in CI
