# Roadmap — chargeback-prediction
ML + LLM pipeline that scores card transactions for dispute risk and explains each score. All data is synthetic. No real cardholder or employer data, ever. Each item is one PR. The daily pipeline takes the first unchecked item.
- [x] 1. Scaffold: Python 3.12, uv/poetry, ruff, pytest, GitHub Actions CI, README with problem statement and architecture sketch
- [x] 2. Synthetic data generator: transactions, subscriptions, customers, and disputes with realistic reason codes (Visa 10.4, 13.1, 13.2; Mastercard 4837, 4853) and fraud/friendly-fraud patterns
- [x] 3. Feature pipeline: velocity, account age, card-on-file updates, retry history, geo/BIN mismatch, prior disputes; documented feature table
- [x] 4. Baseline model: logistic regression + gradient boosting, time-based train/test split, PR-AUC, precision@k, calibration plot
- [x] 5. Explainability: SHAP values per prediction, top-factor summary per transaction
- [x] 6. LLM reasoning layer: turn features + SHAP + reason code into a human-readable risk rationale and a suggested representment strategy (provider-agnostic interface, mock for tests)
- [ ] 7. Representment evidence assembler: given a dispute, gather the relevant evidence fields per reason code and produce a draft response packet
- [ ] 8. FastAPI service: /score and /explain endpoints, Pydantic schemas, OpenAPI docs, Dockerfile
- [ ] 9. Evaluation report: model card, cost-weighted metrics (chargeback fees vs. false declines), notebook with charts
- [ ] 10. Polish: architecture diagram, README walkthrough, demo GIF or recorded CLI run, tagged v1.0 release
