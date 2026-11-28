# Model monitoring

`cbp monitor` checks a trained model against the traffic that arrived **after its training
date**. It is the piece that tells you when a model needs retraining, rather than finding out
from next quarter's chargeback invoice.

```bash
uv run cbp monitor                                 # logreg, 30-day windows, artifacts/monitoring/
uv run cbp monitor --model gbm --window-days 14
uv run cbp monitor --current data/june.csv --fail-on alert   # scheduled job: non-zero exit on alert
```

The generated report for the default dataset is [monitoring/report.md](monitoring/report.md)
(machine-readable: [monitoring/monitoring.json](monitoring/monitoring.json)).

> All data is synthetic.

## Two clocks

A chargeback label arrives 30–120 days after the charge. A model that went wrong today shows up
in dispute counts about four months from now, so monitoring runs on two clocks:

| Clock | Check | Available | Can raise |
|---|---|---|---|
| Leading | **Score PSI** against the training window: the distribution every decision consumes | the day traffic arrives | warn / alert |
| Leading | **Feature PSI** for each of the 30 inputs: *why* the score moved | the day traffic arrives | warn only |
| Leading | **Mean score**: the model's own forecast of the dispute rate | the day traffic arrives | (reported) |
| Lagging | **Observed vs. expected disputes** on rows whose dispute window has closed | ~120 days later | warn / alert |

Feature drift can only warn. A feature can move a lot without moving the score (it may carry
little weight), so it is a diagnosis, not a symptom. The score and calibration checks are the
ones that page someone.

## The statistics

**PSI** is `Σ (aᵦ − eᵦ) · ln(aᵦ / eᵦ)` over bins, i.e. a symmetrised KL divergence.
Implementation choices (`monitoring/drift.py`):

- **Bins come from the training window**, at its deciles, so the yardstick doesn't move with the
  traffic it measures. Outer bins are open-ended, so a value never seen in training (a new
  price point) still lands in a bin.
- **Discrete features get their own bins.** Most inputs are flags or small counts, so their
  deciles repeat. De-duplicating the edges turns each distinct value into a bin instead of
  leaving nine empty ones.
- **Empty bins are floored** at a 0.01% share, so a category that appears or vanishes gives a
  large PSI, not an infinite one.
- **Noise floor.** Two samples from the *same* distribution still differ: under no drift,
  `n·m/(n+m) · PSI` is roughly χ² with `B − 1` degrees of freedom, so the expected PSI is
  `(B − 1)(1/n + 1/m)`. The report prints it for each window. At ~2,000 charges per window it
  is about 0.005, far below the 0.1 warning line, so 30-day windows are big enough. A test
  checks the formula against simulation.

**Calibration** uses a count test, not ECE. If the model is calibrated, a window's dispute
count is a sum of independent Bernoulli(pᵢ) draws, so

```
z = (observed − Σ pᵢ) / sqrt(Σ pᵢ (1 − pᵢ))
```

is approximately standard normal. A 30-day window holds only a few dozen disputes, which is too
few for a stable ten-bin reliability table, but the count test stays well-defined and has a
known false-alarm rate. At |z| ≥ 3 that is about 0.3% of healthy windows; a test checks this by
simulation. `z > 0` means the model is *under*-predicting disputes, which is the expensive
direction.

**Only matured labels count.** A row whose 120-day window is still open has `disputed = false`
because it hasn't been disputed *yet*. Counting those rows would make every recent window look
over-predicted. The test runs once at least 500 of a window's rows have matured. Maturity
depends only on `created_at`, so the matured part of a window is an unbiased sample of it with
respect to the label.

## What it finds on the default dataset

The [generated report](monitoring/report.md) flags every window for score drift, with PSI
around 0.6. The top drivers are `subscription_age_days`, `prior_approved_renewals` and
`prior_subscription_declines`, and the share of first charges (`is_initial`) falls from 21% to
2%. That is a real property of the data, not a bug. The synthetic merchant launches
on 2025-01-01, so the training window (January to July 2025) is a **young book**: mostly first
charges and first renewals. A year later the same customers are on their ninth renewal, so
tenure features sit far outside anything the model trained on.

The lagging check then answers the question that matters: **drift is not degradation.**

| Model | Score PSI | Matured windows | Calibration |
|---|---|---|---|
| Logistic regression | 0.56–0.70 | 3 | holds: z = −1.9, −0.6, −0.9 |
| Gradient boosting | 0.67–0.82 | 3 | **fails** in Dec 2025: 72 disputes vs. 47 expected, z = +4.0 |

The linear model extrapolates the tenure trend smoothly and stays calibrated. Trees can't
extrapolate: every value beyond the training range lands in the same outermost leaf, which is
the likely reason the tree model under-predicts the December renewal wave. That is a second, independent reason to prefer the logistic regression on top of
its PR-AUC (see the [evaluation](evaluation/report.md)). It is also why the score-drift alert
still matters for the calibrated model: a policy that blocks "the riskiest 1% of traffic"
blocks a different population once the score distribution moves, even if each probability
is right.

## Design notes and limits

- **The reference is the training window**, because that is what the model learned from. A
  "first month in production" baseline would hide the tenure drift above, and the tenure
  drift is exactly what a retraining decision needs to see.
- **Windows are judged independently.** With eight windows and |z| ≥ 3 the family-wise
  false-alarm rate is still about 2%. A production version would add a CUSUM or sequential test
  on the dispute count, which catches slow miscalibration sooner than a fixed window can.
- **Labels mature late.** In the default run, the last four months have no matured labels.
  For those months the drift checks are the only evidence, which is why they alert on their own
  instead of waiting for labels.
- **What production would add:** per-segment monitoring (network, BIN country, plan), alert
  routing and de-duplication, and a champion/challenger shadow score so "retrain" has a
  candidate ready.
