"""Ranking and calibration metrics for rare-event scoring, in plain NumPy.

Dispute rates sit around 1–2%, so ROC-AUC flatters every model (it is dominated by the easy
negatives). The metrics that map to decisions are:

* **PR-AUC (average precision)** — area under precision/recall; a random scorer gets the base
  rate, so lift over base rate is the honest headline.
* **precision@k** — "if ops can review (or step-up) the top *k* charges, how many are real
  disputes?" Reported at fixed review budgets as a share of traffic.
* **calibration** — downstream decisions (retry or not, represent or not) multiply the score by
  a dollar amount, so a 0.30 must mean 30%. Reported as a reliability table, Brier score and
  expected calibration error (ECE).

Implemented here rather than imported so the definitions (tie handling, bin edges) are explicit
and unit-tested; the test-suite cross-checks them against scikit-learn.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

ArrayLike = npt.ArrayLike


def _validate(
    y_true: ArrayLike, scores: ArrayLike
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    y = np.asarray(y_true, dtype=np.bool_)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape:
        raise ValueError("y_true and scores must be 1-D arrays of equal length")
    if y.size == 0:
        raise ValueError("need at least one row")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    return y, s


def average_precision(y_true: ArrayLike, scores: ArrayLike) -> float:
    """Area under the precision-recall curve, as the step-wise sum ``Σ (Rₙ − Rₙ₋₁) Pₙ``.

    Tied scores form one threshold, so the result does not depend on row order. Matches
    :func:`sklearn.metrics.average_precision_score`. Returns ``nan`` with no positives.
    """
    y, s = _validate(y_true, scores)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    # Last index of each run of tied scores = one threshold on the curve.
    distinct = np.flatnonzero(np.diff(s_sorted)) if s_sorted.size > 1 else np.array([], int)
    threshold_idx = np.r_[distinct, s_sorted.size - 1]
    tp = np.cumsum(y_sorted)[threshold_idx]
    predicted_pos = threshold_idx + 1
    precision = tp / predicted_pos
    recall = tp / n_pos
    recall_step = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_step * precision))


def precision_at_k(y_true: ArrayLike, scores: ArrayLike, k: int) -> float:
    """Share of disputes among the ``k`` highest-scored rows.

    Ties at the boundary are broken by original row order (stable sort), which is how a review
    queue that takes "the first k" would behave.
    """
    y, s = _validate(y_true, scores)
    if not 1 <= k <= y.size:
        raise ValueError(f"k must be in [1, {y.size}], got {k}")
    top = np.argsort(-s, kind="mergesort")[:k]
    return float(y[top].mean())


def recall_at_k(y_true: ArrayLike, scores: ArrayLike, k: int) -> float:
    """Share of all disputes captured in the ``k`` highest-scored rows."""
    y, s = _validate(y_true, scores)
    if not 1 <= k <= y.size:
        raise ValueError(f"k must be in [1, {y.size}], got {k}")
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    top = np.argsort(-s, kind="mergesort")[:k]
    return float(y[top].sum() / n_pos)


def brier_score(y_true: ArrayLike, scores: ArrayLike) -> float:
    """Mean squared error between predicted probability and outcome (lower is better)."""
    y, s = _validate(y_true, scores)
    return float(np.mean((s - y) ** 2))


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_rate: float


def calibration_bins(
    y_true: ArrayLike, scores: ArrayLike, *, n_bins: int = 10
) -> tuple[CalibrationBin, ...]:
    """Reliability table with equal-*count* (quantile) bins.

    Equal-width bins put almost every row of a 2%-base-rate problem into the first bin;
    quantile bins give each point on the reliability curve the same statistical weight.
    Tied scores never straddle a bin edge, so fewer than ``n_bins`` bins may come back.
    """
    y, s = _validate(y_true, scores)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    order = np.argsort(s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    edges = sorted({round(i * s.size / n_bins) for i in range(n_bins + 1)})
    bins: list[CalibrationBin] = []
    start = 0
    for edge in edges[1:]:
        # Extend the bin so a run of tied scores stays together.
        end = edge
        while end < s.size and end > start and s_sorted[end] == s_sorted[end - 1]:
            end += 1
        if end <= start:
            continue
        chunk_s, chunk_y = s_sorted[start:end], y_sorted[start:end]
        bins.append(
            CalibrationBin(
                lower=float(chunk_s[0]),
                upper=float(chunk_s[-1]),
                count=int(chunk_s.size),
                mean_predicted=float(chunk_s.mean()),
                observed_rate=float(chunk_y.mean()),
            )
        )
        start = end
    return tuple(bins)


def expected_calibration_error(bins: Sequence[CalibrationBin]) -> float:
    """Count-weighted mean of ``|observed − predicted|`` across reliability bins."""
    total = sum(b.count for b in bins)
    if total == 0:
        raise ValueError("bins are empty")
    return float(sum(b.count * abs(b.observed_rate - b.mean_predicted) for b in bins) / total)
