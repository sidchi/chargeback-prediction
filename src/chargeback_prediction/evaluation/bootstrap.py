"""Bootstrap confidence intervals for evaluation metrics.

A test window of a few thousand charges holds only one or two hundred disputes, so a PR-AUC
of 0.35 could easily be 0.28 or 0.42 on another month. Every headline number in the
evaluation report carries a percentile bootstrap interval, and model comparisons use a
*paired* bootstrap (both models scored on the same resampled rows), which is far tighter than
comparing two independent intervals.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

DEFAULT_RESAMPLES = 1_000


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate with a percentile bootstrap confidence interval."""

    estimate: float
    low: float
    high: float
    level: float

    def __post_init__(self) -> None:
        if not 0.0 < self.level < 1.0:
            raise ValueError("level must be in (0, 1)")

    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    def format(self, spec: str = ".3f") -> str:
        return f"{self.estimate:{spec}} [{self.low:{spec}}, {self.high:{spec}}]"


Statistic = Callable[..., float]


def bootstrap(
    statistic: Statistic,
    arrays: Sequence[npt.ArrayLike],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    level: float = 0.95,
    seed: int = 0,
    strata: npt.ArrayLike | None = None,
) -> Interval:
    """Percentile bootstrap of ``statistic(*arrays)`` over rows.

    ``strata`` (e.g. the labels) resamples within each group separately so every replicate
    keeps the observed number of positives; with a 2% base rate an unstratified replicate can
    otherwise draw noticeably more or fewer disputes, and PR-AUC moves with the base rate.
    Replicates where the statistic is ``nan`` are dropped.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    data = [np.asarray(a) for a in arrays]
    if not data:
        raise ValueError("need at least one array")
    n = data[0].shape[0]
    if n == 0 or any(d.shape[0] != n for d in data):
        raise ValueError("arrays must be non-empty and of equal length")
    groups: list[npt.NDArray[np.intp]]
    if strata is None:
        groups = [np.arange(n)]
    else:
        st = np.asarray(strata)
        if st.shape != (n,):
            raise ValueError("strata must have one entry per row")
        groups = [np.flatnonzero(st == v) for v in np.unique(st)]

    rng = np.random.default_rng(seed)
    estimate = float(statistic(*data))
    replicates = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = np.concatenate([g[rng.integers(0, g.size, g.size)] for g in groups])
        replicates[i] = statistic(*(d[idx] for d in data))
    replicates = replicates[np.isfinite(replicates)]
    if replicates.size == 0:
        return Interval(estimate, float("nan"), float("nan"), level)
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(replicates, [tail, 100.0 - tail])
    return Interval(estimate, float(low), float(high), level)
