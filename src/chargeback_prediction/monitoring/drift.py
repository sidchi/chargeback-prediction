"""Population Stability Index (PSI) and the binning it depends on, in plain NumPy.

PSI compares how a variable is distributed in a *reference* sample (the model's training
window) with a *current* sample (this month's traffic)::

    PSI = Σ_b (a_b − e_b) · ln(a_b / e_b)

where ``e_b`` and ``a_b`` are the reference and current shares of bin ``b``. It is the
symmetrised KL divergence over the bins, so it is zero for identical distributions, grows with
any shift, and does not care which direction the shift went. The conventional reading is
``< 0.1`` stable, ``0.1–0.25`` worth a look, ``> 0.25`` a material shift.

Two details decide whether PSI is trustworthy on payment features:

* **Bins come from the reference only**, at its quantiles, so the yardstick does not move with
  the traffic being measured. Most features here are flags or small counts; their quantiles
  repeat, and de-duplicating the edges turns each distinct value (or run of values) into its
  own bin instead of leaving bins empty. Open-ended outer bins catch values the reference
  never saw (a new $2,000 price point lands in the top bin rather than nowhere).
* **Empty bins are smoothed** with a small floor share, otherwise one value that never
  appeared in training makes PSI infinite.

PSI also has a *noise floor*: two samples from the same distribution still differ by chance.
Under no drift, ``n·m/(n+m) · PSI`` is approximately χ² with ``B − 1`` degrees of freedom, so
its expectation is ``(B − 1)·(1/n + 1/m)``. A 30-day window of ~2,000 charges puts that near
0.005, well under the 0.1 warning line; :func:`psi_noise_floor` reports it so a reader can tell
a small window's noise from a real shift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

DEFAULT_BINS = 10
"""Quantile bins for continuous variables; discrete ones get fewer after de-duplication."""

SMOOTHING_SHARE = 1e-4
"""Floor for an empty bin's share, so PSI stays finite when a bin is new or vanishes."""


def _as_1d(values: npt.ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


@dataclass(frozen=True, slots=True)
class Bins:
    """Upper-inclusive bin edges fitted on a reference sample.

    Bin ``i`` holds values in ``(edges[i-1], edges[i]]``; bin 0 is everything ``<= edges[0]``
    and the last bin (index ``len(edges)``) everything above ``edges[-1]``.
    """

    edges: tuple[float, ...]

    @property
    def count(self) -> int:
        return len(self.edges) + 1

    @classmethod
    def fit(cls, reference: npt.ArrayLike, *, n_bins: int = DEFAULT_BINS) -> Bins:
        """Edges at the reference's interior quantiles, de-duplicated for discrete data."""
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        ref = _as_1d(reference, "reference")
        quantiles = np.quantile(ref, np.arange(1, n_bins) / n_bins, method="lower")
        edges = np.unique(quantiles)
        # A constant reference still needs one edge so "anything else" has a bin to land in.
        return cls(edges=tuple(float(e) for e in edges))

    def assign(self, values: npt.ArrayLike) -> npt.NDArray[np.intp]:
        """Bin index for each value."""
        return np.searchsorted(np.asarray(self.edges), _as_1d(values, "values"), side="left")

    def shares(self, values: npt.ArrayLike) -> FloatArray:
        """Share of ``values`` in each bin (sums to 1)."""
        idx = self.assign(values)
        counts = np.bincount(idx, minlength=self.count).astype(np.float64)
        return counts / counts.sum()


def psi_from_shares(expected: FloatArray, actual: FloatArray) -> float:
    """PSI between two vectors of bin shares, with empty bins floored at ``SMOOTHING_SHARE``."""
    if expected.shape != actual.shape or expected.ndim != 1:
        raise ValueError("expected and actual must be 1-D share vectors of equal length")
    e = np.maximum(expected, SMOOTHING_SHARE)
    a = np.maximum(actual, SMOOTHING_SHARE)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def population_stability_index(
    reference: npt.ArrayLike, current: npt.ArrayLike, *, n_bins: int = DEFAULT_BINS
) -> float:
    """PSI of ``current`` against ``reference``, binned on the reference's quantiles."""
    bins = Bins.fit(reference, n_bins=n_bins)
    return psi_from_shares(bins.shares(reference), bins.shares(current))


def psi_noise_floor(n_bins: int, n_reference: int, n_current: int) -> float:
    """Expected PSI between two samples of the *same* distribution (χ² approximation)."""
    if n_bins < 1 or n_reference < 1 or n_current < 1:
        raise ValueError("bin count and sample sizes must be positive")
    return (n_bins - 1) * (1.0 / n_reference + 1.0 / n_current)
