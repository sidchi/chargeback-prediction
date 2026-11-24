"""Turn raw SHAP vectors into ranked, human-readable risk factors per transaction.

A disputes analyst does not want 29 signed floats. They want: how risky is this, compared with
normal, and the two or three things driving it, phrased in the units they think in ("4 declines
in the last 24 h", "account 3 days old"). Contributions are reported as **odds multipliers**
(``exp(phi)``) because those compose: three factors of ×2 make ×8 the odds.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from chargeback_prediction.explain.shapley import ShapValues
from chargeback_prediction.features import FEATURES, FeatureKind, FeatureSpec

DEFAULT_TOP_K = 3

MIN_CONTRIBUTION = 0.05
"""Factors moving the log-odds by less than this (about ±5% odds) are left out of summaries."""

_SPECS: Mapping[str, FeatureSpec] = {f.name: f for f in FEATURES}


class Direction(StrEnum):
    RAISES = "raises"
    LOWERS = "lowers"


def sigmoid(z: float) -> float:
    """Numerically stable logistic function."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def format_probability(p: float) -> str:
    """``12.3%``, but never rounds a near-certain score up to a misleading ``100.0%``."""
    if p > 0.999:
        return ">99.9%"
    if 0.0 < p < 0.001:
        return "<0.1%"
    return f"{p:.1%}"


def format_multiplier(m: float) -> str:
    """``×2.35``, ``×0.41``, ``×1,668``: two decimals until the number is big enough not to care."""
    return f"×{m:,.0f}" if m >= 100 else f"×{m:.2f}"


def format_value(feature: str, value: float) -> str:
    """Render a feature value in its natural unit, e.g. ``yes``, ``$49.99``, ``3.5 days``."""
    spec = _SPECS.get(feature)
    if spec is not None and spec.kind is FeatureKind.FLAG:
        return "yes" if value >= 0.5 else "no"
    if feature.endswith("_usd"):
        return f"${value:,.2f}"
    if spec is not None and spec.kind is FeatureKind.COUNT:
        return f"{value:,.0f}"
    if feature.endswith("_days") or feature.startswith("days_since"):
        return f"{value:,.1f} days"
    return f"{value:,.4g}"


@dataclass(frozen=True, slots=True)
class Factor:
    """One feature's contribution to one transaction's score."""

    feature: str
    value: float
    contribution: float
    """SHAP value in log-odds relative to the background average."""

    @property
    def direction(self) -> Direction:
        return Direction.RAISES if self.contribution >= 0 else Direction.LOWERS

    @property
    def odds_multiplier(self) -> float:
        return math.exp(self.contribution)

    @property
    def description(self) -> str:
        spec = _SPECS.get(self.feature)
        return spec.description if spec else ""

    def render(self) -> str:
        shown = format_value(self.feature, self.value)
        return f"{self.feature} = {shown} ({format_multiplier(self.odds_multiplier)} odds)"


@dataclass(frozen=True, slots=True)
class TransactionExplanation:
    transaction_id: str
    probability: float
    """Model's dispute probability for this transaction."""
    baseline_probability: float
    """Mean predicted probability over the background sample: "a typical transaction"."""
    base_value: float
    """Background mean log-odds; ``base_value + sum(all contributions)`` is this row's log-odds."""
    factors: tuple[Factor, ...]
    """Top factors by ``|contribution|``, largest first."""
    rest: float
    """Combined log-odds contribution of every feature not in ``factors``."""

    @property
    def relative_risk(self) -> float:
        return (
            self.probability / self.baseline_probability if self.baseline_probability else math.nan
        )

    def raising(self) -> tuple[Factor, ...]:
        return tuple(f for f in self.factors if f.direction is Direction.RAISES)

    def lowering(self) -> tuple[Factor, ...]:
        return tuple(f for f in self.factors if f.direction is Direction.LOWERS)

    def summary(self) -> str:
        """One paragraph for an analyst, e.g. in a review queue or a representment note."""
        head = (
            f"{self.transaction_id}: dispute risk {format_probability(self.probability)} "
            f"({self.relative_risk:.1f}× a typical transaction at {self.baseline_probability:.1%})."
        )
        parts = [head]
        if up := self.raising():
            parts.append("Raised by " + "; ".join(f.render() for f in up) + ".")
        if down := self.lowering():
            parts.append("Lowered by " + "; ".join(f.render() for f in down) + ".")
        if not self.factors:
            parts.append("No single factor stands out.")
        return " ".join(parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "transaction_id": self.transaction_id,
            "probability": round(self.probability, 6),
            "baseline_probability": round(self.baseline_probability, 6),
            "relative_risk": round(self.relative_risk, 3),
            "base_log_odds": round(self.base_value, 6),
            "factors": [
                {
                    "feature": f.feature,
                    "value": f.value,
                    "display_value": format_value(f.feature, f.value),
                    "shap_log_odds": round(f.contribution, 6),
                    "odds_multiplier": round(f.odds_multiplier, 4),
                    "direction": str(f.direction),
                    "description": f.description,
                }
                for f in self.factors
            ],
            "other_features_log_odds": round(self.rest, 6),
            "summary": self.summary(),
        }


def top_factors(
    contributions: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    feature_names: Sequence[str],
    *,
    k: int = DEFAULT_TOP_K,
    min_contribution: float = MIN_CONTRIBUTION,
) -> tuple[Factor, ...]:
    """The ``k`` largest-magnitude contributions at or above ``min_contribution``.

    Ties are broken by feature order so output is deterministic.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    order = sorted(range(len(feature_names)), key=lambda i: (-abs(contributions[i]), i))
    return tuple(
        Factor(feature_names[i], float(values[i]), float(contributions[i]))
        for i in order[:k]
        if abs(contributions[i]) >= min_contribution
    )


def explain_rows(
    shap: ShapValues,
    X: npt.NDArray[np.float64],
    transaction_ids: Sequence[str],
    *,
    baseline_probability: float,
    k: int = DEFAULT_TOP_K,
    min_contribution: float = MIN_CONTRIBUTION,
) -> list[TransactionExplanation]:
    """Build a :class:`TransactionExplanation` for each explained row."""
    if len(X) != len(shap) or len(transaction_ids) != len(shap):
        raise ValueError("X, transaction_ids and SHAP values must have the same number of rows")
    out: list[TransactionExplanation] = []
    for i, txn in enumerate(transaction_ids):
        phi = shap.values[i]
        factors = top_factors(phi, X[i], shap.feature_names, k=k, min_contribution=min_contribution)
        shown = sum(f.contribution for f in factors)
        out.append(
            TransactionExplanation(
                transaction_id=txn,
                probability=sigmoid(float(shap.raw[i])),
                baseline_probability=baseline_probability,
                base_value=shap.base_value,
                factors=factors,
                rest=float(phi.sum()) - shown,
            )
        )
    return out


DOMINANCE_LOG_ODDS = math.log(100.0)
"""A single feature multiplying the odds by 100× or more is flagged for review."""


@dataclass(frozen=True, slots=True)
class Importance:
    feature: str
    mean_abs: float
    """Mean ``|phi|``: how much the feature moves scores, in either direction."""
    share: float
    """``mean_abs`` as a share of the total over all features."""
    max_abs: float = 0.0
    """Largest ``|phi|`` on any row: catches rare features with outsized single-row impact."""

    @property
    def dominant(self) -> bool:
        """On some row this feature alone moves the odds by at least ×100.

        Usually a sparse feature the model saw only a handful of times in training (near
        separation), so its weight is extrapolated. Worth a human look before trusting scores.
        """
        return self.max_abs >= DOMINANCE_LOG_ODDS


def global_importance(shap: ShapValues) -> tuple[Importance, ...]:
    """Features ranked by mean absolute SHAP value over the explained rows."""
    mean_abs = shap.mean_abs()
    max_abs = np.abs(shap.values).max(axis=0) if len(shap) else np.zeros_like(mean_abs)
    total = float(mean_abs.sum()) or 1.0
    ranked = sorted(
        zip(shap.feature_names, mean_abs, max_abs, strict=True), key=lambda t: (-t[1], t[0])
    )
    return tuple(Importance(name, float(v), float(v) / total, float(mx)) for name, v, mx in ranked)
