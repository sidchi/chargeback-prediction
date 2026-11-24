"""Cost-weighted evaluation: what a dispute-risk score is worth in dollars.

PR-AUC says how well a model *ranks*; a merchant needs to know what acting on the ranking
*costs*. Every charge ends up in one of four cells, and the two error cells have very different
prices that both scale with the charge amount::

                       disputed                         not disputed
    blocked     dispute avoided (action cost)    false decline: lost margin + churn risk
    approved    chargeback: unrecovered amount   normal sale (no cost)
                + network fee + ops handling

"Block" stands for whatever the merchant does with a high-risk renewal: skip the charge,
require step-up authentication, or hold it for review. Blocking a charge that would have been
disputed costs nothing beyond the action itself, because that revenue would have been clawed
back by the chargeback anyway.

Because the costs depend on the amount, the cost-minimising rule is per transaction, not one
global threshold. With a calibrated probability ``p`` of dispute, blocking is cheaper when::

    p · C_chargeback(amount)  >  (1 − p) · C_false_decline(amount) + C_action

which is why the baseline models keep their probabilities calibrated (no class re-weighting).

All dollar figures in :class:`CostModel` are illustrative defaults for a digital subscription
merchant, not benchmarks; override them with your own unit economics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class CostModel:
    """Unit economics that turn confusion-matrix cells into dollars."""

    dispute_fee_usd: float = 25.0
    """Network/acquirer fee charged per chargeback, win or lose."""
    ops_cost_per_dispute_usd: float = 20.0
    """Analyst time to triage the dispute and assemble a representment packet."""
    representment_recovery_rate: float = 0.25
    """Expected share of the disputed amount recovered through representment (win rate × amount)."""
    gross_margin: float = 0.80
    """Margin lost when a legitimate charge is blocked (digital goods have high margins)."""
    false_decline_churn_rate: float = 0.15
    """Probability a falsely declined subscriber cancels instead of updating their card."""
    customer_lifetime_value_usd: float = 120.0
    """Future margin lost when that subscriber churns."""
    action_cost_usd: float = 0.10
    """Cost of acting on a flag at all: a step-up challenge, a reminder email, a review slot."""

    def __post_init__(self) -> None:
        for name in (
            "dispute_fee_usd",
            "ops_cost_per_dispute_usd",
            "customer_lifetime_value_usd",
            "action_cost_usd",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("representment_recovery_rate", "gross_margin", "false_decline_churn_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def chargeback_cost(self, amounts: npt.ArrayLike) -> FloatArray:
        """Cost of approving a charge that is later disputed."""
        a = np.asarray(amounts, dtype=np.float64)
        return (
            a * (1.0 - self.representment_recovery_rate)
            + self.dispute_fee_usd
            + self.ops_cost_per_dispute_usd
        )

    def false_decline_cost(self, amounts: npt.ArrayLike) -> FloatArray:
        """Cost of blocking a charge that would not have been disputed."""
        a = np.asarray(amounts, dtype=np.float64)
        return (
            a * self.gross_margin + self.false_decline_churn_rate * self.customer_lifetime_value_usd
        )

    def break_even_probability(self, amounts: npt.ArrayLike) -> FloatArray:
        """Dispute probability above which blocking is the cheaper action, per amount."""
        cb, fd = self.chargeback_cost(amounts), self.false_decline_cost(amounts)
        return np.asarray((fd + self.action_cost_usd) / (cb + fd), dtype=np.float64)

    def scaled(self, false_decline_multiplier: float) -> CostModel:
        """Copy with the false-decline side scaled, for sensitivity analysis.

        Scales margin (capped at 1) and lifetime value together, so the false-decline cost of
        every amount scales by the multiplier unless the margin cap binds.
        """
        if false_decline_multiplier <= 0:
            raise ValueError("multiplier must be > 0")
        return replace(
            self,
            gross_margin=min(1.0, self.gross_margin * false_decline_multiplier),
            customer_lifetime_value_usd=self.customer_lifetime_value_usd * false_decline_multiplier,
        )


class PolicyKind(StrEnum):
    APPROVE_ALL = "approve_all"
    EXPECTED_COST = "expected_cost"
    REVIEW_BUDGET = "review_budget"
    HINDSIGHT_THRESHOLD = "hindsight_threshold"
    PERFECT_FORESIGHT = "perfect_foresight"

    @property
    def label(self) -> str:
        return _POLICY_LABELS[self]


_POLICY_LABELS = {
    PolicyKind.APPROVE_ALL: "Approve everything (no model)",
    PolicyKind.EXPECTED_COST: "Expected-cost rule (per amount)",
    PolicyKind.REVIEW_BUDGET: "Fixed review budget (top-k)",
    PolicyKind.HINDSIGHT_THRESHOLD: "Best single threshold, in hindsight",
    PolicyKind.PERFECT_FORESIGHT: "Perfect foresight (lower bound)",
}


@dataclass(frozen=True, slots=True)
class PolicyOutcome:
    """Confusion-matrix counts and dollar costs of one blocking policy on one window."""

    kind: PolicyKind
    n: int
    blocked: int
    disputes: int
    disputes_blocked: int
    chargeback_cost: float
    false_decline_cost: float
    action_cost: float

    @property
    def false_declines(self) -> int:
        return self.blocked - self.disputes_blocked

    @property
    def disputes_approved(self) -> int:
        return self.disputes - self.disputes_blocked

    @property
    def total_cost(self) -> float:
        return self.chargeback_cost + self.false_decline_cost + self.action_cost

    @property
    def cost_per_1k(self) -> float:
        """Total cost per 1,000 charges, comparable across windows of different size."""
        return 1_000.0 * self.total_cost / self.n

    @property
    def block_rate(self) -> float:
        return self.blocked / self.n

    @property
    def residual_dispute_ratio(self) -> float:
        """Disputes per approved charge: the ratio network monitoring programs watch."""
        approved = self.n - self.blocked
        return self.disputes_approved / approved if approved else 0.0

    def savings_vs(self, baseline: PolicyOutcome) -> float:
        return baseline.total_cost - self.total_cost


def _check(
    y_true: npt.ArrayLike, amounts: npt.ArrayLike, other: npt.ArrayLike, name: str
) -> tuple[BoolArray, FloatArray]:
    y = np.asarray(y_true, dtype=np.bool_)
    a = np.asarray(amounts, dtype=np.float64)
    if y.ndim != 1 or a.shape != y.shape or np.shape(other) != y.shape:
        raise ValueError(f"y_true, amounts and {name} must be 1-D arrays of equal length")
    if y.size == 0:
        raise ValueError("need at least one row")
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("amounts must be finite and non-negative")
    return y, a


def policy_outcome(
    kind: PolicyKind,
    y_true: npt.ArrayLike,
    amounts: npt.ArrayLike,
    blocked: npt.ArrayLike,
    costs: CostModel,
) -> PolicyOutcome:
    """Score a set of block decisions against the realised outcomes."""
    y, a = _check(y_true, amounts, blocked, "blocked")
    b = np.asarray(blocked, dtype=np.bool_)
    return PolicyOutcome(
        kind=kind,
        n=int(y.size),
        blocked=int(b.sum()),
        disputes=int(y.sum()),
        disputes_blocked=int((y & b).sum()),
        chargeback_cost=float(costs.chargeback_cost(a[y & ~b]).sum()),
        false_decline_cost=float(costs.false_decline_cost(a[~y & b]).sum()),
        action_cost=float(b.sum() * costs.action_cost_usd),
    )


def expected_cost_decisions(
    probabilities: npt.ArrayLike, amounts: npt.ArrayLike, costs: CostModel
) -> BoolArray:
    """Block exactly the charges whose expected cost of approving exceeds that of blocking."""
    p = np.asarray(probabilities, dtype=np.float64)
    a = np.asarray(amounts, dtype=np.float64)
    if p.shape != a.shape:
        raise ValueError("probabilities and amounts must have equal shape")
    if ((p < 0) | (p > 1) | ~np.isfinite(p)).any():
        raise ValueError("probabilities must be in [0, 1]")
    approve = p * costs.chargeback_cost(a)
    block = (1.0 - p) * costs.false_decline_cost(a) + costs.action_cost_usd
    return np.asarray(approve > block, dtype=np.bool_)


def top_k_decisions(scores: npt.ArrayLike, k: int) -> BoolArray:
    """Block the ``k`` highest-scored charges (stable tie-break by row order)."""
    s = np.asarray(scores, dtype=np.float64)
    if not 0 <= k <= s.size:
        raise ValueError(f"k must be in [0, {s.size}], got {k}")
    out = np.zeros(s.size, dtype=np.bool_)
    out[np.argsort(-s, kind="mergesort")[:k]] = True
    return out


@dataclass(frozen=True, slots=True)
class CostCurve:
    """Total cost of blocking the top-``k`` charges by score, for every ``k`` from 0 to n.

    Only cut points between distinct scores are reachable with a probability threshold; the
    curve includes the others too, so it reads as "block this share of traffic".
    """

    blocked: npt.NDArray[np.int64]
    total_cost: FloatArray
    thresholds: FloatArray
    """Score of the ``k``-th ranked charge (``inf`` for k = 0)."""

    @property
    def best_index(self) -> int:
        return int(np.argmin(self.total_cost))


def cost_curve(
    y_true: npt.ArrayLike, scores: npt.ArrayLike, amounts: npt.ArrayLike, costs: CostModel
) -> CostCurve:
    """Vectorised cost of every top-``k`` blocking policy in O(n log n)."""
    y, a = _check(y_true, amounts, scores, "scores")
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    y_o, a_o = y[order], a[order]
    cb = np.where(y_o, costs.chargeback_cost(a_o), 0.0)
    fd = np.where(~y_o, costs.false_decline_cost(a_o), 0.0)
    k = np.arange(y.size + 1)
    # Blocking the top k: pay false declines among them, chargebacks among the rest.
    fd_blocked = np.r_[0.0, np.cumsum(fd)]
    cb_approved = cb.sum() - np.r_[0.0, np.cumsum(cb)]
    total = fd_blocked + cb_approved + k * costs.action_cost_usd
    thresholds = np.r_[np.inf, s[order]]
    return CostCurve(blocked=k, total_cost=total, thresholds=thresholds)


def hindsight_threshold_decisions(
    y_true: npt.ArrayLike, scores: npt.ArrayLike, amounts: npt.ArrayLike, costs: CostModel
) -> tuple[BoolArray, float]:
    """Cheapest single score threshold chosen *on the evaluation window itself*.

    Not achievable in production (it peeks at the labels); it is reported as an upper bound on
    what any global threshold could have done, to show what the expected-cost rule leaves on
    the table without tuning. Only cut points between distinct scores are considered.
    Returns the decisions and the threshold (block when ``score >= threshold``).
    """
    curve = cost_curve(y_true, scores, amounts, costs)
    s_sorted = curve.thresholds[1:]
    # Valid cut after position k (block k rows) iff k == 0, k == n, or score changes there.
    k = np.arange(s_sorted.size + 1)
    valid = np.ones(k.size, dtype=np.bool_)
    if s_sorted.size > 1:
        valid[1:-1] = s_sorted[:-1] != s_sorted[1:]
    candidates = np.flatnonzero(valid)
    best_k = int(candidates[np.argmin(curve.total_cost[candidates])])
    threshold = float(curve.thresholds[best_k])
    s = np.asarray(scores, dtype=np.float64)
    return np.asarray(s >= threshold, dtype=np.bool_), threshold
