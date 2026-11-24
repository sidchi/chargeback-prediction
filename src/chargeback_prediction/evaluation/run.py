"""Evaluate trained baselines as *decisions*: dollars saved, with uncertainty, per model."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt

from chargeback_prediction.evaluation.bootstrap import DEFAULT_RESAMPLES, Interval, bootstrap
from chargeback_prediction.evaluation.costs import (
    CostCurve,
    CostModel,
    PolicyKind,
    PolicyOutcome,
    cost_curve,
    expected_cost_decisions,
    hindsight_threshold_decisions,
    policy_outcome,
    top_k_decisions,
)
from chargeback_prediction.features import FeatureTable
from chargeback_prediction.modeling import (
    BaselineReport,
    FeatureMatrix,
    ModelEvaluation,
    ModelKind,
    TrainConfig,
    average_precision,
    train_baselines,
)

AMOUNT_FEATURE = "amount_usd"
DEFAULT_SENSITIVITY: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
"""False-decline cost multipliers for the sensitivity table."""


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    costs: CostModel = field(default_factory=CostModel)
    review_budget: float = 0.01
    """Share of traffic the fixed-budget policy blocks (top-k by score)."""
    n_resamples: int = DEFAULT_RESAMPLES
    level: float = 0.95
    sensitivity: tuple[float, ...] = DEFAULT_SENSITIVITY
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.review_budget <= 1.0:
            raise ValueError("review_budget must be in (0, 1]")
        if any(m <= 0 for m in self.sensitivity):
            raise ValueError("sensitivity multipliers must be > 0")


@dataclass(frozen=True, slots=True)
class SensitivityRow:
    false_decline_multiplier: float
    outcome: PolicyOutcome
    approve_all: PolicyOutcome

    @property
    def savings_per_1k(self) -> float:
        return self.approve_all.cost_per_1k - self.outcome.cost_per_1k


@dataclass(frozen=True, slots=True)
class CategoryCapture:
    """How many disputes of one category the expected-cost rule blocked."""

    category: str
    disputes: int
    blocked: int

    @property
    def rate(self) -> float:
        return self.blocked / self.disputes if self.disputes else float("nan")


@dataclass(frozen=True, slots=True)
class ModelCostEvaluation:
    kind: ModelKind
    ranking: ModelEvaluation
    pr_auc: Interval
    savings_per_1k: Interval
    """Expected-cost rule vs. approve-all, dollars per 1,000 charges."""
    policies: Mapping[PolicyKind, PolicyOutcome]
    hindsight_threshold: float
    curve: CostCurve = field(repr=False)
    categories: tuple[CategoryCapture, ...]
    sensitivity: tuple[SensitivityRow, ...]

    @property
    def expected_cost(self) -> PolicyOutcome:
        return self.policies[PolicyKind.EXPECTED_COST]

    def capture_of_achievable(self, approve_all: PolicyOutcome, perfect: PolicyOutcome) -> float:
        """Share of the maximum possible saving (perfect foresight) the expected-cost rule gets."""
        possible = perfect.savings_vs(approve_all)
        return self.expected_cost.savings_vs(approve_all) / possible if possible else float("nan")


@dataclass(frozen=True, slots=True)
class PairedComparison:
    better: ModelKind
    worse: ModelKind
    pr_auc_difference: Interval
    savings_difference_per_1k: Interval


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    baseline: BaselineReport
    config: EvaluationConfig
    amounts: npt.NDArray[np.float64] = field(repr=False)
    """Test-window charge amounts, aligned with ``baseline.split.test``."""
    approve_all: PolicyOutcome
    perfect: PolicyOutcome
    models: tuple[ModelCostEvaluation, ...]
    comparisons: tuple[PairedComparison, ...]

    def recommended(self) -> ModelCostEvaluation:
        """Model whose expected-cost rule is cheapest on the test window."""
        return min(self.models, key=lambda m: m.expected_cost.total_cost)


def dispute_categories(table: FeatureTable) -> dict[str, str]:
    """``transaction_id → dispute_category`` for disputed rows of an in-memory feature table."""
    return {r.transaction_id: r.dispute_category for r in table.rows if r.dispute_category}


def read_dispute_categories(path: Path) -> dict[str, str]:
    """The same mapping from a feature CSV (the evaluation-only ``dispute_category`` column)."""
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "dispute_category" not in reader.fieldnames:
            raise ValueError(f"{path}: missing column 'dispute_category'")
        return {
            row["transaction_id"]: row["dispute_category"]
            for row in reader
            if row["dispute_category"]
        }


def expected_cost_savings_per_1k(
    y_true: npt.NDArray[np.bool_],
    amounts: npt.NDArray[np.float64],
    probabilities: npt.NDArray[np.float64],
    costs: CostModel,
) -> float:
    """Dollars per 1,000 charges the expected-cost rule saves over approving everything."""
    none = np.zeros(y_true.size, dtype=np.bool_)
    base = policy_outcome(PolicyKind.APPROVE_ALL, y_true, amounts, none, costs)
    rule = policy_outcome(
        PolicyKind.EXPECTED_COST,
        y_true,
        amounts,
        expected_cost_decisions(probabilities, amounts, costs),
        costs,
    )
    return base.cost_per_1k - rule.cost_per_1k


def evaluate_costs(
    baseline: BaselineReport,
    *,
    categories: Mapping[str, str] | None = None,
    config: EvaluationConfig | None = None,
) -> EvaluationReport:
    """Cost-weighted, bootstrapped evaluation of every model in a :class:`BaselineReport`."""
    cfg = config or EvaluationConfig()
    test = baseline.split.test
    if AMOUNT_FEATURE not in test.feature_names:
        raise ValueError(f"feature matrix has no {AMOUNT_FEATURE!r} column")
    amounts = test.X[:, test.feature_names.index(AMOUNT_FEATURE)].astype(np.float64)
    y = test.y
    costs = cfg.costs
    n = len(test)
    approve_all = policy_outcome(PolicyKind.APPROVE_ALL, y, amounts, np.zeros(n, bool), costs)
    perfect = policy_outcome(PolicyKind.PERFECT_FORESIGHT, y, amounts, y, costs)
    savings_stat = partial(expected_cost_savings_per_1k, costs=costs)
    k = max(1, round(cfg.review_budget * n))
    row_categories = [
        (categories or {}).get(tid, "unknown") if disputed else ""
        for tid, disputed in zip(test.transaction_ids, y, strict=True)
    ]

    models: list[ModelCostEvaluation] = []
    for ev in baseline.evaluations:
        p = ev.scores
        rule = expected_cost_decisions(p, amounts, costs)
        hindsight, threshold = hindsight_threshold_decisions(y, p, amounts, costs)
        policies = {
            PolicyKind.EXPECTED_COST: policy_outcome(
                PolicyKind.EXPECTED_COST, y, amounts, rule, costs
            ),
            PolicyKind.REVIEW_BUDGET: policy_outcome(
                PolicyKind.REVIEW_BUDGET, y, amounts, top_k_decisions(p, k), costs
            ),
            PolicyKind.HINDSIGHT_THRESHOLD: policy_outcome(
                PolicyKind.HINDSIGHT_THRESHOLD, y, amounts, hindsight, costs
            ),
        }
        captured = tuple(
            CategoryCapture(
                category=c,
                disputes=sum(1 for rc in row_categories if rc == c),
                blocked=int(
                    sum(1 for rc, b in zip(row_categories, rule, strict=True) if rc == c and b)
                ),
            )
            for c in sorted({rc for rc in row_categories if rc})
        )
        sensitivity = tuple(
            SensitivityRow(
                false_decline_multiplier=m,
                outcome=policy_outcome(
                    PolicyKind.EXPECTED_COST,
                    y,
                    amounts,
                    expected_cost_decisions(p, amounts, costs.scaled(m)),
                    costs.scaled(m),
                ),
                approve_all=policy_outcome(
                    PolicyKind.APPROVE_ALL, y, amounts, np.zeros(n, bool), costs.scaled(m)
                ),
            )
            for m in cfg.sensitivity
        )
        models.append(
            ModelCostEvaluation(
                kind=ev.kind,
                ranking=ev,
                pr_auc=bootstrap(
                    average_precision,
                    (y, p),
                    n_resamples=cfg.n_resamples,
                    level=cfg.level,
                    seed=cfg.seed,
                    strata=y,
                ),
                savings_per_1k=bootstrap(
                    savings_stat,
                    (y, amounts, p),
                    n_resamples=cfg.n_resamples,
                    level=cfg.level,
                    seed=cfg.seed,
                    strata=y,
                ),
                policies=policies,
                hindsight_threshold=threshold,
                curve=cost_curve(y, p, amounts, costs),
                categories=captured,
                sensitivity=sensitivity,
            )
        )

    comparisons: list[PairedComparison] = []
    ranked = sorted(models, key=lambda m: m.ranking.pr_auc, reverse=True)
    for worse in ranked[1:]:
        best = ranked[0]

        def ap_diff(
            yy: npt.NDArray[np.bool_],
            s1: npt.NDArray[np.float64],
            s2: npt.NDArray[np.float64],
        ) -> float:
            return average_precision(yy, s1) - average_precision(yy, s2)

        def savings_diff(
            yy: npt.NDArray[np.bool_],
            a: npt.NDArray[np.float64],
            s1: npt.NDArray[np.float64],
            s2: npt.NDArray[np.float64],
        ) -> float:
            return savings_stat(yy, a, s1) - savings_stat(yy, a, s2)

        comparisons.append(
            PairedComparison(
                better=best.kind,
                worse=worse.kind,
                pr_auc_difference=bootstrap(
                    ap_diff,
                    (y, best.ranking.scores, worse.ranking.scores),
                    n_resamples=cfg.n_resamples,
                    level=cfg.level,
                    seed=cfg.seed,
                    strata=y,
                ),
                savings_difference_per_1k=bootstrap(
                    savings_diff,
                    (y, amounts, best.ranking.scores, worse.ranking.scores),
                    n_resamples=cfg.n_resamples,
                    level=cfg.level,
                    seed=cfg.seed,
                    strata=y,
                ),
            )
        )

    return EvaluationReport(
        baseline=baseline,
        config=cfg,
        amounts=amounts,
        approve_all=approve_all,
        perfect=perfect,
        models=tuple(models),
        comparisons=tuple(comparisons),
    )


def run_evaluation(
    matrix: FeatureMatrix,
    *,
    categories: Mapping[str, str] | None = None,
    train_config: TrainConfig | None = None,
    config: EvaluationConfig | None = None,
) -> EvaluationReport:
    """Train the baselines on the time split and evaluate them as blocking policies."""
    return evaluate_costs(
        train_baselines(matrix, train_config), categories=categories, config=config
    )
