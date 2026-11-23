"""Train the baseline models on a time-based split and evaluate them on the held-out window."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from chargeback_prediction.modeling.dataset import FeatureMatrix
from chargeback_prediction.modeling.metrics import (
    CalibrationBin,
    average_precision,
    brier_score,
    calibration_bins,
    expected_calibration_error,
    precision_at_k,
    recall_at_k,
)
from chargeback_prediction.modeling.models import Classifier, ModelKind, build_model, predict_risk
from chargeback_prediction.modeling.split import TimeSplit, time_split
from chargeback_prediction.synthetic.config import MAX_DISPUTE_WINDOW_DAYS

DEFAULT_REVIEW_BUDGETS: tuple[float, ...] = (0.005, 0.01, 0.05)
"""Share of test traffic an ops team could review or step-up, highest-risk first."""


@dataclass(frozen=True, slots=True)
class TrainConfig:
    models: tuple[ModelKind, ...] = (ModelKind.LOGREG, ModelKind.GBM)
    test_fraction: float = 0.25
    label_gap_days: int = MAX_DISPUTE_WINDOW_DAYS
    review_budgets: tuple[float, ...] = DEFAULT_REVIEW_BUDGETS
    calibration_bins: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("at least one model is required")
        if any(not 0.0 < b <= 1.0 for b in self.review_budgets):
            raise ValueError("review budgets must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class AtK:
    budget: float
    """Share of test rows reviewed."""
    k: int
    precision: float
    recall: float

    def lift(self, base_rate: float) -> float:
        """Precision relative to reviewing ``k`` random rows."""
        return self.precision / base_rate if base_rate else float("nan")


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    kind: ModelKind
    pr_auc: float
    brier: float
    ece: float
    at_k: tuple[AtK, ...]
    calibration: tuple[CalibrationBin, ...]
    scores: npt.NDArray[np.float64] = field(repr=False)
    """Test-set probabilities, aligned with ``split.test``."""
    fit_seconds: float


@dataclass(frozen=True, slots=True)
class BaselineReport:
    split: TimeSplit
    config: TrainConfig
    evaluations: tuple[ModelEvaluation, ...]
    models: dict[ModelKind, Classifier] = field(repr=False)

    @property
    def test_base_rate(self) -> float:
        return self.split.test.positive_rate

    def best(self) -> ModelEvaluation:
        """Model with the highest PR-AUC on the test window."""
        return max(self.evaluations, key=lambda e: e.pr_auc)


def evaluate(
    kind: ModelKind,
    y_true: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    *,
    review_budgets: Sequence[float] = DEFAULT_REVIEW_BUDGETS,
    n_bins: int = 10,
    fit_seconds: float = 0.0,
) -> ModelEvaluation:
    """Compute every reported metric for one model's test-set scores."""
    n = len(y_true)
    at_k = tuple(
        AtK(
            budget=b,
            k=(k := max(1, round(b * n))),
            precision=precision_at_k(y_true, scores, k),
            recall=recall_at_k(y_true, scores, k),
        )
        for b in review_budgets
    )
    bins = calibration_bins(y_true, scores, n_bins=n_bins)
    return ModelEvaluation(
        kind=kind,
        pr_auc=average_precision(y_true, scores),
        brier=brier_score(y_true, scores),
        ece=expected_calibration_error(bins),
        at_k=at_k,
        calibration=bins,
        scores=scores,
        fit_seconds=fit_seconds,
    )


def train_baselines(matrix: FeatureMatrix, config: TrainConfig | None = None) -> BaselineReport:
    """Split by time, fit each configured model on the train window, evaluate on the test window."""
    cfg = config or TrainConfig()
    split = time_split(matrix, test_fraction=cfg.test_fraction, label_gap_days=cfg.label_gap_days)
    evaluations: list[ModelEvaluation] = []
    fitted: dict[ModelKind, Classifier] = {}
    for kind in cfg.models:
        model = build_model(kind, seed=cfg.seed)
        started = time.perf_counter()
        model.fit(split.train.X, split.train.y)
        elapsed = time.perf_counter() - started
        scores = predict_risk(model, split.test.X)
        evaluations.append(
            evaluate(
                kind,
                split.test.y,
                scores,
                review_budgets=cfg.review_budgets,
                n_bins=cfg.calibration_bins,
                fit_seconds=elapsed,
            )
        )
        fitted[kind] = model
    return BaselineReport(split=split, config=cfg, evaluations=tuple(evaluations), models=fitted)
