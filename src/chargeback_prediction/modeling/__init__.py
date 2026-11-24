"""Baseline dispute-risk models, a label-aware time split, and evaluation metrics."""

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
from chargeback_prediction.modeling.models import (
    Classifier,
    ModelKind,
    build_model,
    predict_risk,
)
from chargeback_prediction.modeling.split import TimeSplit, time_split
from chargeback_prediction.modeling.train import (
    AtK,
    BaselineReport,
    ModelEvaluation,
    TrainConfig,
    evaluate,
    train_baselines,
)

__all__ = [
    "AtK",
    "BaselineReport",
    "CalibrationBin",
    "Classifier",
    "FeatureMatrix",
    "ModelEvaluation",
    "ModelKind",
    "TimeSplit",
    "TrainConfig",
    "average_precision",
    "brier_score",
    "build_model",
    "calibration_bins",
    "evaluate",
    "expected_calibration_error",
    "precision_at_k",
    "predict_risk",
    "recall_at_k",
    "time_split",
    "train_baselines",
]
