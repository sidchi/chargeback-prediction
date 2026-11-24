"""Baseline classifiers: a regularised logistic regression and a gradient-boosted tree model.

Both are deliberately un-reweighted for class imbalance. Re-weighting (``class_weight``) helps
recall at a fixed 0.5 threshold, but nobody thresholds a 2%-base-rate score at 0.5, and it
destroys calibration: a re-weighted model's 0.30 no longer means 30%. Ranking quality (PR-AUC,
precision@k) is unaffected by re-weighting anyway, so we keep probabilities honest.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, Self

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


class ModelKind(StrEnum):
    LOGREG = "logreg"
    GBM = "gbm"

    @property
    def label(self) -> str:
        return _LABELS[self]


_LABELS = {
    ModelKind.LOGREG: "Logistic regression",
    ModelKind.GBM: "Gradient boosting",
}


class Classifier(Protocol):
    """The subset of the scikit-learn estimator API the pipeline relies on."""

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.bool_]) -> Self: ...

    def predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...


def _signed_log1p(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """``sign(x)·log(1+|x|)``: tames heavy-tailed counts and dollars for a linear model."""
    return np.sign(X) * np.log1p(np.abs(X))


def build_model(kind: ModelKind, *, seed: int = 0) -> Classifier:
    """Return an unfitted baseline model.

    * ``logreg`` — signed-log1p → standardise → L2 logistic regression. The log transform
      matters: velocity counts and amounts are heavy-tailed, and without it a handful of
      card-testing bursts dominate the coefficients.
    * ``gbm`` — :class:`~sklearn.ensemble.HistGradientBoostingClassifier`. Trees need no
      scaling and can pick up interactions such as "renewal *and* recent account-updater
      reissue" that the linear model cannot. A training window holds only a few hundred
      disputes, so the defaults are deliberately conservative (depth 3, at least 100 rows per
      leaf, strong L2, slow learning rate); they were fixed up front, not tuned on the test
      window.
    """
    if kind is ModelKind.LOGREG:
        model: Classifier = Pipeline(
            [
                ("log", FunctionTransformer(_signed_log1p)),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=2_000)),
            ]
        )
        return model
    if kind is ModelKind.GBM:
        gbm: Classifier = HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_iter=150,
            max_depth=3,
            min_samples_leaf=100,
            l2_regularization=5.0,
            early_stopping=False,
            random_state=seed,
        )
        return gbm
    raise AssertionError(f"unhandled model kind {kind!r}")  # pragma: no cover


def predict_risk(model: Classifier, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Probability of dispute (positive class) for each row."""
    proba = model.predict_proba(X)
    return np.asarray(proba[:, 1], dtype=np.float64)
