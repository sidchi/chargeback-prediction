"""Framework-free scoring core: everything the HTTP layer needs, loaded once at startup.

Kept separate from FastAPI so it can be unit-tested directly and reused from a batch job or a
different transport. Everything here is read-only after construction, so one instance is safely
shared by the worker threads FastAPI runs synchronous endpoints on.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from chargeback_prediction.explain.factors import TransactionExplanation, explain_rows
from chargeback_prediction.explain.run import (
    DEFAULT_BACKGROUND_SIZE,
    background_sample,
    load_model,
    model_path,
)
from chargeback_prediction.explain.shapley import shap_values
from chargeback_prediction.features import FEATURE_NAMES
from chargeback_prediction.modeling import Classifier, FeatureMatrix, ModelKind, predict_risk
from chargeback_prediction.modeling.split import time_split
from chargeback_prediction.modeling.train import TrainConfig
from chargeback_prediction.reason_codes import ReasonCode
from chargeback_prediction.reasoning.llm import LLMClient, MockLLM
from chargeback_prediction.reasoning.prompt import DisputeContext
from chargeback_prediction.reasoning.reasoner import RiskRationale, RiskReasoner, feature_values

FloatArray = npt.NDArray[np.float64]


def artifact_id(kind: ModelKind, path: Path) -> str:
    """``logreg@1a2b3c4d5e6f``: model kind plus a content hash of the artifact file.

    Returned with every score so a decision can be traced to the exact model that made it,
    even after the artifact directory is overwritten by a retrain.
    """
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"{kind.value}@{digest[:12]}"


@dataclass(frozen=True, slots=True)
class Scorer:
    """A trained model plus the SHAP background and reasoner needed to explain its scores."""

    model: Classifier = field(repr=False)
    kind: ModelKind
    model_id: str
    background: FloatArray = field(repr=False)
    """Training-window rows that define "a typical transaction" for SHAP."""
    baseline_probability: float
    reasoner: RiskReasoner = field(repr=False)
    llm_provider: str
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @classmethod
    def build(
        cls,
        model: Classifier,
        kind: ModelKind,
        background: FloatArray,
        *,
        model_id: str,
        llm: LLMClient | None = None,
    ) -> Scorer:
        """Validate the pieces and fail fast if the model cannot be explained.

        Raises:
            ValueError: for an empty or mis-shaped background.
            TypeError: for a model with no exact SHAP implementation.
        """
        background = np.asarray(background, dtype=np.float64)
        if background.ndim != 2 or len(background) == 0:
            raise ValueError("background must be a non-empty 2-D array")
        if background.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"background has {background.shape[1]} columns, expected {len(FEATURE_NAMES)}"
            )
        # Surface an unsupported model at startup, not on the first /explain call.
        shap_values(model, background[:1], background[:1], FEATURE_NAMES)
        client = llm or MockLLM()
        return cls(
            model=model,
            kind=kind,
            model_id=model_id,
            background=background,
            baseline_probability=float(predict_risk(model, background).mean()),
            reasoner=RiskReasoner(client),
            llm_provider=client.name,
        )

    @classmethod
    def from_artifacts(
        cls,
        models_dir: Path,
        features_csv: Path,
        kind: ModelKind = ModelKind.LOGREG,
        *,
        background_size: int = DEFAULT_BACKGROUND_SIZE,
        seed: int = 0,
        llm: LLMClient | None = None,
    ) -> Scorer:
        """Load a ``cbp train`` model and sample its SHAP background from the training window.

        The background is drawn exactly as ``cbp explain`` draws it (same split, same seed), so
        the service and the offline reports agree on every explanation.

        Raises:
            FileNotFoundError: if the model or feature table is missing.
        """
        model = load_model(models_dir, kind)
        matrix = FeatureMatrix.read_csv(features_csv)
        split_cfg = TrainConfig()
        split = time_split(
            matrix, test_fraction=split_cfg.test_fraction, label_gap_days=split_cfg.label_gap_days
        )
        background = background_sample(split.train, background_size, seed=seed)
        return cls.build(
            model,
            kind,
            background,
            model_id=artifact_id(kind, model_path(models_dir, kind)),
            llm=llm,
        )

    def score(self, X: FloatArray) -> FloatArray:
        """Dispute probability per row, in one vectorised call."""
        return predict_risk(self.model, np.asarray(X, dtype=np.float64))

    def explain(
        self, X: FloatArray, transaction_ids: Sequence[str], *, top_k: int
    ) -> list[TransactionExplanation]:
        """Exact SHAP explanation per row against the training background."""
        X = np.asarray(X, dtype=np.float64)
        shap = shap_values(self.model, X, self.background, self.feature_names)
        return explain_rows(
            shap,
            X,
            transaction_ids,
            baseline_probability=self.baseline_probability,
            k=top_k,
        )

    def rationale(
        self, explanation: TransactionExplanation, reason: ReasonCode, row: Sequence[float]
    ) -> RiskRationale:
        """Playbook strategy + LLM-written rationale; falls back to a template, never raises."""
        context = DisputeContext(
            explanation=explanation,
            reason=reason,
            features=feature_values(self.feature_names, row),
        )
        return self.reasoner.reason(context)
