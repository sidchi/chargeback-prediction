"""FastAPI app: ``/v1/score``, ``/v1/explain``, ``/v1/model`` and ``/health``.

Run it with ``cbp serve`` (or ``uvicorn --factory chargeback_prediction.service.app:create_app``).
Interactive OpenAPI docs are served at ``/docs``; the raw schema at ``/openapi.json``.

Endpoints are plain ``def``, not ``async def``: scoring and SHAP are CPU-bound numpy work, and
FastAPI runs sync endpoints on its worker thread pool so they never block the event loop.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, Request

from chargeback_prediction import __version__
from chargeback_prediction.explain.factors import TransactionExplanation, format_value
from chargeback_prediction.explain.run import DEFAULT_BACKGROUND_SIZE
from chargeback_prediction.modeling import ModelKind
from chargeback_prediction.reasoning.llm import LLMClient, MockLLM, OpenAICompatibleClient
from chargeback_prediction.reasoning.run import parse_reason_code
from chargeback_prediction.service.schemas import (
    ExplainRequest,
    ExplainResponse,
    FactorOut,
    Health,
    ModelInfo,
    RationaleOut,
    Score,
    ScoreRequest,
    ScoreResponse,
)
from chargeback_prediction.service.scorer import Scorer

ENV_PREFIX = "CBP_"


@dataclass(frozen=True, slots=True)
class Settings:
    """Where to load the model from. Read from ``CBP_*`` environment variables by default."""

    models_dir: Path = Path("artifacts/baseline")
    features: Path = Path("data/features.csv")
    """Feature table the SHAP background is sampled from (its training window)."""
    model: ModelKind = ModelKind.LOGREG
    background_size: int = DEFAULT_BACKGROUND_SIZE
    llm_base_url: str | None = None
    """Unset: the deterministic mock LLM. Set: any OpenAI-compatible chat-completions server."""
    llm_model: str | None = None
    llm_api_key: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """``CBP_MODELS_DIR``, ``CBP_FEATURES``, ``CBP_MODEL``, ``CBP_BACKGROUND_SIZE``,
        ``CBP_LLM_BASE_URL``, ``CBP_LLM_MODEL``, ``CBP_LLM_API_KEY``.

        Raises:
            ValueError: for an unknown model kind or a non-integer background size.
        """
        env = os.environ if env is None else env
        defaults = cls()

        def get(name: str) -> str | None:
            return env.get(ENV_PREFIX + name) or None

        return cls(
            models_dir=Path(get("MODELS_DIR") or defaults.models_dir),
            features=Path(get("FEATURES") or defaults.features),
            model=ModelKind(get("MODEL") or defaults.model),
            background_size=int(get("BACKGROUND_SIZE") or defaults.background_size),
            llm_base_url=get("LLM_BASE_URL"),
            llm_model=get("LLM_MODEL"),
            llm_api_key=get("LLM_API_KEY"),
        )

    def llm_client(self) -> LLMClient:
        """The configured LLM provider.

        Raises:
            ValueError: if only one of ``llm_base_url`` and ``llm_model`` is set.
        """
        if self.llm_base_url is None and self.llm_model is None:
            return MockLLM()
        if not self.llm_base_url or not self.llm_model:
            raise ValueError("set both CBP_LLM_BASE_URL and CBP_LLM_MODEL, or neither")
        return OpenAICompatibleClient(self.llm_base_url, self.llm_model, api_key=self.llm_api_key)

    def load_scorer(self) -> Scorer:
        return Scorer.from_artifacts(
            self.models_dir,
            self.features,
            self.model,
            background_size=self.background_size,
            llm=self.llm_client(),
        )


def get_scorer(request: Request) -> Scorer:
    scorer: Scorer = request.app.state.scorer
    return scorer


ScorerDep = Annotated[Scorer, Depends(get_scorer)]


def _factors(explanation: TransactionExplanation) -> list[FactorOut]:
    return [
        FactorOut(
            feature=f.feature,
            value=f.value,
            display_value=format_value(f.feature, f.value),
            shap_log_odds=f.contribution,
            odds_multiplier=f.odds_multiplier,
            direction=str(f.direction),
            description=f.description,
        )
        for f in explanation.factors
    ]


def create_app(scorer: Scorer | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the app.

    Pass a ready ``scorer`` (tests, embedding) or let the app load one at startup from
    ``settings`` (default: :meth:`Settings.from_env`). Loading at startup rather than per
    request means a missing or broken model fails the deploy's health check, not a customer
    request.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.scorer = scorer or (settings or Settings.from_env()).load_scorer()
        yield

    app = FastAPI(
        title="Chargeback risk scoring",
        version=__version__,
        summary="Dispute-risk scores with exact SHAP explanations (synthetic data only).",
        description=(
            "Scores card-not-present transactions for the probability of a chargeback and "
            "explains each score. Probabilities are calibrated, not decisions: thresholds for "
            "review, step-up or retry suppression are business policy and live with the caller."
        ),
        lifespan=lifespan,
    )

    @app.get("/health", tags=["ops"])
    def health() -> Health:
        """Liveness and readiness: the model is loaded once the app accepts requests."""
        return Health(status="ok")

    @app.get("/v1/model", tags=["model"])
    def model_info(scorer: ScorerDep) -> ModelInfo:
        """The loaded model, its expected feature order and the typical-transaction baseline."""
        return ModelInfo(
            model_id=scorer.model_id,
            kind=scorer.kind.value,
            label=scorer.kind.label,
            features=list(scorer.feature_names),
            baseline_probability=scorer.baseline_probability,
            background_rows=len(scorer.background),
            llm_provider=scorer.llm_provider,
            version=__version__,
        )

    @app.post("/v1/score", tags=["scoring"])
    def score(body: ScoreRequest, scorer: ScorerDep) -> ScoreResponse:
        """Dispute probability for a batch of up to 1,000 transactions."""
        X = np.array(
            [t.features.vector(scorer.feature_names) for t in body.transactions], dtype=np.float64
        )
        probabilities = scorer.score(X)
        base = scorer.baseline_probability
        return ScoreResponse(
            model_id=scorer.model_id,
            baseline_probability=base,
            results=[
                Score(transaction_id=t.transaction_id, probability=float(p), relative_risk=p / base)
                for t, p in zip(body.transactions, probabilities, strict=True)
            ],
        )

    @app.post("/v1/explain", tags=["scoring"])
    def explain(body: ExplainRequest, scorer: ScorerDep) -> ExplainResponse:
        """Score one transaction and explain it: exact SHAP factors, a one-paragraph summary,
        and, given the dispute's reason code, a representment rationale and strategy."""
        txn = body.transaction
        row = txn.features.vector(scorer.feature_names)
        (exp,) = scorer.explain(np.array([row]), [txn.transaction_id], top_k=body.top_k)
        rationale = None
        if body.reason_code is not None:
            r = scorer.rationale(exp, parse_reason_code(body.reason_code), row).to_dict()
            rationale = RationaleOut.model_validate(r)
        return ExplainResponse(
            model_id=scorer.model_id,
            transaction_id=exp.transaction_id,
            probability=exp.probability,
            baseline_probability=exp.baseline_probability,
            relative_risk=exp.relative_risk,
            base_log_odds=exp.base_value,
            factors=_factors(exp),
            other_features_log_odds=exp.rest,
            summary=exp.summary(),
            rationale=rationale,
        )

    return app
