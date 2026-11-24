"""Request and response schemas for the scoring service.

The feature payload is spelled out field by field rather than generated from the catalog at
import time. That keeps the HTTP contract readable in code review and in the OpenAPI docs, and
makes any catalog change a deliberate API change: ``tests/test_service.py`` fails if
:class:`TransactionFeatures` and :data:`~chargeback_prediction.features.FEATURES` drift apart
in name, order or type.

The service scores a feature vector, not a raw transaction. Features are point-in-time
aggregates over the customer's history (velocity, prior disputes, card-on-file updates), so
they belong to whoever owns that history: the feature pipeline offline, a feature store online.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from chargeback_prediction.features import FEATURES
from chargeback_prediction.reasoning.run import parse_reason_code

MAX_BATCH = 1_000
"""Largest ``/v1/score`` batch. Scoring is one vectorised ``predict_proba`` call per batch."""

MAX_TOP_K = 10

_DESCRIPTIONS = {f.name: f.description for f in FEATURES}

Flag = Annotated[int, Field(ge=0, le=1)]
"""0 or 1."""
Count = Annotated[int, Field(ge=0)]
"""Non-negative integer count of prior events."""
Numeric = Annotated[float, Field(ge=0, allow_inf_nan=False)]
"""Continuous, non-negative, finite."""


def _f(name: str) -> Any:
    return Field(description=_DESCRIPTIONS[name])


class TransactionFeatures(BaseModel):
    """All model inputs for one transaction, as built by ``cbp features`` (see docs/features.md).

    Every field is required and unknown fields are rejected: a typo'd or missing feature must
    fail loudly, not be silently scored as zero.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    amount_usd: Numeric = _f("amount_usd")
    is_initial: Flag = _f("is_initial")
    is_renewal: Flag = _f("is_renewal")
    is_one_off: Flag = _f("is_one_off")
    is_mastercard: Flag = _f("is_mastercard")
    attempt: Count = _f("attempt")
    account_age_days: Numeric = _f("account_age_days")
    email_disposable: Flag = _f("email_disposable")
    email_freemail: Flag = _f("email_freemail")
    ip_missing: Flag = _f("ip_missing")
    ip_bin_mismatch: Flag = _f("ip_bin_mismatch")
    billing_bin_mismatch: Flag = _f("billing_bin_mismatch")
    attempts_1h: Count = _f("attempts_1h")
    attempts_24h: Count = _f("attempts_24h")
    attempts_7d: Count = _f("attempts_7d")
    declines_24h: Count = _f("declines_24h")
    declines_30d: Count = _f("declines_30d")
    approved_usd_24h: Numeric = _f("approved_usd_24h")
    days_since_prev_attempt: Numeric = _f("days_since_prev_attempt")
    au_updates_90d: Count = _f("au_updates_90d")
    customer_card_updates_90d: Count = _f("customer_card_updates_90d")
    days_since_card_update: Numeric = _f("days_since_card_update")
    plan_is_annual: Flag = _f("plan_is_annual")
    subscription_age_days: Numeric = _f("subscription_age_days")
    prior_approved_renewals: Count = _f("prior_approved_renewals")
    prior_subscription_declines: Count = _f("prior_subscription_declines")
    charged_after_cancel_request: Flag = _f("charged_after_cancel_request")
    prior_disputes: Count = _f("prior_disputes")
    prior_fraud_disputes: Count = _f("prior_fraud_disputes")
    days_since_last_dispute: Numeric = _f("days_since_last_dispute")

    def vector(self, names: tuple[str, ...]) -> list[float]:
        """Values in the model's column order."""
        return [float(getattr(self, n)) for n in names]


TransactionId = Annotated[str, Field(min_length=1, max_length=128, examples=["txn_00013396"])]


class TransactionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: TransactionId
    features: TransactionFeatures


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[TransactionIn] = Field(min_length=1, max_length=MAX_BATCH)

    @model_validator(mode="after")
    def _unique_ids(self) -> ScoreRequest:
        seen: set[str] = set()
        for t in self.transactions:
            if t.transaction_id in seen:
                raise ValueError(f"duplicate transaction_id {t.transaction_id!r}")
            seen.add(t.transaction_id)
        return self


class Score(BaseModel):
    transaction_id: str
    probability: float = Field(description="Calibrated probability that a dispute is filed.")
    relative_risk: float = Field(
        description="probability / baseline_probability: how many times riskier than a "
        "typical transaction."
    )


class ScoreResponse(BaseModel):
    model_id: str = Field(description="Model kind and artifact hash, for audit trails.")
    baseline_probability: float
    results: list[Score]


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction: TransactionIn
    top_k: int = Field(default=3, ge=0, le=MAX_TOP_K, description="Factors to return.")
    reason_code: str | None = Field(
        default=None,
        description="If the transaction was disputed, e.g. 'visa:13.2' or 'mastercard:4837': "
        "adds a risk rationale and representment strategy.",
        examples=["visa:13.2"],
    )

    @field_validator("reason_code")
    @classmethod
    def _known_reason_code(cls, value: str | None) -> str | None:
        if value is not None:
            parse_reason_code(value)  # raises ValueError -> 422
        return value


class FactorOut(BaseModel):
    feature: str
    value: float
    display_value: str
    shap_log_odds: float
    odds_multiplier: float
    direction: str = Field(description="'raises' or 'lowers'.")
    description: str


class RationaleOut(BaseModel):
    reason_code: str
    reason_title: str
    strategy: str
    playbook_strategy: str
    confidence: str
    rationale: str
    key_factors: list[str]
    evidence: list[str] = Field(description="From the rule-based playbook, never the LLM.")
    source: str = Field(description="'llm', or 'fallback' when the LLM failed validation.")
    provider: str
    warnings: list[str]


class ExplainResponse(BaseModel):
    model_id: str
    transaction_id: str
    probability: float
    baseline_probability: float
    relative_risk: float
    base_log_odds: float
    factors: list[FactorOut]
    other_features_log_odds: float = Field(
        description="Combined contribution of every feature not listed in factors."
    )
    summary: str
    rationale: RationaleOut | None = None


class ModelInfo(BaseModel):
    model_id: str
    kind: str
    label: str
    features: list[str]
    baseline_probability: float
    background_rows: int
    llm_provider: str
    version: str


class Health(BaseModel):
    status: str
