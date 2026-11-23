"""What the model is told: a fixed system prompt plus a structured, numbers-only context.

Everything in the context is derived from our own features, SHAP values and reason-code
catalog: there is no free text from the cardholder or issuer. That keeps the prompt
injection-free by construction; when item 7 adds dispute narratives, they must go in a
separately delimited, explicitly untrusted field.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from chargeback_prediction.explain.factors import (
    TransactionExplanation,
    format_multiplier,
    format_probability,
    format_value,
)
from chargeback_prediction.reason_codes import ReasonCode
from chargeback_prediction.reasoning.llm import CONTEXT_CLOSE, CONTEXT_OPEN, Message
from chargeback_prediction.reasoning.playbook import Recommendation

RATIONALE_MAX_CHARS = 1_200
MAX_KEY_FACTORS = 6


@dataclass(frozen=True, slots=True)
class DisputeContext:
    """Everything known about one disputed transaction at reasoning time."""

    explanation: TransactionExplanation
    reason: ReasonCode
    features: Mapping[str, float] = field(repr=False)
    """All feature values for the transaction (not just the top factors)."""

    @property
    def transaction_id(self) -> str:
        return self.explanation.transaction_id


SYSTEM_PROMPT = f"""\
You are a payments disputes analyst at a subscription merchant. For one disputed card \
transaction you receive, inside {CONTEXT_OPEN} tags, a JSON object with: the dispute reason \
code; the fraud model's pre-dispute risk score; the features that drove that score, as SHAP odds \
multipliers relative to a typical transaction; playbook signals; and the playbook's recommended \
strategy with the strategies you may choose from.

Write a short rationale an analyst can paste into a case note, then pick a strategy.

Rules:
- Use only facts present in the context. Do not invent evidence, amounts, dates or features.
- Choose "strategy" from playbook.allowed_strategies only. Prefer the recommended strategy; \
choose another allowed one only if the context clearly supports it, and say why.
- "key_factors" lists feature names, exactly as written in the context, that your rationale \
relies on (1 to {MAX_KEY_FACTORS}).
- A high pre-dispute score on a fraud-coded dispute supports a stolen card; a low score with a \
long, clean history supports friendly fraud.
- "rationale" is 2 to 5 plain sentences, at most {RATIONALE_MAX_CHARS} characters.

Reply with only a JSON object:
{{"strategy": "represent" | "accept" | "review", "confidence": "low" | "medium" | "high", \
"rationale": "...", "key_factors": ["feature_name", ...]}}"""


def context_payload(ctx: DisputeContext, rec: Recommendation) -> dict[str, Any]:
    """The JSON object the model reasons over. Stable key order for reproducible prompts."""
    exp = ctx.explanation
    return {
        "transaction_id": exp.transaction_id,
        "dispute": {
            "reason_code": ctx.reason.key,
            "title": ctx.reason.title,
            "category": str(ctx.reason.category),
            "friendly_fraud_prone": ctx.reason.friendly_fraud_prone,
        },
        "model": {
            "dispute_probability": format_probability(exp.probability),
            "typical_probability": format_probability(exp.baseline_probability),
            "relative_risk": round(exp.relative_risk, 1),
        },
        "top_factors": [
            {
                "feature": f.feature,
                "value": format_value(f.feature, f.value),
                "effect": str(f.direction),
                "odds_multiplier": format_multiplier(f.odds_multiplier),
                "description": f.description,
            }
            for f in exp.factors
        ],
        "signals": [
            {"feature": s.feature, "leans": str(s.leaning), "detail": s.detail} for s in rec.signals
        ],
        "playbook": {
            "recommended_strategy": str(rec.strategy),
            "allowed_strategies": sorted(str(s) for s in rec.allowed),
            "reasons": list(rec.reasons),
        },
    }


def citable_features(ctx: DisputeContext, rec: Recommendation) -> frozenset[str]:
    """Feature names the model may cite: the ones it was actually shown."""
    return frozenset(f.feature for f in ctx.explanation.factors) | {s.feature for s in rec.signals}


def build_messages(ctx: DisputeContext, rec: Recommendation) -> tuple[Message, ...]:
    payload = json.dumps(context_payload(ctx, rec), indent=2, ensure_ascii=False)
    user = f"{CONTEXT_OPEN}\n{payload}\n{CONTEXT_CLOSE}"
    return (Message("system", SYSTEM_PROMPT), Message("user", user))
