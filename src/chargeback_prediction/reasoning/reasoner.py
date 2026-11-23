"""Turn a scored, explained dispute into a validated rationale and representment strategy.

Division of labour:

* the **model** owns the probability (items 4–5);
* the **playbook** owns the decision space: it recommends a strategy and fixes which strategies
  are allowed (:mod:`.playbook`);
* the **LLM** owns the prose, and may pick among the allowed strategies.

Every reply is validated before it is trusted: it must be a JSON object with a known strategy
inside the allowed set, a confidence level, a bounded rationale, and key factors that name only
features the model was shown. A rejected reply is retried once with the validation error fed
back; if the provider is down or keeps failing validation, a deterministic template rationale
is returned instead, marked ``source="fallback"``. Scoring never blocks on the LLM.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from chargeback_prediction.explain.factors import format_probability
from chargeback_prediction.reason_codes import ReasonCode
from chargeback_prediction.reasoning.llm import CompletionRequest, LLMClient, LLMError, Message
from chargeback_prediction.reasoning.playbook import Recommendation, Strategy, recommend
from chargeback_prediction.reasoning.prompt import (
    MAX_KEY_FACTORS,
    RATIONALE_MAX_CHARS,
    DisputeContext,
    build_messages,
    citable_features,
)


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Source(StrEnum):
    LLM = "llm"
    FALLBACK = "fallback"


class RationaleError(ValueError):
    """An LLM reply failed validation. The message is safe to show back to the model."""


@dataclass(frozen=True, slots=True)
class ParsedReply:
    strategy: Strategy
    confidence: Confidence
    rationale: str
    key_factors: tuple[str, ...]


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_reply(text: str, *, allowed: frozenset[Strategy], citable: frozenset[str]) -> ParsedReply:
    """Validate a model reply against the response contract.

    Tolerates a Markdown code fence around the JSON (a common model habit) and nothing else.

    Raises:
        RationaleError: describing the first problem found.
    """
    body = text.strip()
    if fenced := _FENCE.match(body):
        body = fenced.group(1)
    try:
        data: Any = json.loads(body)
    except json.JSONDecodeError:
        raise RationaleError("reply is not valid JSON") from None
    if not isinstance(data, dict):
        raise RationaleError("reply must be a JSON object")
    missing = {"strategy", "confidence", "rationale", "key_factors"} - data.keys()
    if missing:
        raise RationaleError(f"missing keys: {', '.join(sorted(missing))}")

    try:
        strategy = Strategy(data["strategy"])
    except ValueError:
        raise RationaleError(f"unknown strategy {data['strategy']!r}") from None
    if strategy not in allowed:
        options = ", ".join(sorted(allowed))
        raise RationaleError(f"strategy {strategy!s} is not allowed here; choose from {options}")
    try:
        confidence = Confidence(data["confidence"])
    except ValueError:
        raise RationaleError(f"unknown confidence {data['confidence']!r}") from None

    rationale = data["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise RationaleError("rationale must be a non-empty string")
    rationale = " ".join(rationale.split())
    if len(rationale) > RATIONALE_MAX_CHARS:
        raise RationaleError(f"rationale exceeds {RATIONALE_MAX_CHARS} characters")

    factors = data["key_factors"]
    if not isinstance(factors, list) or not all(isinstance(f, str) for f in factors):
        raise RationaleError("key_factors must be a list of feature names")
    factors = list(dict.fromkeys(factors))
    if citable and not 1 <= len(factors) <= MAX_KEY_FACTORS:
        raise RationaleError(f"key_factors must name 1 to {MAX_KEY_FACTORS} features")
    unknown = [f for f in factors if f not in citable]
    if unknown:
        raise RationaleError(f"key_factors not in the context: {', '.join(unknown)}")
    return ParsedReply(strategy, confidence, rationale, tuple(factors))


@dataclass(frozen=True, slots=True)
class RiskRationale:
    """The reasoning layer's output for one dispute: what to do, why, and with what evidence."""

    transaction_id: str
    reason: ReasonCode
    probability: float
    strategy: Strategy
    playbook_strategy: Strategy
    confidence: Confidence
    rationale: str
    key_factors: tuple[str, ...]
    evidence: tuple[str, ...]
    """From the playbook, never the LLM: an evidence list must not be hallucinated."""
    source: Source
    provider: str
    warnings: tuple[str, ...] = ()

    @property
    def overrides_playbook(self) -> bool:
        return self.strategy is not self.playbook_strategy

    def to_dict(self) -> dict[str, object]:
        return {
            "transaction_id": self.transaction_id,
            "reason_code": self.reason.key,
            "reason_title": self.reason.title,
            "dispute_probability": round(self.probability, 6),
            "strategy": str(self.strategy),
            "playbook_strategy": str(self.playbook_strategy),
            "confidence": str(self.confidence),
            "rationale": self.rationale,
            "key_factors": list(self.key_factors),
            "evidence": list(self.evidence),
            "source": str(self.source),
            "provider": self.provider,
            "warnings": list(self.warnings),
        }

    def render(self) -> str:
        """Plain-text case note."""
        via = f"{self.source} via {self.provider}"
        lines = [
            f"{self.transaction_id} · {self.reason.key} {self.reason.title}",
            f"  strategy:    {self.strategy} (playbook: {self.playbook_strategy}, "
            f"confidence: {self.confidence}, {via})",
            f"  rationale:   {self.rationale}",
            f"  key factors: {', '.join(self.key_factors) or '-'}",
        ]
        if self.evidence:
            lines.append("  evidence:")
            lines += [f"    - {e}" for e in self.evidence]
        lines += [f"  warning:     {w}" for w in self.warnings]
        return "\n".join(lines)


def template_rationale(ctx: DisputeContext, rec: Recommendation) -> str:
    """Deterministic rationale used when no valid LLM reply is available."""
    exp = ctx.explanation
    parts = [
        f"{ctx.reason.key} ({ctx.reason.title}); pre-dispute risk "
        f"{format_probability(exp.probability)}, {exp.relative_risk:.1f}× typical."
    ]
    if exp.factors:
        parts.append("Top factors: " + "; ".join(f.render() for f in exp.factors) + ".")
    if rec.signals:
        parts.append("Signals: " + "; ".join(s.detail for s in rec.signals) + ".")
    parts.append(f"Playbook: {rec.strategy}. " + " ".join(rec.reasons))
    return " ".join(parts)


class RiskReasoner:
    """Playbook + prompt + LLM + validation + fallback, for one dispute at a time."""

    def __init__(self, client: LLMClient, *, max_attempts: int = 2) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._client = client
        self._max_attempts = max_attempts

    def reason(self, ctx: DisputeContext) -> RiskRationale:
        rec = recommend(ctx.reason, ctx.features)
        citable = citable_features(ctx, rec)
        messages: tuple[Message, ...] = build_messages(ctx, rec)
        warnings: list[str] = []

        for attempt in range(1, self._max_attempts + 1):
            try:
                reply = self._client.complete(CompletionRequest(messages))
            except LLMError as exc:
                warnings.append(f"attempt {attempt}: provider error: {exc}")
                continue
            try:
                parsed = parse_reply(reply, allowed=rec.allowed, citable=citable)
            except RationaleError as exc:
                warnings.append(f"attempt {attempt}: rejected reply: {exc}")
                messages = (
                    *messages,
                    Message("assistant", reply),
                    Message(
                        "user",
                        f"Your reply was rejected: {exc}. Reply again with only the JSON object.",
                    ),
                )
                continue
            return self._result(ctx, rec, parsed, Source.LLM, warnings)

        fallback = ParsedReply(
            strategy=rec.strategy,
            confidence=Confidence.LOW,
            rationale=template_rationale(ctx, rec),
            key_factors=tuple(sorted(citable)),
        )
        return self._result(ctx, rec, fallback, Source.FALLBACK, warnings)

    def reason_all(self, contexts: Sequence[DisputeContext]) -> list[RiskRationale]:
        return [self.reason(ctx) for ctx in contexts]

    def _result(
        self,
        ctx: DisputeContext,
        rec: Recommendation,
        parsed: ParsedReply,
        source: Source,
        warnings: list[str],
    ) -> RiskRationale:
        return RiskRationale(
            transaction_id=ctx.transaction_id,
            reason=ctx.reason,
            probability=ctx.explanation.probability,
            strategy=parsed.strategy,
            playbook_strategy=rec.strategy,
            confidence=parsed.confidence,
            rationale=parsed.rationale,
            key_factors=parsed.key_factors,
            evidence=rec.evidence,
            source=source,
            provider=self._client.name,
            warnings=tuple(warnings),
        )


def feature_values(names: Sequence[str], row: Sequence[float]) -> Mapping[str, float]:
    """Pair a feature row with its names."""
    if len(names) != len(row):
        raise ValueError("feature names and values differ in length")
    return {n: float(v) for n, v in zip(names, row, strict=True)}
