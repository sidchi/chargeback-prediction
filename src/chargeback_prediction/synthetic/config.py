"""Tunable knobs for the synthetic data generator."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType

from chargeback_prediction.synthetic.schema import Persona

_DEFAULT_PERSONA_MIX: Mapping[Persona, float] = MappingProxyType(
    {
        Persona.GOOD: 0.89,
        Persona.FORGETFUL: 0.07,
        Persona.SERIAL_DISPUTER: 0.02,
        Persona.FRAUDSTER: 0.02,
    }
)

MAX_DISPUTE_WINDOW_DAYS = 120
"""Visa and Mastercard both give cardholders roughly 120 days from processing to dispute."""


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Configuration for :func:`chargeback_prediction.synthetic.generate`.

    The same config (including ``seed``) always produces an identical dataset.
    """

    n_customers: int = 5_000
    seed: int = 42
    start: datetime = datetime(2025, 1, 1, tzinfo=UTC)
    """Start of the observation window. Accounts may pre-date it; activity does not."""
    days: int = 540
    """Length of the observation window. 540 days lets annual plans renew at least once."""
    persona_mix: Mapping[Persona, float] = field(default=_DEFAULT_PERSONA_MIX)
    max_retry_attempts: int = 3
    """Total authorization attempts for a renewal after soft declines (1 = no retries)."""
    retry_spacing_days: int = 3

    def __post_init__(self) -> None:
        if self.n_customers <= 0:
            raise ValueError("n_customers must be positive")
        if self.days < 30:
            raise ValueError("days must be at least 30")
        if self.start.tzinfo is None:
            raise ValueError("start must be timezone-aware")
        if self.max_retry_attempts < 1:
            raise ValueError("max_retry_attempts must be >= 1")
        if self.retry_spacing_days < 1:
            raise ValueError("retry_spacing_days must be >= 1")
        if set(self.persona_mix) != set(Persona):
            raise ValueError("persona_mix must define a weight for every Persona")
        if any(w < 0 for w in self.persona_mix.values()):
            raise ValueError("persona_mix weights must be non-negative")
        if not math.isclose(sum(self.persona_mix.values()), 1.0, abs_tol=1e-9):
            raise ValueError("persona_mix weights must sum to 1")
