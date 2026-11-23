"""Record types for the synthetic dataset.

Each dataclass maps one-to-one to an output table. Raw events (transactions, card updates,
disputes) are stored rather than pre-computed features, so the feature pipeline (roadmap
item 3) has to do the same point-in-time work it would do against production data.

Two columns are **ground-truth labels, not features**: ``Customer.persona`` and
``Dispute.is_friendly_fraud``. They exist so the evaluation can slice results by behaviour
pattern; feeding them to a model would be target leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from chargeback_prediction.reason_codes import DisputeCategory, Network


class Persona(StrEnum):
    """Hidden behaviour pattern that drives how a synthetic customer transacts and disputes."""

    GOOD = "good"
    """Pays, occasionally churns; rare genuine service disputes."""

    FORGETFUL = "forgetful"
    """Stops using the product but not the subscription, then disputes the renewal
    ("cancelled recurring"). The classic subscription friendly-fraud pattern."""

    SERIAL_DISPUTER = "serial_disputer"
    """Real cardholder who repeatedly claims fraud / non-receipt on their own purchases."""

    FRAUDSTER = "fraudster"
    """Uses a stolen card: new account, card testing, geo/BIN mismatch, velocity bursts.
    Disputes are filed by the *real* cardholder and are true fraud."""


class Plan(StrEnum):
    MONTHLY = "monthly"
    ANNUAL = "annual"

    @property
    def period_days(self) -> int:
        return 30 if self is Plan.MONTHLY else 365


class TransactionKind(StrEnum):
    INITIAL = "initial"
    """First charge of a subscription."""
    RENEWAL = "renewal"
    """Merchant-initiated recurring charge (card on file)."""
    ONE_OFF = "one_off"
    """Customer-initiated purchase outside any subscription."""


class CardUpdateSource(StrEnum):
    ACCOUNT_UPDATER = "account_updater"
    """Network account-updater service pushed a new PAN/expiry (card reissued)."""
    CUSTOMER = "customer"
    """Customer replaced their card on file themselves."""


@dataclass(frozen=True, slots=True)
class Customer:
    customer_id: str
    created_at: datetime
    country: str
    email_domain_type: str
    """``freemail``, ``corporate`` or ``disposable``."""
    card_network: Network
    card_bin: str
    """Synthetic 6-digit issuer prefix (never a real PAN)."""
    bin_country: str
    persona: Persona
    """Ground-truth label. Do not use as a feature."""


@dataclass(frozen=True, slots=True)
class Subscription:
    subscription_id: str
    customer_id: str
    plan: Plan
    price_cents: int
    started_at: datetime
    cancel_requested_at: datetime | None


@dataclass(frozen=True, slots=True)
class Transaction:
    transaction_id: str
    customer_id: str
    subscription_id: str | None
    created_at: datetime
    kind: TransactionKind
    amount_cents: int
    currency: str
    network: Network
    bin_country: str
    ip_country: str | None
    """``None`` for merchant-initiated renewals, where no customer session exists."""
    attempt: int
    """1 for the first authorization of a charge, 2+ for retries after a soft decline."""
    approved: bool
    decline_code: str | None
    """ISO 8583 response code for declines (e.g. ``51`` insufficient funds), else ``None``."""


@dataclass(frozen=True, slots=True)
class CardUpdate:
    customer_id: str
    updated_at: datetime
    source: CardUpdateSource


@dataclass(frozen=True, slots=True)
class Dispute:
    dispute_id: str
    transaction_id: str
    customer_id: str
    network: Network
    reason_code: str
    """Catalog key such as ``visa:13.2``; see :mod:`chargeback_prediction.reason_codes`."""
    category: DisputeCategory
    filed_at: datetime
    amount_cents: int
    is_friendly_fraud: bool
    """Ground-truth label: the real cardholder disputed a charge they benefited from."""
