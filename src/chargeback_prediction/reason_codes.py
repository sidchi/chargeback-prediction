"""Catalog of card-network dispute reason codes used throughout the pipeline.

Reason codes drive two downstream decisions: which features matter for a given dispute
(e.g. velocity for fraud, cancellation history for subscription disputes) and which
evidence a merchant must assemble for representment. Keeping them in one typed catalog
means the data generator, model and evidence assembler all agree on the vocabulary.

Only the subset of codes most relevant to card-not-present subscription merchants is
modelled here; it is intentionally small and easy to extend.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Network(StrEnum):
    """Card network that owns the reason code."""

    VISA = "visa"
    MASTERCARD = "mastercard"


class DisputeCategory(StrEnum):
    """Normalized dispute category, independent of network-specific codes."""

    FRAUD = "fraud"
    """Cardholder claims they did not authorize the transaction."""

    CONSUMER = "consumer"
    """Cardholder authorized the payment but disputes what they received or were billed."""

    PROCESSING = "processing"
    """Error in how the transaction was authorized or settled."""


@dataclass(frozen=True, slots=True)
class ReasonCode:
    """A single network reason code and how the pipeline treats it."""

    network: Network
    code: str
    title: str
    category: DisputeCategory
    friendly_fraud_prone: bool
    """True when disputes under this code are frequently filed by the real cardholder
    (e.g. forgotten subscriptions), which changes the representment strategy."""

    @property
    def key(self) -> str:
        """Stable identifier such as ``visa:10.4``."""
        return f"{self.network}:{self.code}"


_CATALOG: tuple[ReasonCode, ...] = (
    ReasonCode(
        Network.VISA,
        "10.4",
        "Other Fraud - Card Absent Environment",
        DisputeCategory.FRAUD,
        friendly_fraud_prone=True,
    ),
    ReasonCode(
        Network.VISA,
        "13.1",
        "Merchandise/Services Not Received",
        DisputeCategory.CONSUMER,
        friendly_fraud_prone=True,
    ),
    ReasonCode(
        Network.VISA,
        "13.2",
        "Cancelled Recurring Transaction",
        DisputeCategory.CONSUMER,
        friendly_fraud_prone=True,
    ),
    ReasonCode(
        Network.MASTERCARD,
        "4837",
        "No Cardholder Authorization",
        DisputeCategory.FRAUD,
        friendly_fraud_prone=True,
    ),
    ReasonCode(
        Network.MASTERCARD,
        "4853",
        "Cardholder Dispute",
        DisputeCategory.CONSUMER,
        friendly_fraud_prone=True,
    ),
)

_BY_KEY: dict[str, ReasonCode] = {rc.key: rc for rc in _CATALOG}


def all_reason_codes() -> tuple[ReasonCode, ...]:
    """Return every reason code in the catalog, in a stable order."""
    return _CATALOG


def lookup(network: Network | str, code: str) -> ReasonCode:
    """Find a reason code by network and code.

    Raises:
        KeyError: if the code is not in the catalog.
        ValueError: if ``network`` is not a known network.
    """
    key = f"{Network(network)}:{code.strip()}"
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(f"unknown reason code {key!r}") from None


def by_category(category: DisputeCategory) -> tuple[ReasonCode, ...]:
    """Return all reason codes in a normalized category."""
    return tuple(rc for rc in _CATALOG if rc.category is category)
