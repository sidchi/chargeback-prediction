"""Which exhibits a representment needs, per reason code.

An exhibit is one piece of evidence in the response packet. Some can be pulled from the data this
pipeline holds (transactions, subscriptions, card updates, disputes); others live in merchant
systems outside it (gateway authorization responses, login logs, signup screenshots) and are
listed as attachments an analyst must add.

The requirement lists summarise public network guidance (e.g. Visa Compelling Evidence 3.0 for
10.4) in illustrative form. They are not legal or network-compliance advice; check current
network rules and your acquirer's requirements before relying on them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from chargeback_prediction.reason_codes import DisputeCategory, ReasonCode


class ExhibitKind(StrEnum):
    # Assembled from the dataset.
    TRANSACTION_RECORD = "transaction_record"
    ACCOUNT_PROFILE = "account_profile"
    SUBSCRIPTION_RECORD = "subscription_record"
    CANCELLATION_CHECK = "cancellation_check"
    BILLING_HISTORY = "billing_history"
    CE3_PRIOR_TRANSACTIONS = "ce3_prior_transactions"
    PRIOR_UNDISPUTED_HISTORY = "prior_undisputed_history"
    POST_CHARGE_ACTIVITY = "post_charge_activity"
    CARD_CONTINUITY = "card_continuity"
    # Held in merchant systems outside the dataset: the analyst attaches them.
    AVS_CVV = "avs_cvv"
    THREE_DS = "three_ds"
    USAGE_LOGS = "usage_logs"
    TERMS_ACCEPTANCE = "terms_acceptance"
    RENEWAL_REMINDER = "renewal_reminder"
    CUSTOMER_CONTACT = "customer_contact"

    @property
    def label(self) -> str:
        return _LABELS[self]

    @property
    def external_source(self) -> str | None:
        """Where an analyst finds this exhibit, or ``None`` if the assembler builds it."""
        return _EXTERNAL_SOURCES.get(self)

    @property
    def is_external(self) -> bool:
        return self in _EXTERNAL_SOURCES


_LABELS: Mapping[ExhibitKind, str] = MappingProxyType(
    {
        ExhibitKind.TRANSACTION_RECORD: "Disputed transaction",
        ExhibitKind.ACCOUNT_PROFILE: "Customer account",
        ExhibitKind.SUBSCRIPTION_RECORD: "Subscription",
        ExhibitKind.CANCELLATION_CHECK: "Cancellation history",
        ExhibitKind.BILLING_HISTORY: "Billing history on this subscription",
        ExhibitKind.CE3_PRIOR_TRANSACTIONS: "Visa CE 3.0 prior undisputed transactions",
        ExhibitKind.PRIOR_UNDISPUTED_HISTORY: "Prior undisputed transactions on this card",
        ExhibitKind.POST_CHARGE_ACTIVITY: "Activity after the disputed charge",
        ExhibitKind.CARD_CONTINUITY: "Card-on-file continuity (account updater)",
        ExhibitKind.AVS_CVV: "AVS and CVV results",
        ExhibitKind.THREE_DS: "3-D Secure authentication",
        ExhibitKind.USAGE_LOGS: "Login and service-usage logs",
        ExhibitKind.TERMS_ACCEPTANCE: "Terms and cancellation policy accepted at signup",
        ExhibitKind.RENEWAL_REMINDER: "Pre-renewal reminder notice",
        ExhibitKind.CUSTOMER_CONTACT: "Customer contact and refund offers",
    }
)

_EXTERNAL_SOURCES: Mapping[ExhibitKind, str] = MappingProxyType(
    {
        ExhibitKind.AVS_CVV: "gateway authorization response for the disputed charge",
        ExhibitKind.THREE_DS: "3DS server / gateway authentication result (ECI, CAVV)",
        ExhibitKind.USAGE_LOGS: "product analytics or auth logs: logins, IP address, device ID",
        ExhibitKind.TERMS_ACCEPTANCE: "signup flow: terms version, checkbox timestamp, screenshot",
        ExhibitKind.RENEWAL_REMINDER: "email/notification service send log",
        ExhibitKind.CUSTOMER_CONTACT: "support desk tickets and refund ledger",
    }
)


@dataclass(frozen=True, slots=True)
class Requirement:
    kind: ExhibitKind
    required: bool
    why: str
    """What the exhibit proves for this reason code, in the analyst's terms."""


def _req(kind: ExhibitKind, why: str) -> Requirement:
    return Requirement(kind, True, why)


def _opt(kind: ExhibitKind, why: str) -> Requirement:
    return Requirement(kind, False, why)


_TXN = _req(ExhibitKind.TRANSACTION_RECORD, "identifies the charge being disputed")
_ACCOUNT = _req(ExhibitKind.ACCOUNT_PROFILE, "ties the charge to a known customer account")

_REQUIREMENTS: Mapping[str, tuple[Requirement, ...]] = MappingProxyType(
    {
        "visa:10.4": (
            _TXN,
            _ACCOUNT,
            _req(
                ExhibitKind.CE3_PRIOR_TRANSACTIONS,
                "CE 3.0: 2+ undisputed transactions 120–365 days earlier on the same card, "
                "sharing 2+ identifiers with the disputed one, shifts liability to the issuer",
            ),
            _req(ExhibitKind.USAGE_LOGS, "supplies the IP address / device ID that CE 3.0 matches"),
            _opt(ExhibitKind.AVS_CVV, "shows the card details entered matched the issuer's"),
            _opt(ExhibitKind.POST_CHARGE_ACTIVITY, "the customer kept paying after the charge"),
            _opt(ExhibitKind.CARD_CONTINUITY, "the issuer itself kept the card on file current"),
        ),
        "mastercard:4837": (
            _TXN,
            _ACCOUNT,
            _req(
                ExhibitKind.PRIOR_UNDISPUTED_HISTORY,
                "the same card was used on this account before without dispute",
            ),
            _opt(ExhibitKind.THREE_DS, "an authenticated charge shifts fraud liability"),
            _opt(ExhibitKind.USAGE_LOGS, "the cardholder used the service after the charge"),
            _opt(ExhibitKind.POST_CHARGE_ACTIVITY, "the customer kept paying after the charge"),
            _opt(ExhibitKind.CARD_CONTINUITY, "the issuer itself kept the card on file current"),
        ),
        "visa:13.1": (
            _TXN,
            _req(ExhibitKind.SUBSCRIPTION_RECORD, "defines what was bought and when it is due"),
            _req(ExhibitKind.USAGE_LOGS, "the service was available and used after the charge"),
            _opt(ExhibitKind.TERMS_ACCEPTANCE, "the customer accepted what they would receive"),
            _opt(ExhibitKind.POST_CHARGE_ACTIVITY, "the customer kept paying after the charge"),
        ),
        "visa:13.2": (
            _TXN,
            _req(ExhibitKind.SUBSCRIPTION_RECORD, "shows the recurring agreement"),
            _req(
                ExhibitKind.CANCELLATION_CHECK,
                "no cancellation request reached the merchant before the charge",
            ),
            _req(ExhibitKind.TERMS_ACCEPTANCE, "the cancellation policy was disclosed and agreed"),
            _opt(ExhibitKind.BILLING_HISTORY, "earlier renewals were paid without dispute"),
            _opt(ExhibitKind.RENEWAL_REMINDER, "the customer was told the renewal was coming"),
            _opt(ExhibitKind.USAGE_LOGS, "the customer kept using the service"),
        ),
        "mastercard:4853": (
            _TXN,
            _req(ExhibitKind.SUBSCRIPTION_RECORD, "defines what was sold"),
            _req(ExhibitKind.CANCELLATION_CHECK, "no cancellation request before the charge"),
            _req(ExhibitKind.USAGE_LOGS, "the service was delivered and used"),
            _opt(ExhibitKind.TERMS_ACCEPTANCE, "the terms shown at purchase"),
            _opt(ExhibitKind.CUSTOMER_CONTACT, "how the merchant responded to any complaint"),
            _opt(ExhibitKind.BILLING_HISTORY, "earlier charges were paid without dispute"),
        ),
    }
)

_FALLBACK: Mapping[DisputeCategory, tuple[Requirement, ...]] = MappingProxyType(
    {
        DisputeCategory.FRAUD: _REQUIREMENTS["mastercard:4837"],
        DisputeCategory.CONSUMER: _REQUIREMENTS["mastercard:4853"],
        DisputeCategory.PROCESSING: (_TXN, _ACCOUNT),
    }
)


def requirements_for(reason: ReasonCode) -> tuple[Requirement, ...]:
    """Exhibits to assemble for a reason code, required ones first-class.

    Codes without a dedicated list fall back to their category's generic list, so adding a code
    to the catalog never breaks the assembler.
    """
    return _REQUIREMENTS.get(reason.key, _FALLBACK[reason.category])
