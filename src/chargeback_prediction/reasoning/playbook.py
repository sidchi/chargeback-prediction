"""Deterministic representment playbook: the policy the LLM is allowed to explain, not override.

Whether to fight a dispute is a money decision with network-rule consequences, so it is made by
reviewable rules. The rules read the transaction's features as **signals** — facts that lean
towards a stolen card, towards the real cardholder (friendly fraud), or towards a merchant error —
and map them, per reason code, to a recommended :class:`Strategy` plus the set of strategies a
language model may choose between when it writes the rationale.

Evidence lists summarise public network guidance (e.g. Visa Compelling Evidence 3.0) in
illustrative form. They are prompts for the evidence assembler (roadmap item 7), not legal or
network-compliance advice; check current network rules before relying on them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from chargeback_prediction.explain.factors import format_value
from chargeback_prediction.reason_codes import DisputeCategory, ReasonCode


class Strategy(StrEnum):
    REPRESENT = "represent"
    """Fight the dispute: submit compelling evidence to the issuer."""
    ACCEPT = "accept"
    """Accept the chargeback (or refund first): fighting would likely lose or is unjustified."""
    REVIEW = "review"
    """Signals conflict; route to an analyst before the response deadline."""


class Leaning(StrEnum):
    TRUE_FRAUD = "true_fraud"
    """Consistent with a stolen card: the cardholder really did not authorise the charge."""
    CARDHOLDER = "cardholder"
    """Consistent with the real cardholder making (and benefiting from) the charge."""
    MERCHANT_ERROR = "merchant_error"
    """The merchant is at fault, e.g. charging after a cancellation request."""


NEW_ACCOUNT_DAYS = 7.0
ESTABLISHED_ACCOUNT_DAYS = 120.0
"""Matches the start of Visa CE 3.0's 120–365 day look-back for prior undisputed transactions."""
VELOCITY_ATTEMPTS_24H = 4
CARD_TESTING_DECLINES_24H = 2
LOYAL_RENEWALS = 2


@dataclass(frozen=True, slots=True)
class Signal:
    """A fact about the transaction that points the dispute one way."""

    feature: str
    leaning: Leaning
    detail: str


def _get(features: Mapping[str, float], name: str) -> float:
    try:
        return float(features[name])
    except KeyError:
        raise KeyError(f"feature {name!r} is required by the playbook") from None


def _show(name: str, features: Mapping[str, float]) -> str:
    return format_value(name, _get(features, name))


_Rule = Callable[[Mapping[str, float]], Signal | None]


def _flag(feature: str, leaning: Leaning, detail: str) -> _Rule:
    def rule(f: Mapping[str, float]) -> Signal | None:
        return Signal(feature, leaning, detail) if _get(f, feature) >= 0.5 else None

    return rule


def _at_least(feature: str, threshold: float, leaning: Leaning, template: str) -> _Rule:
    """Fires when ``feature >= threshold``; ``template`` is formatted with ``{value}``."""

    def rule(f: Mapping[str, float]) -> Signal | None:
        if _get(f, feature) >= threshold:
            return Signal(feature, leaning, template.format(value=_show(feature, f)))
        return None

    return rule


def _new_account(f: Mapping[str, float]) -> Signal | None:
    if _get(f, "account_age_days") < NEW_ACCOUNT_DAYS:
        detail = f"account created {_show('account_age_days', f)} before the charge"
        return Signal("account_age_days", Leaning.TRUE_FRAUD, detail)
    return None


def _established_account(f: Mapping[str, float]) -> Signal | None:
    if _get(f, "account_age_days") >= ESTABLISHED_ACCOUNT_DAYS and _get(f, "ip_bin_mismatch") < 0.5:
        detail = (
            f"established account ({_show('account_age_days', f)} old) "
            "with no IP/issuer country mismatch"
        )
        return Signal("account_age_days", Leaning.CARDHOLDER, detail)
    return None


_RULES: tuple[_Rule, ...] = (
    _flag(
        "charged_after_cancel_request",
        Leaning.MERCHANT_ERROR,
        "the customer had already asked to cancel before this charge",
    ),
    _new_account,
    _flag("email_disposable", Leaning.TRUE_FRAUD, "signed up with a disposable email address"),
    _flag(
        "ip_bin_mismatch",
        Leaning.TRUE_FRAUD,
        "customer IP country differs from the card's issuing country",
    ),
    _flag(
        "billing_bin_mismatch",
        Leaning.TRUE_FRAUD,
        "billing country differs from the card's issuing country",
    ),
    _at_least(
        "attempts_24h",
        VELOCITY_ATTEMPTS_24H,
        Leaning.TRUE_FRAUD,
        "{value} authorization attempts in the previous 24 h",
    ),
    _at_least(
        "declines_24h",
        CARD_TESTING_DECLINES_24H,
        Leaning.TRUE_FRAUD,
        "{value} declines in the previous 24 h (card testing)",
    ),
    _established_account,
    _at_least(
        "prior_approved_renewals",
        LOYAL_RENEWALS,
        Leaning.CARDHOLDER,
        "{value} approved renewals of this subscription before the charge",
    ),
    _at_least(
        "prior_disputes",
        1,
        Leaning.CARDHOLDER,
        "{value} earlier dispute(s) filed by this customer (possible serial disputer)",
    ),
)


def collect_signals(features: Mapping[str, float]) -> tuple[Signal, ...]:
    """Every playbook signal that fires for this transaction, in rule order.

    Raises:
        KeyError: if a feature the rules need is missing.
    """
    return tuple(s for rule in _RULES if (s := rule(features)) is not None)


_EVIDENCE: Mapping[str, tuple[str, ...]] = {
    "visa:10.4": (
        "Visa Compelling Evidence 3.0: two or more prior undisputed transactions on the same "
        "card, 120–365 days before the dispute, sharing at least two identifiers (IP address, "
        "device ID, user account, shipping address) with the disputed charge",
        "account logins or service usage after the disputed charge",
        "AVS and CVV results from the original authorization",
    ),
    "mastercard:4837": (
        "proof the cardholder participated: prior undisputed transactions on the same card "
        "and account",
        "3-D Secure authentication result, if the charge was authenticated",
        "account logins or service usage after the disputed charge",
    ),
    "visa:13.1": (
        "proof the service was available and used: login and usage logs after the charge",
        "subscription terms the customer accepted at signup",
    ),
    "visa:13.2": (
        "cancellation policy disclosed at signup and the customer's acceptance of it",
        "account and support history showing no cancellation request before the charge",
        "pre-renewal reminder notice, if one was sent",
        "service usage after the renewal",
    ),
    "mastercard:4853": (
        "terms of service and the subscription description shown at purchase",
        "service usage logs",
        "record of customer contact and any refund offered",
    ),
}

_ACCEPT_EVIDENCE = ("refund confirmation, or accept the chargeback without a response",)


@dataclass(frozen=True, slots=True)
class Recommendation:
    """What the playbook recommends for one dispute, and how much latitude the LLM gets."""

    strategy: Strategy
    allowed: frozenset[Strategy]
    """Strategies a model-written rationale may settle on. Always contains ``strategy``."""
    reasons: tuple[str, ...]
    """Why the rules landed here, in plain language."""
    signals: tuple[Signal, ...]
    evidence: tuple[str, ...]
    """Evidence to assemble if the dispute is represented (empty guidance for unknown codes)."""

    def __post_init__(self) -> None:
        if self.strategy not in self.allowed:
            raise ValueError("the recommended strategy must be one of the allowed strategies")

    def count(self, leaning: Leaning) -> int:
        return sum(s.leaning is leaning for s in self.signals)


def recommend(reason: ReasonCode, features: Mapping[str, float]) -> Recommendation:
    """Apply the playbook to a disputed transaction.

    Hard rule: a charge made after the customer asked to cancel is accepted whatever the reason
    code: the merchant would lose, and fighting it invites a compliance complaint.
    """
    signals = collect_signals(features)
    fraud = sum(s.leaning is Leaning.TRUE_FRAUD for s in signals)
    cardholder = sum(s.leaning is Leaning.CARDHOLDER for s in signals)
    merchant_error = any(s.leaning is Leaning.MERCHANT_ERROR for s in signals)
    evidence = _EVIDENCE.get(reason.key, ())

    def rec(strategy: Strategy, allowed: set[Strategy], *reasons: str) -> Recommendation:
        chosen = _ACCEPT_EVIDENCE if strategy is Strategy.ACCEPT else evidence
        return Recommendation(strategy, frozenset(allowed), reasons, signals, chosen)

    if merchant_error:
        return rec(
            Strategy.ACCEPT,
            {Strategy.ACCEPT},
            "The customer asked to cancel before this charge, so the merchant is at fault.",
        )

    if reason.category is DisputeCategory.FRAUD:
        if fraud >= 2 and cardholder == 0:
            return rec(
                Strategy.ACCEPT,
                {Strategy.ACCEPT, Strategy.REVIEW},
                f"{fraud} stolen-card signals and none pointing to the real cardholder.",
            )
        if cardholder >= 2 and fraud == 0:
            return rec(
                Strategy.REPRESENT,
                {Strategy.REPRESENT, Strategy.REVIEW},
                f"{cardholder} signals that the real cardholder made the charge and none of a "
                "stolen card: likely friendly fraud.",
            )
        return rec(
            Strategy.REVIEW,
            set(Strategy),
            f"Mixed or weak signals ({fraud} stolen-card, {cardholder} cardholder).",
        )

    if reason.category is DisputeCategory.CONSUMER:
        if fraud >= 2:
            return rec(
                Strategy.REVIEW,
                set(Strategy),
                f"Filed as a consumer dispute, but {fraud} stolen-card signals fired: check "
                "whether the card was compromised before arguing the service was delivered.",
            )
        return rec(
            Strategy.REPRESENT,
            {Strategy.REPRESENT, Strategy.REVIEW},
            "No cancellation request before the charge, and the service was delivered as "
            "subscribed: representable with usage and terms evidence.",
        )

    return rec(Strategy.REVIEW, set(Strategy), "No playbook for this dispute category yet.")
