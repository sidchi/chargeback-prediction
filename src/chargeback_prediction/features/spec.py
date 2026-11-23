"""The feature catalog: every model input, its group, type and meaning, in one place.

The catalog is the single source of truth for column order in
:class:`~chargeback_prediction.features.pipeline.FeatureTable` and for the documented feature
table in ``docs/features.md`` (regenerate with ``cbp features --doc``; a test fails if the
committed copy drifts).

Every feature is **point-in-time**: computed only from the transaction itself and events that
happened *strictly before* its ``created_at``. In particular, prior-dispute features count
disputes by the time they were *filed*, not by the time of the disputed charge, because a
dispute filed next month is not something the merchant could have known at authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FeatureGroup(StrEnum):
    TRANSACTION = "transaction"
    ACCOUNT = "account"
    GEO = "geo"
    VELOCITY = "velocity"
    CARD_ON_FILE = "card_on_file"
    SUBSCRIPTION = "subscription"
    PRIOR_DISPUTES = "prior_disputes"


class FeatureKind(StrEnum):
    FLAG = "flag"
    """0 or 1."""
    COUNT = "count"
    """Non-negative integer count of prior events."""
    NUMERIC = "numeric"
    """Continuous, non-negative value."""


HISTORY_CAP_DAYS = 365.0
"""Value for "days since X" features when X never happened (or happened longer ago).

A finite cap keeps every feature numeric and non-null for linear models; the model can still
tell "never" from "recently" because recent values are small. Tree models are indifferent.
"""


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    group: FeatureGroup
    kind: FeatureKind
    description: str


_G = FeatureGroup
_K = FeatureKind

FEATURES: tuple[FeatureSpec, ...] = (
    # --- the authorization itself
    FeatureSpec("amount_usd", _G.TRANSACTION, _K.NUMERIC, "Charge amount in dollars."),
    FeatureSpec("is_initial", _G.TRANSACTION, _K.FLAG, "First charge of a subscription."),
    FeatureSpec(
        "is_renewal", _G.TRANSACTION, _K.FLAG, "Merchant-initiated recurring charge (card on file)."
    ),
    FeatureSpec(
        "is_one_off", _G.TRANSACTION, _K.FLAG, "Customer-initiated purchase outside a subscription."
    ),
    FeatureSpec("is_mastercard", _G.TRANSACTION, _K.FLAG, "Card network is Mastercard (vs. Visa)."),
    FeatureSpec(
        "attempt",
        _G.TRANSACTION,
        _K.COUNT,
        "Authorization attempt number for this charge; 2+ means it is a retry after a soft "
        "decline.",
    ),
    # --- the account
    FeatureSpec(
        "account_age_days", _G.ACCOUNT, _K.NUMERIC, "Days between account creation and the charge."
    ),
    FeatureSpec(
        "email_disposable", _G.ACCOUNT, _K.FLAG, "Signed up with a disposable email domain."
    ),
    FeatureSpec(
        "email_freemail",
        _G.ACCOUNT,
        _K.FLAG,
        "Signed up with a free webmail domain (baseline: corporate).",
    ),
    # --- geography vs. the card's issuer
    FeatureSpec(
        "ip_missing",
        _G.GEO,
        _K.FLAG,
        "No customer IP: merchant-initiated charge with no customer session.",
    ),
    FeatureSpec(
        "ip_bin_mismatch",
        _G.GEO,
        _K.FLAG,
        "Customer IP country differs from the card's issuing (BIN) country. 0 when IP is missing.",
    ),
    FeatureSpec(
        "billing_bin_mismatch",
        _G.GEO,
        _K.FLAG,
        "Account billing country differs from the card's issuing (BIN) country.",
    ),
    # --- velocity over all of the customer's earlier authorization attempts
    FeatureSpec(
        "attempts_1h", _G.VELOCITY, _K.COUNT, "Authorization attempts in the previous hour."
    ),
    FeatureSpec(
        "attempts_24h", _G.VELOCITY, _K.COUNT, "Authorization attempts in the previous 24 hours."
    ),
    FeatureSpec(
        "attempts_7d", _G.VELOCITY, _K.COUNT, "Authorization attempts in the previous 7 days."
    ),
    FeatureSpec(
        "declines_24h",
        _G.VELOCITY,
        _K.COUNT,
        "Declined attempts in the previous 24 hours (card-testing signal).",
    ),
    FeatureSpec(
        "declines_30d", _G.VELOCITY, _K.COUNT, "Declined attempts in the previous 30 days."
    ),
    FeatureSpec(
        "approved_usd_24h",
        _G.VELOCITY,
        _K.NUMERIC,
        "Dollars approved in the previous 24 hours (spend burst).",
    ),
    FeatureSpec(
        "days_since_prev_attempt",
        _G.VELOCITY,
        _K.NUMERIC,
        f"Days since the customer's previous attempt; {HISTORY_CAP_DAYS:g} if none.",
    ),
    # --- card on file
    FeatureSpec(
        "au_updates_90d",
        _G.CARD_ON_FILE,
        _K.COUNT,
        "Account-updater card reissues in the previous 90 days (the cardholder may believe the "
        "old card ended the subscription).",
    ),
    FeatureSpec(
        "customer_card_updates_90d",
        _G.CARD_ON_FILE,
        _K.COUNT,
        "Cards on file replaced by the customer in the previous 90 days.",
    ),
    FeatureSpec(
        "days_since_card_update",
        _G.CARD_ON_FILE,
        _K.NUMERIC,
        f"Days since the card on file last changed (any source); {HISTORY_CAP_DAYS:g} if never.",
    ),
    # --- subscription and retry history (zero for one-off purchases)
    FeatureSpec("plan_is_annual", _G.SUBSCRIPTION, _K.FLAG, "Charge belongs to an annual plan."),
    FeatureSpec(
        "subscription_age_days",
        _G.SUBSCRIPTION,
        _K.NUMERIC,
        "Days since the subscription started.",
    ),
    FeatureSpec(
        "prior_approved_renewals",
        _G.SUBSCRIPTION,
        _K.COUNT,
        "Renewals of this subscription that were approved before this charge.",
    ),
    FeatureSpec(
        "prior_subscription_declines",
        _G.SUBSCRIPTION,
        _K.COUNT,
        "Declined attempts on this subscription before this charge (retry history).",
    ),
    FeatureSpec(
        "charged_after_cancel_request",
        _G.SUBSCRIPTION,
        _K.FLAG,
        "The customer had already asked to cancel this subscription: a legitimate "
        "'cancelled recurring' dispute waiting to happen.",
    ),
    # --- the customer's dispute history, by filing date
    FeatureSpec(
        "prior_disputes",
        _G.PRIOR_DISPUTES,
        _K.COUNT,
        "Disputes the customer filed before this charge.",
    ),
    FeatureSpec(
        "prior_fraud_disputes",
        _G.PRIOR_DISPUTES,
        _K.COUNT,
        "Of those, disputes in the fraud category (Visa 10.4, Mastercard 4837).",
    ),
    FeatureSpec(
        "days_since_last_dispute",
        _G.PRIOR_DISPUTES,
        _K.NUMERIC,
        f"Days since the customer last filed a dispute; {HISTORY_CAP_DAYS:g} if never.",
    ),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)


def render_feature_table() -> str:
    """Render the catalog as the Markdown document committed at ``docs/features.md``."""
    lines = [
        "# Feature table",
        "",
        "<!-- Generated by `cbp features --doc`. Do not edit by hand. -->",
        "",
        "One row per **approved** transaction (only approved charges can be disputed). Every",
        "feature is point-in-time: it uses the transaction itself and events strictly before its",
        "`created_at`. Prior disputes count by **filing** date, not by the date of the disputed",
        "charge. Velocity counts include declined attempts.",
        "",
        "Label columns (`disputed`, `reason_code`, `dispute_category`, `label_mature`) sit",
        "alongside the features and must never be used as inputs. `label_mature` is false while",
        "the dispute window is still open, so a negative label there may yet turn positive.",
        "",
        "| # | Feature | Group | Kind | Description |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {i} | `{f.name}` | {f.group} | {f.kind} | {f.description} |"
        for i, f in enumerate(FEATURES, start=1)
    ]
    return "\n".join(lines) + "\n"
