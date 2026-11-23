"""Gather the evidence for a dispute from the raw tables and draft its response packet.

Everything is read **point in time**: only records stamped at or before ``as_of`` (by default
the moment the dispute was filed) are used, so a packet assembled today is the packet the
analyst would have seen then, and tests do not depend on how far the dataset runs.

Two columns in the synthetic data are ground-truth labels (``Customer.persona`` and
``Dispute.is_friendly_fraud``). The assembler never reads them: a real merchant does not know
them, and a packet built from them would be fiction.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta

from chargeback_prediction.reason_codes import DisputeCategory, ReasonCode, lookup
from chargeback_prediction.representment.packet import (
    Exhibit,
    ExhibitStatus,
    ResponsePacket,
    format_money,
)
from chargeback_prediction.representment.requirements import (
    ExhibitKind,
    Requirement,
    requirements_for,
)
from chargeback_prediction.synthetic.generator import SyntheticDataset
from chargeback_prediction.synthetic.schema import (
    CardUpdate,
    CardUpdateSource,
    Customer,
    Dispute,
    Subscription,
    Transaction,
)

CE3_WINDOW_DAYS = (120, 365)
"""Visa CE 3.0 looks at transactions 120–365 days before the disputed one."""
CE3_MIN_TRANSACTIONS = 2
DEFAULT_RESPONSE_DAYS = 20
"""A typical acquirer-imposed deadline for the merchant's response. Acquirers set their own,
shorter than the network's, so treat this as a placeholder and configure it."""


class EvidenceIndex:
    """The dataset indexed for evidence lookups. Build once, assemble many packets."""

    def __init__(self, dataset: SyntheticDataset) -> None:
        self.customers: dict[str, Customer] = {c.customer_id: c for c in dataset.customers}
        self.subscriptions: dict[str, Subscription] = {
            s.subscription_id: s for s in dataset.subscriptions
        }
        self.transactions: dict[str, Transaction] = {
            t.transaction_id: t for t in dataset.transactions
        }
        self.disputes: dict[str, Dispute] = {d.dispute_id: d for d in dataset.disputes}
        self._txns_by_customer: dict[str, list[Transaction]] = defaultdict(list)
        for t in sorted(dataset.transactions, key=lambda t: (t.created_at, t.transaction_id)):
            self._txns_by_customer[t.customer_id].append(t)
        self._updates_by_customer: dict[str, list[CardUpdate]] = defaultdict(list)
        for u in sorted(dataset.card_updates, key=lambda u: u.updated_at):
            self._updates_by_customer[u.customer_id].append(u)
        self._disputes_by_txn: dict[str, list[Dispute]] = defaultdict(list)
        self._disputes_by_customer: dict[str, list[Dispute]] = defaultdict(list)
        for d in sorted(dataset.disputes, key=lambda d: d.filed_at):
            self._disputes_by_txn[d.transaction_id].append(d)
            self._disputes_by_customer[d.customer_id].append(d)

    def customer_transactions(self, customer_id: str, as_of: datetime) -> Iterator[Transaction]:
        return (t for t in self._txns_by_customer[customer_id] if t.created_at <= as_of)

    def card_updates(self, customer_id: str, as_of: datetime) -> list[CardUpdate]:
        return [u for u in self._updates_by_customer[customer_id] if u.updated_at <= as_of]

    def disputed(self, transaction_id: str, as_of: datetime) -> bool:
        return any(d.filed_at <= as_of for d in self._disputes_by_txn[transaction_id])

    def customer_disputes(self, customer_id: str, as_of: datetime) -> list[Dispute]:
        return [d for d in self._disputes_by_customer[customer_id] if d.filed_at <= as_of]


class _Case:
    """One dispute's view of the index, with the shared lookups every collector needs."""

    def __init__(self, index: EvidenceIndex, dispute: Dispute, as_of: datetime) -> None:
        try:
            self.txn = index.transactions[dispute.transaction_id]
            self.customer = index.customers[dispute.customer_id]
        except KeyError as exc:
            raise KeyError(
                f"dispute {dispute.dispute_id} references unknown record {exc.args[0]!r}"
            ) from None
        self.index = index
        self.dispute = dispute
        self.as_of = as_of
        self.sub = index.subscriptions.get(self.txn.subscription_id or "")
        self.history = list(index.customer_transactions(self.customer.customer_id, as_of))
        self.updates = index.card_updates(self.customer.customer_id, as_of)

    @property
    def charged_at(self) -> datetime:
        return self.txn.created_at

    def undisputed_approved(self, txns: Iterable[Transaction]) -> list[Transaction]:
        return [
            t
            for t in txns
            if t.approved
            and t.transaction_id != self.txn.transaction_id
            and not self.index.disputed(t.transaction_id, self.as_of)
        ]

    def same_card_since(self) -> datetime | None:
        """When the card on file became the disputed card.

        A customer replacing their card starts a new card; an account-updater refresh keeps the
        same card account (the issuer reissued it), so it does not break continuity.
        """
        replaced = [
            u.updated_at
            for u in self.updates
            if u.source is CardUpdateSource.CUSTOMER and u.updated_at <= self.charged_at
        ]
        return max(replaced) if replaced else None

    def on_same_card(self, t: Transaction) -> bool:
        since = self.same_card_since()
        return since is None or t.created_at >= since

    def cancellation_before_charge(self) -> datetime | None:
        if self.sub is None or self.sub.cancel_requested_at is None:
            return None
        requested = self.sub.cancel_requested_at
        return requested if requested < self.charged_at else None

    def money(self, cents: int) -> str:
        return format_money(cents, self.txn.currency)


def _day(at: datetime) -> str:
    return f"{at:%Y-%m-%d}"


def _ids(txns: Iterable[Transaction]) -> tuple[str, ...]:
    return tuple(t.transaction_id for t in txns)


# --------------------------------------------------------------------------- collectors
# Each returns (status, summary, facts, transaction ids) for one exhibit kind.

_Collected = tuple[ExhibitStatus, str, tuple[tuple[str, str], ...], tuple[str, ...]]


def _transaction_record(c: _Case) -> _Collected:
    t = c.txn
    facts = (
        ("Transaction", f"`{t.transaction_id}`"),
        ("Date", f"{t.created_at:%Y-%m-%d %H:%M} UTC"),
        ("Amount", c.money(t.amount_cents)),
        (
            "Type",
            {
                "initial": "first subscription charge",
                "renewal": "recurring renewal",
                "one_off": "one-off purchase",
            }[t.kind.value],
        ),
        ("Authorization", f"approved on attempt {t.attempt}" if t.approved else "declined"),
        ("Network / issuer country", f"{t.network.title()} / {t.bin_country}"),
        ("Customer IP country", t.ip_country or "n/a (merchant-initiated, no session)"),
    )
    summary = (
        f"{c.money(t.amount_cents)} {facts[3][1]} on {_day(t.created_at)}, approved by the issuer."
    )
    return ExhibitStatus.FOUND, summary, facts, (t.transaction_id,)


def _account_profile(c: _Case) -> _Collected:
    cu = c.customer
    age = (c.charged_at - cu.created_at).days
    facts = (
        ("Customer", f"`{cu.customer_id}`"),
        ("Account opened", _day(cu.created_at)),
        ("Account age at charge", f"{age} days"),
        ("Billing country / card issuer country", f"{cu.country} / {cu.bin_country}"),
        ("Email domain", cu.email_domain_type),
    )
    summary = f"Account opened {_day(cu.created_at)}, {age} days before the disputed charge"
    if c.txn.ip_country is not None and c.txn.ip_country == cu.bin_country:
        summary += f"; the purchase came from {c.txn.ip_country}, the card's issuing country"
    return ExhibitStatus.FOUND, summary + ".", facts, ()


def _subscription_record(c: _Case) -> _Collected:
    s = c.sub
    if s is None:
        return (
            ExhibitStatus.MISSING,
            "The charge was a one-off purchase, not a subscription.",
            (),
            (),
        )
    facts = (
        ("Subscription", f"`{s.subscription_id}`"),
        ("Plan", f"{s.plan.value}, {c.money(s.price_cents)} per {s.plan.period_days} days"),
        ("Started", _day(s.started_at)),
    )
    summary = (
        f"{s.plan.value.title()} plan at {c.money(s.price_cents)}, started {_day(s.started_at)}."
    )
    return ExhibitStatus.FOUND, summary, facts, ()


def _cancellation_check(c: _Case) -> _Collected:
    s = c.sub
    if s is None:
        return ExhibitStatus.MISSING, "No subscription, so no cancellation history.", (), ()
    before = c.cancellation_before_charge()
    if before is not None:
        return (
            ExhibitStatus.WEAK,
            f"The customer asked to cancel on {_day(before)}, before this charge.",
            (("Cancellation requested", _day(before)),),
            (),
        )
    requested = s.cancel_requested_at
    if requested is not None and requested <= c.as_of:
        summary = (
            f"No cancellation request before the charge. The customer asked to cancel on "
            f"{_day(requested)}, after it."
        )
        return ExhibitStatus.FOUND, summary, (("Cancellation requested", _day(requested)),), ()
    return (
        ExhibitStatus.FOUND,
        f"No cancellation request on record as of {_day(c.as_of)}.",
        (("Cancellation requested", "never"),),
        (),
    )


def _billing_history(c: _Case) -> _Collected:
    if c.sub is None:
        return ExhibitStatus.MISSING, "No subscription billing history.", (), ()
    prior = [
        t
        for t in c.undisputed_approved(c.history)
        if t.subscription_id == c.sub.subscription_id and t.created_at < c.charged_at
    ]
    if not prior:
        return ExhibitStatus.WEAK, "The disputed charge was the first on this subscription.", (), ()
    total = sum(t.amount_cents for t in prior)
    summary = (
        f"{len(prior)} earlier charge(s) on this subscription, {c.money(total)} in total, "
        f"paid without dispute (first {_day(prior[0].created_at)}, last "
        f"{_day(prior[-1].created_at)})."
    )
    return (
        ExhibitStatus.FOUND,
        summary,
        (("Undisputed prior charges", str(len(prior))),),
        _ids(prior),
    )


def _ce3_prior_transactions(c: _Case) -> _Collected:
    lo, hi = CE3_WINDOW_DAYS
    start, end = c.charged_at - timedelta(days=hi), c.charged_at - timedelta(days=lo)
    qualifying = [
        t
        for t in c.undisputed_approved(c.history)
        if start <= t.created_at <= end and c.on_same_card(t)
    ]
    facts = (
        ("Look-back window", f"{_day(start)} to {_day(end)}"),
        ("Qualifying transactions", f"{len(qualifying)} (need {CE3_MIN_TRANSACTIONS})"),
        ("Shared identifiers in data", "user account ID"),
        ("To confirm from logs", "IP address or device ID (CE 3.0 needs 2 identifiers)"),
    )
    if len(qualifying) >= CE3_MIN_TRANSACTIONS:
        summary = (
            f"{len(qualifying)} undisputed transactions on the same card and user account in the "
            f"CE 3.0 window."
        )
        return ExhibitStatus.FOUND, summary, facts, _ids(qualifying)
    have = f"Only {len(qualifying)}" if qualifying else "No"
    summary = (
        f"{have} undisputed transaction(s) on this card in the CE 3.0 window; "
        f"the liability shift needs {CE3_MIN_TRANSACTIONS}."
    )
    return ExhibitStatus.WEAK, summary, facts, _ids(qualifying)


def _prior_undisputed_history(c: _Case) -> _Collected:
    prior = [
        t
        for t in c.undisputed_approved(c.history)
        if t.created_at < c.charged_at and c.on_same_card(t)
    ]
    if not prior:
        return (
            ExhibitStatus.WEAK,
            "No earlier undisputed transactions on this card and account.",
            (),
            (),
        )
    summary = (
        f"{len(prior)} earlier undisputed transaction(s) on the same card and account, "
        f"from {_day(prior[0].created_at)}."
    )
    return ExhibitStatus.FOUND, summary, (("Prior undisputed", str(len(prior))),), _ids(prior)


def _post_charge_activity(c: _Case) -> _Collected:
    after = [t for t in c.undisputed_approved(c.history) if t.created_at > c.charged_at]
    if not after:
        return (
            ExhibitStatus.WEAK,
            f"No further approved charges between the disputed one and {_day(c.as_of)}.",
            (),
            (),
        )
    summary = (
        f"The customer was charged {len(after)} more time(s) on the same account after the "
        f"disputed charge without disputing them (latest {_day(after[-1].created_at)})."
    )
    return ExhibitStatus.FOUND, summary, (("Later undisputed", str(len(after))),), _ids(after)


def _card_continuity(c: _Case) -> _Collected:
    refreshes = [u for u in c.updates if u.source is CardUpdateSource.ACCOUNT_UPDATER]
    if not refreshes:
        return ExhibitStatus.MISSING, "No account-updater refreshes on record.", (), ()
    summary = (
        f"The issuer refreshed this card on file through the account updater "
        f"{len(refreshes)} time(s), latest {_day(refreshes[-1].updated_at)}."
    )
    return ExhibitStatus.FOUND, summary, (("Updater refreshes", str(len(refreshes))),), ()


_COLLECTORS = {
    ExhibitKind.TRANSACTION_RECORD: _transaction_record,
    ExhibitKind.ACCOUNT_PROFILE: _account_profile,
    ExhibitKind.SUBSCRIPTION_RECORD: _subscription_record,
    ExhibitKind.CANCELLATION_CHECK: _cancellation_check,
    ExhibitKind.BILLING_HISTORY: _billing_history,
    ExhibitKind.CE3_PRIOR_TRANSACTIONS: _ce3_prior_transactions,
    ExhibitKind.PRIOR_UNDISPUTED_HISTORY: _prior_undisputed_history,
    ExhibitKind.POST_CHARGE_ACTIVITY: _post_charge_activity,
    ExhibitKind.CARD_CONTINUITY: _card_continuity,
}


def _exhibit(c: _Case, req: Requirement) -> Exhibit:
    if req.kind.is_external:
        summary = f"Attach from {req.kind.external_source}."
        return Exhibit(req.kind, ExhibitStatus.ATTACH, req.required, req.why, summary)
    status, summary, facts, ids = _COLLECTORS[req.kind](c)
    return Exhibit(req.kind, status, req.required, req.why, summary, facts, ids)


# --------------------------------------------------------------------------- packet


def _adverse_facts(c: _Case) -> tuple[str, ...]:
    out: list[str] = []
    if (before := c.cancellation_before_charge()) is not None:
        out.append(
            f"The customer asked to cancel on {_day(before)}, before the {_day(c.charged_at)} "
            "charge. The merchant would lose; refund or accept."
        )
    if c.dispute.amount_cents > c.txn.amount_cents:
        out.append("The disputed amount exceeds the transaction amount; check the dispute record.")
    return tuple(out)


def _internal_notes(c: _Case) -> tuple[str, ...]:
    notes: list[str] = []
    earlier = [
        d
        for d in c.index.customer_disputes(c.customer.customer_id, c.as_of)
        if d.dispute_id != c.dispute.dispute_id
    ]
    if earlier:
        notes.append(
            f"The customer has {len(earlier)} other dispute(s) on file. Useful for internal risk "
            "decisions; never cite prior disputes in the submission."
        )
    if (since := c.same_card_since()) is not None:
        notes.append(
            f"The customer replaced their card on {_day(since)}; only history on "
            "the current card counts as same-card evidence."
        )
    if c.txn.ip_country is not None and c.txn.ip_country != c.customer.bin_country:
        notes.append(
            f"The purchase came from {c.txn.ip_country} but the card was issued in "
            f"{c.customer.bin_country}: a stolen-card signal the issuer will also see."
        )
    if c.dispute.amount_cents < c.txn.amount_cents:
        notes.append(
            f"Partial dispute: {c.money(c.dispute.amount_cents)} of {c.money(c.txn.amount_cents)}."
        )
    return tuple(notes)


def _rebuttal(
    c: _Case, reason: ReasonCode, exhibits: tuple[Exhibit, ...], adverse: tuple[str, ...]
) -> str:
    """A deterministic draft rebuttal built only from exhibits with status ``found``.

    No rebuttal is drafted when an adverse fact applies: arguing it would be misleading.
    """
    if adverse:
        return "Not drafted: an adverse fact applies (see the analyst section). Refund or accept."
    found = {e.kind: e for e in exhibits if e.status is ExhibitStatus.FOUND}
    t = c.txn
    opening = (
        f"The cardholder disputes the {c.money(t.amount_cents)} charge of {_day(t.created_at)} "
        f"under {reason.network.title()} {reason.code} ({reason.title})."
    )
    order: tuple[ExhibitKind, ...]
    if reason.category is DisputeCategory.FRAUD:
        claim = "The merchant contends the charge was made by the cardholder."
        order = (
            ExhibitKind.ACCOUNT_PROFILE,
            ExhibitKind.CE3_PRIOR_TRANSACTIONS,
            ExhibitKind.PRIOR_UNDISPUTED_HISTORY,
            ExhibitKind.POST_CHARGE_ACTIVITY,
            ExhibitKind.CARD_CONTINUITY,
        )
    else:
        claim = "The merchant contends the charge was valid under the agreed terms."
        order = (
            ExhibitKind.SUBSCRIPTION_RECORD,
            ExhibitKind.CANCELLATION_CHECK,
            ExhibitKind.BILLING_HISTORY,
            ExhibitKind.POST_CHARGE_ACTIVITY,
        )
    points = [found[k].summary for k in order if k in found]
    if not points:
        return f"{opening} The data holds no supporting evidence; see the attachments list."
    return " ".join([opening, claim, *points])


def assemble(
    index: EvidenceIndex,
    dispute: Dispute | str,
    *,
    as_of: datetime | None = None,
    response_days: int = DEFAULT_RESPONSE_DAYS,
) -> ResponsePacket:
    """Draft the response packet for one dispute.

    Args:
        dispute: the dispute record, or its id.
        as_of: read only data stamped at or before this moment (default: when it was filed).
        response_days: days after filing the merchant has to respond.

    Raises:
        KeyError: for an unknown dispute id or a dispute referencing missing records.
        ValueError: for an unknown reason code, ``as_of`` before the charge, or
            non-positive ``response_days``.
    """
    if isinstance(dispute, str):
        try:
            dispute = index.disputes[dispute]
        except KeyError:
            raise KeyError(f"unknown dispute id {dispute!r}") from None
    if response_days <= 0:
        raise ValueError("response_days must be positive")
    network, _, code = dispute.reason_code.partition(":")
    try:
        reason = lookup(network, code)
    except KeyError as exc:
        raise ValueError(str(exc.args[0])) from None

    case = _Case(index, dispute, as_of or dispute.filed_at)
    if case.as_of < case.charged_at:
        raise ValueError("as_of is before the disputed charge")
    exhibits = tuple(_exhibit(case, req) for req in requirements_for(reason))
    adverse = _adverse_facts(case)
    return ResponsePacket(
        dispute_id=dispute.dispute_id,
        transaction_id=case.txn.transaction_id,
        reason=reason,
        filed_at=dispute.filed_at,
        respond_by=dispute.filed_at + timedelta(days=response_days),
        as_of=case.as_of,
        amount_cents=dispute.amount_cents,
        currency=case.txn.currency,
        exhibits=exhibits,
        adverse=adverse,
        internal_notes=_internal_notes(case),
        rebuttal=_rebuttal(case, reason, exhibits, adverse),
    )


def assemble_all(
    dataset: SyntheticDataset, *, response_days: int = DEFAULT_RESPONSE_DAYS
) -> list[ResponsePacket]:
    """One packet per dispute, in filing order."""
    index = EvidenceIndex(dataset)
    ordered = sorted(dataset.disputes, key=lambda d: (d.filed_at, d.dispute_id))
    return [assemble(index, d, response_days=response_days) for d in ordered]
