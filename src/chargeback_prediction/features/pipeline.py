"""Point-in-time feature pipeline: raw events in, one feature row per approved charge out.

The pipeline indexes each customer's and subscription's events into sorted timelines with
prefix sums, so every windowed count or sum is two binary searches. Building features for the
default synthetic dataset (~40k attempts) takes well under a second in pure Python, and the
same code path would work on a production export because it only reads the raw tables.

Look-ahead is prevented structurally: every query asks for events in ``[t - window, t)``, so
events at or after the charge's own timestamp are never visible to it.
"""

from __future__ import annotations

import csv
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import accumulate
from pathlib import Path

from chargeback_prediction.features.spec import FEATURE_NAMES, HISTORY_CAP_DAYS
from chargeback_prediction.reason_codes import DisputeCategory, Network
from chargeback_prediction.synthetic.config import MAX_DISPUTE_WINDOW_DAYS
from chargeback_prediction.synthetic.generator import SyntheticDataset
from chargeback_prediction.synthetic.schema import (
    CardUpdateSource,
    Customer,
    Dispute,
    Plan,
    Subscription,
    Transaction,
    TransactionKind,
)

_DAY = timedelta(days=1)
_SECONDS_PER_DAY = 86_400.0

LABEL_COLUMNS: tuple[str, ...] = ("disputed", "reason_code", "dispute_category", "label_mature")
ID_COLUMNS: tuple[str, ...] = ("transaction_id", "customer_id", "created_at")


@dataclass(frozen=True, slots=True)
class FeatureRow:
    """Features and labels for one approved transaction."""

    transaction_id: str
    customer_id: str
    created_at: datetime
    values: tuple[float, ...]
    """Feature values in :data:`~chargeback_prediction.features.spec.FEATURE_NAMES` order."""
    disputed: bool
    """Label: a dispute on this transaction was filed within the observed window."""
    reason_code: str | None
    dispute_category: DisputeCategory | None
    label_mature: bool
    """The full dispute window has elapsed, so ``disputed=False`` is a confirmed negative."""

    def feature(self, name: str) -> float:
        return self.values[FEATURE_NAMES.index(name)]


@dataclass(frozen=True, slots=True)
class FeatureTable:
    rows: tuple[FeatureRow, ...]
    observed_until: datetime
    """Labels reflect disputes filed up to this instant."""

    columns: tuple[str, ...] = FEATURE_NAMES

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[FeatureRow]:
        return iter(self.rows)

    def column(self, name: str) -> list[float]:
        idx = self.columns.index(name)
        return [row.values[idx] for row in self.rows]

    def write_csv(self, path: Path) -> None:
        """Write ids, features and labels to one CSV (labels last, so they are easy to drop)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow((*ID_COLUMNS, *self.columns, *LABEL_COLUMNS))
            for row in self.rows:
                writer.writerow(
                    (
                        row.transaction_id,
                        row.customer_id,
                        row.created_at.isoformat(),
                        *(_fmt(v) for v in row.values),
                        "true" if row.disputed else "false",
                        row.reason_code or "",
                        row.dispute_category or "",
                        "true" if row.label_mature else "false",
                    )
                )


def _fmt(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.6g}"


# --------------------------------------------------------------------- timelines


class _Timeline:
    """Sorted event times with prefix sums, answering "how many / how much in [lo, hi)"."""

    __slots__ = ("_prefix", "_times")

    def __init__(self, events: Sequence[tuple[datetime, float]]) -> None:
        ordered = sorted(events, key=lambda e: e[0])
        self._times = [t for t, _ in ordered]
        self._prefix = [0.0, *accumulate(w for _, w in ordered)]

    def _index(self, at: datetime) -> int:
        return bisect_left(self._times, at)

    def count(self, lo: datetime, hi: datetime) -> int:
        return max(0, self._index(hi) - self._index(lo))

    def total(self, lo: datetime, hi: datetime) -> float:
        i, j = self._index(lo), self._index(hi)
        return self._prefix[j] - self._prefix[i] if j > i else 0.0

    def count_before(self, at: datetime) -> int:
        return self._index(at)

    def last_before(self, at: datetime) -> datetime | None:
        i = self._index(at)
        return self._times[i - 1] if i else None


_EMPTY = _Timeline(())


def _days(delta: timedelta) -> float:
    return max(0.0, delta.total_seconds() / _SECONDS_PER_DAY)


def _days_since(last: datetime | None, at: datetime) -> float:
    return HISTORY_CAP_DAYS if last is None else min(HISTORY_CAP_DAYS, _days(at - last))


@dataclass(frozen=True, slots=True)
class _CustomerHistory:
    attempts: _Timeline
    declines: _Timeline
    approved_usd: _Timeline
    au_updates: _Timeline
    customer_updates: _Timeline
    card_updates: _Timeline
    disputes: _Timeline
    fraud_disputes: _Timeline


@dataclass(frozen=True, slots=True)
class _SubscriptionHistory:
    subscription: Subscription
    approved_renewals: _Timeline
    declines: _Timeline


def _index_customers(ds: SyntheticDataset) -> dict[str, _CustomerHistory]:
    txns: dict[str, list[Transaction]] = defaultdict(list)
    for t in ds.transactions:
        txns[t.customer_id].append(t)
    updates: dict[str, list[tuple[datetime, CardUpdateSource]]] = defaultdict(list)
    for u in ds.card_updates:
        updates[u.customer_id].append((u.updated_at, u.source))
    disputes: dict[str, list[Dispute]] = defaultdict(list)
    for d in ds.disputes:
        disputes[d.customer_id].append(d)

    out: dict[str, _CustomerHistory] = {}
    for c in ds.customers:
        ts = txns.get(c.customer_id, [])
        ups = updates.get(c.customer_id, [])
        dps = disputes.get(c.customer_id, [])
        out[c.customer_id] = _CustomerHistory(
            attempts=_Timeline([(t.created_at, 1.0) for t in ts]),
            declines=_Timeline([(t.created_at, 1.0) for t in ts if not t.approved]),
            approved_usd=_Timeline(
                [(t.created_at, t.amount_cents / 100) for t in ts if t.approved]
            ),
            au_updates=_Timeline(
                [(at, 1.0) for at, src in ups if src is CardUpdateSource.ACCOUNT_UPDATER]
            ),
            customer_updates=_Timeline(
                [(at, 1.0) for at, src in ups if src is CardUpdateSource.CUSTOMER]
            ),
            card_updates=_Timeline([(at, 1.0) for at, _ in ups]),
            # Keyed by filing time: that is when the merchant learns about a dispute.
            disputes=_Timeline([(d.filed_at, 1.0) for d in dps]),
            fraud_disputes=_Timeline(
                [(d.filed_at, 1.0) for d in dps if d.category is DisputeCategory.FRAUD]
            ),
        )
    return out


def _index_subscriptions(ds: SyntheticDataset) -> dict[str, _SubscriptionHistory]:
    txns: dict[str, list[Transaction]] = defaultdict(list)
    for t in ds.transactions:
        if t.subscription_id is not None:
            txns[t.subscription_id].append(t)
    return {
        s.subscription_id: _SubscriptionHistory(
            subscription=s,
            approved_renewals=_Timeline(
                [
                    (t.created_at, 1.0)
                    for t in txns.get(s.subscription_id, [])
                    if t.approved and t.kind is TransactionKind.RENEWAL
                ]
            ),
            declines=_Timeline(
                [(t.created_at, 1.0) for t in txns.get(s.subscription_id, []) if not t.approved]
            ),
        )
        for s in ds.subscriptions
    }


# ---------------------------------------------------------------------- features


def _feature_values(
    txn: Transaction,
    customer: Customer,
    hist: _CustomerHistory,
    sub: _SubscriptionHistory | None,
) -> tuple[float, ...]:
    t = txn.created_at
    ip_missing = txn.ip_country is None

    if sub is None:
        plan_is_annual = sub_age = renewals = sub_declines = after_cancel = 0.0
    else:
        s = sub.subscription
        plan_is_annual = float(s.plan is Plan.ANNUAL)
        sub_age = _days(t - s.started_at)
        renewals = float(sub.approved_renewals.count_before(t))
        sub_declines = float(sub.declines.count_before(t))
        after_cancel = float(s.cancel_requested_at is not None and s.cancel_requested_at < t)

    values = {
        "amount_usd": txn.amount_cents / 100,
        "is_initial": float(txn.kind is TransactionKind.INITIAL),
        "is_renewal": float(txn.kind is TransactionKind.RENEWAL),
        "is_one_off": float(txn.kind is TransactionKind.ONE_OFF),
        "is_mastercard": float(txn.network is Network.MASTERCARD),
        "attempt": float(txn.attempt),
        "account_age_days": _days(t - customer.created_at),
        "email_disposable": float(customer.email_domain_type == "disposable"),
        "email_freemail": float(customer.email_domain_type == "freemail"),
        "ip_missing": float(ip_missing),
        "ip_bin_mismatch": float(not ip_missing and txn.ip_country != txn.bin_country),
        "billing_bin_mismatch": float(customer.country != customer.bin_country),
        "attempts_1h": float(hist.attempts.count(t - timedelta(hours=1), t)),
        "attempts_24h": float(hist.attempts.count(t - _DAY, t)),
        "attempts_7d": float(hist.attempts.count(t - 7 * _DAY, t)),
        "declines_24h": float(hist.declines.count(t - _DAY, t)),
        "declines_30d": float(hist.declines.count(t - 30 * _DAY, t)),
        "approved_usd_24h": hist.approved_usd.total(t - _DAY, t),
        "days_since_prev_attempt": _days_since(hist.attempts.last_before(t), t),
        "au_updates_90d": float(hist.au_updates.count(t - 90 * _DAY, t)),
        "customer_card_updates_90d": float(hist.customer_updates.count(t - 90 * _DAY, t)),
        "days_since_card_update": _days_since(hist.card_updates.last_before(t), t),
        "plan_is_annual": plan_is_annual,
        "subscription_age_days": sub_age,
        "prior_approved_renewals": renewals,
        "prior_subscription_declines": sub_declines,
        "charged_after_cancel_request": after_cancel,
        "prior_disputes": float(hist.disputes.count_before(t)),
        "prior_fraud_disputes": float(hist.fraud_disputes.count_before(t)),
        "days_since_last_dispute": _days_since(hist.disputes.last_before(t), t),
    }
    return tuple(values[name] for name in FEATURE_NAMES)


def _latest_event(ds: SyntheticDataset) -> datetime:
    stamps = [t.created_at for t in ds.transactions[-1:]]
    stamps += [d.filed_at for d in ds.disputes[-1:]]
    stamps += [u.updated_at for u in ds.card_updates[-1:]]
    if not stamps:
        raise ValueError("dataset has no events; cannot infer observed_until")
    return max(stamps)


def build_features(
    dataset: SyntheticDataset, *, observed_until: datetime | None = None
) -> FeatureTable:
    """Build one point-in-time feature row per approved transaction.

    Args:
        dataset: Raw tables. Each table must be sorted by time, as produced by
            :func:`~chargeback_prediction.synthetic.generate` and
            :func:`~chargeback_prediction.synthetic.read_dataset`.
        observed_until: When the data snapshot was taken. Disputes filed later are ignored
            and labels younger than the dispute window are marked immature. Defaults to the
            latest event in the dataset.
    """
    cutoff = observed_until or _latest_event(dataset)
    if cutoff.tzinfo is None:
        raise ValueError("observed_until must be timezone-aware")
    customers = {c.customer_id: c for c in dataset.customers}
    histories = _index_customers(dataset)
    subscriptions = _index_subscriptions(dataset)
    dispute_by_txn = {d.transaction_id: d for d in dataset.disputes if d.filed_at <= cutoff}
    window = timedelta(days=MAX_DISPUTE_WINDOW_DAYS)

    rows: list[FeatureRow] = []
    for txn in dataset.transactions:
        if not txn.approved or txn.created_at > cutoff:
            continue
        sub = subscriptions[txn.subscription_id] if txn.subscription_id else None
        dispute = dispute_by_txn.get(txn.transaction_id)
        rows.append(
            FeatureRow(
                transaction_id=txn.transaction_id,
                customer_id=txn.customer_id,
                created_at=txn.created_at,
                values=_feature_values(
                    txn, customers[txn.customer_id], histories[txn.customer_id], sub
                ),
                disputed=dispute is not None,
                reason_code=dispute.reason_code if dispute else None,
                dispute_category=dispute.category if dispute else None,
                label_mature=txn.created_at + window <= cutoff,
            )
        )
    return FeatureTable(rows=tuple(rows), observed_until=cutoff)
