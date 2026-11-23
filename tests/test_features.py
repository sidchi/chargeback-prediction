from __future__ import annotations

import csv
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from chargeback_prediction.features import (
    FEATURE_NAMES,
    FEATURES,
    HISTORY_CAP_DAYS,
    LABEL_COLUMNS,
    FeatureKind,
    FeatureTable,
    build_features,
    render_feature_table,
)
from chargeback_prediction.reason_codes import DisputeCategory, Network
from chargeback_prediction.synthetic import (
    CardUpdate,
    CardUpdateSource,
    Customer,
    Dispute,
    GeneratorConfig,
    Persona,
    Plan,
    Subscription,
    SyntheticDataset,
    Transaction,
    TransactionKind,
    generate,
)

T0 = datetime(2025, 3, 1, 12, 0, tzinfo=UTC)
DOCS = Path(__file__).resolve().parents[1] / "docs" / "features.md"


# ----------------------------------------------------------------- tiny fixtures


def _customer(**overrides: object) -> Customer:
    base = Customer(
        customer_id="cus_1",
        created_at=T0 - timedelta(days=100),
        country="US",
        email_domain_type="corporate",
        card_network=Network.VISA,
        card_bin="411111",
        bin_country="US",
        persona=Persona.GOOD,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


_txn_counter = 0


def _txn(at: datetime, **overrides: object) -> Transaction:
    global _txn_counter
    _txn_counter += 1
    base = Transaction(
        transaction_id=f"txn_{_txn_counter:04d}",
        customer_id="cus_1",
        subscription_id=None,
        created_at=at,
        kind=TransactionKind.ONE_OFF,
        amount_cents=2500,
        currency="USD",
        network=Network.VISA,
        bin_country="US",
        ip_country="US",
        attempt=1,
        approved=True,
        decline_code=None,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def _dispute(txn: Transaction, filed_at: datetime, *, fraud: bool = True) -> Dispute:
    return Dispute(
        dispute_id=f"dp_{txn.transaction_id}",
        transaction_id=txn.transaction_id,
        customer_id=txn.customer_id,
        network=Network.VISA,
        reason_code="visa:10.4" if fraud else "visa:13.1",
        category=DisputeCategory.FRAUD if fraud else DisputeCategory.CONSUMER,
        filed_at=filed_at,
        amount_cents=txn.amount_cents,
        is_friendly_fraud=False,
    )


def _dataset(
    transactions: list[Transaction],
    *,
    customers: list[Customer] | None = None,
    subscriptions: list[Subscription] | None = None,
    card_updates: list[CardUpdate] | None = None,
    disputes: list[Dispute] | None = None,
) -> SyntheticDataset:
    return SyntheticDataset(
        customers=tuple(customers or [_customer()]),
        subscriptions=tuple(subscriptions or []),
        transactions=tuple(sorted(transactions, key=lambda t: t.created_at)),
        card_updates=tuple(card_updates or []),
        disputes=tuple(disputes or []),
    )


def _row(table: FeatureTable, txn: Transaction) -> dict[str, float]:
    (row,) = (r for r in table if r.transaction_id == txn.transaction_id)
    return dict(zip(FEATURE_NAMES, row.values, strict=True))


# ------------------------------------------------------------------- catalog


def test_catalog_names_are_unique_snake_case() -> None:
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        assert name.isidentifier()
        assert name == name.lower()


def test_ground_truth_labels_are_not_features() -> None:
    forbidden = {"persona", "is_friendly_fraud", *LABEL_COLUMNS}
    assert not forbidden & set(FEATURE_NAMES)


def test_committed_feature_doc_is_up_to_date() -> None:
    assert DOCS.read_text(encoding="utf-8") == render_feature_table(), (
        "docs/features.md is stale: run `uv run cbp features --doc > docs/features.md`"
    )


# ------------------------------------------------------------- point-in-time


def test_only_approved_transactions_get_rows() -> None:
    ok = _txn(T0)
    declined = _txn(T0 + timedelta(minutes=5), approved=False, decline_code="51")
    table = build_features(_dataset([ok, declined]))
    assert [r.transaction_id for r in table] == [ok.transaction_id]


def test_velocity_windows_exclude_the_charge_itself_and_the_future() -> None:
    events = [
        _txn(T0 - timedelta(days=8)),  # outside 7d
        _txn(T0 - timedelta(days=3), approved=False, decline_code="05"),
        _txn(T0 - timedelta(hours=5), amount_cents=1000),
        _txn(T0 - timedelta(minutes=30), approved=False, decline_code="14"),
    ]
    target = _txn(T0, amount_cents=9900)
    simultaneous = _txn(T0, amount_cents=500)  # same timestamp: not "before"
    later = _txn(T0 + timedelta(minutes=1))
    f = _row(build_features(_dataset([*events, target, simultaneous, later])), target)

    assert f["amount_usd"] == pytest.approx(99.0)
    assert f["attempts_1h"] == 1
    assert f["attempts_24h"] == 2
    assert f["attempts_7d"] == 3
    assert f["declines_24h"] == 1
    assert f["declines_30d"] == 2
    assert f["approved_usd_24h"] == pytest.approx(10.0)
    assert f["days_since_prev_attempt"] == pytest.approx(30 / (24 * 60))


def test_first_ever_attempt_uses_history_cap() -> None:
    target = _txn(T0)
    f = _row(build_features(_dataset([target])), target)
    assert f["days_since_prev_attempt"] == HISTORY_CAP_DAYS
    assert f["days_since_card_update"] == HISTORY_CAP_DAYS
    assert f["days_since_last_dispute"] == HISTORY_CAP_DAYS
    assert f["attempts_7d"] == 0


def test_prior_disputes_count_by_filing_date_not_charge_date() -> None:
    early = _txn(T0 - timedelta(days=60))
    middle = _txn(T0 - timedelta(days=40))
    target = _txn(T0)
    disputes = [
        _dispute(early, T0 - timedelta(days=10), fraud=True),  # known at T0
        _dispute(middle, T0 + timedelta(days=5), fraud=False),  # filed after T0: invisible
    ]
    f = _row(build_features(_dataset([early, middle, target], disputes=disputes)), target)
    assert f["prior_disputes"] == 1
    assert f["prior_fraud_disputes"] == 1
    assert f["days_since_last_dispute"] == pytest.approx(10)


def test_card_on_file_features_split_by_source_and_window() -> None:
    target = _txn(T0)
    updates = [
        CardUpdate("cus_1", T0 - timedelta(days=200), CardUpdateSource.ACCOUNT_UPDATER),
        CardUpdate("cus_1", T0 - timedelta(days=20), CardUpdateSource.ACCOUNT_UPDATER),
        CardUpdate("cus_1", T0 - timedelta(days=2), CardUpdateSource.CUSTOMER),
        CardUpdate("cus_1", T0 + timedelta(days=1), CardUpdateSource.CUSTOMER),
    ]
    f = _row(build_features(_dataset([target], card_updates=updates)), target)
    assert f["au_updates_90d"] == 1
    assert f["customer_card_updates_90d"] == 1
    assert f["days_since_card_update"] == pytest.approx(2)


def test_geo_and_account_features() -> None:
    customer = _customer(country="GB", email_domain_type="disposable")
    session = _txn(T0, ip_country="BR")
    renewal = _txn(T0 + timedelta(days=1), ip_country=None, kind=TransactionKind.RENEWAL)
    table = build_features(_dataset([session, renewal], customers=[customer]))

    f = _row(table, session)
    assert f["ip_bin_mismatch"] == 1
    assert f["ip_missing"] == 0
    assert f["billing_bin_mismatch"] == 1
    assert f["email_disposable"] == 1
    assert f["email_freemail"] == 0
    assert f["account_age_days"] == pytest.approx(100)
    assert f["is_one_off"] == 1

    g = _row(table, renewal)
    assert g["ip_missing"] == 1
    assert g["ip_bin_mismatch"] == 0, "a missing IP is not a mismatch"


def test_subscription_and_retry_history() -> None:
    started = T0 - timedelta(days=365)
    sub = Subscription(
        "sub_1", "cus_1", Plan.ANNUAL, 9999, started, cancel_requested_at=T0 - timedelta(days=5)
    )
    on_sub = {"subscription_id": "sub_1", "amount_cents": 9999, "ip_country": None}
    initial = _txn(started, kind=TransactionKind.INITIAL, **{**on_sub, "ip_country": "US"})
    first_try = _txn(
        T0 - timedelta(days=6),
        kind=TransactionKind.RENEWAL,
        approved=False,
        decline_code="51",
        **on_sub,
    )
    retry = _txn(T0, kind=TransactionKind.RENEWAL, attempt=2, **on_sub)
    one_off = _txn(T0 + timedelta(days=1))
    table = build_features(_dataset([initial, first_try, retry, one_off], subscriptions=[sub]))

    f = _row(table, retry)
    assert f["is_renewal"] == 1
    assert f["attempt"] == 2
    assert f["plan_is_annual"] == 1
    assert f["subscription_age_days"] == pytest.approx(365)
    assert f["prior_approved_renewals"] == 0
    assert f["prior_subscription_declines"] == 1
    assert f["charged_after_cancel_request"] == 1

    assert _row(table, initial)["charged_after_cancel_request"] == 0
    g = _row(table, one_off)
    assert (g["plan_is_annual"], g["subscription_age_days"], g["prior_subscription_declines"]) == (
        0,
        0,
        0,
    )


# -------------------------------------------------------------------- labels


def test_labels_respect_snapshot_and_maturity() -> None:
    old = _txn(T0)
    young = _txn(T0 + timedelta(days=100))
    filed_late = _txn(T0 + timedelta(days=10))
    disputes = [
        _dispute(old, T0 + timedelta(days=30)),
        _dispute(filed_late, T0 + timedelta(days=150)),  # after the snapshot
    ]
    cutoff = T0 + timedelta(days=130)
    table = build_features(
        _dataset([old, filed_late, young], disputes=disputes), observed_until=cutoff
    )
    by_id = {r.transaction_id: r for r in table}

    assert by_id[old.transaction_id].disputed
    assert by_id[old.transaction_id].reason_code == "visa:10.4"
    assert by_id[old.transaction_id].dispute_category is DisputeCategory.FRAUD
    assert by_id[old.transaction_id].label_mature
    assert not by_id[filed_late.transaction_id].disputed
    assert by_id[filed_late.transaction_id].label_mature
    assert not by_id[young.transaction_id].label_mature
    assert table.observed_until == cutoff


def test_observed_until_defaults_to_latest_event() -> None:
    txn = _txn(T0)
    table = build_features(_dataset([txn], disputes=[_dispute(txn, T0 + timedelta(days=9))]))
    assert table.observed_until == T0 + timedelta(days=9)


def test_observed_until_validation() -> None:
    with pytest.raises(ValueError, match="no events"):
        build_features(_dataset([]))
    with pytest.raises(ValueError, match="timezone-aware"):
        build_features(_dataset([_txn(T0)]), observed_until=datetime(2025, 6, 1))


# --------------------------------------------------- on the generated dataset


@pytest.fixture(scope="module")
def generated() -> tuple[SyntheticDataset, FeatureTable]:
    ds = generate(GeneratorConfig(n_customers=3_000, seed=7))
    return ds, build_features(ds)


def test_one_row_per_approved_transaction(
    generated: tuple[SyntheticDataset, FeatureTable],
) -> None:
    ds, table = generated
    assert len(table) == sum(t.approved for t in ds.transactions)
    assert sum(r.disputed for r in table) == len(ds.disputes)


def test_feature_values_respect_their_kind(
    generated: tuple[SyntheticDataset, FeatureTable],
) -> None:
    _, table = generated
    for spec in FEATURES:
        values = table.column(spec.name)
        assert min(values) >= 0, spec.name
        if spec.kind is FeatureKind.FLAG:
            assert set(values) <= {0.0, 1.0}, spec.name
        if spec.kind is FeatureKind.COUNT:
            assert all(v.is_integer() for v in values), spec.name


def _dispute_rate(rows: list[bool]) -> float:
    return sum(rows) / len(rows)


@pytest.mark.parametrize(
    "feature",
    ["ip_bin_mismatch", "declines_24h", "charged_after_cancel_request", "au_updates_90d"],
)
def test_signal_features_carry_lift(
    generated: tuple[SyntheticDataset, FeatureTable], feature: str
) -> None:
    """The patterns the generator plants must survive the pipeline as real lift."""
    _, table = generated
    mature = [r for r in table if r.label_mature]
    base = _dispute_rate([r.disputed for r in mature])
    flagged = _dispute_rate([r.disputed for r in mature if r.feature(feature) > 0])
    assert flagged > 2 * base


# ---------------------------------------------------------------------- CSV


def test_write_csv(tmp_path: Path) -> None:
    txn = _txn(T0, amount_cents=1234)
    table = build_features(_dataset([txn], disputes=[_dispute(txn, T0 + timedelta(days=200))]))
    out = tmp_path / "nested" / "features.csv"
    table.write_csv(out)

    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    header = ["transaction_id", "customer_id", "created_at", *FEATURE_NAMES, *LABEL_COLUMNS]
    assert list(rows[0]) == header
    assert rows[0]["amount_usd"] == "12.34"
    assert rows[0]["attempts_24h"] == "0"
    assert rows[0]["disputed"] == "true"
    assert rows[0]["reason_code"] == "visa:10.4"
    assert rows[0]["dispute_category"] == "fraud"
    assert rows[0]["label_mature"] == "true"
