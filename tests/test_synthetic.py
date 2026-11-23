from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import fields
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from chargeback_prediction.reason_codes import DisputeCategory, all_reason_codes
from chargeback_prediction.synthetic import (
    Customer,
    GeneratorConfig,
    Persona,
    SyntheticDataset,
    TransactionKind,
    generate,
    write_dataset,
)
from chargeback_prediction.synthetic.config import MAX_DISPUTE_WINDOW_DAYS


@pytest.fixture(scope="module")
def config() -> GeneratorConfig:
    return GeneratorConfig(n_customers=3_000, seed=7)


@pytest.fixture(scope="module")
def dataset(config: GeneratorConfig) -> SyntheticDataset:
    return generate(config)


def _customers_by_id(ds: SyntheticDataset) -> dict[str, Customer]:
    return {c.customer_id: c for c in ds.customers}


# --------------------------------------------------------------------- determinism


def test_same_seed_is_reproducible() -> None:
    cfg = GeneratorConfig(n_customers=300, seed=1)
    assert generate(cfg) == generate(cfg)


def test_different_seed_changes_output() -> None:
    a = generate(GeneratorConfig(n_customers=300, seed=1))
    b = generate(GeneratorConfig(n_customers=300, seed=2))
    assert a.transactions != b.transactions


# ------------------------------------------------------------------------ integrity


def test_ids_are_unique(dataset: SyntheticDataset) -> None:
    for ids in (
        [c.customer_id for c in dataset.customers],
        [s.subscription_id for s in dataset.subscriptions],
        [t.transaction_id for t in dataset.transactions],
        [d.dispute_id for d in dataset.disputes],
    ):
        assert len(ids) == len(set(ids))


def test_referential_integrity(dataset: SyntheticDataset) -> None:
    customers = _customers_by_id(dataset)
    subs = {s.subscription_id: s for s in dataset.subscriptions}
    txns = {t.transaction_id: t for t in dataset.transactions}

    assert len(customers) == 3_000
    for sub in dataset.subscriptions:
        assert sub.customer_id in customers
    for txn in dataset.transactions:
        assert txn.customer_id in customers
        if txn.subscription_id is not None:
            assert subs[txn.subscription_id].customer_id == txn.customer_id
    for upd in dataset.card_updates:
        assert upd.customer_id in customers
    for dispute in dataset.disputes:
        txn = txns[dispute.transaction_id]
        assert dispute.customer_id == txn.customer_id
        assert dispute.amount_cents == txn.amount_cents


def test_events_fall_inside_observation_window(
    dataset: SyntheticDataset, config: GeneratorConfig
) -> None:
    end = config.start + timedelta(days=config.days)
    for txn in dataset.transactions:
        assert config.start <= txn.created_at <= end
    for upd in dataset.card_updates:
        assert config.start <= upd.updated_at <= end
    for dispute in dataset.disputes:
        assert dispute.filed_at <= end


def test_tables_are_sorted_by_time(dataset: SyntheticDataset) -> None:
    times = [t.created_at for t in dataset.transactions]
    assert times == sorted(times)
    filed = [d.filed_at for d in dataset.disputes]
    assert filed == sorted(filed)


def test_transactions_belong_to_existing_accounts(dataset: SyntheticDataset) -> None:
    customers = _customers_by_id(dataset)
    for txn in dataset.transactions:
        assert txn.created_at >= customers[txn.customer_id].created_at


# ----------------------------------------------------------------- authorizations


def test_decline_codes_only_on_declines(dataset: SyntheticDataset) -> None:
    for txn in dataset.transactions:
        assert (txn.decline_code is None) == txn.approved


def test_renewals_are_merchant_initiated(dataset: SyntheticDataset) -> None:
    for txn in dataset.transactions:
        if txn.kind is TransactionKind.RENEWAL:
            assert txn.ip_country is None
            assert txn.subscription_id is not None
        else:
            assert txn.ip_country is not None
            assert txn.attempt == 1


def test_retries_follow_soft_declines_and_respect_limit(
    dataset: SyntheticDataset, config: GeneratorConfig
) -> None:
    by_sub: dict[str, list[int]] = defaultdict(list)
    for txn in dataset.transactions:
        if txn.kind is TransactionKind.RENEWAL:
            assert 1 <= txn.attempt <= config.max_retry_attempts
            by_sub[txn.subscription_id or ""].append(txn.attempt)
    retried = [attempts for attempts in by_sub.values() if max(attempts) > 1]
    assert retried, "expected some renewals to be retried after soft declines"


def test_no_billing_after_cancellation_except_billing_bug(dataset: SyntheticDataset) -> None:
    customers = _customers_by_id(dataset)
    subs = {s.subscription_id: s for s in dataset.subscriptions}
    late_charges = [
        t
        for t in dataset.transactions
        if t.subscription_id is not None
        and (cancel := subs[t.subscription_id].cancel_requested_at) is not None
        and t.created_at > cancel
    ]
    assert late_charges, "expected the forgetful billing-bug scenario to appear"
    assert {customers[t.customer_id].persona for t in late_charges} == {Persona.FORGETFUL}


# ----------------------------------------------------------------------- disputes


def test_disputes_use_catalog_codes_for_the_card_network(dataset: SyntheticDataset) -> None:
    catalog = {rc.key: rc for rc in all_reason_codes()}
    for dispute in dataset.disputes:
        rc = catalog[dispute.reason_code]
        assert rc.network is dispute.network
        assert rc.category is dispute.category


def test_every_roadmap_reason_code_appears(dataset: SyntheticDataset) -> None:
    assert {d.reason_code for d in dataset.disputes} == {rc.key for rc in all_reason_codes()}


def test_disputes_only_on_approved_transactions_within_window(dataset: SyntheticDataset) -> None:
    txns = {t.transaction_id: t for t in dataset.transactions}
    for dispute in dataset.disputes:
        txn = txns[dispute.transaction_id]
        assert txn.approved
        assert (
            timedelta(0)
            < dispute.filed_at - txn.created_at
            <= timedelta(days=MAX_DISPUTE_WINDOW_DAYS)
        )


def test_at_most_one_dispute_per_transaction(dataset: SyntheticDataset) -> None:
    ids = [d.transaction_id for d in dataset.disputes]
    assert len(ids) == len(set(ids))


def test_dispute_rate_is_realistic(dataset: SyntheticDataset) -> None:
    # High-risk subscription merchants run ~0.5-3% disputes per approved transaction.
    assert 0.005 < dataset.summary().dispute_rate < 0.03


def test_friendly_fraud_label_matches_persona(dataset: SyntheticDataset) -> None:
    customers = _customers_by_id(dataset)
    for dispute in dataset.disputes:
        persona = customers[dispute.customer_id].persona
        if persona is Persona.FRAUDSTER:
            assert not dispute.is_friendly_fraud
            assert dispute.category is DisputeCategory.FRAUD
        elif persona is Persona.SERIAL_DISPUTER:
            assert dispute.is_friendly_fraud
        elif persona is Persona.GOOD:
            assert not dispute.is_friendly_fraud
    summary = dataset.summary()
    assert 0 < summary.friendly_fraud_disputes < summary.n_disputes


# ------------------------------------------------------------------ fraud signals


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values)


def test_fraudsters_show_geo_bin_mismatch(dataset: SyntheticDataset) -> None:
    customers = _customers_by_id(dataset)
    mismatch: dict[Persona, list[bool]] = defaultdict(list)
    for txn in dataset.transactions:
        if txn.ip_country is not None:
            persona = customers[txn.customer_id].persona
            mismatch[persona].append(txn.ip_country != txn.bin_country)
    assert _rate(mismatch[Persona.FRAUDSTER]) > 0.5
    assert _rate(mismatch[Persona.GOOD]) < 0.1


def test_fraudsters_transact_on_new_accounts(dataset: SyntheticDataset) -> None:
    customers = _customers_by_id(dataset)
    age_days: dict[Persona, list[float]] = defaultdict(list)
    for txn in dataset.transactions:
        if txn.kind is TransactionKind.ONE_OFF:
            c = customers[txn.customer_id]
            age_days[c.persona].append((txn.created_at - c.created_at).total_seconds() / 86_400)
    fraud_median = sorted(age_days[Persona.FRAUDSTER])[len(age_days[Persona.FRAUDSTER]) // 2]
    good_median = sorted(age_days[Persona.GOOD])[len(age_days[Persona.GOOD]) // 2]
    assert fraud_median < 4 < good_median


def test_fraudsters_card_test_before_buying(dataset: SyntheticDataset) -> None:
    customers = _customers_by_id(dataset)
    # Card testing = small declined authorizations within an hour of account creation.
    small_declines = {
        t.customer_id
        for t in dataset.transactions
        if not t.approved
        and t.amount_cents <= 500
        and t.kind is TransactionKind.ONE_OFF
        and t.created_at - customers[t.customer_id].created_at <= timedelta(hours=1)
    }
    fraudsters = {c.customer_id for c in dataset.customers if c.persona is Persona.FRAUDSTER}
    assert len(small_declines & fraudsters) / len(fraudsters) > 0.5
    assert all(customers[cid].persona is Persona.FRAUDSTER for cid in small_declines)


def test_forgetful_customers_get_account_updater_hits(dataset: SyntheticDataset) -> None:
    updated = {u.customer_id for u in dataset.card_updates}
    by_persona: dict[Persona, list[bool]] = defaultdict(list)
    for c in dataset.customers:
        by_persona[c.persona].append(c.customer_id in updated)
    assert _rate(by_persona[Persona.FORGETFUL]) > 2 * _rate(by_persona[Persona.GOOD])


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_customers": 0}, "n_customers"),
        ({"days": 10}, "days"),
        ({"start": datetime(2025, 1, 1)}, "timezone-aware"),
        ({"max_retry_attempts": 0}, "max_retry_attempts"),
        ({"retry_spacing_days": 0}, "retry_spacing_days"),
        ({"persona_mix": {Persona.GOOD: 1.0}}, "every Persona"),
        ({"persona_mix": dict.fromkeys(Persona, 0.5)}, "sum to 1"),
        (
            {
                "persona_mix": {
                    Persona.GOOD: 1.5,
                    Persona.FORGETFUL: -0.5,
                    Persona.SERIAL_DISPUTER: 0.0,
                    Persona.FRAUDSTER: 0.0,
                }
            },
            "non-negative",
        ),
    ],
)
def test_config_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GeneratorConfig(**kwargs)  # type: ignore[arg-type]


def test_persona_mix_is_honoured() -> None:
    only_good = dict.fromkeys(Persona, 0.0) | {Persona.GOOD: 1.0}
    ds = generate(GeneratorConfig(n_customers=200, persona_mix=only_good))
    assert {c.persona for c in ds.customers} == {Persona.GOOD}
    assert not any(d.is_friendly_fraud for d in ds.disputes)


# ------------------------------------------------------------------------------- io


def test_write_dataset_round_trips_row_counts(tmp_path: Path) -> None:
    ds = generate(GeneratorConfig(n_customers=200, seed=3, start=datetime(2024, 6, 1, tzinfo=UTC)))
    paths = write_dataset(ds, tmp_path / "out")
    for table, path in paths.items():
        with path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        records = getattr(ds, table)
        assert len(rows) == len(records)
        if records:
            assert list(rows[0]) == [f.name for f in fields(records[0])]

    with paths["transactions"].open(encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    assert row["approved"] in {"true", "false"}
    assert datetime.fromisoformat(row["created_at"]).tzinfo is not None
