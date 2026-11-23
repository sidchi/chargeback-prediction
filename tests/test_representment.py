from __future__ import annotations

import json
from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from chargeback_prediction.cli import main
from chargeback_prediction.reason_codes import (
    DisputeCategory,
    Network,
    ReasonCode,
    all_reason_codes,
    lookup,
)
from chargeback_prediction.representment import (
    CE3_WINDOW_DAYS,
    EvidenceIndex,
    Exhibit,
    ExhibitKind,
    ExhibitStatus,
    Readiness,
    assemble,
    assemble_all,
    format_money,
    readiness,
    readiness_table,
    render_readiness,
    requirements_for,
)
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
    write_dataset,
)

T0 = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
CUSTOMER = Customer(
    customer_id="cus_1",
    created_at=T0 - timedelta(days=500),
    country="US",
    email_domain_type="corporate",
    card_network=Network.VISA,
    card_bin="400000",
    bin_country="US",
    persona=Persona.GOOD,
)
SUB = Subscription(
    subscription_id="sub_1",
    customer_id="cus_1",
    plan=Plan.MONTHLY,
    price_cents=1499,
    started_at=T0 - timedelta(days=90),
    cancel_requested_at=None,
)


def _txn(txn_id: str, days_before: float, **overrides: object) -> Transaction:
    base = Transaction(
        transaction_id=txn_id,
        customer_id="cus_1",
        subscription_id=None,
        created_at=T0 - timedelta(days=days_before),
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


def _renewal(txn_id: str, days_before: float) -> Transaction:
    return _txn(
        txn_id,
        days_before,
        subscription_id="sub_1",
        kind=TransactionKind.RENEWAL,
        amount_cents=1499,
        ip_country=None,
    )


def _dispute(txn: Transaction, code: str, *, days_after: float = 20, dp: str = "dp_1") -> Dispute:
    network, _, num = code.partition(":")
    rc = lookup(network, num)
    return Dispute(
        dispute_id=dp,
        transaction_id=txn.transaction_id,
        customer_id=txn.customer_id,
        network=rc.network,
        reason_code=code,
        category=rc.category,
        filed_at=txn.created_at + timedelta(days=days_after),
        amount_cents=txn.amount_cents,
        is_friendly_fraud=True,
    )


def _dataset(
    txns: list[Transaction],
    disputes: list[Dispute],
    *,
    sub: Subscription | None = SUB,
    updates: list[CardUpdate] | None = None,
    customer: Customer = CUSTOMER,
) -> SyntheticDataset:
    return SyntheticDataset(
        customers=(customer,),
        subscriptions=(sub,) if sub else (),
        transactions=tuple(sorted(txns, key=lambda t: t.created_at)),
        card_updates=tuple(updates or ()),
        disputes=tuple(disputes),
    )


# --------------------------------------------------------------------------- requirements


@pytest.mark.parametrize("reason", all_reason_codes(), ids=lambda rc: rc.key)
def test_every_catalog_code_has_requirements_starting_with_the_transaction(
    reason: ReasonCode,
) -> None:
    reqs = requirements_for(reason)
    assert reqs[0].kind is ExhibitKind.TRANSACTION_RECORD
    assert reqs[0].required
    assert len({r.kind for r in reqs}) == len(reqs), "duplicate exhibit kinds"
    assert all(r.why for r in reqs)


def test_unknown_code_falls_back_to_its_category() -> None:
    novel = ReasonCode(Network.VISA, "99.9", "Hypothetical", DisputeCategory.PROCESSING, False)
    kinds = [r.kind for r in requirements_for(novel)]
    assert kinds == [ExhibitKind.TRANSACTION_RECORD, ExhibitKind.ACCOUNT_PROFILE]


def test_external_kinds_name_their_source_and_internal_kinds_do_not() -> None:
    for kind in ExhibitKind:
        assert kind.label
        assert (kind.external_source is not None) == kind.is_external
    assert ExhibitKind.AVS_CVV.is_external
    assert not ExhibitKind.CE3_PRIOR_TRANSACTIONS.is_external


# --------------------------------------------------------------------------- visa 10.4 / CE 3.0


def test_ce3_counts_only_undisputed_same_card_charges_inside_the_window() -> None:
    lo, hi = CE3_WINDOW_DAYS
    disputed = _txn("txn_d", 0)
    in_window = [_txn("txn_a", lo + 10), _txn("txn_b", hi - 10)]
    too_recent = _txn("txn_recent", lo - 5)
    too_old = _txn("txn_old", hi + 5)
    declined = _txn("txn_declined", lo + 20, approved=False, decline_code="05")
    previously_disputed = _txn("txn_prev", lo + 30)
    ds = _dataset(
        [disputed, *in_window, too_recent, too_old, declined, previously_disputed],
        [
            _dispute(disputed, "visa:10.4"),
            _dispute(previously_disputed, "visa:10.4", dp="dp_0"),
        ],
    )
    packet = assemble(EvidenceIndex(ds), "dp_1")
    ce3 = packet.exhibit(ExhibitKind.CE3_PRIOR_TRANSACTIONS)
    assert ce3.status is ExhibitStatus.FOUND
    assert set(ce3.transaction_ids) == {"txn_a", "txn_b"}
    # Usage logs (IP / device ID) are required for CE 3.0 and live outside the dataset.
    assert packet.readiness is Readiness.NEEDS_ATTACHMENTS
    assert packet.exhibit(ExhibitKind.USAGE_LOGS).status is ExhibitStatus.ATTACH
    assert "txn_a" in packet.render_markdown()
    # The other dispute is internal context, never part of the submission.
    assert any("other dispute" in n for n in packet.internal_notes)
    assert "other dispute" not in packet.rebuttal


def test_ce3_is_weak_with_one_qualifying_charge() -> None:
    disputed = _txn("txn_d", 0)
    ds = _dataset([disputed, _txn("txn_a", 200)], [_dispute(disputed, "visa:10.4")])
    packet = assemble(EvidenceIndex(ds), "dp_1")
    assert packet.exhibit(ExhibitKind.CE3_PRIOR_TRANSACTIONS).status is ExhibitStatus.WEAK
    assert packet.readiness is Readiness.WEAK


def test_customer_card_replacement_breaks_same_card_history_but_updater_does_not() -> None:
    disputed = _txn("txn_d", 0)
    prior = [_txn("txn_a", 200), _txn("txn_b", 150)]
    updater = CardUpdate("cus_1", T0 - timedelta(days=180), CardUpdateSource.ACCOUNT_UPDATER)
    replaced = CardUpdate("cus_1", T0 - timedelta(days=160), CardUpdateSource.CUSTOMER)

    kept = _dataset([disputed, *prior], [_dispute(disputed, "visa:10.4")], updates=[updater])
    packet = assemble(EvidenceIndex(kept), "dp_1")
    assert packet.exhibit(ExhibitKind.CE3_PRIOR_TRANSACTIONS).status is ExhibitStatus.FOUND
    assert packet.exhibit(ExhibitKind.CARD_CONTINUITY).status is ExhibitStatus.FOUND

    broken = _dataset(
        [disputed, *prior], [_dispute(disputed, "visa:10.4")], updates=[updater, replaced]
    )
    packet = assemble(EvidenceIndex(broken), "dp_1")
    ce3 = packet.exhibit(ExhibitKind.CE3_PRIOR_TRANSACTIONS)
    assert ce3.transaction_ids == ("txn_b",)
    assert ce3.status is ExhibitStatus.WEAK
    assert any("replaced their card" in n for n in packet.internal_notes)


def test_ip_mismatch_stays_out_of_the_rebuttal() -> None:
    disputed = _txn("txn_d", 0, ip_country="RO")
    ds = _dataset([disputed], [_dispute(disputed, "visa:10.4")])
    packet = assemble(EvidenceIndex(ds), "dp_1")
    assert "RO" not in packet.rebuttal
    assert any("RO" in n for n in packet.internal_notes)


# --------------------------------------------------------------------------- mastercard 4837


def test_4837_is_ready_from_data_alone_when_card_history_exists() -> None:
    disputed = _txn("txn_d", 0, network=Network.MASTERCARD)
    ds = _dataset([disputed, _txn("txn_a", 40)], [_dispute(disputed, "mastercard:4837")])
    packet = assemble(EvidenceIndex(ds), "dp_1")
    assert packet.exhibit(ExhibitKind.PRIOR_UNDISPUTED_HISTORY).transaction_ids == ("txn_a",)
    assert packet.readiness is Readiness.READY
    assert "made by the cardholder" in packet.rebuttal


def test_4837_with_no_history_is_weak_and_rebuttal_says_so() -> None:
    disputed = _txn("txn_d", 0, network=Network.MASTERCARD)
    fresh = replace(CUSTOMER, created_at=T0 - timedelta(days=1))
    ds = _dataset([disputed], [_dispute(disputed, "mastercard:4837")], customer=fresh)
    packet = assemble(EvidenceIndex(ds), "dp_1")
    assert packet.readiness is Readiness.WEAK
    assert packet.exhibit(ExhibitKind.POST_CHARGE_ACTIVITY).status is ExhibitStatus.WEAK


# --------------------------------------------------------------------------- visa 13.2


def test_13_2_with_paid_renewals_and_no_cancellation_needs_only_terms() -> None:
    renewals = [_renewal(f"txn_r{i}", 30 * i) for i in (3, 2, 1)]
    disputed = _renewal("txn_d", 0)
    later = _renewal("txn_later", -15)
    ds = _dataset([*renewals, disputed, later], [_dispute(disputed, "visa:13.2")])
    packet = assemble(EvidenceIndex(ds), "dp_1")

    assert packet.exhibit(ExhibitKind.CANCELLATION_CHECK).status is ExhibitStatus.FOUND
    billing = packet.exhibit(ExhibitKind.BILLING_HISTORY)
    assert billing.transaction_ids == ("txn_r3", "txn_r2", "txn_r1")
    assert "USD 44.97" in billing.summary
    assert packet.readiness is Readiness.NEEDS_ATTACHMENTS
    assert [e.kind for e in packet.to_attach] == [
        ExhibitKind.TERMS_ACCEPTANCE,
        ExhibitKind.RENEWAL_REMINDER,
        ExhibitKind.USAGE_LOGS,
    ]
    assert "valid under the agreed terms" in packet.rebuttal


def test_13_2_charged_after_cancellation_is_not_represented() -> None:
    disputed = _renewal("txn_d", 0)
    cancelled = replace(SUB, cancel_requested_at=T0 - timedelta(days=5))
    ds = _dataset(
        [_renewal("txn_r1", 30), disputed], [_dispute(disputed, "visa:13.2")], sub=cancelled
    )
    packet = assemble(EvidenceIndex(ds), "dp_1")
    assert packet.exhibit(ExhibitKind.CANCELLATION_CHECK).status is ExhibitStatus.WEAK
    assert packet.readiness is Readiness.DO_NOT_REPRESENT
    assert packet.rebuttal.startswith("Not drafted")
    assert "Adverse facts" in packet.render_markdown()


def test_cancellation_after_the_charge_supports_the_merchant() -> None:
    disputed = _renewal("txn_d", 0)
    cancelled = replace(SUB, cancel_requested_at=T0 + timedelta(days=3))
    ds = _dataset([disputed], [_dispute(disputed, "visa:13.2")], sub=cancelled)
    check = assemble(EvidenceIndex(ds), "dp_1").exhibit(ExhibitKind.CANCELLATION_CHECK)
    assert check.status is ExhibitStatus.FOUND
    assert "after it" in check.summary


def test_assembly_is_point_in_time() -> None:
    """A cancellation logged after ``as_of`` is invisible, as it was to the analyst then."""
    disputed = _renewal("txn_d", 0)
    cancelled = replace(SUB, cancel_requested_at=T0 + timedelta(days=60))
    ds = _dataset([disputed], [_dispute(disputed, "visa:13.2")], sub=cancelled)
    index = EvidenceIndex(ds)
    now = assemble(index, "dp_1")
    assert "never" in dict(now.exhibit(ExhibitKind.CANCELLATION_CHECK).facts).values()
    later = assemble(index, "dp_1", as_of=T0 + timedelta(days=90))
    assert "after it" in later.exhibit(ExhibitKind.CANCELLATION_CHECK).summary


# --------------------------------------------------------------------------- one-offs, partials


def test_consumer_dispute_on_a_one_off_purchase_is_missing_subscription_evidence() -> None:
    disputed = _txn("txn_d", 0)
    ds = _dataset([disputed], [_dispute(disputed, "visa:13.1")], sub=None)
    packet = assemble(EvidenceIndex(ds), "dp_1")
    assert packet.exhibit(ExhibitKind.SUBSCRIPTION_RECORD).status is ExhibitStatus.MISSING
    assert packet.readiness is Readiness.WEAK


def test_partial_and_overclaimed_disputes_are_flagged() -> None:
    disputed = _txn("txn_d", 0, network=Network.MASTERCARD)
    partial = replace(_dispute(disputed, "mastercard:4837"), amount_cents=1000)
    packet = assemble(EvidenceIndex(_dataset([disputed], [partial])), "dp_1")
    assert any("Partial dispute: USD 10.00 of USD 25.00" in n for n in packet.internal_notes)

    over = replace(partial, amount_cents=9999)
    packet = assemble(EvidenceIndex(_dataset([disputed], [over])), "dp_1")
    assert packet.readiness is Readiness.DO_NOT_REPRESENT


# --------------------------------------------------------------------------- errors


def test_errors() -> None:
    disputed = _txn("txn_d", 0)
    good = _dispute(disputed, "visa:10.4")
    index = EvidenceIndex(_dataset([disputed], [good]))
    with pytest.raises(KeyError, match="unknown dispute id"):
        assemble(index, "dp_nope")
    with pytest.raises(ValueError, match="response_days"):
        assemble(index, good, response_days=0)
    with pytest.raises(ValueError, match="before the disputed charge"):
        assemble(index, good, as_of=T0 - timedelta(days=1))
    with pytest.raises(ValueError, match="unknown reason code"):
        assemble(index, replace(good, reason_code="visa:99.9"))
    with pytest.raises(KeyError, match="unknown record"):
        assemble(index, replace(good, transaction_id="txn_ghost"))


# --------------------------------------------------------------------------- packet


def test_readiness_precedence() -> None:
    def ex(status: ExhibitStatus, required: bool = True) -> Exhibit:
        return Exhibit(ExhibitKind.TRANSACTION_RECORD, status, required, "why", "summary")

    assert readiness((ex(ExhibitStatus.FOUND),), ()) is Readiness.READY
    assert readiness((ex(ExhibitStatus.FOUND), ex(ExhibitStatus.WEAK, False)), ()) is (
        Readiness.READY
    )
    assert readiness((ex(ExhibitStatus.ATTACH),), ()) is Readiness.NEEDS_ATTACHMENTS
    assert readiness((ex(ExhibitStatus.ATTACH), ex(ExhibitStatus.MISSING)), ()) is Readiness.WEAK
    assert readiness((ex(ExhibitStatus.FOUND),), ("adverse",)) is Readiness.DO_NOT_REPRESENT


def test_format_money() -> None:
    assert format_money(2500, "USD") == "USD 25.00"
    assert format_money(123456789, "EUR") == "EUR 1,234,567.89"
    assert format_money(-5, "USD") == "USD -0.05"


def test_packet_dict_is_json_ready_and_missing_kind_raises() -> None:
    disputed = _txn("txn_d", 0)
    packet = assemble(
        EvidenceIndex(_dataset([disputed], [_dispute(disputed, "visa:10.4")])), "dp_1"
    )
    data = json.loads(json.dumps(packet.to_dict()))
    assert data["reason_code"] == "visa:10.4"
    assert data["respond_by"] > data["filed_at"]
    assert {e["kind"] for e in data["exhibits"]} >= {"transaction_record", "usage_logs"}
    with pytest.raises(KeyError):
        packet.exhibit(ExhibitKind.CANCELLATION_CHECK)


# --------------------------------------------------------------------------- generated data


@pytest.fixture(scope="module")
def generated() -> SyntheticDataset:
    return generate(GeneratorConfig(n_customers=800, seed=7, days=400))


def test_every_generated_dispute_assembles_without_ground_truth(
    generated: SyntheticDataset,
) -> None:
    packets = assemble_all(generated)
    assert len(packets) == len(generated.disputes) > 0
    assert len({p.readiness for p in packets}) > 1
    # Ground-truth labels must never leak into a packet.
    labels = {str(p) for p in Persona} - {"good"}
    for p in packets:
        text = p.render_markdown() + json.dumps(p.to_dict())
        assert "is_friendly_fraud" not in text
        assert "persona" not in text.lower()
        assert not any(label in text for label in labels)
    # Every cited transaction exists, was approved, and precedes the packet's as-of time.
    txns = {t.transaction_id: t for t in generated.transactions}
    for p in packets:
        for e in p.exhibits:
            for tid in e.transaction_ids:
                assert txns[tid].approved
                assert txns[tid].created_at <= p.as_of


def test_readiness_table_counts_every_dispute(generated: SyntheticDataset) -> None:
    rows = readiness_table(assemble_all(generated), generated.disputes)
    assert sum(r.disputes for r in rows) == len(generated.disputes)
    assert sum(r.friendly_fraud for r in rows) == sum(
        d.is_friendly_fraud for d in generated.disputes
    )
    assert all(0.0 <= r.friendly_share <= 1.0 for r in rows)
    # Charged-after-cancellation disputes are genuine merchant errors, never friendly fraud.
    assert all(r.friendly_fraud == 0 for r in rows if r.readiness is Readiness.DO_NOT_REPRESENT)
    table = render_readiness(rows)
    assert table.startswith("| Reason code | Readiness |")
    assert table.count("\n") == len(rows) + 2


def test_dispute_dataclass_fields_unchanged() -> None:
    """The assembler relies on these fields; fail loudly if the schema moves."""
    names = {f.name for f in fields(Dispute)}
    assert {"dispute_id", "transaction_id", "customer_id", "reason_code", "filed_at"} <= names


# --------------------------------------------------------------------------- CLI


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory, generated: SyntheticDataset) -> Path:
    out = tmp_path_factory.mktemp("evidence") / "synthetic"
    write_dataset(generated, out)
    return out


def test_cli_summary(data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["evidence", "--data", str(data_dir)]) == 0
    assert "| Reason code | Readiness |" in capsys.readouterr().out


def test_cli_single_packet_markdown_json_and_out(
    data_dir: Path, generated: SyntheticDataset, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dp = generated.disputes[0].dispute_id
    assert main(["evidence", "--data", str(data_dir), "--dispute-id", dp]) == 0
    assert f"# Representment draft — {dp}" in capsys.readouterr().out

    args = ["evidence", "--data", str(data_dir), "--dispute-id", dp, "--dispute-id", dp]
    assert main([*args, "--json"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1, "duplicate ids are assembled once"
    assert json.loads(lines[0])["dispute_id"] == dp

    assert main([*args, "--out", str(tmp_path / "packets")]) == 0
    assert (tmp_path / "packets" / f"{dp}.md").read_text(encoding="utf-8").startswith("# ")


def test_cli_errors(data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["evidence", "--data", str(data_dir), "--dispute-id", "dp_nope"])
    assert "unknown dispute id" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["evidence", "--data", str(data_dir), "--response-days", "0"])
    assert "response_days" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["evidence", "--data", str(tmp_path / "missing")])
    assert "cbp generate" in capsys.readouterr().err
