from __future__ import annotations

from pathlib import Path

import pytest

from chargeback_prediction import __version__
from chargeback_prediction.cli import main


def test_reason_codes_command_lists_catalog(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["reason-codes"]) == 0
    out = capsys.readouterr().out
    assert "visa:13.2" in out
    assert "Cancelled Recurring Transaction" in out


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_command_is_required() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_generate_command_writes_csvs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "data"
    assert main(["generate", "--customers", "150", "--seed", "5", "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == [
        "card_updates.csv",
        "customers.csv",
        "disputes.csv",
        "subscriptions.csv",
        "transactions.csv",
    ]
    assert "customers" in capsys.readouterr().out


def test_generate_rejects_invalid_config(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["generate", "--customers", "0", "--out", str(tmp_path)])
    assert exc.value.code == 2


def test_generate_rejects_bad_date(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["generate", "--start", "not-a-date", "--out", str(tmp_path)])
    assert exc.value.code == 2


def test_features_command_builds_table_from_generated_csvs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "synthetic"
    assert main(["generate", "--customers", "300", "--seed", "3", "--out", str(data)]) == 0
    capsys.readouterr()

    out = tmp_path / "features.csv"
    assert main(["features", "--data", str(data), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "mature labels" in printed
    header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[:4] == ["transaction_id", "customer_id", "created_at", "amount_usd"]
    assert header[-1] == "label_mature"


def test_features_doc_prints_markdown(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["features", "--doc"]) == 0
    assert capsys.readouterr().out.startswith("# Feature table")


def test_features_reports_missing_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["features", "--data", str(tmp_path / "nope")])
    assert exc.value.code == 2
    assert "run `cbp generate` first" in capsys.readouterr().err


def test_features_reports_corrupt_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["generate", "--customers", "50", "--out", str(tmp_path)]) == 0
    (tmp_path / "customers.csv").write_text("id\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["features", "--data", str(tmp_path)])
    assert exc.value.code == 2
    assert "customers.csv" in capsys.readouterr().err
