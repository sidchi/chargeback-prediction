from __future__ import annotations

from pathlib import Path

import pytest

from chargeback_prediction import __version__
from chargeback_prediction.cli import main
from chargeback_prediction.features import build_features
from chargeback_prediction.synthetic import GeneratorConfig, generate


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


def test_train_command_writes_report_and_models(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data, features, out = tmp_path / "synthetic", tmp_path / "features.csv", tmp_path / "art"
    assert main(["generate", "--customers", "1500", "--seed", "11", "--out", str(data)]) == 0
    assert main(["features", "--data", str(data), "--out", str(features)]) == 0
    capsys.readouterr()

    assert main(["train", "--features", str(features), "--out", str(out), "--model", "logreg"]) == 0
    printed = capsys.readouterr().out
    assert "Logistic regression" in printed
    assert "Gradient boosting" not in printed
    assert sorted(p.name for p in out.iterdir()) == [
        "calibration.png",
        "metrics.json",
        "model_logreg.joblib",
        "report.md",
    ]


def test_train_reports_missing_features(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["train", "--features", str(tmp_path / "nope.csv")])
    assert exc.value.code == 2
    assert "run `cbp features` first" in capsys.readouterr().err


def test_train_rejects_bad_split(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    features = tmp_path / "features.csv"
    features.write_text("transaction_id\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["train", "--features", str(features)])
    assert exc.value.code == 2
    assert "missing columns" in capsys.readouterr().err

    data = tmp_path / "synthetic"
    assert main(["generate", "--customers", "200", "--out", str(data)]) == 0
    assert main(["features", "--data", str(data), "--out", str(features)]) == 0
    with pytest.raises(SystemExit) as exc:
        main(["train", "--features", str(features), "--test-fraction", "1.5"])
    assert exc.value.code == 2
    assert "test_fraction" in capsys.readouterr().err


def test_evaluate_command_writes_cost_report_and_model_card(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data, features, out = tmp_path / "synthetic", tmp_path / "features.csv", tmp_path / "eval"
    assert main(["generate", "--customers", "1500", "--seed", "11", "--out", str(data)]) == 0
    assert main(["features", "--data", str(data), "--out", str(features)]) == 0
    capsys.readouterr()

    args = ["evaluate", "--features", str(features), "--out", str(out), "--resamples", "20"]
    assert main([*args, "--ltv", "300", "--model", "logreg"]) == 0
    printed = capsys.readouterr().out
    assert "recommended: Logistic regression" in printed
    assert "Gradient boosting" not in printed
    assert sorted(p.name for p in out.iterdir()) == [
        "cost_curve.png",
        "evaluation.json",
        "model_card.md",
        "report.md",
    ]
    assert "$300 lifetime value" in (out / "report.md").read_text(encoding="utf-8")


def test_evaluate_reports_missing_features(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["evaluate", "--features", str(tmp_path / "nope.csv")])
    assert exc.value.code == 2
    assert "run `cbp features` first" in capsys.readouterr().err


def test_evaluate_rejects_invalid_costs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    features = tmp_path / "features.csv"
    build_features(generate(GeneratorConfig(n_customers=300, seed=5))).write_csv(features)
    with pytest.raises(SystemExit) as exc:
        main(["evaluate", "--features", str(features), "--margin", "1.5"])
    assert exc.value.code == 2
    assert "gross_margin" in capsys.readouterr().err
