from __future__ import annotations

from pathlib import Path

import pytest

from chargeback_prediction.synthetic import GeneratorConfig, generate, read_dataset, write_dataset


def test_read_dataset_is_exact_inverse_of_write(tmp_path: Path) -> None:
    ds = generate(GeneratorConfig(n_customers=400, seed=11))
    write_dataset(ds, tmp_path)
    assert read_dataset(tmp_path) == ds


def test_read_dataset_reports_missing_table(tmp_path: Path) -> None:
    write_dataset(generate(GeneratorConfig(n_customers=50, seed=1)), tmp_path)
    (tmp_path / "disputes.csv").unlink()
    with pytest.raises(FileNotFoundError):
        read_dataset(tmp_path)


def test_read_dataset_rejects_wrong_header(tmp_path: Path) -> None:
    write_dataset(generate(GeneratorConfig(n_customers=50, seed=1)), tmp_path)
    (tmp_path / "card_updates.csv").write_text("customer_id,when,source\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"card_updates\.csv: expected columns"):
        read_dataset(tmp_path)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("cus_000001,2025-01-01T00:00:00+00:00,carrier_pigeon", "carrier_pigeon"),
        ("cus_000001,2025-01-01T00:00:00,customer", "missing a UTC offset"),
        ("cus_000001,2025-01-01T00:00:00+00:00", "expected 3 cells"),
    ],
)
def test_read_dataset_points_at_the_bad_line(tmp_path: Path, row: str, message: str) -> None:
    write_dataset(generate(GeneratorConfig(n_customers=50, seed=1)), tmp_path)
    (tmp_path / "card_updates.csv").write_text(
        f"customer_id,updated_at,source\n{row}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=rf"card_updates\.csv:2: .*{message}"):
        read_dataset(tmp_path)


def test_read_dataset_rejects_bad_boolean(tmp_path: Path) -> None:
    ds = generate(GeneratorConfig(n_customers=50, seed=1))
    write_dataset(ds, tmp_path)
    path = tmp_path / "transactions.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace(",true,", ",yes,").replace(",false,", ",yes,")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"transactions\.csv:2: expected 'true' or 'false'"):
        read_dataset(tmp_path)
