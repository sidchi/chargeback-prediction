"""Write a :class:`SyntheticDataset` to disk as one CSV file per table.

CSV keeps the dataset diff-able and readable with any tool; the feature pipeline is free to
convert to a columnar format later. Encoding rules: datetimes are ISO-8601 UTC, ``None`` is
an empty cell, booleans are ``true``/``false``, enums are their string value.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import astuple, fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from chargeback_prediction.synthetic import schema
from chargeback_prediction.synthetic.generator import SyntheticDataset

TABLES: tuple[str, ...] = (
    "customers",
    "subscriptions",
    "transactions",
    "card_updates",
    "disputes",
)


def _encode(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _write_table(path: Path, rows: Iterable[Any], record_type: type) -> int:
    count = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(f.name for f in fields(record_type))
        for row in rows:
            writer.writerow(_encode(v) for v in astuple(row))
            count += 1
    return count


def write_dataset(dataset: SyntheticDataset, out_dir: Path) -> dict[str, Path]:
    """Write every table to ``out_dir/<table>.csv`` and return the paths by table name."""
    record_types: dict[str, type] = {
        "customers": schema.Customer,
        "subscriptions": schema.Subscription,
        "transactions": schema.Transaction,
        "card_updates": schema.CardUpdate,
        "disputes": schema.Dispute,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for table in TABLES:
        path = out_dir / f"{table}.csv"
        _write_table(path, getattr(dataset, table), record_types[table])
        paths[table] = path
    return paths
