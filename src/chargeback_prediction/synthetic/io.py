"""Read and write a :class:`SyntheticDataset` as one CSV file per table.

CSV keeps the dataset diff-able and readable with any tool; the feature pipeline is free to
convert to a columnar format later. Encoding rules: datetimes are ISO-8601 UTC, ``None`` is
an empty cell, booleans are ``true``/``false``, enums are their string value.

:func:`read_dataset` is the exact inverse of :func:`write_dataset`, so downstream stages
(features, training) run against files on disk the same way they would against an export
from a production warehouse.
"""

from __future__ import annotations

import csv
import types
from collections.abc import Callable, Iterable
from dataclasses import astuple, fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

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


_RECORD_TYPES: dict[str, type] = {
    "customers": schema.Customer,
    "subscriptions": schema.Subscription,
    "transactions": schema.Transaction,
    "card_updates": schema.CardUpdate,
    "disputes": schema.Dispute,
}


def write_dataset(dataset: SyntheticDataset, out_dir: Path) -> dict[str, Path]:
    """Write every table to ``out_dir/<table>.csv`` and return the paths by table name."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for table in TABLES:
        path = out_dir / f"{table}.csv"
        _write_table(path, getattr(dataset, table), _RECORD_TYPES[table])
        paths[table] = path
    return paths


def _parse_bool(cell: str) -> bool:
    if cell == "true":
        return True
    if cell == "false":
        return False
    raise ValueError(f"expected 'true' or 'false', got {cell!r}")


def _parse_datetime(cell: str) -> datetime:
    value = datetime.fromisoformat(cell)
    if value.tzinfo is None:
        raise ValueError(f"datetime {cell!r} is missing a UTC offset")
    return value


def _decoder(annotation: Any) -> Callable[[str], Any]:
    """Build a cell decoder for one dataclass field annotation (the inverse of ``_encode``)."""
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        (inner,) = (a for a in get_args(annotation) if a is not type(None))
        decode_inner = _decoder(inner)
        return lambda cell: None if cell == "" else decode_inner(cell)
    if annotation is bool:
        return _parse_bool
    if annotation is datetime:
        return _parse_datetime
    if annotation is int:
        return int
    if annotation is str:
        return str
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    raise TypeError(f"no CSV decoder for field type {annotation!r}")  # pragma: no cover


def _read_table(path: Path, record_type: type) -> tuple[Any, ...]:
    hints = get_type_hints(record_type)
    names = [f.name for f in fields(record_type)]
    decoders = [_decoder(hints[name]) for name in names]
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header != names:
            raise ValueError(f"{path.name}: expected columns {names}, got {header}")
        rows = []
        for line_no, row in enumerate(reader, start=2):
            if len(row) != len(names):
                raise ValueError(f"{path.name}:{line_no}: expected {len(names)} cells")
            try:
                rows.append(record_type(*(d(c) for d, c in zip(decoders, row, strict=True))))
            except ValueError as exc:
                raise ValueError(f"{path.name}:{line_no}: {exc}") from None
    return tuple(rows)


def read_dataset(data_dir: Path) -> SyntheticDataset:
    """Load a dataset previously written by :func:`write_dataset`.

    Raises:
        FileNotFoundError: if a table file is missing.
        ValueError: if a header or cell does not match the schema.
    """
    tables = {
        table: _read_table(data_dir / f"{table}.csv", _RECORD_TYPES[table]) for table in TABLES
    }
    return SyntheticDataset(**tables)
