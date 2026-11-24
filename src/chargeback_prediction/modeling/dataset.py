"""Numeric design matrix for model training, built from a feature table or its CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

from chargeback_prediction.features import FEATURE_NAMES, FeatureTable

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    """Features as a dense ``float64`` matrix plus the labels and timestamps a split needs.

    Rows are sorted by ``created_at`` so time-based slicing is a contiguous range.
    """

    X: FloatArray
    y: BoolArray
    """``True`` when a dispute was filed on the transaction within the observed window."""
    mature: BoolArray
    """The dispute window had fully elapsed at snapshot time: a negative label is final."""
    created_at: tuple[datetime, ...]
    transaction_ids: tuple[str, ...]
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        n = len(self.created_at)
        if self.X.shape != (n, len(self.feature_names)):
            raise ValueError(
                f"X has shape {self.X.shape}, expected ({n}, {len(self.feature_names)})"
            )
        if self.y.shape != (n,) or self.mature.shape != (n,) or len(self.transaction_ids) != n:
            raise ValueError("labels, maturity flags and ids must have one entry per row")
        if any(a > b for a, b in zip(self.created_at, self.created_at[1:], strict=False)):
            raise ValueError("rows must be sorted by created_at")

    def __len__(self) -> int:
        return len(self.created_at)

    @property
    def positive_rate(self) -> float:
        return float(self.y.mean()) if len(self) else 0.0

    def subset(self, mask: BoolArray) -> FeatureMatrix:
        """Rows where ``mask`` is true, preserving order."""
        idx = np.flatnonzero(mask)
        return FeatureMatrix(
            X=self.X[idx],
            y=self.y[idx],
            mature=self.mature[idx],
            created_at=tuple(self.created_at[i] for i in idx),
            transaction_ids=tuple(self.transaction_ids[i] for i in idx),
            feature_names=self.feature_names,
        )

    @classmethod
    def from_table(cls, table: FeatureTable) -> FeatureMatrix:
        rows = sorted(table.rows, key=lambda r: r.created_at)
        return cls(
            X=np.array([r.values for r in rows], dtype=np.float64).reshape(
                len(rows), len(table.columns)
            ),
            y=np.array([r.disputed for r in rows], dtype=np.bool_),
            mature=np.array([r.label_mature for r in rows], dtype=np.bool_),
            created_at=tuple(r.created_at for r in rows),
            transaction_ids=tuple(r.transaction_id for r in rows),
            feature_names=tuple(table.columns),
        )

    @classmethod
    def read_csv(cls, path: Path) -> FeatureMatrix:
        """Load a CSV written by :meth:`FeatureTable.write_csv`.

        Only the id, timestamp, feature and ``disputed``/``label_mature`` columns are read;
        reason code and category are evaluation-only labels and stay out of the matrix.
        """
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError(f"{path}: empty file") from None
            missing = [c for c in (*FEATURE_NAMES, "disputed", "label_mature") if c not in header]
            if missing:
                raise ValueError(f"{path}: missing columns {missing}")
            col = {name: i for i, name in enumerate(header)}
            feat_idx = [col[name] for name in FEATURE_NAMES]
            records: list[tuple[datetime, str, list[float], bool, bool]] = []
            for line_no, row in enumerate(reader, start=2):
                try:
                    records.append(
                        (
                            datetime.fromisoformat(row[col["created_at"]]),
                            row[col["transaction_id"]],
                            [float(row[i]) for i in feat_idx],
                            _parse_bool(row[col["disputed"]]),
                            _parse_bool(row[col["label_mature"]]),
                        )
                    )
                except (ValueError, IndexError) as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from None
        records.sort(key=lambda r: r[0])
        n = len(records)
        return cls(
            X=np.array([r[2] for r in records], dtype=np.float64).reshape(n, len(FEATURE_NAMES)),
            y=np.array([r[3] for r in records], dtype=np.bool_),
            mature=np.array([r[4] for r in records], dtype=np.bool_),
            created_at=tuple(r[0] for r in records),
            transaction_ids=tuple(r[1] for r in records),
        )


def _parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"expected 'true' or 'false', got {value!r}")
