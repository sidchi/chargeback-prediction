"""Time-based train/test split that respects how dispute labels arrive.

A random split is wrong for chargebacks twice over: it trains on the future (customer
behaviour drifts) and it trains on labels the merchant would not have had yet. Disputes are
filed up to ~120 days after the charge, so on the day a model is trained, only charges older
than the dispute window carry final labels. :func:`time_split` reproduces that:

::

    ──────── train ────────┤◀── label gap ──▶├──────── test ────────▶ time
                       train_end          cutoff

* **test** is the newest ``test_fraction`` of mature rows (``created_at >= cutoff``);
* **train** is mature rows with ``created_at < cutoff - label_gap_days``, so every training
  dispute was filed *before* the test period starts;
* rows inside the gap are dropped (their labels were still in flight at ``cutoff``).

Immature rows (dispute window still open at snapshot time) are never used: a ``False`` label
there means "not yet", not "no".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from chargeback_prediction.modeling.dataset import FeatureMatrix
from chargeback_prediction.synthetic.config import MAX_DISPUTE_WINDOW_DAYS


@dataclass(frozen=True, slots=True)
class TimeSplit:
    train: FeatureMatrix
    test: FeatureMatrix
    train_end: datetime
    """Exclusive upper bound on training ``created_at``."""
    cutoff: datetime
    """Inclusive lower bound on test ``created_at``; the simulated model-training date."""
    dropped_in_gap: int
    dropped_immature: int


def time_split(
    matrix: FeatureMatrix,
    *,
    test_fraction: float = 0.25,
    label_gap_days: int = MAX_DISPUTE_WINDOW_DAYS,
) -> TimeSplit:
    """Split mature rows into an older training window and a newer test window.

    Args:
        matrix: Rows sorted by ``created_at`` (guaranteed by :class:`FeatureMatrix`).
        test_fraction: Share of mature rows, newest first, held out for testing.
        label_gap_days: Gap between the end of training and the start of testing. The default
            equals the network dispute window, so training labels are final as of ``cutoff``.
            Set to 0 to measure how optimistic an un-gapped split is.

    Raises:
        ValueError: on bad parameters, or if either side ends up empty or without a positive.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    if label_gap_days < 0:
        raise ValueError("label_gap_days must be >= 0")

    mature = matrix.subset(matrix.mature)
    if len(mature) < 2:
        raise ValueError("need at least two mature rows to split")
    cutoff = mature.created_at[int(len(mature) * (1.0 - test_fraction))]
    train_end = cutoff - timedelta(days=label_gap_days)

    stamps = np.array([t.timestamp() for t in mature.created_at])
    train_mask = stamps < train_end.timestamp()
    test_mask = stamps >= cutoff.timestamp()
    train, test = mature.subset(train_mask), mature.subset(test_mask)

    for name, part in (("train", train), ("test", test)):
        if len(part) == 0:
            raise ValueError(f"{name} split is empty; lower label_gap_days or test_fraction")
        if not part.y.any():
            raise ValueError(f"{name} split has no disputes; use a larger dataset")
    return TimeSplit(
        train=train,
        test=test,
        train_end=train_end,
        cutoff=cutoff,
        dropped_in_gap=int(len(mature) - train_mask.sum() - test_mask.sum()),
        dropped_immature=len(matrix) - len(mature),
    )
