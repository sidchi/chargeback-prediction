from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from chargeback_prediction.modeling import (
    average_precision,
    brier_score,
    calibration_bins,
    expected_calibration_error,
    precision_at_k,
    recall_at_k,
)
from chargeback_prediction.modeling.metrics import CalibrationBin


@pytest.mark.parametrize("seed", range(5))
def test_average_precision_matches_sklearn_including_ties(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.random(500) < 0.05
    y[0] = True
    # Coarse rounding creates many tied scores, which is where hand-rolled AP usually drifts.
    scores = np.round(rng.random(500) * 0.5 + y * 0.3, 1)
    assert average_precision(y, scores) == pytest.approx(average_precision_score(y, scores))


def test_average_precision_known_values() -> None:
    assert average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx(
        0.5 * 1.0 + 0.5 * (2 / 3)
    )
    assert average_precision([0, 1], [0.2, 0.9]) == pytest.approx(1.0)
    # All scores tied: one threshold, precision = base rate.
    assert average_precision([1, 0, 0, 0], [0.5] * 4) == pytest.approx(0.25)


def test_average_precision_is_order_invariant_under_ties() -> None:
    y = np.array([1, 0, 0, 1, 0])
    s = np.array([0.5, 0.5, 0.5, 0.2, 0.1])
    perm = np.array([1, 2, 0, 4, 3])
    assert average_precision(y, s) == pytest.approx(average_precision(y[perm], s[perm]))


def test_average_precision_without_positives_is_nan() -> None:
    assert math.isnan(average_precision([0, 0], [0.1, 0.2]))


@pytest.mark.parametrize(
    ("y", "s"),
    [
        ([], []),
        ([1, 0], [0.1]),
        ([1, 0], [0.1, float("nan")]),
        ([[1, 0]], [[0.1, 0.2]]),
    ],
)
def test_metric_input_validation(y: list[Any], s: list[Any]) -> None:
    with pytest.raises(ValueError, match=r"1-D|at least one|finite"):
        average_precision(y, s)


def test_precision_and_recall_at_k() -> None:
    y = [1, 0, 1, 0, 0, 1]
    s = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    assert precision_at_k(y, s, 1) == 1.0
    assert precision_at_k(y, s, 3) == pytest.approx(2 / 3)
    assert recall_at_k(y, s, 3) == pytest.approx(2 / 3)
    assert recall_at_k(y, s, 6) == 1.0
    assert math.isnan(recall_at_k([0, 0], [0.1, 0.2], 1))


def test_precision_at_k_breaks_ties_by_row_order() -> None:
    assert precision_at_k([0, 1], [0.5, 0.5], 1) == 0.0
    assert precision_at_k([1, 0], [0.5, 0.5], 1) == 1.0


@pytest.mark.parametrize("k", [0, 4])
def test_k_out_of_range(k: int) -> None:
    with pytest.raises(ValueError, match="k must be"):
        precision_at_k([1, 0, 0], [0.3, 0.2, 0.1], k)
    with pytest.raises(ValueError, match="k must be"):
        recall_at_k([1, 0, 0], [0.3, 0.2, 0.1], k)


def test_brier_score() -> None:
    assert brier_score([1, 0], [1.0, 0.0]) == 0.0
    assert brier_score([1, 0], [0.5, 0.5]) == pytest.approx(0.25)


def test_calibration_bins_are_equal_count_and_cover_every_row() -> None:
    rng = np.random.default_rng(0)
    s = rng.random(1_000)
    y = rng.random(1_000) < s
    bins = calibration_bins(y, s, n_bins=10)
    assert len(bins) == 10
    assert {b.count for b in bins} == {100}
    assert all(a.upper <= b.lower for a, b in pairwise(bins))
    # Outcomes drawn from the scores themselves: well calibrated by construction.
    assert expected_calibration_error(bins) < 0.06


def test_calibration_bins_keep_ties_together() -> None:
    s = [0.1, 0.1, 0.1, 0.1, 0.9, 0.9]
    y = [0, 0, 1, 0, 1, 1]
    bins = calibration_bins(y, s, n_bins=3)
    assert [b.count for b in bins] == [4, 2]
    assert bins[0].observed_rate == pytest.approx(0.25)
    assert bins[1].mean_predicted == pytest.approx(0.9)


def test_calibration_bins_more_bins_than_rows() -> None:
    bins = calibration_bins([1, 0], [0.8, 0.2], n_bins=10)
    assert [b.count for b in bins] == [1, 1]


def test_calibration_validation() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        calibration_bins([1], [0.5], n_bins=0)
    with pytest.raises(ValueError, match="empty"):
        expected_calibration_error([])


def test_expected_calibration_error_is_count_weighted() -> None:
    bins = [
        CalibrationBin(lower=0, upper=0.1, count=3, mean_predicted=0.1, observed_rate=0.1),
        CalibrationBin(lower=0.5, upper=0.5, count=1, mean_predicted=0.5, observed_rate=0.9),
    ]
    assert expected_calibration_error(bins) == pytest.approx(0.4 / 4)
