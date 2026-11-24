from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pytest

from chargeback_prediction.features import FEATURE_NAMES, build_features
from chargeback_prediction.modeling import (
    BaselineReport,
    FeatureMatrix,
    ModelKind,
    TrainConfig,
    build_model,
    evaluate,
    predict_risk,
    time_split,
    train_baselines,
)
from chargeback_prediction.modeling.report import render_markdown, report_to_dict, write_artifacts
from chargeback_prediction.synthetic import GeneratorConfig, generate

T0 = datetime(2025, 1, 1, tzinfo=UTC)
N_FEATURES = len(FEATURE_NAMES)


def _matrix(n: int, *, mature_until: int | None = None, positives_every: int = 5) -> FeatureMatrix:
    """``n`` daily rows; row ``i`` is mature iff ``i < mature_until``."""
    mature_until = n if mature_until is None else mature_until
    return FeatureMatrix(
        X=np.arange(n * N_FEATURES, dtype=np.float64).reshape(n, N_FEATURES),
        y=np.array([i % positives_every == 0 for i in range(n)]),
        mature=np.array([i < mature_until for i in range(n)]),
        created_at=tuple(T0 + timedelta(days=i) for i in range(n)),
        transaction_ids=tuple(f"txn_{i}" for i in range(n)),
    )


# ------------------------------------------------------------------ FeatureMatrix


def test_feature_matrix_validates_shapes_and_order() -> None:
    m = _matrix(3)
    with pytest.raises(ValueError, match="X has shape"):
        FeatureMatrix(m.X[:, :2], m.y, m.mature, m.created_at, m.transaction_ids)
    with pytest.raises(ValueError, match="one entry per row"):
        FeatureMatrix(m.X, m.y[:2], m.mature, m.created_at, m.transaction_ids)
    with pytest.raises(ValueError, match="sorted"):
        FeatureMatrix(m.X, m.y, m.mature, m.created_at[::-1], m.transaction_ids)


def test_subset_preserves_alignment() -> None:
    m = _matrix(10)
    sub = m.subset(np.arange(10) % 2 == 1)
    assert sub.transaction_ids == ("txn_1", "txn_3", "txn_5", "txn_7", "txn_9")
    assert sub.X[0, 0] == m.X[1, 0]
    assert len(sub) == 5


@pytest.fixture(scope="module")
def default_table_matrix() -> FeatureMatrix:
    return FeatureMatrix.from_table(build_features(generate(GeneratorConfig(n_customers=2_500))))


def test_csv_roundtrip_matches_in_memory_matrix(tmp_path: Path) -> None:
    table = build_features(generate(GeneratorConfig(n_customers=200, seed=3)))
    path = tmp_path / "features.csv"
    table.write_csv(path)
    from_csv = FeatureMatrix.read_csv(path)
    in_memory = FeatureMatrix.from_table(table)
    assert from_csv.transaction_ids == in_memory.transaction_ids
    assert from_csv.created_at == in_memory.created_at
    np.testing.assert_allclose(from_csv.X, in_memory.X, rtol=1e-5)
    np.testing.assert_array_equal(from_csv.y, in_memory.y)
    np.testing.assert_array_equal(from_csv.mature, in_memory.mature)


def test_read_csv_errors(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty file"):
        FeatureMatrix.read_csv(empty)

    no_labels = tmp_path / "no_labels.csv"
    no_labels.write_text("transaction_id,created_at\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        FeatureMatrix.read_csv(no_labels)

    header = ["transaction_id", "created_at", *FEATURE_NAMES, "disputed", "label_mature"]
    bad = tmp_path / "bad.csv"
    row = ["t1", T0.isoformat(), *["0"] * N_FEATURES, "yes", "true"]
    bad.write_text(",".join(header) + "\n" + ",".join(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.csv:2: expected 'true' or 'false'"):
        FeatureMatrix.read_csv(bad)


# --------------------------------------------------------------------- time split


def test_time_split_orders_windows_and_enforces_label_gap() -> None:
    # 400 days, last 100 immature -> 300 mature; test = newest 25% of mature (75 rows).
    split = time_split(_matrix(400, mature_until=300), test_fraction=0.25, label_gap_days=30)
    assert split.dropped_immature == 100
    assert len(split.test) == 75
    assert split.cutoff == T0 + timedelta(days=225)
    assert split.train_end == split.cutoff - timedelta(days=30)
    assert max(split.train.created_at) < split.train_end
    assert min(split.test.created_at) >= split.cutoff
    assert split.dropped_in_gap == 30
    assert len(split.train) + len(split.test) + split.dropped_in_gap == 300
    assert split.test.mature.all()
    assert split.train.mature.all()
    assert not set(split.train.transaction_ids) & set(split.test.transaction_ids)


def test_zero_gap_keeps_every_mature_row() -> None:
    split = time_split(_matrix(100), test_fraction=0.2, label_gap_days=0)
    assert split.dropped_in_gap == 0
    assert (len(split.train), len(split.test)) == (80, 20)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"test_fraction": 0.0}, "test_fraction"),
        ({"test_fraction": 1.0}, "test_fraction"),
        ({"label_gap_days": -1}, "label_gap_days"),
        ({"label_gap_days": 500}, "train split is empty"),
    ],
)
def test_time_split_rejects_bad_parameters(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        time_split(_matrix(100), **kwargs)  # type: ignore[arg-type]


def test_time_split_needs_mature_rows_and_positives() -> None:
    with pytest.raises(ValueError, match="at least two mature"):
        time_split(_matrix(10, mature_until=1))
    with pytest.raises(ValueError, match="no disputes"):
        time_split(_matrix(20, positives_every=1_000), label_gap_days=0)


# ------------------------------------------------------------------------ models


@pytest.mark.parametrize("kind", list(ModelKind))
def test_models_learn_a_planted_signal(kind: ModelKind) -> None:
    rng = np.random.default_rng(1)
    X = rng.poisson(1.0, size=(3_000, 5)).astype(np.float64)
    y = rng.random(3_000) < np.where(X[:, 0] > 2, 0.5, 0.02)
    model = build_model(kind)
    model.fit(X, y)
    scores = predict_risk(model, X)
    assert scores.shape == (3_000,)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert scores[X[:, 0] > 2].mean() > 5 * scores[X[:, 0] <= 2].mean()


def test_kind_labels() -> None:
    assert ModelKind("gbm").label == "Gradient boosting"
    assert ModelKind.LOGREG.label == "Logistic regression"


def test_evaluate_sizes_review_budgets() -> None:
    y = np.array([True] + [False] * 199)
    scores = np.linspace(1, 0, 200)
    ev = evaluate(ModelKind.LOGREG, y, scores, review_budgets=(0.001, 0.05))
    assert [a.k for a in ev.at_k] == [1, 10]
    assert ev.at_k[0].precision == 1.0
    assert ev.at_k[0].lift(0.005) == pytest.approx(200)
    assert np.isnan(ev.at_k[0].lift(0.0))
    assert ev.pr_auc == 1.0


def test_train_config_validation() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        TrainConfig(models=())
    with pytest.raises(ValueError, match="review budgets"):
        TrainConfig(review_budgets=(0.0,))


# ------------------------------------------------------- end-to-end on synthetic data


@pytest.fixture(scope="module")
def report(default_table_matrix: FeatureMatrix) -> BaselineReport:
    return train_baselines(default_table_matrix)


def test_baselines_beat_random_on_the_future_window(report: BaselineReport) -> None:
    base = report.test_base_rate
    assert 0.0 < base < 0.1
    for ev in report.evaluations:
        # A random scorer's PR-AUC equals the base rate; the planted patterns are strong.
        assert ev.pr_auc > 4 * base, ev.kind
        assert ev.at_k[0].precision > 5 * base, ev.kind
        # Unweighted training keeps probabilities near the observed rate.
        assert ev.ece < 0.05, ev.kind
        assert ev.scores.shape == (len(report.split.test),)
    assert report.best().pr_auc == max(e.pr_auc for e in report.evaluations)


def test_training_only_sees_rows_whose_labels_were_final_at_cutoff(
    report: BaselineReport,
) -> None:
    split = report.split
    assert max(split.train.created_at) + timedelta(days=120) <= split.cutoff


def test_training_is_deterministic(default_table_matrix: FeatureMatrix) -> None:
    cfg = TrainConfig(models=(ModelKind.GBM,))
    a = train_baselines(default_table_matrix, cfg).evaluations[0]
    b = train_baselines(default_table_matrix, cfg).evaluations[0]
    np.testing.assert_array_equal(a.scores, b.scores)


def test_markdown_and_dict_reports(report: BaselineReport) -> None:
    md = render_markdown(report, figure="calibration.png")
    assert md.startswith("# Baseline model report")
    for needle in ("Logistic regression", "Gradient boosting", "*Random scorer*", "top 1.0%"):
        assert needle in md
    assert "![Calibration and precision-recall curves](calibration.png)" in md
    assert "calibration.png" not in render_markdown(report)

    data = report_to_dict(report)
    assert set(data["models"]) == {"logreg", "gbm"}
    assert data["split"]["label_gap_days"] == 120
    json.dumps(data)  # serialisable


def test_write_artifacts_round_trips_models(report: BaselineReport, tmp_path: Path) -> None:
    paths = write_artifacts(report, tmp_path / "out")
    assert paths["figure"].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert json.loads(paths["metrics"].read_text())["test_base_rate"] == report.test_base_rate
    for ev in report.evaluations:
        model = joblib.load(paths[f"model_{ev.kind.value}"])
        np.testing.assert_allclose(predict_risk(model, report.split.test.X), ev.scores)
