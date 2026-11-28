from __future__ import annotations

import dataclasses
import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chargeback_prediction.cli import main
from chargeback_prediction.features import build_features
from chargeback_prediction.modeling import (
    FeatureMatrix,
    ModelKind,
    build_model,
    predict_risk,
    time_split,
)
from chargeback_prediction.monitoring import (
    Bins,
    MonitorConfig,
    MonitorReport,
    Status,
    calibration_check,
    monitor,
    population_stability_index,
    psi_from_shares,
    psi_noise_floor,
)
from chargeback_prediction.monitoring.report import render_markdown, to_json, write_artifacts
from chargeback_prediction.synthetic import GeneratorConfig, generate

# ------------------------------------------------------------------------------- PSI


def test_psi_is_zero_for_identical_samples() -> None:
    x = np.random.default_rng(0).lognormal(size=5_000)
    assert population_stability_index(x, x) == pytest.approx(0.0, abs=1e-12)


def test_psi_matches_hand_computation() -> None:
    expected = np.array([0.5, 0.3, 0.2])
    actual = np.array([0.2, 0.3, 0.5])
    by_hand = sum((a - e) * math.log(a / e) for e, a in zip(expected, actual, strict=True))
    assert psi_from_shares(expected, actual) == pytest.approx(by_hand)
    # PSI is symmetric in its arguments (it is a symmetrised KL divergence).
    assert psi_from_shares(actual, expected) == pytest.approx(by_hand)


def test_psi_grows_with_the_size_of_a_shift() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(size=20_000)
    psis = [population_stability_index(ref, rng.normal(loc=mu, size=5_000)) for mu in (0, 0.2, 1)]
    assert psis[0] < 0.01 < psis[1] < 0.1 < 0.25 < psis[2]


def test_psi_stays_finite_when_a_bin_empties_or_appears() -> None:
    ref = np.zeros(1_000)
    assert math.isfinite(population_stability_index(ref, np.ones(1_000)))
    assert population_stability_index(ref, np.ones(1_000)) > 1.0


def test_bins_collapse_to_distinct_values_for_flags_and_counts() -> None:
    flag = np.r_[np.zeros(950), np.ones(50)]
    bins = Bins.fit(flag)
    assert bins.edges == (0.0,)
    assert bins.shares(flag).tolist() == pytest.approx([0.95, 0.05])
    counts = np.repeat([0.0, 1.0, 2.0, 3.0], [700, 200, 80, 20])
    assert set(Bins.fit(counts).edges) <= {0.0, 1.0, 2.0, 3.0}


def test_values_outside_the_reference_range_land_in_the_outer_bins() -> None:
    bins = Bins.fit(np.arange(100.0))
    assert bins.assign(np.array([-1e9, 1e9])).tolist() == [0, bins.count - 1]


def test_bins_validate_input() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        Bins.fit(np.arange(10.0), n_bins=1)
    with pytest.raises(ValueError, match="empty"):
        Bins.fit(np.array([]))
    with pytest.raises(ValueError, match="finite"):
        Bins.fit(np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="1-D"):
        Bins.fit(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="equal length"):
        psi_from_shares(np.array([1.0]), np.array([0.5, 0.5]))


def test_noise_floor_predicts_psi_between_samples_of_one_distribution() -> None:
    rng = np.random.default_rng(2)
    n_ref, n_cur = 10_000, 2_000
    ref = rng.gamma(2.0, size=n_ref)
    bins = Bins.fit(ref)
    observed = np.mean(
        [
            psi_from_shares(bins.shares(ref), bins.shares(rng.gamma(2.0, size=n_cur)))
            for _ in range(300)
        ]
    )
    assert observed == pytest.approx(psi_noise_floor(bins.count, n_ref, n_cur), rel=0.15)
    with pytest.raises(ValueError, match="positive"):
        psi_noise_floor(10, 0, 5)


# ------------------------------------------------------------------- calibration check


def test_calibration_check_is_centred_for_a_calibrated_model() -> None:
    rng = np.random.default_rng(3)
    zs = []
    for _ in range(400):
        p = rng.uniform(0.0, 0.1, size=2_000)
        zs.append(calibration_check(rng.random(p.size) < p, p, MonitorConfig()).z)
    assert abs(np.mean(zs)) < 0.15
    assert np.std(zs) == pytest.approx(1.0, abs=0.1)
    # A 3-sigma alert should fire on well under 1% of healthy windows.
    assert np.mean(np.abs(zs) >= 3.0) < 0.01


def test_calibration_check_by_hand() -> None:
    p = np.full(100, 0.1)
    y = np.zeros(100, dtype=bool)
    y[:19] = True
    check = calibration_check(y, p, MonitorConfig())
    assert check.expected == pytest.approx(10.0)
    assert check.z == pytest.approx(9 / 3)  # sqrt(100 · 0.1 · 0.9) = 3
    assert check.status is Status.ALERT
    assert check.ratio == pytest.approx(1.9)
    assert check.observed_rate == pytest.approx(0.19)
    assert check.expected_rate == pytest.approx(0.1)
    with pytest.raises(ValueError, match="aligned"):
        calibration_check(y[:5], p, MonitorConfig())


def test_calibration_check_with_degenerate_scores() -> None:
    check = calibration_check(np.zeros(10, dtype=bool), np.zeros(10), MonitorConfig())
    assert check.z == 0.0
    assert math.isnan(check.ratio)


# ------------------------------------------------------------------------------ config


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"window_days": 0}, "window_days"),
        ({"psi_warn": 0.3, "psi_alert": 0.2}, "psi_warn"),
        ({"z_warn": 0.0}, "z_warn"),
        ({"min_labeled_rows": 0}, "minimum"),
        ({"top_features": -1}, "top_features"),
    ],
)
def test_config_validation(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MonitorConfig(**kwargs)


def test_status_thresholds_and_ordering() -> None:
    cfg = MonitorConfig()
    assert [cfg.psi_status(x) for x in (0.05, 0.1, 0.3)] == [
        Status.OK,
        Status.WARN,
        Status.ALERT,
    ]
    assert [cfg.z_status(z) for z in (1.0, -2.5, 3.0)] == [Status.OK, Status.WARN, Status.ALERT]
    assert max(Status.PENDING, Status.OK) is Status.OK


# ------------------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def matrix() -> FeatureMatrix:
    return FeatureMatrix.from_table(build_features(generate(GeneratorConfig(n_customers=2_500))))


@pytest.fixture(scope="module")
def model(matrix: FeatureMatrix) -> Any:
    split = time_split(matrix)
    return build_model(ModelKind.LOGREG).fit(split.train.X, split.train.y)


@pytest.fixture(scope="module")
def report(model: Any, matrix: FeatureMatrix) -> MonitorReport:
    return monitor(model, matrix)


def _after_cutoff(matrix: FeatureMatrix) -> np.ndarray:
    cutoff = time_split(matrix).cutoff
    return np.array([t >= cutoff for t in matrix.created_at])


def _with(matrix: FeatureMatrix, **changes: Any) -> FeatureMatrix:
    return dataclasses.replace(matrix, **changes)


def test_windows_tile_the_traffic_after_the_training_date(
    report: MonitorReport, matrix: FeatureMatrix
) -> None:
    split = time_split(matrix)
    assert report.monitored_from == split.cutoff
    assert report.reference_rows == len(split.train)
    assert sum(w.rows for w in report.windows) == int(_after_cutoff(matrix).sum())
    for prev, nxt in zip(report.windows, report.windows[1:], strict=False):
        assert nxt.start >= prev.end
        assert prev.end - prev.start == timedelta(days=report.config.window_days)


def test_calibration_only_counts_matured_labels(
    report: MonitorReport, matrix: FeatureMatrix
) -> None:
    labeled = report.labeled_windows
    assert labeled, "the default history should have some matured windows"
    newest = report.windows[-1]
    assert newest.calibration is None
    assert newest.calibration_status is Status.PENDING
    for w in labeled:
        assert w.calibration is not None
        assert w.calibration.rows == round(w.label_coverage * w.rows)


def test_healthy_model_on_its_own_history_stays_calibrated(report: MonitorReport) -> None:
    assert report.calibration_status is not Status.ALERT
    assert "Calibration held" in report.headline() or "Calibration failed" in report.headline()


def test_feature_drift_alone_only_warns(report: MonitorReport) -> None:
    for w in report.windows:
        assert all(d.status <= Status.WARN for d in w.features)
        assert [d.psi for d in w.features] == sorted((d.psi for d in w.features), reverse=True)


def test_current_defaults_to_reference(
    model: Any, matrix: FeatureMatrix, report: MonitorReport
) -> None:
    assert monitor(model, matrix, matrix) == report


def test_injected_feature_shift_is_detected(model: Any, matrix: FeatureMatrix) -> None:
    j = matrix.feature_names.index("amount_usd")
    X = matrix.X.copy()
    X[_after_cutoff(matrix), j] *= 4.0  # a price change the model never saw
    shifted = monitor(model, matrix, _with(matrix, X=X))
    for w in shifted.windows:
        if w.drift_status is Status.PENDING:
            continue
        amount = next(d for d in w.features if d.feature == "amount_usd")
        assert amount.psi > 0.25
        assert amount.status is Status.WARN
        assert amount.current_mean == pytest.approx(4 * amount.reference_mean, rel=0.5)
        assert w.drift_status >= Status.WARN


def test_injected_label_shift_fails_calibration(model: Any, matrix: FeatureMatrix) -> None:
    # A dispute wave the model did not anticipate: 5% of matured charges become disputes.
    rng = np.random.default_rng(4)
    y = matrix.y.copy()
    wave = _after_cutoff(matrix) & matrix.mature & (rng.random(len(matrix)) < 0.05)
    y[wave] = True
    shifted = monitor(model, matrix, _with(matrix, y=y))
    assert shifted.calibration_status is Status.ALERT
    for w in shifted.labeled_windows:
        assert w.calibration is not None
        assert w.calibration.z > 3
        assert any("under-predicting" in r for r in w.reasons)
    assert "Calibration failed" in shifted.headline()


def test_small_windows_are_not_judged(model: Any, matrix: FeatureMatrix) -> None:
    r = monitor(model, matrix, config=MonitorConfig(window_days=1, min_window_rows=10_000))
    assert all(w.drift_status is Status.PENDING for w in r.windows)
    assert "No window was large enough" in r.headline()
    assert all("not judged" in w.reasons[0] for w in r.windows)


def test_unlabeled_traffic_reports_calibration_unverified(
    model: Any, matrix: FeatureMatrix
) -> None:
    unlabeled = _with(matrix, mature=np.zeros(len(matrix), dtype=bool))
    r = monitor(model, matrix, unlabeled)
    assert r.labeled_windows == ()
    assert r.calibration_status is Status.PENDING
    assert "calibration is unverified" in r.headline()


def test_stable_traffic_headline(model: Any, matrix: FeatureMatrix) -> None:
    # Loose thresholds make the same traffic read as stable.
    r = monitor(model, matrix, config=MonitorConfig(psi_warn=50.0, psi_alert=100.0))
    assert r.drift_status is Status.OK
    assert r.headline().startswith("Scores are stable")
    assert r.top_drivers() == ()


def test_monitor_rejects_bad_inputs(model: Any, matrix: FeatureMatrix) -> None:
    before = matrix.subset(~_after_cutoff(matrix))
    with pytest.raises(ValueError, match="no monitored rows"):
        monitor(model, matrix, before)
    renamed = _with(matrix, feature_names=tuple(f"x{i}" for i in range(len(matrix.feature_names))))
    with pytest.raises(ValueError, match="different columns"):
        monitor(model, matrix, renamed)


def test_scores_use_the_model(model: Any, matrix: FeatureMatrix, report: MonitorReport) -> None:
    split = time_split(matrix)
    assert report.reference_mean_score == pytest.approx(
        float(predict_risk(model, split.train.X).mean())
    )


# ----------------------------------------------------------------------------- report


def test_markdown_and_json(report: MonitorReport, tmp_path: Path) -> None:
    md = render_markdown(report, ModelKind.LOGREG)
    for heading in ("# Model monitoring report", "## Windows", "Logistic regression"):
        assert heading in md
    assert md.count("\n| 20") == len(report.windows)
    data = to_json(report, ModelKind.LOGREG)
    assert data["status"] == report.status.label
    assert len(data["windows"]) == len(report.windows)
    assert json.loads(json.dumps(data)) == data

    paths = write_artifacts(report, ModelKind.LOGREG, tmp_path / "mon")
    assert paths["report"].read_text(encoding="utf-8").startswith("# Model monitoring report")
    assert json.loads(paths["metrics"].read_text(encoding="utf-8"))["model"] == "logreg"


def test_markdown_without_feature_drift(model: Any, matrix: FeatureMatrix) -> None:
    r = monitor(model, matrix, config=MonitorConfig(psi_warn=50.0, psi_alert=100.0))
    assert "No feature reached the warning threshold" in render_markdown(r, ModelKind.LOGREG)


# -------------------------------------------------------------------------------- CLI


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("monitor-cli")
    features = root / "features.csv"
    build_features(generate(GeneratorConfig(n_customers=1_500, seed=9))).write_csv(features)
    models = root / "baseline"
    assert (
        main(["train", "--features", str(features), "--out", str(models), "--model", "logreg"]) == 0
    )
    return features, models


def test_monitor_command(
    trained: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    features, models = trained
    out = tmp_path / "mon"
    argv = ["monitor", "--features", str(features), "--models", str(models), "--out", str(out)]
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert "status:" in printed
    assert (out / "report.md").exists()
    status = json.loads((out / "monitoring.json").read_text(encoding="utf-8"))["status"]
    expected_exit = 1 if status in {"warn", "alert"} else 0
    assert main([*argv, "--current", str(features), "--fail-on", "warn"]) == expected_exit


def test_monitor_fail_on_reflects_status(trained: tuple[Path, Path], tmp_path: Path) -> None:
    features, models = trained
    base = ["monitor", "--features", str(features), "--models", str(models)]
    base += ["--out", str(tmp_path)]
    # Impossible-to-reach thresholds: nothing can warn or alert, so the job passes.
    loose = ["--psi-warn", "50", "--psi-alert", "100", "--z-warn", "50", "--z-alert", "100"]
    assert main([*base, *loose, "--fail-on", "warn"]) == 0
    # Hair-trigger thresholds: every judged window warns or alerts, so the job fails.
    tight = ["--psi-warn", "0.0001", "--psi-alert", "0.0002"]
    assert main([*base, *tight, "--fail-on", "warn"]) == 1


def test_monitor_command_errors(
    trained: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    features, models = trained
    with pytest.raises(SystemExit) as exc:
        main(["monitor", "--features", str(features), "--models", str(tmp_path)])
    assert exc.value.code == 2
    assert "cbp train" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main(["monitor", "--features", str(tmp_path / "nope.csv"), "--models", str(models)])
    assert "cbp features" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main(
            ["monitor", "--features", str(features), "--models", str(models), "--window-days", "0"]
        )
    assert "window_days" in capsys.readouterr().err
    bad = tmp_path / "bad.csv"
    bad.write_text("transaction_id\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["monitor", "--features", str(bad), "--models", str(models)])
    assert "missing columns" in capsys.readouterr().err
