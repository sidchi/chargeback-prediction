from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from chargeback_prediction.cli import main
from chargeback_prediction.explain import (
    Direction,
    ExplainConfig,
    Factor,
    ShapValues,
    TransactionExplanation,
    explain_rows,
    explain_test_window,
    explain_transactions,
    format_value,
    global_importance,
    load_model,
    shap_values,
    sigmoid,
    top_factors,
)
from chargeback_prediction.explain.factors import format_multiplier, format_probability
from chargeback_prediction.explain.run import background_sample, render_markdown, write_artifacts
from chargeback_prediction.explain.shapley import _coalition_weight
from chargeback_prediction.features import build_features
from chargeback_prediction.modeling import (
    FeatureMatrix,
    ModelKind,
    build_model,
    predict_risk,
    time_split,
)
from chargeback_prediction.synthetic import GeneratorConfig, generate

FloatArray = npt.NDArray[np.float64]

# ------------------------------------------------------------- brute-force reference


def _brute_force_shap(model: Any, x: FloatArray, background: FloatArray) -> FloatArray:
    """Interventional Shapley values by enumerating every coalition (2**m model calls)."""
    m = len(x)

    def value(coalition: tuple[int, ...]) -> float:
        hybrid = background.copy()
        hybrid[:, list(coalition)] = x[list(coalition)]
        return float(np.mean(model.decision_function(hybrid)))

    phi = np.zeros(m)
    for i in range(m):
        others = [j for j in range(m) if j != i]
        for size in range(m):
            weight = math.factorial(size) * math.factorial(m - size - 1) / math.factorial(m)
            for s in combinations(others, size):
                phi[i] += weight * (value((*s, i)) - value(s))
    return phi


@pytest.fixture(scope="module")
def toy() -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Five features with an interaction, a heavy tail and a constant column."""
    rng = np.random.default_rng(7)
    n = 3_000
    X = np.column_stack(
        [
            rng.integers(0, 2, n),  # flag
            rng.poisson(1.5, n),  # count
            rng.lognormal(3, 1, n),  # heavy-tailed amount
            rng.uniform(0, 365, n),  # days
            np.zeros(n),  # constant
        ]
    ).astype(np.float64)
    logit = -3 + 1.5 * X[:, 0] * (X[:, 1] > 1) + 0.01 * X[:, 2] - 0.004 * X[:, 3]
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    return X, y


NAMES = ("flag", "count", "amount_usd", "age_days", "constant")


@pytest.mark.parametrize("kind", list(ModelKind))
def test_exact_shap_matches_brute_force_enumeration(
    kind: ModelKind, toy: tuple[FloatArray, npt.NDArray[np.bool_]]
) -> None:
    X, y = toy
    model = build_model(kind, seed=1).fit(X, y)
    background = X[:40]
    explained = X[100:106]
    result = shap_values(model, explained, background, NAMES)
    for row, phi in zip(explained, result.values, strict=True):
        np.testing.assert_allclose(phi, _brute_force_shap(model, row, background), atol=1e-10)
    assert result.additivity_error() < 1e-10
    np.testing.assert_allclose(result.values[:, 4], 0.0)  # constant feature gets no credit


def test_deeper_trees_and_repeated_splits_stay_exact(
    toy: tuple[FloatArray, npt.NDArray[np.bool_]],
) -> None:
    """Depth 5 makes paths split on the same feature twice and on up to five features."""
    X, y = toy
    model = HistGradientBoostingClassifier(max_depth=5, max_iter=15, random_state=0).fit(X, y)
    background = X[:25]
    result = shap_values(model, X[200:203], background, NAMES)
    for row, phi in zip(X[200:203], result.values, strict=True):
        np.testing.assert_allclose(phi, _brute_force_shap(model, row, background), atol=1e-10)


def test_missing_values_follow_the_learned_direction(
    toy: tuple[FloatArray, npt.NDArray[np.bool_]],
) -> None:
    X, y = toy
    X = X.copy()
    X[::7, 2] = np.nan
    model = HistGradientBoostingClassifier(max_depth=3, max_iter=20, random_state=0).fit(X, y)
    rows = X[[0, 7, 8]]  # two with a missing amount
    result = shap_values(model, rows, X[1:30], NAMES)
    assert result.additivity_error() < 1e-10
    for row, phi in zip(rows, result.values, strict=True):
        np.testing.assert_allclose(phi, _brute_force_shap(model, row, X[1:30]), atol=1e-10)


def test_coalition_weights_sum_to_one_over_positions() -> None:
    # sum over coalition sizes s of C(m-1, s) * s!(m-s-1)!/m! == 1
    for m in range(1, 7):
        total = sum(math.comb(m - 1, s) * _coalition_weight(s, m - 1 - s) for s in range(m))
        assert total == pytest.approx(1.0)


def test_unsupported_models_and_shapes_are_rejected(
    toy: tuple[FloatArray, npt.NDArray[np.bool_]],
) -> None:
    X, y = toy
    model = build_model(ModelKind.LOGREG).fit(X, y)
    with pytest.raises(ValueError, match="non-empty"):
        shap_values(model, X[:2], X[:0], NAMES)
    with pytest.raises(ValueError, match="feature columns"):
        shap_values(model, X[:2, :3], X[:5], NAMES)
    with pytest.raises(TypeError, match="no exact SHAP"):
        shap_values(LogisticRegression().fit(X, y), X[:2], X[:5], NAMES)

    mixing = Pipeline(
        [
            ("mix", FunctionTransformer(lambda a: a + a.sum(axis=1, keepdims=True))),
            ("clf", LogisticRegression()),
        ]
    ).fit(X, y)
    with pytest.raises(TypeError, match="mixes features"):
        shap_values(mixing, X[:2], X[:5], NAMES)

    multiclass = Pipeline([("s", StandardScaler()), ("clf", LogisticRegression())]).fit(
        X, np.arange(len(X)) % 3
    )
    with pytest.raises(TypeError, match="binary linear"):
        shap_values(multiclass, X[:2], X[:5], NAMES)
    gbm3 = HistGradientBoostingClassifier(max_iter=2).fit(X, np.arange(len(X)) % 3)
    with pytest.raises(TypeError, match="binary"):
        shap_values(gbm3, X[:2], X[:5], NAMES)


def test_shap_values_container_validates_and_aggregates() -> None:
    values = np.array([[0.5, -1.0], [-0.5, 0.0]])
    sv = ShapValues(values, base_value=-2.0, raw=np.array([-2.5, -2.5]), feature_names=("a", "b"))
    assert len(sv) == 2
    assert sv.additivity_error() == pytest.approx(0.0)
    np.testing.assert_allclose(sv.mean_abs(), [0.5, 0.5])
    with pytest.raises(ValueError, match="names"):
        ShapValues(values, 0.0, np.zeros(2), ("a",))
    with pytest.raises(ValueError, match="one entry per row"):
        ShapValues(values, 0.0, np.zeros(3), ("a", "b"))
    empty = ShapValues(np.zeros((0, 2)), 0.0, np.zeros(0), ("a", "b"))
    assert empty.additivity_error() == 0.0


# ------------------------------------------------------------------ factors


def test_sigmoid_is_stable_at_extremes() -> None:
    assert sigmoid(0.0) == 0.5
    assert sigmoid(800.0) == 1.0
    assert sigmoid(-800.0) == 0.0
    assert sigmoid(-2.0) == pytest.approx(1 / (1 + math.exp(2.0)))


@pytest.mark.parametrize(
    ("feature", "value", "expected"),
    [
        ("is_renewal", 1.0, "yes"),
        ("email_disposable", 0.0, "no"),
        ("amount_usd", 1234.5, "$1,234.50"),
        ("declines_24h", 4.0, "4"),
        ("account_age_days", 2.25, "2.2 days"),
        ("days_since_prev_attempt", 0.04, "0.0 days"),
        ("unknown_feature", 0.123456, "0.1235"),
    ],
)
def test_format_value_uses_natural_units(feature: str, value: float, expected: str) -> None:
    assert format_value(feature, value) == expected


def test_probability_and_multiplier_formatting() -> None:
    assert format_probability(0.12345) == "12.3%"
    assert format_probability(0.99999) == ">99.9%"
    assert format_probability(0.00001) == "<0.1%"
    assert format_probability(0.0) == "0.0%"
    assert format_multiplier(2.345) == "×2.35"
    assert format_multiplier(1668.2) == "×1,668"


def test_top_factors_ranks_by_magnitude_and_drops_noise() -> None:
    names = ("a", "b", "c", "d")
    phi = np.array([0.2, -0.9, 0.01, 0.9])
    x = np.array([1.0, 2.0, 3.0, 4.0])
    factors = top_factors(phi, x, names, k=3)
    assert [f.feature for f in factors] == ["b", "d", "a"]  # tie broken by feature order
    assert top_factors(phi, x, names, k=10, min_contribution=0.5) == factors[:2]
    assert top_factors(phi, x, names, k=0) == ()
    with pytest.raises(ValueError, match="k must be"):
        top_factors(phi, x, names, k=-1)


def test_factor_properties() -> None:
    up = Factor("declines_24h", 4.0, math.log(3.0))
    down = Factor("account_age_days", 800.0, -0.5)
    assert up.direction is Direction.RAISES
    assert down.direction is Direction.LOWERS
    assert up.odds_multiplier == pytest.approx(3.0)
    assert "card-testing" in up.description
    assert Factor("unknown", 1.0, 0.1).description == ""
    assert up.render() == "declines_24h = 4 (×3.00 odds)"


def test_explain_rows_builds_summaries_that_add_up() -> None:
    names = ("declines_24h", "account_age_days", "amount_usd")
    values = np.array([[1.2, -0.4, 0.02], [0.0, 0.0, 0.0]])
    base = -4.0
    sv = ShapValues(values, base, base + values.sum(axis=1), names)
    X = np.array([[5.0, 900.0, 10.0], [0.0, 30.0, 10.0]])
    risky, plain = explain_rows(sv, X, ["txn_a", "txn_b"], baseline_probability=0.02, k=3)

    assert risky.probability == pytest.approx(sigmoid(base + 0.82))
    assert [f.feature for f in risky.factors] == ["declines_24h", "account_age_days"]
    assert risky.rest == pytest.approx(0.02)  # the sub-threshold amount
    assert risky.base_value + sum(f.contribution for f in risky.factors) + risky.rest == (
        pytest.approx(math.log(risky.probability / (1 - risky.probability)))
    )
    text = risky.summary()
    assert text.startswith("txn_a: dispute risk")
    assert "Raised by declines_24h = 5" in text
    assert "Lowered by account_age_days = 900.0 days" in text
    assert plain.factors == ()
    assert "No single factor stands out." in plain.summary()

    payload = risky.to_dict()
    json.dumps(payload)
    factors = payload["factors"]
    assert isinstance(factors, list)
    assert factors[0]["direction"] == "raises"
    assert factors[0]["display_value"] == "5"

    with pytest.raises(ValueError, match="same number of rows"):
        explain_rows(sv, X[:1], ["x"], baseline_probability=0.02)


def test_relative_risk_handles_zero_baseline() -> None:
    e = TransactionExplanation("t", 0.1, 0.0, -2.0, (), 0.0)
    assert math.isnan(e.relative_risk)


def test_global_importance_ranks_and_flags_dominant_features() -> None:
    values = np.array([[0.1, 6.0, -0.3], [-0.1, 0.0, 0.3]])
    sv = ShapValues(values, 0.0, values.sum(axis=1), ("a", "b", "c"))
    ranked = global_importance(sv)
    assert [i.feature for i in ranked] == ["b", "c", "a"]
    assert sum(i.share for i in ranked) == pytest.approx(1.0)
    assert ranked[0].max_abs == pytest.approx(6.0)
    assert [i.feature for i in ranked if i.dominant] == ["b"]
    empty = global_importance(ShapValues(np.zeros((0, 2)), 0.0, np.zeros(0), ("a", "b")))
    assert all(i.mean_abs == 0.0 and not i.dominant for i in empty)


# ------------------------------------------------------- end-to-end on synthetic data


@pytest.fixture(scope="module")
def matrix() -> FeatureMatrix:
    return FeatureMatrix.from_table(build_features(generate(GeneratorConfig(n_customers=2_000))))


@pytest.fixture(scope="module")
def fitted(matrix: FeatureMatrix) -> dict[ModelKind, Any]:
    split = time_split(matrix)
    return {k: build_model(k).fit(split.train.X, split.train.y) for k in ModelKind}


@pytest.mark.parametrize("kind", list(ModelKind))
def test_explain_test_window_end_to_end(
    kind: ModelKind, matrix: FeatureMatrix, fitted: dict[ModelKind, Any]
) -> None:
    model = fitted[kind]
    report = explain_test_window(
        model, matrix, ExplainConfig(kind=kind, background_size=150, top_risky=5)
    )
    split = time_split(matrix)
    assert len(report.shap) == len(split.test)
    assert report.shap.additivity_error() < 1e-9
    # SHAP log-odds reproduce the model's probabilities exactly.
    np.testing.assert_allclose(
        [sigmoid(r) for r in report.shap.raw], predict_risk(model, split.test.X), atol=1e-12
    )
    assert len(report.riskiest) == 5
    probs = [e.probability for e in report.riskiest]
    assert probs == sorted(probs, reverse=True)
    assert probs[0] == pytest.approx(predict_risk(model, split.test.X).max())
    assert all(e.relative_risk > 1 for e in report.riskiest)
    assert all(e.factors for e in report.riskiest)
    # The planted patterns should surface among the most important features.
    top = {i.feature for i in report.importance[:8]}
    assert top & {"prior_approved_renewals", "account_age_days", "is_one_off", "amount_usd"}

    md = render_markdown(report)
    assert md.startswith(f"# Explainability report: {kind.label}")
    assert "## Global importance" in md


def test_background_sample_is_seeded_training_data(matrix: FeatureMatrix) -> None:
    train = time_split(matrix).train
    a = background_sample(train, 50, seed=1)
    b = background_sample(train, 50, seed=1)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (50, train.X.shape[1])
    assert len(background_sample(train, len(train) + 10, seed=0)) == len(train)


def test_explain_transactions_matches_window_explanations(
    matrix: FeatureMatrix, fitted: dict[ModelKind, Any]
) -> None:
    model = fitted[ModelKind.LOGREG]
    cfg = ExplainConfig(background_size=100, top_risky=1)
    window = explain_test_window(model, matrix, cfg)
    riskiest = window.riskiest[0]
    (single,) = explain_transactions(model, matrix, [riskiest.transaction_id], cfg)
    assert single.probability == pytest.approx(riskiest.probability)
    assert single.factors == riskiest.factors
    with pytest.raises(KeyError):
        explain_transactions(model, matrix, ["txn_nope"], cfg)


def test_explain_config_validates() -> None:
    with pytest.raises(ValueError, match="background_size"):
        ExplainConfig(background_size=0)
    with pytest.raises(ValueError, match="top_k"):
        ExplainConfig(top_k=-1)


def test_write_artifacts_and_dominance_warning(tmp_path: Path) -> None:
    names = ("attempts_1h", "amount_usd")
    values = np.array([[9.0, 0.1], [0.0, -0.1]])
    sv = ShapValues(values, -4.0, -4.0 + values.sum(axis=1), names)
    riskiest = explain_rows(
        sv, np.array([[2.0, 5.0], [0.0, 5.0]]), ["t1", "t2"], baseline_probability=0.02
    )
    from chargeback_prediction.explain import ExplanationReport

    report = ExplanationReport(
        config=ExplainConfig(),
        shap=sv,
        importance=global_importance(sv),
        riskiest=tuple(riskiest),
        baseline_probability=0.02,
        background_rows=10,
    )
    md = render_markdown(report)
    assert "## Warnings: single-feature dominance" in md
    assert "| `attempts_1h` | 9.00 | ×8,103 |" in md

    paths = write_artifacts(report, tmp_path / "out")
    lines = paths["explanations"].read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["transaction_id"] for line in lines] == ["t1", "t2"]
    importance = json.loads(paths["importance"].read_text(encoding="utf-8"))
    assert importance["features"][0] == {
        "feature": "attempts_1h",
        "mean_abs_shap": 4.5,
        "max_abs_shap": 9.0,
        "share": pytest.approx(4.5 / 4.6),
        "dominant": True,
    }
    assert paths["report"].name == "report_logreg.md"


def test_load_model_roundtrip(
    tmp_path: Path, toy: tuple[FloatArray, npt.NDArray[np.bool_]]
) -> None:
    X, y = toy
    model = build_model(ModelKind.GBM).fit(X, y)
    joblib.dump(model, tmp_path / "model_gbm.joblib")
    loaded = load_model(tmp_path, ModelKind.GBM)
    np.testing.assert_allclose(predict_risk(loaded, X[:5]), predict_risk(model, X[:5]))
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path, ModelKind.LOGREG)


# ------------------------------------------------------------------ CLI


def test_explain_command_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data, features = tmp_path / "synthetic", tmp_path / "features.csv"
    models, out = tmp_path / "models", tmp_path / "explain"
    assert main(["generate", "--customers", "1500", "--seed", "11", "--out", str(data)]) == 0
    assert main(["features", "--data", str(data), "--out", str(features)]) == 0
    assert main(["train", "--features", str(features), "--out", str(models)]) == 0
    capsys.readouterr()

    args = ["explain", "--features", str(features), "--models", str(models)]
    assert main([*args, "--model", "gbm", "--out", str(out), "--top-risky", "4"]) == 0
    printed = capsys.readouterr().out
    assert "Gradient boosting: explained" in printed
    assert "highest-risk test transactions" in printed
    assert sorted(p.name for p in out.iterdir()) == [
        "explanations_gbm.jsonl",
        "importance_gbm.json",
        "report_gbm.md",
    ]
    first = json.loads((out / "explanations_gbm.jsonl").read_text().splitlines()[0])

    assert main([*args, "--transaction-id", first["transaction_id"], "--top-factors", "2"]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith(f"{first['transaction_id']}: dispute risk")

    with pytest.raises(SystemExit) as exc:
        main([*args, "--transaction-id", "txn_missing"])
    assert exc.value.code == 2
    assert "unknown transaction id 'txn_missing'" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:
        main([*args, "--background", "0"])
    assert exc.value.code == 2
    assert "background_size" in capsys.readouterr().err


def test_explain_command_reports_missing_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["explain", "--models", str(tmp_path)])
    assert exc.value.code == 2
    assert "run `cbp train` first" in capsys.readouterr().err

    joblib.dump(build_model(ModelKind.LOGREG), tmp_path / "model_logreg.joblib")
    with pytest.raises(SystemExit) as exc:
        main(["explain", "--models", str(tmp_path), "--features", str(tmp_path / "nope.csv")])
    assert exc.value.code == 2
    assert "run `cbp features` first" in capsys.readouterr().err

    bad = tmp_path / "bad.csv"
    bad.write_text("transaction_id\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["explain", "--models", str(tmp_path), "--features", str(bad)])
    assert exc.value.code == 2
    assert "missing columns" in capsys.readouterr().err
