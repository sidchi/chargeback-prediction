from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chargeback_prediction.evaluation import (
    CostModel,
    EvaluationConfig,
    EvaluationReport,
    Interval,
    PolicyKind,
    bootstrap,
    cost_curve,
    dispute_categories,
    expected_cost_decisions,
    expected_cost_savings_per_1k,
    hindsight_threshold_decisions,
    policy_outcome,
    read_dispute_categories,
    run_evaluation,
    top_k_decisions,
)
from chargeback_prediction.evaluation.report import (
    render_cost_report,
    render_model_card,
    report_to_dict,
    write_artifacts,
)
from chargeback_prediction.features import build_features
from chargeback_prediction.modeling import FeatureMatrix, ModelKind, TrainConfig, average_precision
from chargeback_prediction.synthetic import GeneratorConfig, generate

# Round numbers so the expected dollars are easy to check by hand.
COSTS = CostModel(
    dispute_fee_usd=20.0,
    ops_cost_per_dispute_usd=10.0,
    representment_recovery_rate=0.5,
    gross_margin=0.5,
    false_decline_churn_rate=0.1,
    customer_lifetime_value_usd=100.0,
    action_cost_usd=1.0,
)

# ------------------------------------------------------------------------ CostModel


def test_unit_costs_follow_the_documented_formulas() -> None:
    # chargeback: 100 × (1 − 0.5) + 20 + 10 = 80;  false decline: 100 × 0.5 + 0.1 × 100 = 60
    assert COSTS.chargeback_cost([100.0]).tolist() == [80.0]
    assert COSTS.false_decline_cost([100.0]).tolist() == [60.0]
    # block iff p·80 > (1 − p)·60 + 1  ⇔  p > 61 / 140
    assert COSTS.break_even_probability([100.0]).tolist() == pytest.approx([61 / 140])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dispute_fee_usd", -1.0),
        ("action_cost_usd", -0.01),
        ("gross_margin", 1.5),
        ("representment_recovery_rate", -0.1),
        ("false_decline_churn_rate", 2.0),
    ],
)
def test_cost_model_rejects_out_of_range_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        CostModel(**{field: value})


def test_scaled_multiplies_false_decline_cost_and_caps_margin() -> None:
    doubled = COSTS.scaled(2.0)
    assert doubled.false_decline_cost([100.0]).tolist() == [120.0]
    assert doubled.chargeback_cost([100.0]).tolist() == [80.0]
    assert COSTS.scaled(4.0).gross_margin == 1.0
    with pytest.raises(ValueError, match="multiplier"):
        COSTS.scaled(0.0)


# ------------------------------------------------------------------------- policies


def test_policy_outcome_prices_each_confusion_cell() -> None:
    y = np.array([True, True, False, False, False])
    amounts = np.array([100.0, 100.0, 100.0, 100.0, 50.0])
    blocked = np.array([True, False, True, False, False])
    o = policy_outcome(PolicyKind.EXPECTED_COST, y, amounts, blocked, COSTS)
    assert (o.n, o.blocked, o.disputes, o.disputes_blocked) == (5, 2, 2, 1)
    assert (o.false_declines, o.disputes_approved) == (1, 1)
    assert o.chargeback_cost == 80.0
    assert o.false_decline_cost == 60.0
    assert o.action_cost == 2.0
    assert o.total_cost == 142.0
    assert o.cost_per_1k == pytest.approx(142.0 * 200)
    assert o.residual_dispute_ratio == pytest.approx(1 / 3)
    approve_all = policy_outcome(PolicyKind.APPROVE_ALL, y, amounts, np.zeros(5, bool), COSTS)
    assert o.savings_vs(approve_all) == pytest.approx(160.0 - 142.0)


def test_policy_outcome_validates_inputs() -> None:
    with pytest.raises(ValueError, match="equal length"):
        policy_outcome(PolicyKind.APPROVE_ALL, [True], [1.0, 2.0], [False], COSTS)
    with pytest.raises(ValueError, match="at least one row"):
        policy_outcome(PolicyKind.APPROVE_ALL, [], [], [], COSTS)
    with pytest.raises(ValueError, match="non-negative"):
        policy_outcome(PolicyKind.APPROVE_ALL, [True], [-1.0], [False], COSTS)


def test_blocking_everything_with_full_residual_ratio_edge_case() -> None:
    o = policy_outcome(PolicyKind.REVIEW_BUDGET, [True], [10.0], [True], COSTS)
    assert o.residual_dispute_ratio == 0.0


def test_expected_cost_rule_switches_at_the_break_even_probability() -> None:
    p_star = 61 / 140
    p = np.array([p_star - 1e-9, p_star + 1e-9, 0.0, 1.0])
    blocked = expected_cost_decisions(p, np.full(4, 100.0), COSTS)
    assert blocked.tolist() == [False, True, False, True]


def test_expected_cost_rule_is_amount_aware() -> None:
    # Same probability; the fixed fees make a dispute on a small charge relatively costlier.
    blocked = expected_cost_decisions([0.4, 0.4], [5.0, 500.0], COSTS)
    assert blocked.tolist() == [True, False]


def test_expected_cost_rule_validates_probabilities() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        expected_cost_decisions([1.2], [10.0], COSTS)
    with pytest.raises(ValueError, match="equal shape"):
        expected_cost_decisions([0.1, 0.2], [10.0], COSTS)


def test_top_k_breaks_ties_by_row_order() -> None:
    assert top_k_decisions([0.5, 0.9, 0.5, 0.1], 2).tolist() == [True, True, False, False]
    assert not top_k_decisions([0.5], 0).any()
    with pytest.raises(ValueError, match="k must be"):
        top_k_decisions([0.5], 2)


def _random_case(seed: int, n: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.1
    scores = np.round(np.clip(0.1 * y + rng.random(n) * 0.5, 0, 1), 2)  # rounding → ties
    amounts = rng.choice([9.99, 14.99, 49.0, 120.0], size=n)
    return y, scores, amounts


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cost_curve_matches_brute_force(seed: int) -> None:
    y, scores, amounts = _random_case(seed)
    curve = cost_curve(y, scores, amounts, COSTS)
    assert curve.blocked.tolist() == list(range(len(y) + 1))
    for k in (0, 1, 7, 50, len(y)):
        expected = policy_outcome(
            PolicyKind.REVIEW_BUDGET, y, amounts, top_k_decisions(scores, k), COSTS
        ).total_cost
        assert curve.total_cost[k] == pytest.approx(expected)
    assert curve.thresholds[0] == np.inf


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hindsight_threshold_is_the_cheapest_reachable_threshold(seed: int) -> None:
    y, scores, amounts = _random_case(seed)
    blocked, threshold = hindsight_threshold_decisions(y, scores, amounts, COSTS)
    assert blocked.tolist() == (scores >= threshold).tolist()
    best = policy_outcome(PolicyKind.HINDSIGHT_THRESHOLD, y, amounts, blocked, COSTS).total_cost
    candidates = [np.inf, *np.unique(scores)]
    brute = min(
        policy_outcome(PolicyKind.HINDSIGHT_THRESHOLD, y, amounts, scores >= t, COSTS).total_cost
        for t in candidates
    )
    assert best == pytest.approx(brute)


def test_hindsight_threshold_never_splits_tied_scores() -> None:
    # Blocking only the first of two tied rows would be cheapest, but no threshold does that.
    y = np.array([True, False])
    blocked, _ = hindsight_threshold_decisions(y, [0.5, 0.5], [100.0, 100.0], COSTS)
    assert blocked[0] == blocked[1]


# ------------------------------------------------------------------------ bootstrap


def test_bootstrap_of_a_constant_is_degenerate() -> None:
    iv = bootstrap(lambda x: 3.0, (np.arange(10),), n_resamples=20)
    assert (iv.estimate, iv.low, iv.high) == (3.0, 3.0, 3.0)
    assert iv.excludes_zero()
    assert iv.format(".1f") == "3.0 [3.0, 3.0]"


def test_bootstrap_mean_interval_brackets_the_estimate_and_is_reproducible() -> None:
    x = np.random.default_rng(0).normal(10.0, 2.0, 400)
    a = bootstrap(lambda v: float(v.mean()), (x,), n_resamples=300, seed=1)
    b = bootstrap(lambda v: float(v.mean()), (x,), n_resamples=300, seed=1)
    assert a == b
    assert a.low < a.estimate < a.high
    # Standard error of the mean is 0.1, so a 95% interval is roughly ±0.2.
    assert 0.25 < a.high - a.low < 0.55


def test_stratified_bootstrap_keeps_the_positive_count() -> None:
    y = np.array([True] * 5 + [False] * 95)
    iv = bootstrap(lambda v: float(v.sum()), (y,), n_resamples=50, strata=y)
    assert (iv.low, iv.high) == (5.0, 5.0)
    loose = bootstrap(lambda v: float(v.sum()), (y,), n_resamples=200)
    assert loose.high > loose.low


def test_bootstrap_drops_nan_replicates() -> None:
    y = np.array([True] + [False] * 30)
    # Unstratified replicates often miss the only positive; PR-AUC is then nan and dropped.
    iv = bootstrap(average_precision, (y, np.linspace(1, 0, 31)), n_resamples=100)
    assert np.isfinite([iv.low, iv.high]).all()
    all_nan = bootstrap(lambda v: float("nan"), (y,), n_resamples=5)
    assert np.isnan(all_nan.low)


def test_bootstrap_validates_inputs() -> None:
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap(lambda v: 0.0, (np.arange(3),), n_resamples=0)
    with pytest.raises(ValueError, match="at least one array"):
        bootstrap(lambda: 0.0, ())
    with pytest.raises(ValueError, match="equal length"):
        bootstrap(lambda a, b: 0.0, (np.arange(3), np.arange(4)))
    with pytest.raises(ValueError, match="strata"):
        bootstrap(lambda v: 0.0, (np.arange(3),), strata=[1, 2])
    with pytest.raises(ValueError, match="level"):
        Interval(0.0, 0.0, 0.0, level=1.0)


# ------------------------------------------------------------------ end to end


@pytest.fixture(scope="module")
def report() -> EvaluationReport:
    table = build_features(generate(GeneratorConfig(n_customers=1_500, seed=11)))
    return run_evaluation(
        FeatureMatrix.from_table(table),
        categories=dispute_categories(table),
        config=EvaluationConfig(n_resamples=60),
    )


def test_report_covers_every_model_and_policy(report: EvaluationReport) -> None:
    assert [m.kind for m in report.models] == [ModelKind.LOGREG, ModelKind.GBM]
    base, perfect = report.approve_all, report.perfect
    assert base.blocked == 0
    assert base.false_decline_cost == 0.0
    assert perfect.total_cost == pytest.approx(base.disputes * report.config.costs.action_cost_usd)
    for m in report.models:
        assert set(m.policies) == {
            PolicyKind.EXPECTED_COST,
            PolicyKind.REVIEW_BUDGET,
            PolicyKind.HINDSIGHT_THRESHOLD,
        }
        # In hindsight, the best threshold is at least as cheap as approving everything.
        assert m.policies[PolicyKind.HINDSIGHT_THRESHOLD].total_cost <= base.total_cost
        assert m.pr_auc.estimate == pytest.approx(m.ranking.pr_auc)
        assert m.savings_per_1k.estimate == pytest.approx(
            base.cost_per_1k - m.expected_cost.cost_per_1k
        )
        assert sum(c.disputes for c in m.categories) == base.disputes
        assert sum(c.blocked for c in m.categories) == m.expected_cost.disputes_blocked
        assert [r.false_decline_multiplier for r in m.sensitivity] == [0.5, 1.0, 2.0, 4.0]
        blocked = [r.outcome.blocked for r in m.sensitivity]
        assert blocked == sorted(blocked, reverse=True)  # pricier false declines → block less
    assert report.recommended() in report.models


def test_paired_comparison_orders_models_by_pr_auc(report: EvaluationReport) -> None:
    (c,) = report.comparisons
    by_kind = {m.kind: m for m in report.models}
    assert by_kind[c.better].ranking.pr_auc >= by_kind[c.worse].ranking.pr_auc
    assert c.pr_auc_difference.estimate == pytest.approx(
        by_kind[c.better].ranking.pr_auc - by_kind[c.worse].ranking.pr_auc
    )


def test_savings_statistic_matches_policy_outcomes(report: EvaluationReport) -> None:
    m = report.models[0]
    y = report.baseline.split.test.y
    got = expected_cost_savings_per_1k(y, report.amounts, m.ranking.scores, report.config.costs)
    assert got == pytest.approx(m.savings_per_1k.estimate)


def test_rendered_reports_and_artifacts(report: EvaluationReport, tmp_path: Path) -> None:
    md = render_cost_report(report)
    for heading in (
        "## Cost assumptions",
        "## Headline",
        "## Model comparison",
        "blocking policies",
    ):
        assert heading in md
    assert "![" not in md
    card = render_model_card(report)
    for heading in ("## Intended use", "## Limitations", "## Ethical considerations"):
        assert heading in card
    assert report.recommended().kind.label.lower() in card

    paths = write_artifacts(report, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "cost_curve.png",
        "evaluation.json",
        "model_card.md",
        "report.md",
    ]
    assert paths["figure"].read_bytes().startswith(b"\x89PNG")
    assert "![Cost curves and sensitivity](cost_curve.png)" in paths["report"].read_text()
    data = json.loads(paths["metrics"].read_text())
    assert data == json.loads(json.dumps(report_to_dict(report)))
    assert data["recommended"] in {"logreg", "gbm"}
    assert set(data["models"]["logreg"]["policies"]) == {
        "expected_cost",
        "review_budget",
        "hindsight_threshold",
    }


def test_single_model_report_has_no_comparison() -> None:
    table = build_features(generate(GeneratorConfig(n_customers=800, seed=3)))
    r = run_evaluation(
        FeatureMatrix.from_table(table),
        train_config=TrainConfig(models=(ModelKind.LOGREG,)),
        config=EvaluationConfig(n_resamples=10, sensitivity=(1.0,)),
    )
    assert r.comparisons == ()
    # Without categories every dispute is "unknown".
    assert [c.category for c in r.models[0].categories] == ["unknown"]
    assert "Model comparison" not in render_cost_report(r)


def test_evaluation_config_validation() -> None:
    with pytest.raises(ValueError, match="review_budget"):
        EvaluationConfig(review_budget=0.0)
    with pytest.raises(ValueError, match="sensitivity"):
        EvaluationConfig(sensitivity=(1.0, -2.0))


def test_read_dispute_categories(tmp_path: Path) -> None:
    table = build_features(generate(GeneratorConfig(n_customers=300, seed=5)))
    path = tmp_path / "features.csv"
    table.write_csv(path)
    assert read_dispute_categories(path) == dispute_categories(table)
    assert set(dispute_categories(table).values()) <= {"consumer", "fraud"}
    bad = tmp_path / "bad.csv"
    bad.write_text("transaction_id\ntxn_1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dispute_category"):
        read_dispute_categories(bad)


def test_model_card_flags_an_under_predicting_lowest_bin(report: EvaluationReport) -> None:
    import dataclasses

    from chargeback_prediction.evaluation.report import _weakest_bin_note
    from chargeback_prediction.modeling import CalibrationBin

    m = report.models[0]

    def with_first_bin(observed: float) -> list[str]:
        first = CalibrationBin(0.0, 0.01, 100, mean_predicted=0.002, observed_rate=observed)
        ranking = dataclasses.replace(m.ranking, calibration=(first,))
        return _weakest_bin_note(dataclasses.replace(m, ranking=ranking))

    assert "3.6% against 0.2% predicted" in with_first_bin(0.036)[0]
    assert with_first_bin(0.003) == []
    no_bins = dataclasses.replace(m, ranking=dataclasses.replace(m.ranking, calibration=()))
    assert _weakest_bin_note(no_bins) == []
