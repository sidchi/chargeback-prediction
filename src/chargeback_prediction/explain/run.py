"""Explain a trained baseline on the held-out window and write the explainability artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from chargeback_prediction.explain.factors import (
    DEFAULT_TOP_K,
    Importance,
    TransactionExplanation,
    explain_rows,
    global_importance,
)
from chargeback_prediction.explain.shapley import ShapValues, shap_values
from chargeback_prediction.modeling import (
    Classifier,
    FeatureMatrix,
    ModelKind,
    predict_risk,
    time_split,
)
from chargeback_prediction.modeling.train import TrainConfig

DEFAULT_BACKGROUND_SIZE = 300


def model_path(models_dir: Path, kind: ModelKind) -> Path:
    """Where ``cbp train`` saves a model of this kind."""
    return models_dir / f"model_{kind.value}.joblib"


def load_model(models_dir: Path, kind: ModelKind) -> Classifier:
    """Load a model written by ``cbp train``.

    joblib uses pickle: only load artifacts you produced yourself (see the item-4 PR notes).
    """
    path = model_path(models_dir, kind)
    if not path.exists():
        raise FileNotFoundError(path)
    model: Classifier = joblib.load(path)
    return model


@dataclass(frozen=True, slots=True)
class ExplainConfig:
    kind: ModelKind = ModelKind.LOGREG
    top_k: int = DEFAULT_TOP_K
    background_size: int = DEFAULT_BACKGROUND_SIZE
    top_risky: int = 20
    """How many of the highest-scored test rows get a written explanation."""
    seed: int = 0
    split: TrainConfig = field(default_factory=TrainConfig)
    """Must match the split the model was trained with, so the background is training data."""

    def __post_init__(self) -> None:
        if self.background_size < 1:
            raise ValueError("background_size must be >= 1")
        if self.top_k < 0 or self.top_risky < 0:
            raise ValueError("top_k and top_risky must be >= 0")


@dataclass(frozen=True, slots=True)
class ExplanationReport:
    config: ExplainConfig
    shap: ShapValues = field(repr=False)
    """Attributions for every row of the test window."""
    importance: tuple[Importance, ...]
    riskiest: tuple[TransactionExplanation, ...]
    baseline_probability: float
    background_rows: int


def background_sample(train: FeatureMatrix, size: int, *, seed: int) -> np.ndarray:
    """Uniform sample (without replacement) of training rows, in time order."""
    n = len(train)
    if size >= n:
        return train.X.copy()
    idx = np.sort(np.random.default_rng(seed).choice(n, size=size, replace=False))
    return train.X[idx]


def explain_test_window(
    model: Classifier, matrix: FeatureMatrix, config: ExplainConfig | None = None
) -> ExplanationReport:
    """SHAP for every test row; written explanations for the ``top_risky`` highest scores."""
    cfg = config or ExplainConfig()
    split = time_split(
        matrix, test_fraction=cfg.split.test_fraction, label_gap_days=cfg.split.label_gap_days
    )
    background = background_sample(split.train, cfg.background_size, seed=cfg.seed)
    baseline_p = float(predict_risk(model, background).mean())
    test = split.test
    shap = shap_values(model, test.X, background, test.feature_names)
    order = np.argsort(-shap.raw, kind="stable")[: cfg.top_risky]
    riskiest = explain_rows(
        ShapValues(shap.values[order], shap.base_value, shap.raw[order], shap.feature_names),
        test.X[order],
        [test.transaction_ids[i] for i in order],
        baseline_probability=baseline_p,
        k=cfg.top_k,
    )
    return ExplanationReport(
        config=cfg,
        shap=shap,
        importance=global_importance(shap),
        riskiest=tuple(riskiest),
        baseline_probability=baseline_p,
        background_rows=len(background),
    )


def explain_transactions(
    model: Classifier,
    matrix: FeatureMatrix,
    transaction_ids: Sequence[str],
    config: ExplainConfig | None = None,
) -> list[TransactionExplanation]:
    """Explain specific transactions (any window) against a training-window background.

    Raises:
        KeyError: naming the first id not found in ``matrix``.
    """
    cfg = config or ExplainConfig()
    index = {txn: i for i, txn in enumerate(matrix.transaction_ids)}
    rows = []
    for txn in transaction_ids:
        if txn not in index:
            raise KeyError(txn)
        rows.append(index[txn])
    split = time_split(
        matrix, test_fraction=cfg.split.test_fraction, label_gap_days=cfg.split.label_gap_days
    )
    background = background_sample(split.train, cfg.background_size, seed=cfg.seed)
    X = matrix.X[rows]
    shap = shap_values(model, X, background, matrix.feature_names)
    return explain_rows(
        shap,
        X,
        list(transaction_ids),
        baseline_probability=float(predict_risk(model, background).mean()),
        k=cfg.top_k,
    )


def render_markdown(report: ExplanationReport, *, top_features: int = 15) -> str:
    cfg = report.config
    lines = [
        f"# Explainability report: {cfg.kind.label}",
        "",
        "<!-- Generated by `cbp explain`. Synthetic data only. -->",
        "",
        f"Exact interventional SHAP values in log-odds for {len(report.shap):,} test-window "
        f"transactions, against a background of {report.background_rows:,} training-window "
        f"rows (mean predicted risk {report.baseline_probability:.2%}). Local accuracy holds to "
        f"{report.shap.additivity_error():.1e} log-odds.",
        "",
        "## Global importance (mean |SHAP|)",
        "",
        "| # | Feature | Mean \\|SHAP\\| (log-odds) | Share |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {i} | `{imp.feature}` | {imp.mean_abs:.3f} | {imp.share:.1%} |"
        for i, imp in enumerate(report.importance[:top_features], start=1)
    ]
    dominant = sorted((imp for imp in report.importance if imp.dominant), key=lambda i: -i.max_abs)
    if dominant:
        lines += [
            "",
            "## Warnings: single-feature dominance",
            "",
            "On at least one row, each of these features alone multiplies the odds of a dispute "
            "by 100× or more. That usually means a sparse feature seen only a handful of times "
            "in training (near separation), so its weight is extrapolated and scores driven by "
            "it are likely overconfident.",
            "",
            "| Feature | Max \\|SHAP\\| (log-odds) | Max odds multiplier |",
            "|---|---|---|",
        ]
        lines += [
            f"| `{imp.feature}` | {imp.max_abs:.2f} | ×{math.exp(imp.max_abs):,.0f} |"
            for imp in dominant
        ]
    lines += [
        "",
        f"## The {len(report.riskiest)} highest-risk test transactions",
        "",
        "Odds multipliers compose: ×2 and ×3 together mean ×6 the odds of a typical transaction.",
        "",
    ]
    lines += [f"- {e.summary()}" for e in report.riskiest]
    return "\n".join(lines) + "\n"


def write_artifacts(report: ExplanationReport, out_dir: Path) -> dict[str, Path]:
    """Write ``explanations.jsonl`` (top-risk rows), ``importance.json`` and ``report.md``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = report.config.kind.value
    jsonl = out_dir / f"explanations_{kind}.jsonl"
    jsonl.write_text(
        "".join(json.dumps(e.to_dict()) + "\n" for e in report.riskiest), encoding="utf-8"
    )
    importance = out_dir / f"importance_{kind}.json"
    importance.write_text(
        json.dumps(
            {
                "model": kind,
                "base_log_odds": report.shap.base_value,
                "baseline_probability": report.baseline_probability,
                "background_rows": report.background_rows,
                "explained_rows": len(report.shap),
                "additivity_error": report.shap.additivity_error(),
                "features": [
                    {
                        "feature": i.feature,
                        "mean_abs_shap": i.mean_abs,
                        "max_abs_shap": i.max_abs,
                        "share": i.share,
                        "dominant": i.dominant,
                    }
                    for i in report.importance
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    md = out_dir / f"report_{kind}.md"
    md.write_text(render_markdown(report), encoding="utf-8")
    return {"explanations": jsonl, "importance": importance, "report": md}
