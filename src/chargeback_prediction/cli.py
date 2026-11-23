"""Command-line entry point (``cbp``).

Subcommands are added as pipeline stages land (score to follow):

``cbp generate``  write a synthetic dataset as CSV tables
``cbp features``  build the point-in-time feature table from those CSVs
``cbp train``     fit baseline models on a time-based split and write an evaluation report
``cbp explain``   SHAP values and top risk factors per transaction for a trained model
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from chargeback_prediction import __version__
from chargeback_prediction.explain.factors import DEFAULT_TOP_K
from chargeback_prediction.explain.run import DEFAULT_BACKGROUND_SIZE
from chargeback_prediction.features import build_features, render_feature_table
from chargeback_prediction.modeling import FeatureMatrix, ModelKind, TrainConfig, train_baselines
from chargeback_prediction.reason_codes import all_reason_codes
from chargeback_prediction.synthetic import GeneratorConfig, generate, read_dataset, write_dataset


def _utc_date(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value).replace(tzinfo=UTC)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}, expected YYYY-MM-DD") from None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cbp",
        description="Score card transactions for chargeback risk (synthetic data only).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("reason-codes", help="List the dispute reason codes the pipeline models.")

    defaults = GeneratorConfig()
    gen = sub.add_parser("generate", help="Generate a synthetic dataset as CSV files.")
    gen.add_argument("--out", type=Path, default=Path("data/synthetic"), help="output directory")
    gen.add_argument("--customers", type=int, default=defaults.n_customers)
    gen.add_argument("--seed", type=int, default=defaults.seed)
    gen.add_argument(
        "--start",
        type=_utc_date,
        default=defaults.start,
        help="start of the observation window, YYYY-MM-DD (UTC)",
    )
    gen.add_argument("--days", type=int, default=defaults.days, help="length of the window")

    feat = sub.add_parser(
        "features", help="Build the point-in-time feature table from a generated dataset."
    )
    feat.add_argument(
        "--data", type=Path, default=Path("data/synthetic"), help="directory of dataset CSVs"
    )
    feat.add_argument(
        "--out", type=Path, default=Path("data/features.csv"), help="feature table CSV to write"
    )
    feat.add_argument(
        "--doc",
        action="store_true",
        help="print the documented feature table (docs/features.md) instead of building",
    )

    train_defaults = TrainConfig()
    train = sub.add_parser(
        "train", help="Train baseline models on a time-based split and write an evaluation report."
    )
    train.add_argument(
        "--features", type=Path, default=Path("data/features.csv"), help="feature table CSV"
    )
    train.add_argument(
        "--out", type=Path, default=Path("artifacts/baseline"), help="artifact directory"
    )
    train.add_argument(
        "--test-fraction",
        type=float,
        default=train_defaults.test_fraction,
        help="share of mature rows (newest) held out for testing",
    )
    train.add_argument(
        "--gap-days",
        type=int,
        default=train_defaults.label_gap_days,
        help="days between train end and test start, so training labels are final",
    )
    train.add_argument(
        "--model",
        dest="models",
        action="append",
        choices=[k.value for k in ModelKind],
        help="model to train (repeatable; default: all)",
    )
    train.add_argument("--seed", type=int, default=train_defaults.seed)

    exp = sub.add_parser(
        "explain",
        help="Explain a trained model: global SHAP importance and top factors per transaction.",
    )
    exp.add_argument(
        "--features", type=Path, default=Path("data/features.csv"), help="feature table CSV"
    )
    exp.add_argument(
        "--models", type=Path, default=Path("artifacts/baseline"), help="`cbp train` output dir"
    )
    exp.add_argument(
        "--model",
        choices=[k.value for k in ModelKind],
        default=ModelKind.LOGREG.value,
        help="which trained model to explain",
    )
    exp.add_argument(
        "--transaction-id",
        dest="transaction_ids",
        action="append",
        metavar="ID",
        help="explain this transaction and print its summary (repeatable); "
        "default: explain the whole test window and write a report",
    )
    exp.add_argument("--top-factors", type=int, default=DEFAULT_TOP_K, help="factors per row")
    exp.add_argument("--top-risky", type=int, default=20, help="highest-risk test rows to write up")
    exp.add_argument(
        "--background",
        type=int,
        default=DEFAULT_BACKGROUND_SIZE,
        help="training rows sampled as the SHAP background",
    )
    exp.add_argument(
        "--out", type=Path, default=Path("artifacts/explain"), help="artifact directory"
    )
    exp.add_argument("--seed", type=int, default=0)
    return parser


def _cmd_reason_codes() -> int:
    for rc in all_reason_codes():
        print(f"{rc.key:<16} {rc.category:<10} {rc.title}")
    return 0


def _cmd_generate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        config = GeneratorConfig(
            n_customers=args.customers, seed=args.seed, start=args.start, days=args.days
        )
    except ValueError as exc:
        parser.error(str(exc))
    dataset = generate(config)
    paths = write_dataset(dataset, args.out)
    print(dataset.summary().render())
    print(f"\nwrote {len(paths)} tables to {args.out}/")
    return 0


def _cmd_features(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.doc:
        print(render_feature_table(), end="")
        return 0
    try:
        dataset = read_dataset(args.data)
    except FileNotFoundError as exc:
        parser.error(f"{exc.filename} not found; run `cbp generate` first")
    except ValueError as exc:
        parser.error(str(exc))
    table = build_features(dataset)
    table.write_csv(args.out)
    mature = [row for row in table if row.label_mature]
    disputed = sum(row.disputed for row in mature)
    rate = disputed / len(mature) if mature else 0.0
    print(f"rows            {len(table):>8}  ({len(table.columns)} features)")
    print(f"mature labels   {len(mature):>8}  ({disputed} disputed, {rate:.2%})")
    print(f"observed until  {table.observed_until.isoformat()}")
    print(f"\nwrote {args.out}")
    return 0


def _cmd_train(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    # Imported lazily: matplotlib adds noticeable startup time the other commands don't need.
    from chargeback_prediction.modeling.report import write_artifacts

    try:
        matrix = FeatureMatrix.read_csv(args.features)
    except FileNotFoundError:
        parser.error(f"{args.features} not found; run `cbp features` first")
    except ValueError as exc:
        parser.error(str(exc))
    try:
        config = TrainConfig(
            models=tuple(ModelKind(m) for m in (args.models or [k.value for k in ModelKind])),
            test_fraction=args.test_fraction,
            label_gap_days=args.gap_days,
            seed=args.seed,
        )
        report = train_baselines(matrix, config)
    except ValueError as exc:
        parser.error(str(exc))
    paths = write_artifacts(report, args.out)

    split, base = report.split, report.test_base_rate
    print(f"train  {len(split.train):>7} rows  {int(split.train.y.sum()):>5} disputes")
    print(f"test   {len(split.test):>7} rows  {int(split.test.y.sum()):>5} disputes  ({base:.2%})")
    print()
    top = report.evaluations[0].at_k[-1].budget
    print(f"{'model':<22}{'PR-AUC':>8}{'lift':>7}{'Brier':>9}{'ECE':>8}  P@{top:.0%}")
    for ev in report.evaluations:
        print(
            f"{ev.kind.label:<22}{ev.pr_auc:>8.3f}{ev.pr_auc / base:>6.1f}x"
            f"{ev.brier:>9.4f}{ev.ece:>8.4f}  {ev.at_k[-1].precision:.1%}"
        )
    print(
        f"\nwrote {paths['report']}, {paths['figure'].name}, metrics.json and models to {args.out}/"
    )
    return 0


def _cmd_explain(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from chargeback_prediction.explain import (
        ExplainConfig,
        explain_test_window,
        explain_transactions,
        load_model,
    )
    from chargeback_prediction.explain.run import write_artifacts

    kind = ModelKind(args.model)
    try:
        model = load_model(args.models, kind)
    except FileNotFoundError as exc:
        parser.error(f"{exc} not found; run `cbp train` first")
    try:
        matrix = FeatureMatrix.read_csv(args.features)
    except FileNotFoundError:
        parser.error(f"{args.features} not found; run `cbp features` first")
    except ValueError as exc:
        parser.error(str(exc))
    try:
        config = ExplainConfig(
            kind=kind,
            top_k=args.top_factors,
            background_size=args.background,
            top_risky=args.top_risky,
            seed=args.seed,
        )
        if args.transaction_ids:
            for explanation in explain_transactions(model, matrix, args.transaction_ids, config):
                print(explanation.summary())
            return 0
        report = explain_test_window(model, matrix, config)
    except KeyError as exc:
        parser.error(f"unknown transaction id {exc.args[0]!r}")
    except ValueError as exc:
        parser.error(str(exc))
    paths = write_artifacts(report, args.out)

    print(f"{kind.label}: explained {len(report.shap):,} test rows ", end="")
    print(f"(additivity error {report.shap.additivity_error():.1e} log-odds)\n")
    print(f"{'feature':<30}{'mean |SHAP|':>12}{'share':>8}")
    for imp in report.importance[:10]:
        print(f"{imp.feature:<30}{imp.mean_abs:>12.3f}{imp.share:>8.1%}")
    for imp in (i for i in report.importance if i.dominant):
        print(
            f"warning: {imp.feature} alone moves some rows' odds by ×{math.exp(imp.max_abs):,.0f}"
        )
    if report.riskiest:
        print("\nhighest-risk test transactions:")
        for explanation in report.riskiest[:3]:
            print(f"  {explanation.summary()}")
    print(f"\nwrote {paths['report'].name}, {paths['explanations'].name}, ", end="")
    print(f"{paths['importance'].name} to {args.out}/")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "reason-codes":
        return _cmd_reason_codes()
    if args.command == "generate":
        return _cmd_generate(args, parser)
    if args.command == "features":
        return _cmd_features(args, parser)
    if args.command == "train":
        return _cmd_train(args, parser)
    if args.command == "explain":
        return _cmd_explain(args, parser)
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
