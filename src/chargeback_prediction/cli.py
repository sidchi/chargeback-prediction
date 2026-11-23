"""Command-line entry point (``cbp``).

Subcommands are added as pipeline stages land (train, score, explain to follow).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from chargeback_prediction import __version__
from chargeback_prediction.reason_codes import all_reason_codes
from chargeback_prediction.synthetic import GeneratorConfig, generate, write_dataset


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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "reason-codes":
        return _cmd_reason_codes()
    if args.command == "generate":
        return _cmd_generate(args, parser)
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
