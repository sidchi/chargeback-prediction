"""Command-line entry point (``cbp``).

Subcommands are added as pipeline stages land (generate, train, score, explain).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from chargeback_prediction import __version__
from chargeback_prediction.reason_codes import all_reason_codes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cbp",
        description="Score card transactions for chargeback risk (synthetic data only).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("reason-codes", help="List the dispute reason codes the pipeline models.")
    return parser


def _cmd_reason_codes() -> int:
    for rc in all_reason_codes():
        print(f"{rc.key:<16} {rc.category:<10} {rc.title}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = _build_parser().parse_args(argv)
    if args.command == "reason-codes":
        return _cmd_reason_codes()
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
