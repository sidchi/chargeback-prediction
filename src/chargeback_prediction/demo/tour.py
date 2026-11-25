"""A narrated, end-to-end run of the ``cbp`` pipeline on a fresh synthetic dataset.

The tour runs the real commands, in-process and in pipeline order, inside a working directory,
so every path in the output is the relative default a user would type:

``generate`` → ``features`` → ``train`` → ``evaluate`` → ``explain`` → ``reason`` → ``evidence``

Each step's command line and captured output is kept as a :class:`StepRecord`, which is what
:mod:`chargeback_prediction.demo.recording` turns into an asciinema cast and an animated SVG.
"""

from __future__ import annotations

import contextlib
import io
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from chargeback_prediction.synthetic import GeneratorConfig, read_dataset
from chargeback_prediction.synthetic.schema import Dispute

DEMO_REASON_CODE = "visa:13.2"
"""Reason code the tour follows through explain → reason → evidence.

Visa 13.2 (cancelled recurring) is the classic subscription-merchant dispute, and the one where
the playbook, the evidence assembler and the cancellation check all have something to say.
"""

REBUTTAL_HEADING = "## Rebuttal"


def packet_preview(markdown: str) -> list[str]:
    """The lines of a response packet up to the end of its rebuttal paragraph.

    That is the header table, the readiness verdict and the rebuttal: what an analyst reads
    first. Falls back to the whole packet if it has no rebuttal section.
    """
    lines = markdown.splitlines()
    try:
        start = lines.index(REBUTTAL_HEADING) + 1
    except ValueError:
        return lines
    while start < len(lines) and not lines[start].strip():
        start += 1
    end = start
    while end < len(lines) and lines[end].strip():
        end += 1
    return lines[:end]


@dataclass(frozen=True)
class TourConfig:
    """Size and seed of the dataset the tour generates."""

    customers: int = GeneratorConfig().n_customers
    seed: int = GeneratorConfig().seed

    def __post_init__(self) -> None:
        if self.customers < 1:
            raise ValueError("customers must be at least 1")


@dataclass(frozen=True)
class StepRecord:
    """One executed step: what it shows, what was typed and what it printed."""

    narration: str
    command: str
    output: str
    exit_code: int
    seconds: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


@dataclass
class Tour:
    """The outcome of :func:`run_tour`."""

    workdir: Path
    config: TourConfig
    steps: list[StepRecord] = field(default_factory=list)
    dispute: Dispute | None = None

    @property
    def ok(self) -> bool:
        return all(step.ok for step in self.steps)


class TourError(RuntimeError):
    """A pipeline step failed; the partial :class:`Tour` is attached."""

    def __init__(self, message: str, tour: Tour) -> None:
        super().__init__(message)
        self.tour = tour


def pick_dispute(disputes: Sequence[Dispute], reason_code: str = DEMO_REASON_CODE) -> Dispute:
    """Pick the dispute the tour follows: the earliest-filed one under ``reason_code``.

    Falls back to the earliest-filed dispute of any code, so tiny datasets still work. Ties
    on filing time are broken by id, so the choice is deterministic for a given seed.
    """
    if not disputes:
        raise ValueError("the dataset has no disputes; generate more customers")
    ordered = sorted(disputes, key=lambda d: (d.filed_at, d.dispute_id))
    return next((d for d in ordered if d.reason_code == reason_code), ordered[0])


def _run_cli(argv: Sequence[str]) -> tuple[int, str]:
    # Imported here: the CLI imports this module lazily for `cbp demo`.
    from chargeback_prediction.cli import main

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(list(argv))
        except SystemExit as exc:  # argparse errors exit with 2
            code = exc.code if isinstance(exc.code, int) else 1
    return code, out.getvalue() + err.getvalue()


def run_tour(
    workdir: Path,
    config: TourConfig | None = None,
    *,
    on_step: Callable[[StepRecord], None] | None = None,
) -> Tour:
    """Run every pipeline step inside ``workdir`` and return what each one printed.

    ``on_step`` is called after each step (the CLI uses it to stream output live). Raises
    :class:`TourError` on the first failing step, with the steps so far attached.
    """
    config = config or TourConfig()
    workdir.mkdir(parents=True, exist_ok=True)
    tour = Tour(workdir=workdir, config=config)

    def step(narration: str, argv: Sequence[str]) -> str:
        started = time.perf_counter()
        code, output = _run_cli(argv)
        record = StepRecord(
            narration=narration,
            command="cbp " + " ".join(argv),
            output=output,
            exit_code=code,
            seconds=time.perf_counter() - started,
        )
        tour.steps.append(record)
        if on_step is not None:
            on_step(record)
        if not record.ok:
            raise TourError(f"`{record.command}` exited with {code}:\n{output}", tour)
        return output

    with contextlib.chdir(workdir):
        step(
            "Simulate a subscription merchant: customers, renewals, retries, disputes.",
            ["generate", "--customers", str(config.customers), "--seed", str(config.seed)],
        )
        step(
            "Point-in-time features: nothing at or after the charge is visible.",
            ["features"],
        )
        step(
            "Baselines on a future test window, with a 120-day label gap.",
            ["train"],
        )
        step(
            "Price the decisions in dollars, with bootstrap intervals.",
            ["evaluate", "--resamples", "200"],
        )

        dispute = pick_dispute(read_dataset(Path("data/synthetic")).disputes)
        tour.dispute = dispute
        txn = dispute.transaction_id
        step(
            f"A {dispute.reason_code} dispute lands on {txn}. What did the model see?",
            ["explain", "--transaction-id", txn],
        )
        step(
            "Rules pick the strategy; the (mock) LLM writes the case note.",
            ["reason", "--transaction-id", txn],
        )
        step(
            "Draft the representment packet from evidence the merchant holds.",
            ["evidence", "--dispute-id", dispute.dispute_id, "--out", "packets"],
        )

    packet = workdir / "packets" / f"{dispute.dispute_id}.md"
    preview = packet_preview(packet.read_text(encoding="utf-8"))
    shown = StepRecord(
        narration="The top of the packet: readiness, deadline and rebuttal.",
        command=f"head -{len(preview)} packets/{dispute.dispute_id}.md",
        output="\n".join(preview) + "\n",
        exit_code=0,
        seconds=0.0,
    )
    tour.steps.append(shown)
    if on_step is not None:
        on_step(shown)
    return tour
