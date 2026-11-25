from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import pytest

from chargeback_prediction.cli import main
from chargeback_prediction.demo import (
    StepRecord,
    Timing,
    Tour,
    TourConfig,
    TourError,
    build_session,
    pick_dispute,
    render_cast,
    render_svg,
    run_tour,
    transcript,
    wrap,
    write_recording,
)
from chargeback_prediction.demo.recording import LineKind
from chargeback_prediction.demo.tour import packet_preview
from chargeback_prediction.reason_codes import DisputeCategory, Network
from chargeback_prediction.synthetic.schema import Dispute

DOCS_DEMO = Path(__file__).resolve().parents[1] / "docs" / "demo"
SVG = "{http://www.w3.org/2000/svg}"


def _dispute(dispute_id: str, reason_code: str, day: int) -> Dispute:
    network, _ = reason_code.split(":")
    return Dispute(
        dispute_id=dispute_id,
        transaction_id=f"txn_{dispute_id}",
        customer_id="cus_1",
        network=Network(network),
        reason_code=reason_code,
        category=DisputeCategory.CONSUMER,
        filed_at=datetime(2025, 1, day, tzinfo=UTC),
        amount_cents=999,
        is_friendly_fraud=True,
    )


def _tour(*steps: tuple[str, str, str]) -> Tour:
    tour = Tour(workdir=Path("."), config=TourConfig())
    tour.steps.extend(
        StepRecord(narration=n, command=c, output=o, exit_code=0, seconds=0.1) for n, c, o in steps
    )
    return tour


@pytest.fixture(scope="module")
def small_tour(tmp_path_factory: pytest.TempPathFactory) -> Tour:
    return run_tour(tmp_path_factory.mktemp("demo"), TourConfig(customers=1_500, seed=11))


# --- picking the dispute -------------------------------------------------------------------


def test_pick_dispute_prefers_earliest_visa_13_2() -> None:
    disputes = [
        _dispute("dp_3", "visa:13.2", 9),
        _dispute("dp_1", "visa:10.4", 1),
        _dispute("dp_2", "visa:13.2", 5),
    ]
    assert pick_dispute(disputes).dispute_id == "dp_2"


def test_pick_dispute_falls_back_to_earliest_of_any_code() -> None:
    disputes = [_dispute("dp_b", "visa:10.4", 4), _dispute("dp_a", "mastercard:4837", 4)]
    assert pick_dispute(disputes).dispute_id == "dp_a"  # tie on time broken by id


def test_pick_dispute_needs_a_dispute() -> None:
    with pytest.raises(ValueError, match="no disputes"):
        pick_dispute([])


def test_packet_preview_stops_after_the_rebuttal_paragraph() -> None:
    md = "# Draft\n\n| a |\n\n## Rebuttal\n\nFirst line.\nSecond line.\n\n## Exhibits\n\nx\n"
    assert packet_preview(md) == [
        "# Draft",
        "",
        "| a |",
        "",
        "## Rebuttal",
        "",
        "First line.",
        "Second line.",
    ]
    assert packet_preview("no rebuttal\nhere") == ["no rebuttal", "here"]


def test_tour_config_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="customers"):
        TourConfig(customers=0)


# --- the tour itself -----------------------------------------------------------------------


def test_tour_runs_every_pipeline_step_in_order(small_tour: Tour) -> None:
    assert small_tour.ok
    commands = [s.command.split()[1] for s in small_tour.steps]
    assert commands == [
        "generate",
        "features",
        "train",
        "evaluate",
        "explain",
        "reason",
        "evidence",
        f"-{len(small_tour.steps[-1].output.splitlines())}",
    ]
    assert small_tour.steps[-1].command.startswith("head -")


def test_tour_follows_one_dispute_end_to_end(small_tour: Tour) -> None:
    dispute = small_tour.dispute
    assert dispute is not None
    by_name = {s.command.split()[1]: s for s in small_tour.steps}
    assert dispute.transaction_id in by_name["explain"].output
    assert dispute.reason_code in by_name["reason"].output
    assert "strategy:" in by_name["reason"].output
    packet = small_tour.workdir / "packets" / f"{dispute.dispute_id}.md"
    assert packet.exists()
    preview = small_tour.steps[-1].output
    assert preview.startswith(f"# Representment draft — {dispute.dispute_id}")
    assert "## Rebuttal" in preview
    assert preview.splitlines()[-1].strip(), "preview should end on the rebuttal text"


def test_tour_output_uses_relative_paths(small_tour: Tour) -> None:
    for step in small_tour.steps:
        assert str(small_tour.workdir) not in step.output
    assert "wrote data/features.csv" in small_tour.steps[1].output


def test_tour_stops_on_first_failure(tmp_path: Path) -> None:
    seen: list[StepRecord] = []
    # 3 customers produce no mature labels, so `cbp train` fails with a usage error.
    with pytest.raises(TourError) as exc:
        run_tour(tmp_path, TourConfig(customers=3), on_step=seen.append)
    assert exc.value.tour.steps == seen
    assert not seen[-1].ok
    assert seen[-1].exit_code == 2
    assert all(step.ok for step in seen[:-1])


# --- recording -----------------------------------------------------------------------------


def test_wrap_hangs_continuations_under_label_values() -> None:
    line = "  rationale:   " + " ".join(["word"] * 30)
    lines = wrap(line, 40)
    assert all(len(part) <= 40 for part in lines)
    assert lines[0].startswith("  rationale:   word")
    assert all(part.startswith(" " * 15 + "word") for part in lines[1:])


def test_wrap_leaves_short_lines_and_indents_plain_ones() -> None:
    assert wrap("short", 40) == ["short"]
    lines = wrap("  " + "x " * 30, 20)
    assert all(part.startswith("    x") for part in lines[1:])


def test_session_timeline_is_monotonic_and_ordered() -> None:
    tour = _tour(("Step one.", "cbp one", "a\nb\n"), ("Step two.", "cbp two", "c\n"))
    session = build_session(tour, Timing(), title="t")
    times = [line.at for line in session.lines]
    assert times == sorted(times)
    assert [(line.kind, line.text) for line in session.lines] == [
        (LineKind.NARRATION, "# Step one."),
        (LineKind.COMMAND, "$ cbp one"),
        (LineKind.OUTPUT, "a"),
        (LineKind.OUTPUT, "b"),
        (LineKind.OUTPUT, ""),
        (LineKind.NARRATION, "# Step two."),
        (LineKind.COMMAND, "$ cbp two"),
        (LineKind.OUTPUT, "c"),
    ]
    assert session.duration > times[-1]
    assert transcript(session.lines).splitlines()[1] == "$ cbp one"


def test_cast_is_valid_asciicast_v2() -> None:
    tour = _tour(("Hello.", "cbp go", "out → 1×\n"))
    timing = Timing()
    session = build_session(tour, timing, title="demo")
    header, *events = render_cast(session, width=80, height=20, timing=timing).splitlines()
    meta = json.loads(header)
    assert (meta["version"], meta["width"], meta["height"], meta["title"]) == (2, 80, 20, "demo")
    parsed = [json.loads(e) for e in events]
    assert all(kind == "o" for _, kind, _ in parsed)
    stamps = [t for t, _, _ in parsed]
    assert stamps == sorted(stamps)
    played = "".join(data for _, _, data in parsed)
    assert "".join(d for _, _, d in parsed if len(d) == 1) == "cbp go"  # one keystroke each
    assert "out → 1×\r\n" in played


def test_svg_is_well_formed_and_animates_every_line() -> None:
    lines = "\n".join(f"line {i} <&>" for i in range(40))
    tour = _tour(("Many lines.", "cbp many", lines + "\n"))
    session = build_session(tour, title='a "quoted" <title>')
    svg = render_svg(session, cols=60, rows=10)
    root = ET.fromstring(svg)
    texts = [t for t in root.iter(f"{SVG}text") if t.get("opacity") == "0"]
    assert len(texts) == len(session.lines)
    durations = {a.get("dur") for a in root.iter(f"{SVG}animate")}
    durations |= {a.get("dur") for a in root.iter(f"{SVG}animateTransform")}
    assert durations == {f"{session.duration:g}s"}, "every animation shares one loop"
    (scroll,) = root.iter(f"{SVG}animateTransform")
    key_times = [float(k) for k in (scroll.get("keyTimes") or "").split(";")]
    assert key_times[0] == 0.0
    assert key_times == sorted(key_times)
    assert len(key_times) == len((scroll.get("values") or "").split(";"))
    assert "line 39 &lt;&amp;&gt;" in svg
    assert "<script" not in svg
    assert "href" not in svg


def test_write_recording_writes_all_three_files(small_tour: Tour, tmp_path: Path) -> None:
    paths = write_recording(small_tour, tmp_path / "rec")
    assert sorted(p.name for p in paths.values()) == ["demo.cast", "demo.svg", "transcript.txt"]
    text = paths["transcript"].read_text(encoding="utf-8")
    assert text.startswith("# Simulate a subscription merchant")
    assert "$ cbp generate --customers 1500 --seed 11" in text


# --- CLI ------------------------------------------------------------------------------------


def test_demo_command_streams_steps_and_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work, rec = tmp_path / "work", tmp_path / "rec"
    argv = ["demo", "--customers", "1500", "--seed", "11", "--workdir", str(work)]
    assert main([*argv, "--record", str(rec)]) == 0
    out = capsys.readouterr().out
    assert "$ cbp train" in out
    assert "# 8 steps in" in out
    assert f"outputs kept in {work}/" in out
    assert (work / "data" / "features.csv").exists()
    assert sorted(p.name for p in rec.iterdir()) == ["demo.cast", "demo.svg", "transcript.txt"]


def test_demo_command_reports_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo", "--customers", "3"]) == 1
    captured = capsys.readouterr()
    assert "demo stopped: cbp train failed" in captured.err


def test_demo_command_rejects_bad_size() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["demo", "--customers", "0"])
    assert exc.value.code == 2


# --- the committed recording ---------------------------------------------------------------


def test_committed_recording_matches_the_pipeline(tmp_path: Path) -> None:
    """docs/demo/ is generated by `cbp demo --record docs/demo`; fail if the pipeline drifts."""
    tour = run_tour(tmp_path / "work", TourConfig())
    fresh = write_recording(tour, tmp_path / "rec")
    for path in fresh.values():
        committed = DOCS_DEMO / path.name
        assert committed.read_text(encoding="utf-8") == path.read_text(encoding="utf-8"), (
            f"{committed} is stale; regenerate with `uv run cbp demo --record docs/demo`"
        )


def test_wrap_falls_back_to_a_small_hang_on_narrow_terminals() -> None:
    lines = wrap("  key factors: " + "abc " * 10, 20)
    assert all(part.startswith("  abc") for part in lines[1:])


def test_demo_command_reports_tour_value_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import chargeback_prediction.demo as demo

    def no_disputes(*_: object, **__: object) -> Tour:
        raise ValueError("the dataset has no disputes; generate more customers")

    monkeypatch.setattr(demo, "run_tour", no_disputes)
    with pytest.raises(SystemExit) as exc:
        main(["demo", "--customers", "5"])
    assert exc.value.code == 2
    assert "no disputes" in capsys.readouterr().err
