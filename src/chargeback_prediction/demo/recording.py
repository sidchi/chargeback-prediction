"""Turn a :class:`~chargeback_prediction.demo.tour.Tour` into a terminal recording.

Two outputs, both plain text and fully deterministic for a given tour (timings are scripted,
not measured, so re-recording an unchanged pipeline produces an identical file):

* an `asciinema <https://docs.asciinema.org/manual/asciicast/v2/>`_ v2 ``.cast`` file, which
  ``asciinema play`` replays in a real terminal with the typing and pauses intact;
* a self-contained animated ``.svg`` of the same session for the README. It uses SMIL
  animations only (no scripts, fonts or external references), which GitHub renders in an
  ``<img>`` tag, and loops forever.
"""

from __future__ import annotations

import json
import re
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from xml.sax.saxutils import escape

from chargeback_prediction.demo.tour import Tour

PROMPT = "$ "


class LineKind(StrEnum):
    NARRATION = "narration"
    COMMAND = "command"
    OUTPUT = "output"


@dataclass(frozen=True)
class Timing:
    """Scripted pacing of the recording, in seconds."""

    keystroke: float = 0.035
    after_command: float = 0.5
    output_line: float = 0.04
    after_step: float = 1.6
    narration: float = 0.6
    final_hold: float = 5.0


@dataclass(frozen=True)
class Line:
    """One rendered line of the session and the time it appears."""

    at: float
    kind: LineKind
    text: str


@dataclass(frozen=True)
class Session:
    """A tour laid out on a timeline: what appears when."""

    lines: tuple[Line, ...]
    duration: float
    title: str


def build_session(tour: Tour, timing: Timing | None = None, *, title: str = "") -> Session:
    """Lay the tour's steps out on a scripted timeline."""
    timing = timing or Timing()
    clock = 0.4
    lines: list[Line] = []
    for i, step in enumerate(tour.steps):
        if i:
            lines.append(Line(clock, LineKind.OUTPUT, ""))
        lines.append(Line(clock, LineKind.NARRATION, f"# {step.narration}"))
        clock += timing.narration
        # The command line is fully visible once typed; the cast also replays the keystrokes.
        clock += timing.keystroke * len(step.command)
        lines.append(Line(clock, LineKind.COMMAND, PROMPT + step.command))
        clock += timing.after_command
        for text in step.output.rstrip("\n").splitlines():
            lines.append(Line(clock, LineKind.OUTPUT, text))
            clock += timing.output_line
        clock += timing.after_step
    return Session(tuple(lines), round(clock + timing.final_hold, 3), title)


# --- asciinema ---------------------------------------------------------------------------

_ANSI = {
    LineKind.NARRATION: "\x1b[2;3m",  # dim italic
    LineKind.COMMAND: "",
    LineKind.OUTPUT: "",
}
_RESET = "\x1b[0m"
_GREEN = "\x1b[32m"


def render_cast(
    session: Session, *, width: int = 100, height: int = 30, timing: Timing | None = None
) -> str:
    """Render the session as asciinema v2 (a JSON header line, then one event per line)."""
    timing = timing or Timing()
    header = {
        "version": 2,
        "width": width,
        "height": height,
        "title": session.title,
        "env": {"SHELL": "/bin/bash", "TERM": "xterm-256color"},
    }
    events: list[tuple[float, str]] = []
    for line in session.lines:
        if line.kind is LineKind.COMMAND:
            command = line.text.removeprefix(PROMPT)
            typed_from = line.at - timing.keystroke * len(command)
            events.append((typed_from, f"{_GREEN}{PROMPT}{_RESET}"))
            events.extend(
                (typed_from + timing.keystroke * (i + 1), ch) for i, ch in enumerate(command)
            )
            events.append((line.at, "\r\n"))
        else:
            style = _ANSI[line.kind]
            text = f"{style}{line.text}{_RESET}" if style else line.text
            events.append((line.at, text + "\r\n"))
    out = [json.dumps(header, ensure_ascii=False)]
    out.extend(json.dumps([round(t, 3), "o", data], ensure_ascii=False) for t, data in events)
    return "\n".join(out) + "\n"


# --- SVG ---------------------------------------------------------------------------------

_LABEL = re.compile(r"^(\s*[\w ]{1,16}:\s+)\S")


def wrap(text: str, width: int) -> list[str]:
    """Wrap one terminal line to ``width`` columns, keeping its indentation.

    Continuation lines hang under the value of ``label:   value`` lines (as ``cbp reason``
    prints them) and otherwise under the first character of the line.
    """
    if len(text) <= width:
        return [text]
    match = _LABEL.match(text)
    hang = len(match.group(1)) if match else len(text) - len(text.lstrip()) + 2
    if hang > width // 2:
        hang = 2
    return textwrap.wrap(
        text,
        width=width,
        subsequent_indent=" " * hang,
        break_long_words=True,
        break_on_hyphens=False,
        drop_whitespace=True,
    ) or [text]


@dataclass(frozen=True)
class Theme:
    background: str = "#0d1117"
    chrome: str = "#161b22"
    border: str = "#30363d"
    text: str = "#e6edf3"
    prompt: str = "#3fb950"
    narration: str = "#8b949e"
    title: str = "#8b949e"


def _visual_rows(session: Session, cols: int) -> list[Line]:
    rows: list[Line] = []
    for line in session.lines:
        rows.extend(Line(line.at, line.kind, part) for part in wrap(line.text, cols))
    return rows


def _fraction(at: float, duration: float) -> str:
    return f"{min(at / duration, 1.0):.5f}"


def _scroll_keyframes(
    rows: Sequence[Line], visible: int, line_height: float, duration: float
) -> tuple[str, str]:
    times, offsets = ["0"], ["0 0"]
    for i, row in enumerate(rows):
        if i < visible:
            continue
        offset = f"0 {-(i - visible + 1) * line_height:g}"
        frac = _fraction(row.at, duration)
        if times[-1] == frac:  # several rows at the same instant: keep the last offset
            offsets[-1] = offset
        else:
            times.append(frac)
            offsets.append(offset)
    return ";".join(times), ";".join(offsets)


def render_svg(
    session: Session,
    *,
    cols: int = 100,
    rows: int = 30,
    font_size: float = 13.0,
    theme: Theme | None = None,
) -> str:
    """Render the session as a looping, animated, self-contained SVG terminal window."""
    theme = theme or Theme()
    char_w = font_size * 0.6
    line_h = round(font_size * 1.45, 2)
    pad, bar = 14.0, 30.0
    width = round(cols * char_w + 2 * pad, 1)
    height = round(rows * line_h + 2 * pad + bar, 1)
    dur = f"{session.duration:g}s"
    visual = _visual_rows(session, cols)

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}" height="{height:g}" '
        f'viewBox="0 0 {width:g} {height:g}" role="img" '
        f'aria-label="{escape(session.title, {chr(34): "&quot;"})}">',
        f"<title>{escape(session.title)}</title>",
        "<style>"
        f"text{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
        f"font-size:{font_size:g}px;white-space:pre;fill:{theme.text}}}"
        f".n{{fill:{theme.narration};font-style:italic}}"
        f".p{{fill:{theme.prompt};font-weight:bold}}"
        f".t{{fill:{theme.title};font-size:12px}}"
        "</style>",
        f'<rect width="{width:g}" height="{height:g}" rx="8" fill="{theme.background}" '
        f'stroke="{theme.border}"/>',
        f'<path d="M0 8a8 8 0 0 1 8-8h{width - 16:g}a8 8 0 0 1 8 8v{bar - 8:g}h-{width:g}z" '
        f'fill="{theme.chrome}"/>',
    ]
    for i, color in enumerate(("#ff5f56", "#ffbd2e", "#27c93f")):
        out.append(f'<circle cx="{pad + 6 + i * 20:g}" cy="15" r="6" fill="{color}"/>')
    if session.title:
        out.append(
            f'<text class="t" x="{width / 2:g}" y="19" text-anchor="middle">'
            f"{escape(session.title)}</text>"
        )
    out.append(
        f'<clipPath id="screen"><rect x="0" y="{bar + pad - 2:g}" width="{width:g}" '
        f'height="{height - bar - pad + 2:g}"/></clipPath>'
    )
    out.append('<g clip-path="url(#screen)"><g>')
    key_times, offsets = _scroll_keyframes(visual, rows, line_h, session.duration)
    out.append(
        f'<animateTransform attributeName="transform" type="translate" calcMode="discrete" '
        f'values="{offsets}" keyTimes="{key_times}" dur="{dur}" repeatCount="indefinite"/>'
    )
    for i, row in enumerate(visual):
        y = round(bar + pad + font_size + i * line_h, 2)
        out.append(_svg_line(row, pad, y, dur, session.duration))
    out.append("</g></g></svg>")
    return "\n".join(out) + "\n"


def _svg_line(row: Line, x: float, y: float, dur: str, duration: float) -> str:
    show = (
        f'<animate attributeName="opacity" values="0;1" keyTimes="0;{_fraction(row.at, duration)}"'
        f' calcMode="discrete" dur="{dur}" repeatCount="indefinite"/>'
    )
    if row.kind is LineKind.COMMAND and row.text.startswith(PROMPT):
        body = f'<tspan class="p">{PROMPT.strip()}</tspan> {escape(row.text[len(PROMPT) :])}'
        cls = ""
    else:
        body = escape(row.text)
        cls = ' class="n"' if row.kind is LineKind.NARRATION else ""
    return f'<text{cls} x="{x:g}" y="{y:g}" opacity="0">{show}{body}</text>'


def write_recording(
    tour: Tour,
    out_dir: Path,
    *,
    title: str = "chargeback-prediction: cbp demo",
    cols: int = 100,
    rows: int = 30,
) -> dict[str, Path]:
    """Write ``demo.cast``, ``demo.svg`` and a plain ``transcript.txt`` into ``out_dir``."""
    session = build_session(tour, title=title)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "cast": out_dir / "demo.cast",
        "svg": out_dir / "demo.svg",
        "transcript": out_dir / "transcript.txt",
    }
    paths["transcript"].write_text(transcript(session.lines), encoding="utf-8")
    paths["cast"].write_text(render_cast(session, width=cols, height=rows), encoding="utf-8")
    paths["svg"].write_text(render_svg(session, cols=cols, rows=rows), encoding="utf-8")
    return paths


def transcript(lines: Iterable[Line]) -> str:
    """The session as plain text, exactly as it would read in a terminal."""
    return "\n".join(line.text for line in lines) + "\n"
