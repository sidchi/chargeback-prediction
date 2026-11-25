"""``cbp demo``: a narrated end-to-end run of the pipeline, and a recorder for it.

:func:`run_tour` executes every ``cbp`` step on a fresh synthetic dataset;
:func:`write_recording` turns the run into an asciinema cast, an animated SVG and a transcript.
"""

from chargeback_prediction.demo.recording import (
    Session,
    Timing,
    build_session,
    render_cast,
    render_svg,
    transcript,
    wrap,
    write_recording,
)
from chargeback_prediction.demo.tour import (
    StepRecord,
    Tour,
    TourConfig,
    TourError,
    pick_dispute,
    run_tour,
)

__all__ = [
    "Session",
    "StepRecord",
    "Timing",
    "Tour",
    "TourConfig",
    "TourError",
    "build_session",
    "pick_dispute",
    "render_cast",
    "render_svg",
    "run_tour",
    "transcript",
    "wrap",
    "write_recording",
]
