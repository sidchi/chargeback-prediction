"""The evaluation notebook must stay runnable: execute its code cells on a small dataset.

Runs the cells with plain ``exec`` (no Jupyter dependency in CI) in one shared namespace, in
order, the way a kernel would.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "evaluation.ipynb"


def _cells() -> list[dict[str, Any]]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    cells: list[dict[str, Any]] = nb["cells"]
    return cells


def _source(cell: dict[str, Any]) -> str:
    src = cell["source"]
    return src if isinstance(src, str) else "".join(src)


def test_notebook_is_committed_with_outputs_and_no_errors() -> None:
    code = [c for c in _cells() if c["cell_type"] == "code"]
    assert code
    outputs = [o for c in code for o in c["outputs"]]
    assert not [o for o in outputs if o["output_type"] == "error"]
    charts = [o for o in outputs if "image/png" in o.get("data", {})]
    assert len(charts) >= 5
    assert not any("magic" in _source(c) or _source(c).startswith("%") for c in code)


def test_notebook_code_runs_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CBP_NB_CUSTOMERS", "1500")
    monkeypatch.setenv("CBP_NB_RESAMPLES", "20")
    matplotlib.use("Agg")
    monkeypatch.setattr(plt, "show", lambda *a, **k: plt.close("all"))
    namespace: dict[str, Any] = {"__name__": "__notebook__"}
    for i, cell in enumerate(_cells()):
        if cell["cell_type"] == "code":
            exec(compile(_source(cell), f"<cell {i}>", "exec"), namespace)
    assert namespace["report"].models
