from __future__ import annotations

import pytest

from chargeback_prediction import __version__
from chargeback_prediction.cli import main


def test_reason_codes_command_lists_catalog(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["reason-codes"]) == 0
    out = capsys.readouterr().out
    assert "visa:13.2" in out
    assert "Cancelled Recurring Transaction" in out


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_command_is_required() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
