"""Release hygiene: one version everywhere, and notes for it in the changelog."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

from chargeback_prediction import __version__

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "changelog_section.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("changelog_section", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


changelog_section = _load_script()

SAMPLE = """# Changelog

## [Unreleased]

## [2.0.0] - 2027-01-01

### Changed
- Big thing.

## [1.0.0] - 2026-11-25

### Added
- First thing.

[2.0.0]: https://example.invalid/v2
[1.0.0]: https://example.invalid/v1
"""


def _pyproject_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        version: str = tomllib.load(fh)["project"]["version"]
    return version


def test_installed_version_matches_pyproject() -> None:
    assert __version__ == _pyproject_version()


def test_changelog_has_notes_for_the_current_version() -> None:
    notes = changelog_section.section(
        (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), _pyproject_version()
    )
    assert "cbp demo" in notes
    assert "](https://" not in notes.splitlines()[-1], "link definitions are not notes"


def test_section_extracts_between_headings() -> None:
    assert changelog_section.section(SAMPLE, "2.0.0") == "### Changed\n- Big thing.\n"
    assert changelog_section.section(SAMPLE, "1.0.0") == "### Added\n- First thing.\n"


@pytest.mark.parametrize("version", ["3.0.0", "Unreleased"])
def test_section_rejects_missing_or_empty(version: str) -> None:
    with pytest.raises(LookupError):
        changelog_section.section(SAMPLE, version)


def test_script_cli(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(SAMPLE, encoding="utf-8")

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
        )

    ok = run("1.0.0", str(changelog))
    assert (ok.returncode, ok.stdout) == (0, "### Added\n- First thing.\n")
    missing = run("9.9.9", str(changelog))
    assert missing.returncode == 1
    assert "no '## [9.9.9]'" in missing.stderr
    assert run().returncode == 2
