"""Print one version's section of CHANGELOG.md, for use as GitHub release notes.

Usage: ``python scripts/changelog_section.py 1.0.0 [CHANGELOG.md]``

Exits with status 1 if the version has no section or the section is empty, so a release can't
go out without notes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\]")


def section(changelog: str, version: str) -> str:
    """Return the body under ``## [version]`` up to the next version heading or link block."""
    lines = changelog.splitlines()
    body: list[str] | None = None
    for line in lines:
        match = HEADING.match(line)
        if match:
            if body is not None:
                break
            if match.group("version") == version:
                body = []
            continue
        if body is not None:
            if re.match(r"^\[[^\]]+\]: ", line):  # reference-style link definitions
                break
            body.append(line)
    if body is None:
        raise LookupError(f"no '## [{version}]' section in the changelog")
    text = "\n".join(body).strip()
    if not text:
        raise LookupError(f"the '## [{version}]' section is empty")
    return text + "\n"


def main(argv: list[str]) -> int:
    if len(argv) not in (1, 2):
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(argv[1]) if len(argv) == 2 else Path(__file__).resolve().parents[1] / "CHANGELOG.md"
    try:
        sys.stdout.write(section(path.read_text(encoding="utf-8"), argv[0]))
    except LookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
