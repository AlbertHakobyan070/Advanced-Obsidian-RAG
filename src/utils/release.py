"""
release.py — the product version, the release notes, and what each part of an
install is running.

ONE VERSION, SET IN ONE PLACE. `VERSION` at the repository root holds it.
Everything that reports a version reads it from here: both FastAPI apps, GET
/health, both capability maps, the console's version chip. CHANGELOG.md's
newest entry and the `Noetrix version:` line in each agent skill must name
the same version; tests/test_versioning.py fails when they drift, so a bump
cannot forget one of them.

A checkout or a package without VERSION is broken, not "unversioned": the
packagers copy explicit file lists, and a list that dropped VERSION would
otherwise ship an install that cannot say what it is. So this raises.

WHY COMMITS TOO. Between two releases the version stays the same while the code
moves. A service started before a pull still reports the same version as the
files on disk, so drift between a RUNNING process and the checkout is judged on
the git commit it started from as well (None outside a git checkout, e.g. in a
container).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION_FILE = ROOT / "VERSION"
CHANGELOG_FILE = ROOT / "CHANGELOG.md"

_SEMVER = re.compile(r"\d+\.\d+\.\d+")
# "## [0.27.0] - 2026-10-14" — Keep a Changelog's heading shape.
_ENTRY = re.compile(r"^## \[(\d+\.\d+\.\d+)\] - (\d{4}-\d{2}-\d{2})\s*$")
# The line each SKILL.md carries under its title.
SKILL_VERSION_RE = re.compile(r"^Noetrix version: (\d+\.\d+\.\d+)\b", re.M)


def read_version(path: Path = VERSION_FILE) -> str:
    """The MAJOR.MINOR.PATCH in `path`. Raises when it is missing or malformed."""
    try:
        v = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise RuntimeError(f"{path} is missing: it holds the product version "
                           "(CHANGELOG.md says what each one contains)") from None
    if not _SEMVER.fullmatch(v):
        raise RuntimeError(f"{path} must hold a MAJOR.MINOR.PATCH version, got {v!r}")
    return v


PRODUCT_VERSION = read_version()


def git_head(root: Path = ROOT) -> dict | None:
    """{commit, dirty} of the checkout at `root`, or None when it is not a git
    checkout or git is not installed (a container, an unpacked archive)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                capture_output=True, text=True, timeout=10)
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                cwd=root, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if commit.returncode != 0 or status.returncode != 0:
        return None
    return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}


def parse_changelog(text: str) -> list[dict]:
    """Keep-a-Changelog entries, newest first as written:
    [{version, date, preamble, sections: {"Added": [item, ...], ...}}].
    A bullet's continuation lines are folded into it. Anything before the
    first entry (the file's own introduction) is skipped."""
    entries: list[dict] = []
    section = None
    for line in text.splitlines():
        m = _ENTRY.match(line)
        if m:
            entries.append({"version": m.group(1), "date": m.group(2),
                            "preamble": "", "sections": {}})
            section = None
            continue
        if not entries:
            continue
        cur = entries[-1]
        if line.startswith("### "):
            section = line[4:].strip()
            cur["sections"].setdefault(section, [])
        elif line.startswith("- ") and section:
            cur["sections"][section].append(line[2:].strip())
        elif line.startswith("  ") and line.strip() and section and cur["sections"][section]:
            cur["sections"][section][-1] += " " + line.strip()
        elif line.strip() and section is None:
            cur["preamble"] = (cur["preamble"] + " " + line.strip()).strip()
    return entries


def skill_version(path: Path) -> str | None:
    """The `Noetrix version:` a SKILL.md declares, or None when it has none."""
    m = SKILL_VERSION_RE.search(path.read_text(encoding="utf-8"))
    return m.group(1) if m else None
