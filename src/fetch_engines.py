"""
fetch_engines.py — version status, upgrade preview and upgrade of the web-fetch
packages (Settings > Fetch engines; GET /api/fetch-engines, the engine_update job).

An upgrade of one of these can drag numpy, torch or pydantic along and break
the venv the whole RAG runs in, so the console always shows a DRY RUN first
(`check`), and the real upgrade runs as a logged console job:

    python -m src.fetch_engines <name>

runs the install, then the package's browser installs, and exits non-zero on
the first command that fails.
"""
from __future__ import annotations

import importlib.metadata as metadata
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"

# Short: a GET to PyPI's JSON API. The dry run resolves (pip downloads wheels
# to read their metadata), so it gets minutes, not seconds.
PYPI_TIMEOUT_S = 10
DRY_RUN_TIMEOUT_S = 300

# The browser installs requirements.txt documents. Playwright first; patchright
# is the patched driver crawl4ai's and scrapling's stealth modes drive, and
# crawlee reuses the playwright Chromium.
_PLAYWRIGHT = ("-m", "playwright", "install", "chromium")
_PATCHRIGHT = ("-m", "patchright", "install", "chromium")

# One row per package: the pip spec with the extras requirements.txt installs,
# and the commands (run with this interpreter) that must follow an upgrade.
ENGINES: dict[str, dict] = {
    "crawl4ai":   {"spec": "crawl4ai", "post": (_PLAYWRIGHT, _PATCHRIGHT)},
    "scrapling":  {"spec": "scrapling[fetchers]", "post": (_PLAYWRIGHT, _PATCHRIGHT)},
    "crawlee":    {"spec": "crawlee[playwright,beautifulsoup]", "post": (_PLAYWRIGHT,)},
    "playwright": {"spec": "playwright", "post": (_PLAYWRIGHT,),
                   "note": "keep patchright on the same release"},
    "patchright": {"spec": "patchright", "post": (_PATCHRIGHT,),
                   "note": "keep playwright on the same release"},
    "markitdown": {"spec": "markitdown[pdf,docx,pptx,xlsx]", "post": ()},
}

CONTAINER_NOTE = ("Running in a container: an update changes this container only and is "
                  "gone when the image is rebuilt or the container recreated. The image runs "
                  "as a non-root user, so pip may refuse or install into the user site. To "
                  "keep a new version, bump its pin in requirements.txt and rebuild the image.")

_REQ_LINE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*==\s*([^\s#;]+)")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_UV_LINE = re.compile(r"^\s*([+\-~])\s+([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==(\S+)")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _require(name: str) -> dict:
    if name not in ENGINES:
        raise ValueError(f"unknown fetch engine {name!r}; expected one of {list(ENGINES)}")
    return ENGINES[name]


def installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def pins(path: Path = REQUIREMENTS) -> dict[str, str]:
    """{normalized name: pinned version} for every `name==version` line."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _REQ_LINE.match(line)
        if m:
            out[_norm(m.group(1))] = m.group(2)
    return out


def environment() -> dict:
    """Where an update would land. /.dockerenv is Docker's own marker and
    KUBERNETES_SERVICE_HOST is set in every Kubernetes pod."""
    container = Path("/.dockerenv").exists() or bool(os.environ.get("KUBERNETES_SERVICE_HOST"))
    return {"python": sys.executable, "prefix": sys.prefix,
            "venv": sys.prefix != sys.base_prefix, "container": container,
            "installer": "uv" if shutil.which("uv") else "pip",
            "note": CONTAINER_NOTE if container else None}


def status(installed: str | None, pin: str | None, latest: str | None = None) -> list[str]:
    """Status codes, most urgent first. `latest` is None until a check ran."""
    if installed is None:
        return ["not_installed"]
    codes = []
    if latest is not None and latest != installed:   # ponytail: string compare, not PEP 440 ordering
        codes.append("update_available")
    if pin is not None and pin != installed:
        codes.append("differs_from_pin")
    if latest is not None and latest == installed:
        codes.append("up_to_date")
    return codes


def rows() -> list[dict]:
    pinned = pins()
    out = []
    for name, eng in ENGINES.items():
        inst, pin = installed_version(name), pinned.get(_norm(name))
        out.append({"name": name, "spec": eng["spec"], "installed": inst, "pin": pin,
                    "post_install": [" ".join(("python",) + c) for c in eng["post"]],
                    "note": eng.get("note"), "status": status(inst, pin)})
    return out


def install_argv(name: str, dry_run: bool) -> list[str]:
    """The upgrade command. uv gets --upgrade-package, which upgrades this
    package and moves its dependencies only as far as it needs (pip's -U
    default); uv's own -U would let every installed package move."""
    spec = _require(name)["spec"]
    dry = ["--dry-run"] if dry_run else []
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", *dry, "--python", sys.executable,
                "--upgrade-package", name, spec]
    return [sys.executable, "-m", "pip", "install", *dry, "-U", spec]


def update_commands(name: str) -> list[list[str]]:
    """Every command the engine_update job runs, in order."""
    eng = _require(name)
    return [install_argv(name, dry_run=False)] + [[sys.executable, *c] for c in eng["post"]]


def parse_dry_run(output: str) -> list[dict]:
    """[{package, from, to}] from uv's ` - x==1` / ` + x==2` lines or pip's
    `Would install x-2 y-3` line (pip names no removals, so `from` is read
    from the installed metadata)."""
    changes: dict[str, dict] = {}
    # uv colours its output even into a pipe (" \x1b[31m-\x1b[39m \x1b[1mscrapling"):
    # left in, no line matched and a real upgrade read as "no changes".
    for line in _ANSI.sub("", output).splitlines():
        m = _UV_LINE.match(line)
        if m:
            sign, pkg, ver = m.groups()
            row = changes.setdefault(_norm(pkg), {"package": pkg, "from": None, "to": None})
            row["from" if sign == "-" else "to"] = ver
            if sign == "~":
                row["from"] = ver
            continue
        if line.strip().startswith("Would install "):
            for tok in line.strip()[len("Would install "):].split():
                pkg, _, ver = tok.rpartition("-")
                if not pkg:                  # uv's summary line: "Would install 2 packages"
                    continue
                changes[_norm(pkg)] = {"package": pkg, "from": installed_version(pkg), "to": ver}
    return list(changes.values())


def latest_version(name: str) -> str:
    import requests
    r = requests.get(f"https://pypi.org/pypi/{name}/json", timeout=PYPI_TIMEOUT_S)
    r.raise_for_status()
    return str(r.json()["info"]["version"])


def check(name: str) -> dict:
    """Latest on PyPI plus a dry run of the upgrade. Both failures are
    reported in `errors`; neither hides the other's result."""
    eng = _require(name)
    inst, pin = installed_version(name), pins().get(_norm(name))
    errors, latest, changes, tail = [], None, [], ""
    try:
        latest = latest_version(name)
    except Exception as e:
        errors.append(f"PyPI lookup failed: {type(e).__name__}: {e}")
    argv = install_argv(name, dry_run=True)
    try:
        p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=DRY_RUN_TIMEOUT_S)
        out = (p.stdout or "") + (p.stderr or "")
        tail = "\n".join(_ANSI.sub("", out).strip().splitlines()[-40:])
        if p.returncode != 0:
            if "no such option: --dry-run" in out:
                errors.append("this venv's pip is older than 22.2 and has no --dry-run: "
                              "upgrade pip first (python -m pip install -U pip)")
            else:
                errors.append(f"dry run failed (exit {p.returncode})")
        else:
            changes = parse_dry_run(out)
    except subprocess.TimeoutExpired:
        errors.append(f"dry run timed out after {DRY_RUN_TIMEOUT_S} s")
    except OSError as e:
        errors.append(f"dry run could not start: {e}")
    codes = ["check_failed"] if errors else []
    return {"ok": not errors, "name": name, "spec": eng["spec"], "installed": inst,
            "pin": pin, "latest": latest, "changes": changes,
            "status": codes + status(inst, pin, latest),
            "command": " ".join(argv), "raw_tail": tail,
            "error": "; ".join(errors) or None, "environment": environment()}


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise SystemExit(f"usage: python -m src.fetch_engines <{'|'.join(ENGINES)}>")
    for cmd in update_commands(argv[0]):
        print("$ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    print(f"{argv[0]}: now {installed_version(argv[0])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
