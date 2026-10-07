"""
console.py — Force UTF-8 on stdout/stderr so output cannot kill a finished run.

Windows picks the stream encoding from the ANSI codepage (GetACP, cp1251 on
this machine) whenever stdout is not a real console — a pipe, a redirect, or
Git Bash's MinTTY. Printing '✅' then raises UnicodeEncodeError, and because
the success messages come LAST, a fully successful ingest exits non-zero with
a traceback after its JSONL is already on disk.

`chcp 65001` does not help: it changes the console output codepage, not
GetACP, so a piped or redirected child still gets cp1251. The job runner
already works around this per-subprocess (PYTHONIOENCODING in manage_api.py);
this is the same fix for hand-run entry points, which inherit no such env.

Not just emoji: the interpolated values are vault paths, filenames and
citation notes lifted from textbooks, so ASCII-only source would still crash.

Usage — first line of every CLI entry point, before anything is written:
    from src.utils.console import force_utf8_console
    force_utf8_console()
"""
from __future__ import annotations

import sys


def force_utf8_console() -> None:
    """Reconfigure stdout/stderr to UTF-8 in place. Idempotent.

    Reconfigures in place rather than rebinding sys.stdout, so streams already
    captured by something else — logging's StreamHandler grabs sys.stderr at
    configure_logging() time — pick the change up through the same object.
    """
    for stream in (sys.stdout, sys.stderr):
        # Test harnesses swap in stream objects that are not TextIOWrapper
        # (pytest's capture) and have nothing to reconfigure.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
