"""One guard for every writer of a chunk file: the console's jobs (manage_api),
main.py's ingest commands, and the loaders' `python -m` entry points.

WHY. Every loader opens its output with "w", and the md export and OCR Studio's
write_chunks replace their whole file. A run that writes a lane's canonical file
from a narrowed scope, or ANOTHER lane's file at all, truncates it: those rows
leave the sparse index (re-derived from the JSONLs) while they stay in the dense
one -- a silent dense/sparse drift from a job that reported success. A scoped md
parse also replaces the whole-vault parents sidecar with a partial one.

Names are compared case-insensitively, and only AFTER the alias spellings are
refused: NTFS opens a file under spellings the by-name checks miss.

Stdlib only (src.index_ops is too); callers pass `cfg` (the console passes its
live CFG, main.py the config it just loaded).
"""
from __future__ import annotations

import fnmatch
import re
import sys
from pathlib import Path

from src.index_ops import HYPE_SIDECAR

# Every chunk-writing lane's canonical chunk file: (the config key that names it,
# the loader's built-in default). One table, so every output guard agrees on which
# file names are protected. "markdown" is the whole-vault parse, chunks.jsonl.
LANE_CHUNK_FILES = {
    "pdf": ("pdf.output_file", "pdf_chunks.jsonl"),
    "notebooks": ("notebooks.output_file", "ipynb_chunks.jsonl"),
    "code": ("code.output_file", "code_chunks.jsonl"),
    "canvas": ("canvas.output_file", "canvas_chunks.jsonl"),
    "markdown": ("paths.chunks_file", "chunks.jsonl"),
}

# One chunk file PER DOCUMENT, so no fixed name to put in the table above, but each is
# owned by one lane just the same: OCR Studio's ocrstudio_<slug>_<key>_chunks.jsonl and the
# canvas re-index's canvas_edit_<slug>_<key>_chunks.jsonl. Another lane pointed at one
# truncates it. Lowercase, matched against the lowercased name.
LANE_CHUNK_PATTERNS = {
    "ocrstudio": "ocrstudio_*_chunks.jsonl",
    "canvas": "canvas_edit_*_chunks.jsonl",
}

PARENTS_FILE = ("retrieval.parents_file", "parents_md.jsonl")


def sidecar_names(cfg) -> set[str]:
    """The lowercased names of the parents and HyPE sidecars: whole-vault stores that a
    delete and a relocate rewrite, so no lane's output and no report may take them."""
    return {Path(cfg.get(PARENTS_FILE[0]) or PARENTS_FILE[1]).name.lower(), HYPE_SIDECAR.lower()}


def refuse_alias(p: str) -> None:
    """Refuse the NTFS alias spellings of a path.

    The console refuses drives and `..` before this; an absolute CLI path is
    legitimate, so one leading drive or UNC share (//server/share, which the
    console's resolve()d paths carry for a data folder on a network or mapped
    drive) and then one leading "/" are dropped first.
    """
    p = p.replace("\\", "/")
    tail = re.sub(r"^(?:[A-Za-z]:|//[^/]+/[^/]+(?=/))", "", p)
    tail = tail[1:] if tail.startswith("/") else tail
    # Windows opens a file under other spellings than the name the lane guards
    # compare: it drops a segment's trailing dots and spaces ("chunks.jsonl." and
    # "chunks.jsonl " are chunks.jsonl, "data./" is data/) and reads "x::$DATA" as x
    # itself. An empty segment ("data//x") makes the tail rooted, and DATA_DIR / "/x"
    # is the drive root.
    if ":" in tail or any(not s or s.endswith((".", " ")) for s in tail.split("/")):
        raise ValueError(f"path must name its file plainly (no ':', no empty "
                         f"segment, no segment ending in '.' or a space): {p!r}")


def refuse_scoped_canonical(cfg, lane: str, out, scoped: bool) -> None:
    """A run narrowed to one folder, file or page range must not write `lane`'s
    canonical file (`out` None = the loader's own default, which IS that file).

    The canonical name is the one the loader would pick itself (config's output
    file, else its built-in default).
    """
    key, default = LANE_CHUNK_FILES[lane]
    canonical = Path(cfg.get(key) or default).name
    if scoped and (out is None or Path(out).name.lower() == canonical.lower()):
        raise ValueError(
            f"a scoped {lane} run must not write "
            f"{canonical} — the loader truncates its output, so everything "
            f"outside this scope would drop out of the sparse index while "
            f"staying in the dense one, and out of this lane's only chunk "
            f"file. Give this run its own output file (name it *_chunks.jsonl "
            f"so the sparse rebuild picks it up).")


def refuse_other_lanes_file(cfg, out: str | None, lane: str) -> None:
    """Refuse an --output that is ANOTHER lane's canonical chunk file.

    Every loader opens its output with "w", so a run pointed at a different
    lane's file (or at chunks.jsonl) replaces that lane's rows with its own: they
    drop out of the sparse index, which is re-derived from the JSONLs, while they
    stay in the dense one. This holds for a whole-lane run as much as a scoped
    one. `lane`'s OWN file is the caller's business (a whole-lane run may write
    it; a scoped run is refused by refuse_scoped_canonical). A `lane` outside the
    table (OCR Studio) makes every lane "other". Names are compared
    case-insensitively because NTFS is, and against the file each lane would
    write itself (config's output file, else the built-in default) or, for the
    per-document files, the pattern its lane names them by (LANE_CHUNK_PATTERNS).
    The parents and HyPE sidecars (sidecar_names) are no lane's to write either, nor is
    the route pointer sidecar (kept out of sidecar_names: delete and relocate never
    rewrite it).
    """
    if out is None:
        return
    name = Path(out).name.lower()
    if name in sidecar_names(cfg):
        raise ValueError(
            f"{Path(out).name} is the parents or HyPE sidecar, and the loader truncates its "
            f"output: this {lane} run would replace that store with chunk rows. Give this run "
            f"its own output file (name it *_chunks.jsonl so the sparse rebuild picks it up).")
    if name == Path(cfg.get("retrieval.route.pointers_file") or "route_pointers.json").name.lower():
        raise ValueError(
            f"{Path(out).name} is the route pointer sidecar (retrieval.route.pointers_file), and "
            f"the loader truncates its output: this {lane} run would replace it with chunk rows. "
            f"Give this run its own output file (name it *_chunks.jsonl so the sparse rebuild "
            f"picks it up).")
    owners = sorted({other for other, (key, default) in LANE_CHUNK_FILES.items()
                     if other != lane
                     and Path(cfg.get(key) or default).name.lower() == name}
                    | {other for other, pattern in LANE_CHUNK_PATTERNS.items()
                       if other != lane and fnmatch.fnmatchcase(name, pattern)})
    if owners:
        raise ValueError(
            f"{Path(out).name} is the chunk file of the {' and '.join(owners)} lane, "
            f"and the loader truncates its output: this {lane} run would wipe that "
            f"lane's rows out of the sparse index while they stay in the dense one. "
            f"Give this run its own output file (name it *_chunks.jsonl so the "
            f"sparse rebuild picks it up).")


def refuse_scoped_parents(cfg, parents_out, scoped: bool) -> None:
    """A scoped md parse writes a PARTIAL parents sidecar, which replaces the
    whole-vault one."""
    if (scoped and parents_out
            and Path(parents_out).name.lower()
            == Path(cfg.get(PARENTS_FILE[0]) or PARENTS_FILE[1]).name.lower()):
        raise ValueError("a scoped md parse writes a partial parents sidecar, "
                         "which replaces the whole-vault one: pass --parents-out ''")


def refuse_parents_out_clash(cfg, parents_out, out) -> None:
    """The md export opens --parents-out with "w" too: it must not be any lane's chunk
    file (this run's own -o included), a *_chunks.jsonl (the sparse union would read
    parent rows as chunks) or the HyPE sidecar."""
    name = Path(parents_out).name.lower()
    taken = {Path(cfg.get(key) or default).name.lower() for key, default in LANE_CHUNK_FILES.values()}
    taken |= {HYPE_SIDECAR.lower()} | ({Path(out).name.lower()} if out is not None else set())
    if name in taken or name.endswith("_chunks.jsonl"):
        raise ValueError(
            f"--parents-out {Path(parents_out).name} would overwrite a chunk file or the HyPE "
            f"sidecar, or join the sparse union: name it {PARENTS_FILE[1]} (a whole-vault "
            f"run) or pass --parents-out ''")


def check_output(cfg, lane: str, out, *, scoped: bool, parents_out=None) -> None:
    """Every refusal that applies to one run's outputs. Raises ValueError."""
    if out is not None:
        refuse_alias(str(out))
    if lane in LANE_CHUNK_FILES:
        refuse_scoped_canonical(cfg, lane, out, scoped)
    refuse_other_lanes_file(cfg, out, lane)
    if parents_out:
        refuse_alias(str(parents_out))
        refuse_parents_out_clash(cfg, parents_out, out)
    refuse_scoped_parents(cfg, parents_out, scoped)


def exit_if_refused(cfg, lane: str, out, *, scoped: bool, parents_out=None,
                    note: str = "") -> None:
    """The one entry point for every CLI caller: check_output, and on a refusal
    print `ERROR: <message>` (then `note`, when given) to stderr and exit 2."""
    try:
        check_output(cfg, lane, out, scoped=scoped, parents_out=parents_out)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if note:
            print(note, file=sys.stderr)
        raise SystemExit(2) from e
