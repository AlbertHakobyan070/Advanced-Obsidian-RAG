"""Metadata-only maintenance of the stores a chunk lives in: every chunk JSONL,
parents, the HyPE sidecar, every Chroma collection.

Stdlib only at import time. Chroma is reached only through the handle the caller
opened with src.utils.chroma_client.persistent_client (tests/test_chroma_telemetry.py
fails any other way in), and src.embeddings.sidecar is imported where it is needed.

THE .bak NAMING SCHEME for every JSONL rewrite (rewrite_jsonl): the original is
copied to <name>.<tag>-<YYYYmmdd-HHMMSS>.bak beside it. That name never matches
*_chunks.jsonl, so the sparse union (build_sparse_union, rebuild_bm25) cannot
pick a backup up. Nothing deletes them: only the author does.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

HYPE_SIDECAR = "hype_questions.jsonl"   # build_hype.py:160 writes it beside the chunk files
PAGE = 5000
DROP = object()                         # rewrite_jsonl: a fix that returns DROP removes the line
# rag.bat and main.py serve run "streamlit run app.py" with no --server.port and the repo has no
# .streamlit/config.toml, so Streamlit's default applies (config.yaml:630 names it)
STREAMLIT_PORT = 8501


# ---- where the rows live ------------------------------------------------------------

def parents_path(cfg) -> Path:
    """The parents sidecar: the same rule as ParentContext.from_config."""
    return (cfg.path("retrieval.parents_file") if cfg.get("retrieval.parents_file")
            else cfg.project_root / "data" / "parents_md.jsonl")


def chunk_jsonls(cfg) -> list[Path]:
    """chunks.jsonl plus every *_chunks.jsonl beside it: the same union as
    embedder.build_sparse_union and manage_api.chunk_files."""
    base = cfg.path("paths.chunks_file")
    files = [base] if base.exists() else []
    return files + sorted(p for p in base.parent.glob("*_chunks.jsonl") if p != base)


def rewrite_jsonl(path, fix, *, dry_run: bool = False, tag: str = "bak") -> int:
    """Stream `path` through `fix` and replace the file only when a line changed.

    `fix(rec)` gets each parsed record and returns a new dict (replace the line),
    None (unchanged) or DROP (omit the line). Unchanged, blank and unparseable lines
    pass through byte-for-byte; a replaced line keeps its own terminator (the live
    JSONLs are CRLF and a rewrite never mixes endings). Lines split on b"\\n" only,
    like manage_api.iter_jsonl_lines. Returns the replaced plus dropped line count.

    Only when that count is > 0 and not dry_run: the new file is written beside the
    old as <name>.tmp, the original is copied to <name>.<tag>-<stamp>.bak
    (FileExistsError if that name is taken: a second rewrite inside one second;
    every caller is idempotent, so a retry heals), then the tmp replaces it."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    out = None if dry_run else open(tmp, "wb")
    try:
        with open(path, "rb") as f:
            for raw in f:
                keep = raw
                if raw.strip():
                    try:
                        rec = json.loads(raw.decode("utf-8", errors="replace"))
                    except json.JSONDecodeError:
                        rec = None
                    new = fix(rec) if isinstance(rec, dict) else None
                    if new is DROP:
                        n += 1
                        continue
                    if new is not None:
                        if not isinstance(new, dict):
                            raise TypeError(f"{path.name}: a fix returns a dict, None or DROP, "
                                            f"got {type(new).__name__}")
                        n += 1
                        end = b"\r\n" if raw.endswith(b"\r\n") else b"\n" if raw.endswith(b"\n") else b""
                        keep = json.dumps(new, ensure_ascii=False).encode("utf-8") + end
                if out is not None:
                    out.write(keep)
        if out is not None:
            out.close()
            if n:
                bak = path.with_name(f"{path.name}.{tag}-{time.strftime('%Y%m%d-%H%M%S')}.bak")
                if bak.exists():
                    raise FileExistsError(f"{bak} exists: {path.name} was rewritten twice inside "
                                          f"one second. Retry in a second; {path.name} is as it was.")
                shutil.copy2(path, bak)
                os.replace(tmp, path)
    finally:
        if out is not None:
            out.close()
            tmp.unlink(missing_ok=True)
    return n


# ---- who else is using the store ----------------------------------------------------

def answering_ports(cfg) -> list[int]:
    """The ports of :8051 (webui.rag_api), :8052 (webui.port) and Streamlit that
    accept a connection. A live reader blocks Chroma's writes, so a CLI writer asks
    this first. A port is up iff a TCP connect succeeds inside 0.5 s.

    The defaults mirror manage_api.py (8051, 8052). 8501 is an assumption (see
    STREAMLIT_PORT): a session started with --server.port elsewhere is invisible here."""
    api = urlsplit(cfg.get("webui.rag_api", "http://127.0.0.1:8051"))
    if not api.hostname:
        raise ValueError(f"webui.rag_api must be a URL such as http://127.0.0.1:8051, "
                         f"got {cfg.get('webui.rag_api')!r}")
    targets = [(api.hostname, api.port or (443 if api.scheme == "https" else 80)),
               ("127.0.0.1", int(cfg.get("webui.port", 8052))),
               ("127.0.0.1", STREAMLIT_PORT)]
    up = []
    for host, port in targets:
        try:
            socket.create_connection((host, port), timeout=0.5).close()
        except OSError:                  # refused, timed out, unresolvable: nothing answers there
            continue
        up.append(port)
    return up


class StoreBusy(RuntimeError):
    """A collection is mid re-embed, so a write now would race its upserts."""


def refuse_if_building(chroma_dir, client) -> None:
    """Raise StoreBusy when any collection's sidecar says status "building". A
    re-embed writes "building" before it creates the collection (reembed.py), and a
    --limit smoke test is left "building" on purpose (main.py), so it blocks too until
    that collection is deleted. An unreadable sidecar raises read_sidecar's ValueError:
    it is never read as "not building". The graph twin (<collection>__graph) is built by
    the graph_embed job, which --rebuild-into refuses, so its remedy names that job."""
    from src.embeddings.sidecar import GRAPH_SUFFIX, read_sidecar
    building = [c.name for c in client.list_collections()
                if (read_sidecar(chroma_dir, c.name) or {}).get("status") == "building"]
    if building:
        how = "; ".join(
            f"{n}: wait for the running graph_embed job, or run main.py graph-embed (it rebuilds "
            f"the twin from scratch)" if n.endswith(GRAPH_SUFFIX)
            else f"{n}: finish it (main.py index --rebuild-into {n} resumes)" for n in building)
        raise StoreBusy(
            f"{', '.join(building)} mid re-embed (status building). {how}; or delete that "
            f"collection, then retry. A write now would race its upserts.")


# ---- the duplicate-safe append --------------------------------------------------------

class ReplaceRequired(RuntimeError):
    """`index --append` would keep rows that a REPLACE removes."""


def append_conflicts(col, chunks) -> list[str]:
    """One line per source in `chunks` whose indexed rows would survive beside the
    new ones: rows that are in the collection but not in the incoming file. A re-ingest
    of an edited or relabelled file mints new doc_ids (they hash the text and the
    header), so an append would leave the old rows beside the new ones.

    Only the sources PRESENT in `chunks` are compared. A file renamed or moved in the
    vault arrives under another source_file, its old rows are not found here, and the
    append passes: such a file goes through `main.py relocate` (or a delete of the old
    source) BEFORE it is re-ingested.

    Whole-file lanes (md, code, notebooks, canvas, image): every indexed id of the
    source that is not incoming. pdf: only the indexed ids on a page the incoming file
    covers, so a page-range top-up of NEW pages passes; a conflicting pdf line also counts
    the indexed rows on pages the file does not bring, which a delete of the source drops.

    And an incoming id already stored under ANOTHER source_file (relocate keeps the ids
    it moves, which hash the old path): the upsert would move that row back silently."""
    incoming: dict[str, set[str]] = {}
    pages: dict[str, set] = {}
    source_of: dict[str, str] = {}
    for c in chunks:
        meta = c.metadata or {}
        sf = str(meta.get("source_file", ""))
        incoming.setdefault(sf, set()).add(c.id)
        source_of[c.id] = sf
        if meta.get("file_type") == "pdf":
            pages.setdefault(sf, set()).add(meta.get("page_start"))
    sources = sorted(incoming)
    lines = []
    for k in range(0, len(sources), 100):
        got = col.get(where={"source_file": {"$in": sources[k:k + 100]}}, include=["metadatas"])
        stale: dict[str, list[str]] = {}
        uncovered: dict[str, int] = {}
        for i, m in zip(got["ids"], got["metadatas"]):
            sf = str((m or {}).get("source_file", ""))
            if i in incoming[sf]:
                continue
            if sf in pages and (m or {}).get("page_start") not in pages[sf]:
                uncovered[sf] = uncovered.get(sf, 0) + 1
                continue
            stale.setdefault(sf, []).append(i)
        lines += [f"{sf}: {len(ids)} indexed row(s) are not in this file (e.g. {min(ids)})"
                  + (f"; {uncovered[sf]} more on pages this file does not cover, which deleting "
                     f"the source drops too: first re-ingest a page range that covers every "
                     f"indexed page" if uncovered.get(sf) else "")
                  for sf, ids in sorted(stale.items())]
    every_id = sorted(source_of)
    moved: dict[tuple[str, str], list[str]] = {}
    for k in range(0, len(every_id), 500):
        got = col.get(ids=every_id[k:k + 500], include=["metadatas"])
        for i, m in zip(got["ids"], got["metadatas"]):
            stored = str((m or {}).get("source_file", ""))
            if stored != source_of[i]:
                moved.setdefault((source_of[i], stored), []).append(i)
    lines += [f"{sf}: {len(ids)} incoming id(s) are indexed under {other} (e.g. {min(ids)}): the "
              f"same text at another path, e.g. a relocated copy, and appending would move those "
              f"rows to {sf}; delete one of the two first" for (sf, other), ids in sorted(moved.items())]
    return lines


# ---- relocate: the path metadata of indexed files, and nothing else ------------------------
# Ids, text and vectors are never touched. main.py relocate is the only caller.

class RelocateError(RuntimeError):
    """A relocate precondition or check failed."""


def _posix(sf: str) -> str:
    return sf.replace("\\", "/")


def _link_stem(link) -> str:
    """The note name a wikilink points at: before any #heading or |alias, without the '\\' a
    markdown table escapes the alias pipe with ([[note\\|x]] is stored "note\\"), last path segment."""
    return str(link).split("#")[0].split("|")[0].rstrip("\\").rsplit("/", 1)[-1].strip().lower()


def load_map(path) -> tuple[list[dict], list[dict]]:
    """Parse a rename_map.jsonl into (moves, quarantined).

    moves: kind "file", indexed true, op move or rename; each must carry non-empty str
    old, new, old_sf, new_sf and lane. quarantined: the indexed op "quarantine" rows,
    which only need old_sf: they are reported, not handled, because their index side is
    a delete. kind "dir" rows are filesystem-only and ignored. A malformed line raises
    RelocateError naming its line number."""
    moves, quarantined = [], []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise RelocateError(f"{path}: line {n} is not valid JSON ({e.msg}): fix or remove it") from e
        if not isinstance(row, dict):
            raise RelocateError(f"{path}: line {n} is not a JSON object: fix or remove it")
        op = row.get("op")
        if row.get("kind") != "file" or row.get("indexed") is not True \
                or op not in ("move", "rename", "quarantine"):
            continue
        need = ("old_sf",) if op == "quarantine" else ("old", "new", "old_sf", "new_sf", "lane")
        bad = [k for k in need if not (isinstance(row.get(k), str) and row[k])]
        if bad:
            raise RelocateError(f"{path}: line {n}: {', '.join(bad)} must be a non-empty string")
        if type(row.get("n_chunks") or 0) is not int:
            raise RelocateError(f"{path}: line {n}: n_chunks must be an integer, got {row['n_chunks']!r}")
        (quarantined if op == "quarantine" else moves).append(row)
    return moves, quarantined


_MD_TYPES = ("note", "daily_note")


def _relocate_canvas_refs(meta: dict, old_sf: str, new_sf: str) -> dict | None:
    """A canvas row's references to a moved file, rewritten to new_sf: canvas_edge_files
    (the key bridges read) and canvas_file_refs, each kept in the shape it was read in (a
    JSONL list, or Chroma's ", "-joined string). None = not a canvas, or it cites nothing
    at old_sf. The text stays: its footer keeps the old path until that canvas is next
    REPLACEd (contract 2.6)."""
    if meta.get("file_type") != "canvas":
        return None
    from src.ingestion.canvas_loader import decode_canvas_files, encode_canvas_edge, norm_vault_path
    old_n, new_p = norm_vault_path(old_sf), _posix(new_sf)      # a canvas file field uses /
    changed = {}
    files = decode_canvas_files(meta.get("canvas_edge_files"))
    if any(norm_vault_path(e["file"]) == old_n for e in files):
        enc = [encode_canvas_edge(e["node"], new_p if norm_vault_path(e["file"]) == old_n else e["file"],
                                  e["subpath"]) for e in files]
        changed["canvas_edge_files"] = enc if isinstance(meta["canvas_edge_files"], list) else ", ".join(enc)
    refs = meta.get("canvas_file_refs")
    # ponytail: a moved path that itself contains ", " is not found in Chroma's lossy canvas_file_refs
    # join; canvas_edge_files, the key bridges read, is escaped and always rewritten
    parts = refs if isinstance(refs, list) else str(refs).split(", ") if refs else []
    if any(norm_vault_path(r) == old_n for r in parts):
        moved = [new_p if norm_vault_path(r) == old_n else r for r in parts]
        changed["canvas_file_refs"] = moved if isinstance(refs, list) else ", ".join(moved)
    return dict(meta, **changed) if changed else None


def relocate_meta(meta: dict, old_sf: str, new_sf: str) -> dict | None:
    """The path-bearing keys of one chunk's metadata, moved from old_sf to new_sf.
    None = not this file. doc_id, text and every other key are untouched.

    Deviation from contract 2.3 step 2 ("nothing else"): `heading` is also rewritten
    when it equals the old title. Code and notebook citations are `course > heading`
    (generator.py) and their heading is the file title; a pdf with no chapter has
    heading == stem. Without this, moved code and notebooks keep citing the old name.

    A canvas row that cites old_sf (never one of its own rows: canvas moves are refused)
    falls through to _relocate_canvas_refs."""
    if meta.get("source_file") != old_sf:
        return _relocate_canvas_refs(meta, old_sf, new_sf)
    out = dict(meta, source_file=new_sf)
    old_p, new_p = PurePosixPath(_posix(old_sf)), PurePosixPath(_posix(new_sf))
    for old_t, new_t in ((old_p.name, new_p.name), (old_p.stem, new_p.stem)):  # code keeps the extension
        if meta.get("filename") != old_t:
            continue
        out["filename"] = new_t
        if meta.get("file_type") not in _MD_TYPES:   # md heading_path holds headings, not the stem
            if meta.get("heading") == old_t:
                out["heading"] = new_t
            hp = meta.get("heading_path")
            if isinstance(hp, str) and (hp == old_t or hp.startswith(old_t + " > ")):
                out["heading_path"] = new_t + hp[len(old_t):]
        break
    return out


def _lines(path):
    """(raw line, parsed record) for every non-blank line of a JSONL; the record is None
    when the line is not a JSON object (rewrite_jsonl passes such a line through)."""
    with open(path, "rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                rec = None
            yield raw, rec if isinstance(rec, dict) else None


def _side_files(cfg) -> list[tuple[str, Path]]:
    """The sidecars that exist and keep source_file at the TOP of each row: parents, HyPE."""
    hype = cfg.path("paths.chunks_file").parent / HYPE_SIDECAR
    return [(k, p) for k, p in (("parents", parents_path(cfg)), ("hype", hype)) if p.exists()]


def _ocr_docs(cfg) -> dict:
    """OCR Studio: the `source` of every doc.json -> its folder."""
    from src.ocr_studio.pages import store_root
    out = {}
    for f in sorted(store_root(cfg).glob("*/doc.json")):
        try:
            out[json.loads(f.read_text(encoding="utf-8"))["source"]] = f.parent
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise RelocateError(f"{f}: not a readable OCR Studio doc.json ({e!r}): repair or "
                                f"remove that document's folder, then retry") from e
    return out


def plan(cfg, moves, quarantined, client) -> dict:
    """Read-only. {"rows": [...], "totals": {...}, "blockers": [str], "warnings": [str],
    "canvas_ref_rows": [(jsonl name, doc_id)]}.

    One row per move with the counts in every store (chunk JSONLs, parents, HyPE sidecar,
    every Chroma collection), the OCR Studio folder, the disk state and the eval gold
    lines. totals holds the batch sums the dry-run prints: "jsonl" (all JSONL rows under
    an old_sf), "parents", "hype" and one entry per collection. apply() refuses to run
    while blockers is non-empty and checks its rewrites against totals["jsonl"] plus
    canvas_ref_rows, one entry per canvas LINE that cites a moved file (rewrite_jsonl rewrites
    lines, so an id on two lines, of one JSONL or of two, is two entries; a line citing two
    moves is one; rows[i]["canvas_refs"] counts them per move)."""
    from src.embeddings.sidecar import role_for
    from src.ingestion.obsidian_parser import MAX_WIKILINKS, extract_wikilinks, parse_frontmatter
    from src.ocr_studio.pages import doc_key, store_root

    chroma_dir, vault = cfg.path("paths.chroma_dir"), cfg.path("parser.vault_path")
    active = cfg.get("paths.collection_name")
    blockers, global_warnings = [], []

    # -- preconditions and the shape of the map
    up = answering_ports(cfg)
    if up:
        ports = ", ".join(":" + str(p) for p in up)
        blockers.append(f"{ports} answers. Stop :8051, :8052 and Streamlit first "
                        f"(a live reader blocks Chroma's writes)")
    try:
        refuse_if_building(chroma_dir, client)
    except StoreBusy as e:
        blockers.append(str(e))
    olds = Counter(m["old_sf"] for m in moves)
    news = Counter(m["new_sf"].lower() for m in moves)
    blockers += [f"duplicate old_sf in the map: {k}" for k, c in olds.items() if c > 1]
    blockers += [f"{k}: two rows share this new_sf (they may differ only in case: NTFS treats "
                 f"them as one file)" for k, c in news.items() if c > 1]
    blockers += [f"{m['new_sf']}: new_sf is also an old_sf in this map (a chain, or no move at "
                 f"all): split the batch" for m in moves if m["new_sf"] in olds]

    rows = [{"n": n, "old": m["old"], "new": m["new"], "old_sf": m["old_sf"], "new_sf": m["new_sf"],
             "lane": m["lane"], "n_chunks": m.get("n_chunks") or 0, "jsonl": {}, "types": set(),
             "parents": 0, "hype": 0, "cols": {}, "new_in": {}, "eval": 0, "warnings": [], "canvas_refs": 0}
            for n, m in enumerate(moves, 1)]
    ids = [set() for _ in moves]
    by_old = {m["old_sf"]: i for i, m in enumerate(moves)}
    by_new = {m["new_sf"]: i for i, m in enumerate(moves)}
    # old stem AND old basename -> the moves whose basename changes: Obsidian links a pdf, code
    # or notebook file only with its extension ([[Book.pdf]]), and a note with or without it
    stems = {}
    for i, m in enumerate(moves):
        old = PurePosixPath(m["old"])
        if old.name != PurePosixPath(m["new"]).name:
            for key in {old.stem.lower(), old.name.lower()}:
                stems.setdefault(key, []).append(i)

    # -- one streaming pass per chunk JSONL for every row, then the two sidecars
    cites, links, unparseable, capped = set(), set(), {}, set()
    canvas_ref_rows = []                        # (jsonl name, doc_id) per canvas line apply rewrites
    for f in chunk_jsonls(cfg):
        for raw, rec in _lines(f):
            if rec is None:
                unparseable[f.name] = unparseable.get(f.name, 0) + 1
                continue
            meta = rec.get("metadata") or {}
            sf = meta.get("source_file")
            if sf in by_old:
                r = rows[by_old[sf]]
                r["jsonl"][f.name] = r["jsonl"].get(f.name, 0) + 1
                ids[by_old[sf]].add(rec.get("doc_id"))
                r["types"].add(meta.get("file_type"))
            if sf in by_new:
                new_in = rows[by_new[sf]]["new_in"]
                new_in[f.name] = new_in.get(f.name, 0) + 1
            if meta.get("file_type") == "canvas":   # only canvas rows meet every move: apply's predicate
                hits = [i for i, m in enumerate(moves)
                        if relocate_meta(meta, m["old_sf"], m["new_sf"]) is not None]
                if hits:
                    canvas_ref_rows.append((f.name, rec.get("doc_id")))
                for i in hits:
                    cites.add((sf, i))
                    rows[i]["canvas_refs"] += 1
            if isinstance(meta.get("wikilinks"), list):
                if isinstance(sf, str) and len(meta["wikilinks"]) >= MAX_WIKILINKS:
                    capped.add(sf)
                for link in meta["wikilinks"]:
                    links.update((sf, i) for i in stems.get(_link_stem(link), ()))
    # The metadata keeps only a note's first MAX_WIKILINKS links, so a note at the cap may link
    # the renamed file further down: read the note itself (a missing one is said, not skipped).
    for sf in (sorted(capped) if stems else ()):          # only a basename change breaks a link
        note = vault / _posix(sf)
        if not note.is_file():
            global_warnings.append(
                f"{sf}: the index keeps only its first {MAX_WIKILINKS} links and the note is not on "
                f"disk ({note}), so a later link to a renamed file could not be checked")
            continue
        body = parse_frontmatter(note.read_text(encoding="utf-8", errors="replace"))[1]
        for link in extract_wikilinks(body):
            links.update((sf, i) for i in stems.get(_link_stem(link), ()))
    for key, f in _side_files(cfg):
        for raw, rec in _lines(f):
            if rec is None:
                unparseable[f.name] = unparseable.get(f.name, 0) + 1
                continue
            sf = rec.get("source_file")
            if sf in by_old:
                rows[by_old[sf]][key] += 1
            if sf in by_new:
                new_in = rows[by_new[sf]]["new_in"]
                new_in[key] = new_in.get(key, 0) + 1

    # -- every collection in the store (re-embeds, HyPE questions and the graph twin too)
    cols = {c.name: client.get_collection(c.name) for c in client.list_collections()}
    if active not in cols:
        blockers.append(f"paths.collection_name {active!r} is not a collection in {chroma_dir}")
    for r in rows:
        for name, col in cols.items():
            r["cols"][name] = len(col.get(where={"source_file": r["old_sf"]}, include=[])["ids"])
            k = len(col.get(where={"source_file": r["new_sf"]}, include=[])["ids"])
            if k:
                r["new_in"][name] = k

    # -- OCR Studio, the disk, the eval gold
    docs, ocr_root = _ocr_docs(cfg), store_root(cfg)
    gold = [line for y in sorted(cfg.path("eval.sets_dir", "eval/sets").glob("*.yaml"))
            for line in y.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        r["ocr_dir"] = docs.get(r["old"]) or docs.get(_posix(r["old_sf"]))
        r["ocr_target"] = ocr_root / doc_key(r["new"])
        r["disk_new"] = (vault / r["new"]).exists()
        r["disk_old_gone"] = not any((vault / p).exists() for p in {r["old"], _posix(r["old_sf"])})
        r["eval"] = sum(r["old_sf"] in line for line in gold)

    # -- blockers and warnings, row by row
    for r, m, rid in zip(rows, moves, ids):
        tag = f"[{r['n']}] {r['old_sf']}: "
        r["ids"], r["types"] = len(rid), sorted(t for t in r["types"] if t)
        found = r["jsonl"] or r["parents"] or r["hype"] or any(r["cols"].values())
        why = []
        if m.get("approved") is not True:
            why.append("not approved (only the author sets approved: true)")
        if r["lane"] == "canvas" or "canvas" in r["types"]:
            why.append("canvas moves are REPLACE: the path is part of every node id (contract 1.4)")
        elif r["new_sf"] != r["new"].replace("/", "\\"):
            why.append(f"new_sf must be new with every / replaced by a backslash, got {r['new_sf']!r}")
        if not r["disk_new"]:
            why.append(f"new file missing on disk: {r['new']}")
        if not r["disk_old_gone"]:
            why.append(f"old file still on disk: {r['old']}")
        if r["new_in"]:
            why.append("new_sf already present (" + ", ".join(f"{k} {v}" for k, v in r["new_in"].items())
                       + "): a collision, or a half-done run; restore the relocate .bak files")
        if not found:
            why.append("old_sf found in no store")
        if r["n_chunks"] > 0 and r["ids"] != r["n_chunks"]:
            why.append(f"n_chunks {r['n_chunks']} but the JSONLs hold {r['ids']} unique id(s)")
        if active in cols and r["cols"][active] != r["ids"]:
            why.append(f"{active} holds {r['cols'][active]} row(s) but the JSONLs hold {r['ids']} "
                       f"unique id(s): dense and sparse disagree for this file")
        if r["ocr_target"].exists():
            why.append(f"OCR Studio target folder exists: {r['ocr_target']}")
        blockers += [tag + w for w in why]
        for name, n in r["cols"].items():        # HyPE holds questions and the graph twin only canvas rows
            if name != active and role_for(cfg, name) == "chunks" and n != r["ids"]:
                r["warnings"].append(f"{tag}{name} holds {n} row(s), the JSONLs {r['ids']}: it lags the "
                                     f"active collection (re-embed top-up, D12)")
        if r["eval"]:
            r["warnings"].append(f"{tag}eval gold names it on {r['eval']} line(s): update the gold (D10)")
    for sf, i in sorted(cites, key=lambda t: (str(t[0]), t[1])):
        rows[i]["warnings"].append(
            f"{sf} cites {_posix(moves[i]['old_sf'])}: its canvas_edge_files / canvas_file_refs are "
            f"rewritten to {_posix(moves[i]['new_sf'])}; its footer text keeps the old path until that "
            f"canvas is next REPLACEd (console delete, then ingest-canvas + index --append with every "
            f"server down)")
    for sf, i in sorted(links, key=lambda t: (str(t[0]), t[1])):
        stem = PurePosixPath(moves[i]["old"]).stem
        rows[i]["warnings"].append(
            f"{sf} links [[{stem}]]: a rename REPLACEs the referring notes (decision 7)")
    global_warnings += [f"{name}: {n} unparseable line(s); relocate passes them through untouched and "
                        f"cannot see a path in them" for name, n in sorted(unparseable.items())]
    if quarantined:
        global_warnings.append(
            f"{len(quarantined)} indexed quarantine row(s) are not handled here: their sources need "
            f"POST /api/documents/delete (first: {quarantined[0]['old_sf']})")

    totals = {"jsonl": sum(sum(r["jsonl"].values()) for r in rows),
              "parents": sum(r["parents"] for r in rows), "hype": sum(r["hype"] for r in rows)}
    totals.update({name: sum(r["cols"][name] for r in rows) for name in cols})
    return {"rows": rows, "totals": totals, "blockers": blockers,
            "warnings": [w for r in rows for w in r["warnings"]] + global_warnings,
            "canvas_ref_rows": sorted(canvas_ref_rows, key=lambda t: (t[0], str(t[1])))}


def apply(cfg, moves, quarantined, client) -> dict:
    """Move the path metadata of every file in `moves`, in the contract's order: every
    chunk JSONL, parents, the HyPE sidecar, every Chroma collection, OCR Studio. Runs
    plan() first and raises RelocateError (blockers, one per line) if there are any:
    nothing is written before that point. Returns the per-store totals.

    Not resumable by design: an interrupted apply is rolled back by restoring BOTH
    backups (the relocate .bak files and the store backup, contract 2.5) and re-running;
    a half-done state trips the "new_sf already present" blocker."""
    from src.embeddings.sidecar import role_for
    from src.ocr_studio.pages import atomic_write
    p = plan(cfg, moves, quarantined, client)
    if p["blockers"]:
        raise RelocateError("relocate refused, nothing was written:\n" + "\n".join(p["blockers"]))
    by_old = {m["old_sf"]: m["new_sf"] for m in moves}

    def cite_moves(meta):                        # a canvas can cite several moved files
        hit = False
        for mv in moves:
            r = relocate_meta(meta, mv["old_sf"], mv["new_sf"])
            if r is not None:
                meta, hit = r, True
        return meta if hit else None

    def fix(rec):                                # one dict lookup per row; the fold is canvas-only
        meta = rec.get("metadata") or {}
        new_sf = by_old.get(meta.get("source_file"))
        if new_sf is not None:
            return dict(rec, metadata=relocate_meta(meta, meta["source_file"], new_sf))
        if meta.get("file_type") != "canvas":
            return None
        moved = cite_moves(meta)
        return None if moved is None else dict(rec, metadata=moved)

    def fix_top(rec):                            # parents and HyPE rows keep source_file at the top
        new_sf = by_old.get(rec.get("source_file"))
        return None if new_sf is None else dict(rec, source_file=new_sf)

    done = {"jsonl": {}, "parents": 0, "hype": 0, "collections": {}, "ocr_studio": 0}
    for f in chunk_jsonls(cfg):
        n = rewrite_jsonl(f, fix, tag="relocate")
        if n:
            done["jsonl"][f.name] = n
    changed, own, refs = sum(done["jsonl"].values()), p["totals"]["jsonl"], len(p["canvas_ref_rows"])
    if changed != own + refs:                    # contract 2.3 step 2, before any other store is written
        raise RelocateError(f"changed rows {changed} ≠ planned rows {own + refs} (own {own} + canvas refs "
                            f"{refs}): restore the relocate .bak files (contract 2.5)")
    for key, f in _side_files(cfg):
        done[key] = rewrite_jsonl(f, fix_top, tag="relocate")
    for c in client.list_collections():
        col = client.get_collection(c.name)
        done["collections"][c.name] = 0
        for m in moves:
            got = col.get(where={"source_file": m["old_sf"]}, include=["metadatas"])
            for k in range(0, len(got["ids"]), PAGE):
                col.update(ids=got["ids"][k:k + PAGE],
                           metadatas=[relocate_meta(x, m["old_sf"], m["new_sf"])
                                      for x in got["metadatas"][k:k + PAGE]])
            done["collections"][c.name] += len(got["ids"])
    for c in client.list_collections():          # canvas rows citing a moved file; the twin's metadata is never read
        if role_for(cfg, c.name) != "chunks":
            continue
        col = client.get_collection(c.name)
        got = col.get(where={"file_type": "canvas"}, include=["metadatas"])
        upd = [(i, moved) for i, m in zip(got["ids"], got["metadatas"])
               if (moved := cite_moves(m or {})) is not None]
        for k in range(0, len(upd), PAGE):
            col.update(ids=[i for i, _ in upd[k:k + PAGE]], metadatas=[m for _, m in upd[k:k + PAGE]])
    for r in p["rows"]:
        if r["ocr_dir"] is None:
            continue
        os.rename(r["ocr_dir"], r["ocr_target"])
        doc = r["ocr_target"] / "doc.json"
        meta = json.loads(doc.read_text(encoding="utf-8"))      # a missing file raises: no fallback
        meta["source"] = r["new"]
        atomic_write(doc, json.dumps(meta, ensure_ascii=False).encode("utf-8"))
        done["ocr_studio"] += 1
    return done


def verify(cfg, moves, quarantined, client) -> list[str]:
    """After apply and rebuild_bm25.py: the list of failures (empty = pass). No old
    stored path remains in any parsed row of a chunk JSONL, parents or HyPE file, nor in
    any collection; no unparseable line of those files still holds one (as its quoted
    JSON form: parseable lines are left to the parsed check, because canvas reference
    keys legitimately hold vault paths); the new files exist, their ids match n_chunks
    and the active collection; no OCR Studio doc.json names an old path; every
    quarantine row's source is gone; no canvas row of a chunk JSONL or of the active
    collection still cites a moved old_sf in canvas_edge_files; the sparse index is not
    older than the JSONLs."""
    from src.ingestion.canvas_loader import decode_canvas_files, norm_vault_path
    vault, active = cfg.path("parser.vault_path"), cfg.get("paths.collection_name")
    moved_n = {norm_vault_path(m["old_sf"]): m["old_sf"] for m in moves}

    def still_cites(store, meta) -> list[str]:   # a canvas whose bridges would still aim at a moved file
        if meta.get("file_type") != "canvas":
            return []
        hit = dict.fromkeys(norm_vault_path(e["file"]) for e in decode_canvas_files(meta.get("canvas_edge_files")))
        return [f"{store}: canvas {meta.get('source_file')} node {meta.get('canvas_node_id')} still cites "
                f"{moved_n[n]}" for n in hit if n in moved_n]

    gone = moves + quarantined
    olds = {sf for m in gone for sf in (m["old_sf"], _posix(m["old_sf"]))}
    quoted = [(m["old_sf"], [json.dumps(m["old_sf"], ensure_ascii=a).encode("utf-8") for a in (False, True)])
              for m in gone]
    new_ids = {m["new_sf"]: set() for m in moves}
    fails, left, named, stale = [], {}, {}, []
    chunk_files = chunk_jsonls(cfg)
    for f, top in [(f, False) for f in chunk_files] + [(f, True) for _, f in _side_files(cfg)]:
        for raw, rec in _lines(f):
            if rec is None:
                for sf, qs in quoted:
                    if any(q in raw for q in qs):
                        named[(f.name, sf)] = named.get((f.name, sf), 0) + 1
                continue
            sf = (rec if top else rec.get("metadata") or {}).get("source_file")
            if sf in olds:
                left[(f.name, sf)] = left.get((f.name, sf), 0) + 1
            elif sf in new_ids and not top:
                new_ids[sf].add(rec.get("doc_id"))
            if not top:
                stale += still_cites(f.name, rec.get("metadata") or {})
    fails += [f"{name}: {n} row(s) still under {sf}" for (name, sf), n in sorted(left.items())]
    fails += [f"{name}: {n} unparseable line(s) still name {sf}" for (name, sf), n in sorted(named.items())]
    fails += stale

    cols = {c.name: client.get_collection(c.name) for c in client.list_collections()}
    for name, col in cols.items():
        for sf in sorted(olds):
            n = len(col.get(where={"source_file": sf}, include=[])["ids"])
            if n:
                fails.append(f"{name}: {n} row(s) still under {sf}")
    if active not in cols:
        fails.append(f"paths.collection_name {active!r} is not a collection in the store")
    else:
        for m in cols[active].get(where={"file_type": "canvas"}, include=["metadatas"])["metadatas"]:
            fails += still_cites(active, m or {})
    for m in moves:
        found = new_ids[m["new_sf"]]
        if m.get("n_chunks") and len(found) != m["n_chunks"]:
            fails.append(f"{m['new_sf']}: the JSONLs hold {len(found)} unique id(s), the map says {m['n_chunks']}")
        if active in cols:
            n = len(cols[active].get(where={"source_file": m["new_sf"]}, include=[])["ids"])
            if n != len(found):
                fails.append(f"{m['new_sf']}: {active} holds {n} row(s), the JSONLs {len(found)} unique id(s)")
        if not (vault / m["new"]).exists():
            fails.append(f"{m['new']}: missing on disk")
    docs = _ocr_docs(cfg)
    for m in moves:
        fails += [f"OCR Studio: {docs[src]} still names {src}"
                  for src in sorted({m["old"], _posix(m["old_sf"])}) if src in docs]

    bm25 = cfg.path("paths.bm25_index")
    if not bm25.exists():
        fails.append(f"sparse index {bm25} is missing: run rebuild_bm25.py")
    elif chunk_files and bm25.stat().st_mtime < max(f.stat().st_mtime for f in chunk_files):
        fails.append("sparse index predates the relocation: run rebuild_bm25.py")
    return fails


# ---- stamp-provenance: provenance generated on the rows the rule names, and nothing else ----
# Ids, text and vectors are never touched. main.py stamp-provenance is the only caller.

class ProvenanceStampError(RuntimeError):
    """stamp-provenance cannot run as the store stands; the message names the file and the fix.
    Not src.embeddings.sidecar.StampError, which is `rag stamp`'s."""


class StampIncomplete(ProvenanceStampError):
    """The re-plan after a stamp still finds work, or its totals moved."""


def provenance_plan(cfg, client, rule) -> dict:
    """Read-only. The candidates are the chunk JSONL rows provenance_of calls generated.
    {"files": {source_file: candidate rows needing work}, "jsonl": {file name: rows to
    stamp}, "chroma": {chunk collection: rows to stamp}, "ids": sorted doc_ids needing
    either, "unparseable": {file name: lines}, "totals": {"jsonl_ids", "chroma":
    {collection: count()}, "bm25"}}. A JSONL row needs work when its own key is not
    "generated"; a collection row (role chunks only: HyPE and the graph twin are never
    read) when a candidate id's stored key is not. A row's id is the sparse union's
    (Chunk.from_jsonl_record), so jsonl_ids and bm25 count the same rows. Unparseable lines
    are counted, not read: rewrite_jsonl passes them through byte-for-byte. totals["bm25"]
    is the build sidecar's count: the payload is never unpickled.

    Raises ProvenanceStampError, naming the file, for a missing or unreadable sidecar, a stored
    provenance outside PROVENANCE, a candidate with no doc_id of its own (the union's
    positional fallback id restarts in every file, so it can name another file's row), and
    a candidate id that is also on a row of another file: either way the stamp could write
    a row it did not plan."""
    from src.embeddings.embedder import Chunk
    from src.embeddings.sidecar import role_for
    from src.utils.provenance import provenance_of
    sidecar = Path(str(cfg.path("paths.bm25_index")) + ".meta.json")   # write_sparse_meta's name
    redo = "run rebuild_bm25.py (servers down) to rewrite it, then retry"
    if not sidecar.is_file():
        raise ProvenanceStampError(f"{sidecar} is missing, so the sparse row count is unknown: {redo}")
    try:
        side = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:            # unreadable; JSONDecodeError and UnicodeDecodeError both
        raise ProvenanceStampError(f"{sidecar} is not readable JSON ({e}): {redo}") from e
    if not isinstance(side, dict) or type(side.get("count")) is not int:
        raise ProvenanceStampError(f"{sidecar} holds no integer count: {redo}")
    bm25 = side["count"]
    first, cand, cand_set, jsonl, unparseable = {}, [], set(), {}, {}
    for f in chunk_jsonls(cfg):
        # ponytail: idx counts dict records; the union's counts every parsed JSON value, which
        # differs only where a line is valid non-object JSON, and such a line fails the union
        idx = 0
        with open(f, "rb") as fh:                 # _lines' parse, keeping the line number
            for n, raw in enumerate(fh, 1):
                if not raw.strip():
                    continue
                try:
                    rec = json.loads(raw.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    rec = None
                if not isinstance(rec, dict):
                    unparseable[f.name] = unparseable.get(f.name, 0) + 1
                    continue
                did = Chunk.from_jsonl_record(rec, idx).id
                idx += 1
                meta = rec.get("metadata") or {}
                try:
                    generated = provenance_of(meta, rule) == "generated"
                except ValueError as e:
                    raise ProvenanceStampError(f"{f.name}: line {n}: {e}") from e
                prior = first.setdefault(did, (f.name, n))
                if generated and not rec.get("doc_id"):
                    raise ProvenanceStampError(
                        f"{f.name}: line {n}: a row of {meta.get('source_file')!r} the rule calls generated "
                        f"has no doc_id, and a fallback id can name another file's row: re-ingest that "
                        f"file, then retry")
                if prior[0] != f.name and (generated or did in cand_set):
                    raise ProvenanceStampError(
                        f"doc_id {did!r} is in both {prior[0]}: line {prior[1]} and {f.name}: line {n}, one "
                        f"of them a row the rule calls generated: a stamp by that id could write the other "
                        f"row's metadata, so it refuses and writes nothing. Back up both files, remove the "
                        f"duplicate row (not the file) by hand, then retry")
                if not generated:
                    continue
                cand_set.add(did)
                own = meta.get("provenance") != "generated"
                cand.append((meta.get("source_file") or "", did, own))
                if own:
                    jsonl[f.name] = jsonl.get(f.name, 0) + 1
    cand_ids = sorted(cand_set)
    chroma, in_chroma, counts = {}, set(), {}
    for c in client.list_collections():
        if role_for(cfg, c.name) != "chunks":
            continue
        col = client.get_collection(c.name)
        counts[c.name] = col.count()
        for k in range(0, len(cand_ids), PAGE):
            got = col.get(ids=cand_ids[k:k + PAGE], include=["metadatas"])
            todo = [i for i, m in zip(got["ids"], got["metadatas"]) if (m or {}).get("provenance") != "generated"]
            if todo:
                chroma[c.name] = chroma.get(c.name, 0) + len(todo)
                in_chroma.update(todo)
    files = Counter(sf for sf, d, own in cand if own or d in in_chroma)
    return {"files": dict(files), "jsonl": jsonl, "chroma": chroma,
            "ids": sorted({d for _, d, own in cand if own} | in_chroma), "unparseable": unparseable,
            "totals": {"jsonl_ids": len(first), "chroma": counts, "bm25": bm25}}


def stamp_provenance(cfg, client, rule) -> dict:
    """Write provenance: generated on every row provenance_plan finds, Chroma first (the
    collection's own metadata, merged), then the chunk JSONLs (rewrite_jsonl, tag provstamp,
    with a .bak). A crash between the two leaves only JSONL work, which a re-run finds and
    finishes. Re-plans after: raises StampIncomplete("stamp incomplete: ...") unless no id is
    left and the totals are unchanged. Returns the plan it applied. The caller rebuilds the
    sparse index (rebuild_bm25.py) and restarts :8051."""
    from src.utils.provenance import provenance_of
    p = provenance_plan(cfg, client, rule)
    for name in p["chroma"]:
        col = client.get_collection(name)
        for k in range(0, len(p["ids"]), PAGE):
            got = col.get(ids=p["ids"][k:k + PAGE], include=["metadatas"])
            todo = [(i, m or {}) for i, m in zip(got["ids"], got["metadatas"])
                    if (m or {}).get("provenance") != "generated"]
            if todo:
                col.update(ids=[i for i, _ in todo],
                           metadatas=[{**m, "provenance": "generated"} for _, m in todo])

    def fix(rec):
        meta = rec.get("metadata") or {}
        if meta.get("provenance") == "generated" or provenance_of(meta, rule) != "generated":
            return None
        return dict(rec, metadata={**meta, "provenance": "generated"})

    for f in chunk_jsonls(cfg):
        if f.name in p["jsonl"]:
            rewrite_jsonl(f, fix, tag="provstamp")
    after = provenance_plan(cfg, client, rule)
    if after["ids"] or after["totals"] != p["totals"]:
        raise StampIncomplete(f"stamp incomplete: {len(after['ids'])} id(s) still unstamped "
                              f"(e.g. {after['ids'][:3]}), totals {p['totals']} before, {after['totals']} "
                              f"after. Re-run main.py stamp-provenance --yes with every server down")
    return p
