"""route.py: the pointer sidecar that route mode reads (route-then-cite §5.4).

Each generated chunk (a map node, a chapter note) cites primary pages. The sidecar maps
its doc_id to those targets, {source_file, page_start|None, page_end|None}, so a
generated router hit can bring the pages it cites into the pool. Built by
`main.py route-pointers` and the route_pointers job, read by RoutePointers.load.
Format, version 1:

    {"version": 1, "built_at": "<UTC>Z",
     "pointers": {"<doc_id>": [<target>, ...]},     # only ids with at least one target
     "unrouted": {"<doc_id>": "<why>"}}

Canvas rows read their node's Source: lines; note rows read their inline page cites,
else the note's frontmatter source: + pages:. A row the rule cannot read is unrouted
with a reason, never guessed (Review Focus 3).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import yaml

from src.index_ops import chunk_jsonls
from src.ingestion.canvas_loader import SOURCE_LINE_RE, norm_vault_path
from src.utils.provenance import provenance_of

VERSION = 1
Target = dict   # {"source_file": str, "page_start": int | None, "page_end": int | None}
TARGET_KEYS = frozenset(("source_file", "page_start", "page_end"))

_RANGE = re.compile(r"(\d+)(?:-(\d+))?")
_ITEM = r"\d+(?:-\d+)?"


def _part_re(prefix: str) -> re.Pattern:
    """One cite part: [ChN ] then a page list, and `and <page list>` adds to the same part.
    A page list is comma-separated a / a-b items; an item may repeat the prefix."""
    plist = rf"{prefix}{_ITEM}(?:, ?(?:pp?\. ?)?{_ITEM})*"
    return re.compile(rf"(?:Ch(?P<ch>\d+) )?(?P<pages>{plist}(?: and {plist})*)")


_CITE_PART = _part_re(r"pp?\. ?")         # inline: the p./pp. prefix is required
_PAGES_PART = _part_re(r"(?:pp?\. ?)?")   # frontmatter pages: it is optional ("1-29")
_GROUP = re.compile(r"\(([^()]*)\)")      # an inline cite is a parenthesised group


def _spans(pages: str) -> list[tuple[int, int]] | None:
    """a-b items as (a, b), a as (a, a); None when a range runs backwards (a typo, never swapped)."""
    spans = [(int(a), int(b or a)) for a, b in _RANGE.findall(pages)]
    return None if any(a > b for a, b in spans) else spans


def _add(out: list[Target], path: str, spans) -> None:
    for a, b in spans:
        t = {"source_file": path, "page_start": a, "page_end": b}
        if t not in out:
            out.append(t)


def canvas_targets(text: str) -> tuple[list[Target], bool]:
    """(targets, whether a Source line was skipped for a backwards range). The targets are
    the Source: lines of a canvas chunk's body, with full page ranges ("59-63, 68" is
    59-63 and 68-68), paths as raw text. The Connections footer inlines neighbour text, so
    it is cut off; the context header line can never match SOURCE_LINE_RE."""
    out: list[Target] = []
    backwards = False
    for line in text.split("\n\nConnections:", 1)[0].splitlines():
        m = SOURCE_LINE_RE.fullmatch(line.strip())
        if not m:
            continue
        spans = _spans(m["pages"]) if m["pages"] else [(None, None)]
        if spans:                               # None: a backwards range, skipped
            _add(out, m["path"], spans)
        else:
            backwards = True
    return out, backwards


def _parts(group: str, part_re: re.Pattern) -> list[tuple[str | None, list] | None]:
    """A ;-separated group's parts as (chapter, spans), None where a part does not match
    (a backwards range included). A part with no chapter inherits the previous part's."""
    out, ch = [], None
    for raw in group.split(";"):
        m = part_re.fullmatch(raw.strip())
        spans = _spans(m["pages"]) if m else None
        if spans:
            ch = m["ch"] or ch
        out.append((ch, spans) if spans else None)
    return out


def _resolve(parts, rel: str, folder: Path | None, pages) -> tuple[list[Target], str]:
    """Parts against the note's source: a file (folder None) takes chapterless parts, a
    folder maps ChN to its one 'ChN - *.pdf' file, matched case-blind on every OS and kept
    in its on-disk spelling. The union of what resolves, else the first part's reason."""
    targets: list[Target] = []
    why = ""
    named = set(re.findall(r"Ch(\d+)", str(pages))) if pages is not None else set()
    for ch, spans in parts:
        path = reason = None
        if folder is None:
            path, reason = (None, "chapter pages under a file source") if ch else (rel, None)
        else:
            if ch is None and len(named) == 1:
                ch = next(iter(named))
            if ch is None:
                reason = "bare pages under a folder source"
            else:
                decks = [p for p in folder.iterdir() if p.is_file()
                         and re.fullmatch(rf"Ch{ch} - .*\.pdf", p.name, re.I)]
                if len(decks) == 1:
                    path = f"{rel}/{decks[0].name}"
                else:
                    reason = f"Ch{ch} matches {len(decks)} decks"
        if path:
            _add(targets, path, spans)
        else:
            why = why or reason
    return (targets, "") if targets else ([], why)


def note_targets(text: str, frontmatter: dict, vault: Path) -> tuple[list[Target], str]:
    """(targets, "") or ([], why). The chunk's own cites first; only a chunk with none
    falls back to the frontmatter's pages:."""
    source = frontmatter.get("source")
    if not isinstance(source, str) or not source.strip():
        return [], "no source"
    rel = source.replace("\\", "/").strip("/")
    where = vault / rel
    if where.is_file():
        folder = None
    elif where.is_dir():
        folder = where
    else:
        return [], "source not found"
    pages = frontmatter.get("pages")
    cites = [p for g in _GROUP.findall(text) for p in _parts(g, _CITE_PART) if p]
    if cites:
        return _resolve(cites, rel, folder, pages)
    if pages is None:
        return ([{"source_file": rel, "page_start": None, "page_end": None}], "") \
            if folder is None else ([], "no pages")
    if str(pages).strip().lower().startswith("none"):
        return [], "no pages"
    parts = _parts(str(pages), _PAGES_PART)
    if not all(parts):
        return [], "malformed pages"
    return _resolve(parts, rel, folder, pages)


def _frontmatter(path: Path) -> tuple[dict | None, str]:
    """(frontmatter, "") or (None, why). parse_frontmatter's slicing, but a YAML error is
    reported, not read as {}."""
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, "note file not found"
    except (OSError, UnicodeDecodeError):
        return None, "note file unreadable"
    end = content.find("---", 3)
    if not content.startswith("---") or end == -1:
        return {}, ""
    try:
        fm = yaml.safe_load(content[3:end])
    except yaml.YAMLError:
        return None, "frontmatter unreadable"
    if fm is None:
        return {}, ""
    return (fm, "") if isinstance(fm, dict) else (None, "frontmatter unreadable")


def build_pointers(cfg, rule) -> dict:
    """The version-1 sidecar for every chunk-JSONL row the rule calls generated, ids as the
    sparse union reads them (first row of an id wins). A target the rule calls generated (an
    agent file, another map) can never add a primary page, so it is dropped. Reads the JSONLs
    and the vault only."""
    from src.embeddings.embedder import Chunk, iter_jsonl_records
    vault = cfg.path("parser.vault_path")
    pointers: dict[str, list[Target]] = {}
    unrouted: dict[str, str] = {}
    seen: set[str] = set()
    nodes: dict[tuple, list[Target]] = {}       # (source_file, canvas_node_id) -> targets
    backwards: set[tuple] = set()               # nodes with a Source line skipped as backwards
    node_of: dict[str, tuple] = {}
    notes: dict[str, tuple[dict | None, str]] = {}

    def settle(did: str, targets: list[Target], why: str) -> None:
        kept = [t for t in targets if not rule.generated_by_path(t["source_file"])]
        if kept:
            pointers[did] = kept
        else:
            unrouted[did] = "cites only generated files" if targets else why

    for f in chunk_jsonls(cfg):
        for i, rec in enumerate(iter_jsonl_records(f)):
            c = Chunk.from_jsonl_record(rec, i)
            if c.id in seen:
                continue
            seen.add(c.id)
            meta = c.metadata
            if provenance_of(meta, rule) != "generated":
                continue
            sf = meta.get("source_file") or ""
            ft = meta.get("file_type")
            if rule.instruction_file(sf):
                unrouted[c.id] = "agent instruction file"
            elif ft == "canvas":
                # split parts share one node: only one part carries its Source: lines
                key = (sf, meta["canvas_node_id"])
                targets, skipped = canvas_targets(c.text)
                for t in targets:
                    if t not in nodes.setdefault(key, []):
                        nodes[key].append(t)
                if skipped:
                    backwards.add(key)
                node_of[c.id] = key
            elif ft in ("note", "daily_note"):
                if sf not in notes:
                    notes[sf] = _frontmatter(vault / sf.replace("\\", "/"))
                fm, why = notes[sf]
                targets, why = note_targets(c.text, fm, vault) if fm is not None else ([], why)
                settle(c.id, targets, why)
            else:
                unrouted[c.id] = f"no pointer format for {ft or 'none'}"
    for did, key in node_of.items():
        settle(did, nodes.get(key, []), "backwards page range" if key in backwards else "no Source line")
    return {"version": VERSION, "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pointers": pointers, "unrouted": unrouted}


def write_pointers(path: Path, data: dict) -> None:
    """Temp file + replace (retried while :8051 holds it open on Windows)."""
    from src.ocr_studio.pages import atomic_write
    atomic_write(path, json.dumps(data, ensure_ascii=False).encode("utf-8"))


class RoutePointers:
    """A loaded sidecar. targets() is None for an id with no targets, unrouted or absent."""

    def __init__(self, data: dict):
        self._pointers: dict[str, list[Target]] = data["pointers"]

    @classmethod
    def load(cls, path: Path) -> "RoutePointers":
        with open(path, encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise ValueError(f"{path}: route pointer sidecar is not JSON ({e}): rebuild it "
                                 f"with python main.py route-pointers") from e
        version = data.get("version") if isinstance(data, dict) else None
        if version != VERSION:
            raise ValueError(f"{path}: route pointer sidecar version {version!r}, expected "
                             f"{VERSION}: rebuild it with python main.py route-pointers")
        # The shape every reader indexes blindly (the bench fingerprint reads built_at), so a
        # hand-edited file fails here, naming the fix.
        pointers = data.get("pointers")
        if not (isinstance(data.get("built_at"), str) and isinstance(pointers, dict)
                and isinstance(data.get("unrouted"), dict) and all(
                    isinstance(ts, list) and all(isinstance(t, dict) and TARGET_KEYS <= t.keys() for t in ts)
                    for ts in pointers.values())):
            raise ValueError(f"{path}: route pointer sidecar is malformed (built_at must be a string, "
                             f"pointers map each id to a list of {{source_file, page_start, page_end}}, "
                             f"unrouted each id to a reason): rebuild it with python main.py route-pointers")
        return cls(data)

    def targets(self, doc_id: str) -> list[Target] | None:
        return self._pointers.get(doc_id)

    def all_targets(self) -> list[Target]:
        """Every target, deduplicated by (norm_vault_path, page_start, page_end), first seen first."""
        seen, out = set(), []
        for targets in self._pointers.values():
            for t in targets:
                key = (norm_vault_path(t["source_file"]), t["page_start"], t["page_end"])
                if key not in seen:
                    seen.add(key)
                    out.append(t)
        return out

    def __len__(self) -> int:
        return len(self._pointers)
