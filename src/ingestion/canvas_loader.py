"""
canvas_loader.py — Obsidian .canvas ingestion.

WHY THIS EXISTS
  Obsidian `.canvas` files are JSON: nodes plus edges, where the edges are
  hand-drawn semantic relationships carrying optional labels. They are the
  one place in the vault where the owner has already written down how two
  ideas relate, and the pipeline ignored them entirely before this loader.

THE KEY DESIGN DECISION
  The graph is flattened into each chunk AT INGEST TIME — both as a
  human-readable "Connections" footer inside the chunk text, and as edge
  metadata (canvas_edges / canvas_neighbors / canvas_edge_labels /
  canvas_file_refs / canvas_degree). canvas_edges is the ALIGNED one — one
  entry per edge, so a traversal can pair a label with the edge it belongs
  to; the other two are unaligned lists of different lengths.

  The graph is therefore visible to the embedder, the reranker and the LLM
  with ZERO retrieval-time code in the MAIN query path. (The optional Graph
  RAG mode, src/retrieval/graph_expand.py, is a separate endpoint that reads
  the same metadata — it never touches /query or /search.)

WHAT BECOMES A CHUNK
  Only `text` nodes. `file`, `link` and `group` nodes never get their own
  chunk — they exist as edge endpoints (a "Connections" line target) and,
  for `group`, as the enclosing context of whichever text nodes sit inside
  its box.

  Emits the identical JSONL shape as the sibling loaders:
      {"doc_id": "<16hex>", "text": "<context-header + text + connections>",
       "metadata": {...}}

INGESTION HYPERPARAMETERS (config.yaml `canvas:`)
  min_chunk_size   floor on the COMPOSED text (header + body + connections)
  max_chunk_size   split oversized node BODIES; null = off, the behaviour
                   every already-indexed canvas chunk was produced with
  chunking         which splitter does that (heading|fixed|document|none)
  context_depth    how much of a NEIGHBOUR is inlined at ingest time
                   (0 titles only | 1 neighbour text | 2 + second-hop titles)
  context_chars    per-neighbour truncation when context_depth >= 1

  Depth at QUERY time is deliberately not here: it belongs to the Graph RAG
  mode (POST /graph/expand), where the caller asked for a traversal. A depth
  knob on /query would make every ordinary query pay for graph expansion it
  never requested.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# Single source of truth for course/domain detection AND text cleaning — a
# from-reference import (not a copy) so a taxonomy override
# (configure_taxonomy) or a fix to clean_text reaches this loader too,
# exactly like pdf_loader/ipynb_loader/code_loader.
from src.ingestion.obsidian_parser import (
    CHUNKING_STRATEGIES,
    detect_course_from_path,
    clean_text,
    apply_forced_meta,
    split_section,
)
from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)

# Directories to never walk into, mirroring ObsidianParser's default
# skip_dirs (canvas files live in the same vault tree as markdown notes, not
# in code/build directories, so code_loader's skip set doesn't apply here).
_DEFAULT_SKIP_DIRS = {
    ".obsidian", ".trash", ".git", "node_modules",
    ".smart-connections", ".obsidian-git", "_Backups",
}


def iter_canvas_files(root: Path, skip_dirs: set[str] | None = None):
    """Every .canvas file under `root`, pruning skipped directories as it goes.

    os.walk, NOT Path.rglob("*.canvas"). The pattern form does per-entry
    matching plus extra stat calls, which on Windows dominates the walk
    itself. Measured on this vault (124 canvases among ~136k entries):
    warm filesystem cache 4.2s vs 2.4s, cold 45.5s vs ~5s. So the win ranges
    from useful to large depending on cache state, and it is largest exactly
    when someone is waiting on a first scan.

    Pruning skipped directories during the walk saves more again, because
    they are never descended into at all.

    Pruning also fixes a latent bug in the old membership test: it compared
    EVERY path component, including the ones above the vault root, so a vault
    living under a directory that happened to be named `.git` or
    `node_modules` would have matched nothing at all.
    """
    skip = set(skip_dirs) if skip_dirs is not None else set(_DEFAULT_SKIP_DIRS)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for name in filenames:
            if name.lower().endswith(".canvas"):
                yield Path(dirpath) / name


# ─────────────────────────────────────────────────────────────────────
# Small text helpers (composed-text construction only — no retrieval logic)
# ─────────────────────────────────────────────────────────────────────

def _collapse(label) -> str:
    """Collapse any whitespace run (including embedded newlines — edge labels
    can legitimately contain them) into a single space."""
    if not label:
        return ""
    return re.sub(r"\s+", " ", str(label)).strip()


# ---------------------------------------------------------------------
# canvas_edges codec — the ONE machine-readable edge format.
#
# WHY IT EXISTS: canvas_neighbors and canvas_edge_labels are two lists of
# DIFFERENT lengths (labels drop empties, neighbours don't), so nothing
# downstream can tell which label belongs to which edge. canvas_edges pairs
# them: one entry per edge, always, in edge order.
#
# WHY IT ESCAPES: Embedder._clean_meta coerces a list into ChromaDB by
# ", ".join(...) — so on read-back the list is split on ", ". Real edge
# labels in this vault contain commas: one Armenian label is literally
# "այսինքն, հետևում է, որ", which a naive split turns into three labels for
# one edge. Escaping ',' (and '%' and the '|' field separator) makes the
# round-trip through Chroma lossless.
# ---------------------------------------------------------------------

_EDGE_FIELD_SEP = "|"
_EDGE_ESCAPES = {"%": "%25", ",": "%2C", "|": "%7C"}
_EDGE_UNESCAPE_RE = re.compile(r"%(25|2C|7C)")
_EDGE_UNESCAPES = {"25": "%", "2C": ",", "7C": "|"}


def _edge_escape(value: str) -> str:
    # '%' first: it is the escape character, so escaping it afterwards would
    # double-escape the sequences introduced by the other two.
    out = str(value).replace("%", _EDGE_ESCAPES["%"])
    return out.replace(",", _EDGE_ESCAPES[","]).replace("|", _EDGE_ESCAPES["|"])


def _edge_unescape(value: str) -> str:
    # One left-to-right pass, so an escaped '%' can never be re-consumed by a
    # later rule (which sequential str.replace calls would do).
    return _EDGE_UNESCAPE_RE.sub(lambda m: _EDGE_UNESCAPES[m.group(1)], value)


def encode_canvas_edge(neighbor_id: str, direction: str, label: str) -> str:
    """One edge as `neighbor_id|direction|label`, separators escaped."""
    return _EDGE_FIELD_SEP.join(
        _edge_escape(part) for part in (neighbor_id, direction, label))


def decode_canvas_edges(value) -> list[dict]:
    """Read canvas_edges back from EITHER shape it is stored in.

    JSONL keeps a real list; ChromaDB/BM25 keep the ", "-joined string
    Embedder._clean_meta produced. Both decode to
    ``[{"neighbor": id, "direction": "->"|"<-", "label": str}, ...]``.

    Raises ValueError on an entry that is not three fields — that can only
    come from hand-edited metadata, and silently dropping it would make a
    traversal quietly lose edges.
    """
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        entries = list(value)
    else:
        entries = [part for part in str(value).split(", ") if part]
    edges: list[dict] = []
    for entry in entries:
        fields = str(entry).split(_EDGE_FIELD_SEP, 2)
        if len(fields) != 3:
            raise ValueError(
                f"malformed canvas_edges entry {entry!r}: expected "
                f"neighbor_id|direction|label")
        neighbor, direction, label = (_edge_unescape(f) for f in fields)
        edges.append({"neighbor": neighbor, "direction": direction,
                      "label": label})
    return edges


def _first_nonempty_line(text: str) -> str:
    for ln in (text or "").splitlines():
        if ln.strip():
            return ln.strip()
    return ""


def _truncate(s: str, limit: int = 80) -> str:
    s = s.strip()
    return s if len(s) <= limit else s[:limit].rstrip() + "…"


def _describe_target(node: dict | None) -> str:
    """Short human label for an edge's OTHER endpoint, used in the
    Connections footer. text -> first line (truncated); file -> its vault
    path; link -> its url; group -> its label."""
    if not node:
        return "(missing node)"
    ntype = node.get("type")
    if ntype == "text":
        return _truncate(_first_nonempty_line(node.get("text", "")))
    if ntype == "file":
        return node.get("file", "") or ""
    if ntype == "link":
        return node.get("url", "") or ""
    if ntype == "group":
        return _collapse(node.get("label", ""))
    return str(node.get("id", ""))


def _smallest_containing_group(node: dict, groups: list[dict]) -> dict | None:
    """A node belongs to a group when its box is fully contained in the
    group's box. Several groups can qualify (nested knowledge maps) — the
    smallest by area wins, since that's the most specific context."""
    nx, ny = node.get("x", 0), node.get("y", 0)
    nw, nh = node.get("width", 0), node.get("height", 0)
    best, best_area = None, None
    for g in groups:
        gx, gy = g.get("x", 0), g.get("y", 0)
        gw, gh = g.get("width", 0), g.get("height", 0)
        if gx <= nx and gy <= ny and nx + nw <= gx + gw and ny + nh <= gy + gh:
            area = gw * gh
            if best_area is None or area < best_area:
                best, best_area = g, area
    return best


def _context_header(canvas_name: str, group_label: str) -> str:
    """Bracket-style context prefix, modeled on
    obsidian_parser.build_context_header's "[Key: value | Key: value]" form."""
    parts = [f"Canvas: {canvas_name}"]
    if group_label:
        parts.append(f"Group: {group_label}")
    return "[" + " | ".join(parts) + "]\n"


# ─────────────────────────────────────────────────────────────────────
# Chunk dataclass — mirrors Document (obsidian_parser.py) / CodeChunk
# (code_loader.py), except doc_id is ALWAYS set explicitly by the loader.
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CanvasChunk:
    text: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        # The loader always passes doc_id explicitly (see _node_doc_id below)
        # so this text-hash fallback is dead code in normal operation — kept
        # only so the dataclass behaves like its siblings when constructed
        # directly (e.g. in a test) without an id.
        if not self.doc_id:
            sig = f"{self.metadata.get('source_file', '')}::{self.text[:500]}"
            self.doc_id = hashlib.sha256(sig.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "text": self.text, "metadata": self.metadata}


def _node_doc_id(rel_path: str, node_id: str) -> str:
    """doc_id keyed on (canvas path, Obsidian node id) rather than the
    text-hash scheme every other loader uses, deliberately:
      (a) two text nodes in one canvas can legitimately hold identical short
          text (e.g. two nodes both saying "outputs"), which would collide
          under a text hash;
      (b) keying on the node id means editing a node's text UPSERTS that
          chunk on the next ingest rather than orphaning the old id and
          adding a new one.
    """
    return hashlib.sha256(f"{rel_path}#{node_id}".encode()).hexdigest()[:16]


class CanvasLoader:
    """Turn every `text` node of every .canvas file into a chunk carrying its
    graph context (Connections footer + edge metadata). `file`/`link`/`group`
    nodes never get their own chunk."""

    def __init__(
        self,
        vault_path: Path,
        output_file: Path,
        min_chunk: int = 50,
        skip_dirs: set[str] | None = None,
        skip_roots: set[str] | None = None,
        include_path: str | None = None,
        max_chunk: int | None = None,
        chunking: str = "heading",
        overlap: int = 150,
        context_depth: int = 0,
        context_chars: int = 400,
    ):
        self.vault_path = Path(vault_path)
        self.output_file = Path(output_file)
        self.min_chunk = min_chunk
        self.skip_dirs = set(skip_dirs) if skip_dirs is not None else set(_DEFAULT_SKIP_DIRS)
        self.skip_roots = set(skip_roots) if skip_roots is not None else set()
        self.include_path = include_path.lower() if include_path else None
        # Oversized-node splitting. None = OFF, which is the behaviour every
        # already-indexed canvas chunk was produced with. Measured on the
        # node's OWN text: the context header and Connections footer ride
        # along on every part, so a part can exceed max_chunk by their size
        # (splitting the footer across parts would destroy the graph context
        # that is the whole point of this lane).
        self.max_chunk = int(max_chunk) if max_chunk else None
        chunking = (chunking or "heading").lower()
        if chunking not in CHUNKING_STRATEGIES:
            raise ValueError(f"canvas chunking must be one of "
                             f"{CHUNKING_STRATEGIES}, got {chunking!r}")
        self.chunking = chunking
        self.overlap = int(overlap)
        # How much of a NEIGHBOUR is inlined into this chunk at ingest time.
        #   0 = target titles only (first line, truncated) — today's default
        #   1 = the immediate neighbours' text, truncated to context_chars
        #   2 = 1, plus each neighbour's OWN neighbour titles (second hop)
        # Depths above 0 duplicate text across chunks and inflate the index;
        # ingest_vault() reports the measured cost rather than assuming it.
        if int(context_depth) not in (0, 1, 2):
            raise ValueError("canvas context_depth must be 0, 1 or 2, got "
                             f"{context_depth!r}")
        self.context_depth = int(context_depth)
        self.context_chars = int(context_chars)

        # Per-file / batch metadata overrides — metadata only, doc_ids unaffected.
        self.force_domain: str | None = None
        self.force_tags: list[str] = []

        self.stats = {
            "files_found": 0,
            "files_processed": 0,
            "files_skipped": 0,
            "chunks_total": 0,
            "nodes_skipped": 0,
            "edges_total": 0,
            "nodes_split": 0,
            "chars_total": 0,
            "chars_context": 0,
        }

    @classmethod
    def from_config(cls, cfg: Config) -> "CanvasLoader":
        vault = (cfg.get("canvas.vault_path")
                 or cfg.get("code.vault_path")
                 or cfg.get("notebooks.vault_path")
                 or cfg.get("pdf.vault_path")
                 or cfg.get("parser.vault_path"))
        out = (cfg.path("canvas.output_file") if cfg.get("canvas.output_file")
               else cfg.project_root / "data" / "canvas_chunks.jsonl")
        skip_dirs = cfg.get("canvas.skip_dirs", None)
        skip_roots = cfg.get("canvas.skip_roots", None)
        return cls(
            vault_path=Path(vault),
            output_file=out,
            # Default 50, not the 200 the markdown/code loaders use: canvas
            # nodes are inherently short, and their value is the graph
            # context a short node can still carry via its Connections block.
            min_chunk=cfg.get("canvas.min_chunk_size", 50),
            skip_dirs=set(skip_dirs) if skip_dirs is not None else None,
            skip_roots=set(skip_roots) if skip_roots is not None else None,
            include_path=cfg.get("canvas.include_path"),
            max_chunk=cfg.get("canvas.max_chunk_size", None),
            chunking=cfg.get("canvas.chunking", "heading"),
            overlap=cfg.get("canvas.chunk_overlap", 150),
            context_depth=cfg.get("canvas.context_depth", 0),
            context_chars=cfg.get("canvas.context_chars", 400),
        )

    # ---- discovery ----

    def discover_files(self) -> list[Path]:
        files = []
        for f in iter_canvas_files(self.vault_path, self.skip_dirs):
            rel_parts = f.relative_to(self.vault_path).parts
            if rel_parts and rel_parts[0] in self.skip_roots:
                self.stats["files_skipped"] += 1
                continue
            if self.include_path and self.include_path not in \
                    f.relative_to(self.vault_path).as_posix().lower():
                self.stats["files_skipped"] += 1
                continue
            files.append(f)
        self.stats["files_found"] = len(files)
        return sorted(files)

    # ---- composed-text context (canvas.context_depth) ----

    def _inlined_context(self, other: dict | None, other_id: str | None,
                         nodes_by_id: dict, edges_by_node: dict,
                         origin_id: str) -> list[str]:
        """Extra Connections lines for one edge, per canvas.context_depth.

        depth 0 -> nothing (the `-> [label] Title` line already written is the
                   whole footer entry)
        depth 1 -> that neighbour's text, truncated to context_chars
        depth 2 -> also the neighbour's OWN neighbours, as titles

        Everything here is DUPLICATED text: a neighbour's body ends up inside
        every chunk that links to it. That is the point (chunks become
        self-contained) and it is also the cost — ingest_vault() measures it.
        """
        if self.context_depth < 1 or not other or other.get("type") != "text":
            return []
        body = clean_text(other.get("text", "") or "")
        if not body:
            return []
        lines = [f"     {_truncate(_collapse(body), self.context_chars)}"]
        if self.context_depth < 2:
            return lines
        second: list[str] = []
        for e in edges_by_node.get(other_id, []):
            hop_id = (e.get("toNode") if e.get("fromNode") == other_id
                      else e.get("fromNode"))
            # Never walk back to the node we are composing for, and never
            # repeat a title: canvas graphs are small, dense and cyclic.
            if not hop_id or hop_id in (other_id, origin_id):
                continue
            title = _describe_target(nodes_by_id.get(hop_id))
            if title and title not in second:
                second.append(title)
        if second:
            lines.append("     also links: " + " · ".join(second))
        return lines

    # ---- per-file ----

    def load_file(self, filepath: Path) -> list[CanvasChunk]:
        """Parse one .canvas file into chunks (one per `text` node).

        Raises ValueError, naming the file, when the JSON is malformed or the
        top-level `nodes` key is missing — never skipped silently.
        """
        rel_path = filepath.relative_to(self.vault_path).as_posix()
        raw = filepath.read_text(encoding="utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed canvas JSON in {rel_path}: {e}") from e
        if not isinstance(data, dict) or "nodes" not in data:
            raise ValueError(f"Canvas file missing required 'nodes' key: {rel_path}")

        nodes = data.get("nodes") or []
        edges = data.get("edges") or []
        self.stats["edges_total"] += len(edges)

        nodes_by_id = {n["id"]: n for n in nodes if isinstance(n, dict) and n.get("id")}
        groups = [n for n in nodes if isinstance(n, dict) and n.get("type") == "group"]

        edges_by_node: dict[str, list[dict]] = {}
        for e in edges:
            for key in ("fromNode", "toNode"):
                nid = e.get(key)
                if nid:
                    edges_by_node.setdefault(nid, []).append(e)

        canvas_name = filepath.stem
        course_meta = detect_course_from_path(list(filepath.relative_to(self.vault_path).parts))

        chunks: list[CanvasChunk] = []
        for node in nodes:
            if not isinstance(node, dict) or node.get("type") != "text":
                continue
            node_id = node.get("id")
            if not node_id:
                continue

            group = _smallest_containing_group(node, groups)
            group_label = _collapse(group.get("label", "")) if group else ""

            touching = edges_by_node.get(node_id, [])
            neighbors: list[str] = []
            edge_labels: list[str] = []
            edges_aligned: list[str] = []
            file_refs: list[str] = []
            conn_lines: list[str] = []
            for e in touching:
                if e.get("fromNode") == node_id:
                    direction, other_id = "->", e.get("toNode")
                else:
                    direction, other_id = "<-", e.get("fromNode")
                other = nodes_by_id.get(other_id) if other_id else None
                if other_id:
                    neighbors.append(other_id)
                label = _collapse(e.get("label"))
                if label:
                    edge_labels.append(label)
                # ONE entry per edge, labelled or not — this is the list a
                # graph traversal can actually pair up. The two lists above
                # are kept because the composed footer and existing indexed
                # chunks are built from them.
                if other_id:
                    edges_aligned.append(
                        encode_canvas_edge(other_id, direction, label))
                if other and other.get("type") == "file" and other.get("file"):
                    file_refs.append(other["file"])
                desc = _describe_target(other)
                conn_lines.append(f"  {direction} [{label}] {desc}" if label
                                  else f"  {direction} {desc}")
                conn_lines.extend(
                    self._inlined_context(other, other_id, nodes_by_id,
                                          edges_by_node, node_id))

            header = _context_header(canvas_name, group_label)
            body = clean_text(node.get("text", "") or "")
            footer = ("\n\nConnections:\n" + "\n".join(conn_lines)
                      if conn_lines else "")

            # Count the COMPOSED text (header + body + connections), not the
            # raw node text — a short node with rich connections is exactly
            # the case worth keeping.
            if len(f"{header}{body}{footer}".strip()) < self.min_chunk:
                self.stats["nodes_skipped"] += 1
                continue

            # Split the BODY only; header and footer ride along on every part.
            # Cutting the Connections footer across parts would leave some
            # parts of a node with no graph context at all, which is the one
            # thing this lane exists to provide.
            if self.max_chunk and len(body) > self.max_chunk:
                bodies = split_section(body, self.max_chunk, self.overlap,
                                       strategy=self.chunking)
                self.stats["nodes_split"] += 1
            else:
                bodies = [body]

            meta = {
                "source_file": rel_path,
                "filename": filepath.stem,
                "file_type": "canvas",
                "vault_path": str(self.vault_path),
                "canvas_node_id": node_id,
                "canvas_neighbors": neighbors,
                "canvas_edge_labels": edge_labels,
                "canvas_edges": edges_aligned,
                "canvas_file_refs": file_refs,
                "canvas_degree": len(touching),
            }
            if group_label:
                meta["canvas_group"] = group_label
            meta.update(course_meta)

            for part, part_body in enumerate(bodies, start=1):
                part_meta = dict(meta)
                if len(bodies) > 1:
                    # canvas_node_id stays IDENTICAL across parts on purpose:
                    # a graph hop onto this node must reach all of it.
                    part_meta["canvas_part"] = part
                    part_meta["canvas_part_count"] = len(bodies)
                text = f"{header}{part_body}{footer}"
                chunks.append(CanvasChunk(
                    text=text, metadata=part_meta,
                    # Part 1 keeps the node's own id, so turning splitting on
                    # UPSERTS the chunk that already exists and only ADDS the
                    # tail parts — the project's stable-doc_id rule.
                    doc_id=_node_doc_id(
                        rel_path,
                        node_id if part == 1 else f"{node_id}#p{part}"),
                ))
                self.stats["chunks_total"] += 1
                self.stats["chars_total"] += len(text)
                self.stats["chars_context"] += len(footer)

        return chunks

    # ---- vault-wide ----

    def ingest_vault(self, verbose: bool = True) -> Path:
        """Walk the vault and write every canvas chunk to output_file.

        The output is opened with "w", so that file is REPLACED (the console
        refuses a scoped run that would write the canonical
        canvas_chunks.jsonl). And load_file RAISES on a malformed canvas
        rather than skipping it — which, because the file is already open,
        leaves it holding only the canvases written before the bad one. Fix
        the canvas and re-run before any append or sparse rebuild reads it."""
        files = self.discover_files()
        if verbose:
            log.info("Found %d canvas file(s).", len(files))
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        seen_ids: set[str] = set()
        dupes = 0
        with open(self.output_file, "w", encoding="utf-8") as out_f:
            for idx, f in enumerate(files, 1):
                if verbose:
                    log.info("[%d/%d] %s", idx, len(files), f.relative_to(self.vault_path))
                for ch in self.load_file(f):
                    if ch.doc_id in seen_ids:
                        dupes += 1
                        continue
                    seen_ids.add(ch.doc_id)
                    if self.force_domain or self.force_tags:
                        apply_forced_meta(ch.metadata, self.force_domain, self.force_tags)
                    out_f.write(json.dumps(ch.to_dict(), ensure_ascii=False) + "\n")
                    written += 1
                self.stats["files_processed"] += 1
        if dupes:
            log.info("Skipped %d duplicate-id chunk(s) at write time.", dupes)
        if verbose:
            log.info("Canvas ingestion complete: %d chunk(s) -> %s",
                     written, self.output_file)
            # The context_depth cost, MEASURED rather than assumed: how much
            # of the corpus this run wrote is duplicated neighbour context.
            total = self.stats["chars_total"]
            if total:
                ctx = self.stats["chars_context"]
                log.info("  composed %s chars; %s (%.1f%%) are Connections "
                         "context at context_depth=%d",
                         f"{total:,}", f"{ctx:,}", 100.0 * ctx / total,
                         self.context_depth)
            if self.stats["nodes_split"]:
                log.info("  split %d oversized node(s) at max_chunk_size=%s "
                         "(chunking=%s)", self.stats["nodes_split"],
                         self.max_chunk, self.chunking)
        return self.output_file
