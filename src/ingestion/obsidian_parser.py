"""
obsidian_parser.py — Obsidian Vault Ingestion for Personal RAG Pipeline
=========================================================================
Parses an Obsidian vault into semantically meaningful chunks ready for
embedding. Handles:
  - Daily notes with multiple back-to-back course lectures (heading-split)
  - Course-specific markdown notes (section-split)
  - Frontmatter (YAML) metadata extraction
  - Wikilink / backlink resolution
  - Tag extraction and propagation
  - Hierarchical context inheritance (parent headings cascade into children)

Output: List[Document] where each Document carries text + rich metadata
suitable for ChromaDB / BM25 dual indexing.

Usage:
    parser = ObsidianParser("/path/to/vault")
    documents = parser.parse_all()
    # Each document has .text, .metadata (dict), .doc_id (deterministic hash)
"""

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


# ─────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Document:
    """A single chunk ready for embedding + vector store insertion."""
    text: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            # Deterministic ID from content + source for dedup
            sig = f"{self.metadata.get('source_file', '')}::{self.text[:500]}"
            self.doc_id = hashlib.sha256(sig.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "text": self.text, "metadata": self.metadata}


# ─────────────────────────────────────────────────────────────────────
# Course taxonomy — user-extensible
# ─────────────────────────────────────────────────────────────────────
# Every dictionary in this section is config-driven. The default values below
# are GENERIC placeholders that cover common lecture-note conventions
# (course codes, common subject names, "Lecture N" patterns). For your own
# vault, override any of them in config.yaml under `parser.course_taxonomy.*`.
# All matching is case-insensitive.
#
#   COURSE_HEADING_PATTERNS  : regexes that flag a heading as a "course lecture"
#                              (used to split daily notes into lecture segments
#                              and to recognise course-named folders/filenames)
#   COURSE_MAP               : course code -> canonical course name
#   DOMAIN_MAP               : canonical course name -> domain code
#   FOLDER_COURSE_MAP        : lowercase folder name -> canonical course name
#   COURSE_KEYWORDS          : ordered (keyword_substr, course_name) fallback
#                              for the variant-rich human folder names that
#                              exact matching can't enumerate
#   COURSE_CODE_REGEX        : regex for codes like "CS 251", "DS 110", "ECON 101"
#   SKIP_HEADING_PATTERNS    : regexes for headings that look course-ish but
#                              are NOT courses (todos, placeholders, schedules…)
#   ABBREVIATIONS            : ordered (heading_pattern, course_name) for short
#                              course-name forms (NLP, ML, RL, …)
#   DAILY_NOTE_DIRS          : folder names that identify a daily-note folder
# ─────────────────────────────────────────────────────────────────────

_DEFAULT_COURSE_TAXONOMY: dict = {
    "course_code_regex": r"(CS|DS|ENGS|BSDS|ECON)\s*\d{2,3}",
    "course_heading_patterns": [
        # Course codes: "# CS 251", "## DS 223", …
        r"^#{1,3}\s+(CS|DS|ENGS|BSDS|ECON)\s*\d{2,3}",
        # Generic abbreviation / short-name pattern: "# NLP L2", "# RL L3", …
        r"^#{1,3}\s+[A-Za-z]{2,6}\s+L\d+",
        # "# Lecture N" / "## Lecture N:" — strongest generic signal of a lecture
        r"^#{1,3}\s+Lecture\s+\d+",
        # Generic: heading with dash/colon + lecture/week keyword
        r"^#{1,3}\s+\w+\s*[-—:]\s+(?:Lecture|Week|Session|Lab|Tutorial|Intro|Quiz|Midterm|Final|Review|HW)",
    ],
    "course_map": {
        # Code -> canonical name. Add your own institution's codes here.
        # Example: "CS 251": "Machine Learning",
    },
    "domain_map": {
        # Canonical course name -> domain code. Domain codes are free-form —
        # they're used by retrieval scope routing (config.yaml domain_signals).
        # Add your own canonical course names here.
        # Example: "Machine Learning": "ml",
    },
    "folder_course_map": {
        # Lowercase folder name -> canonical course name.
        # Example: "machine learning": "Machine Learning",
    },
    "course_keywords": [
        # (keyword_substring, course_name) — most-specific first.
        # Example: ("machine learning", "Machine Learning"),
    ],
    "skip_heading_patterns": [
        # Headings that look course-ish but are NOT courses.
        r"^todo\b",
        r"^to[\s-]?do",
        r"^day\s+planner",
        r"^plan\b",
        r"^routine",
        r"^schedule\b",
        r"^pre-?study\b",
        r"^sunday",
        r"^saturday",
        r"^tomorrow",
        r"^before\b",
        r"^after\b",
        r"^now\b",
        r"^main\s+tasks",
        r"^current\s+(study|focus|job)",
        r"^what\s+should",
        r"^convo\s+with",
        r"^\.\.\.$",                       # Placeholder headings
        r"^excalidraw\s+data",              # Raw Excalidraw JSON in .md files
        r"^library\s+schedule",
        r"^genius\s+idea",
        r"^do\s+not\s+include",             # Personal "do not include" notes (generic pattern)
        r"^appeal\s+letter",                # Personal admin documents
        r"^licensing\s+&\s+legal",          # One-off business docs
    ],
    "abbreviations": [
        # (heading_regex, canonical_course_name) — add short-name forms here.
        # Example: (r"^nlp\b", "Natural Language Processing"),
    ],
    "daily_note_dirs": [
        "daily notes", "daily", "dailies", "journal",
    ],
}


def _load_taxonomy() -> dict:
    """Load the course taxonomy from config (if available) and merge over
    the generic defaults above. Returns a single dict — see schema in
    _DEFAULT_COURSE_TAXONOMY. Used by the detect_* helpers below.
    """
    try:
        from src.utils.config_loader import load_config  # local import: avoids cycle at module import
        cfg = load_config()
        user_tax = cfg.get("parser.course_taxonomy", {}) or {}
    except Exception:
        user_tax = {}

    out = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
           for k, v in _DEFAULT_COURSE_TAXONOMY.items()}
    for k, v in (user_tax or {}).items():
        if k not in out:
            continue
        if isinstance(out[k], dict) and isinstance(v, dict):
            out[k].update(v)
        elif isinstance(out[k], list) and isinstance(v, list):
            out[k] = list(v) + out[k]  # user entries tried first
        else:
            out[k] = v
    return out


_TAXONOMY: dict | None = None


def _tax() -> dict:
    """Lazy-loaded taxonomy singleton."""
    global _TAXONOMY
    if _TAXONOMY is None:
        _TAXONOMY = _load_taxonomy()
    return _TAXONOMY


def reset_taxonomy_cache() -> None:
    """Drop the cached taxonomy so the next call re-reads config (useful in
    tests, or after live-editing config.yaml)."""
    global _TAXONOMY
    _TAXONOMY = None


# Convenience views over the active taxonomy (read each time so config
# edits are picked up on reset_taxonomy_cache()).
def COURSE_HEADING_PATTERNS() -> list[str]:
    return _tax()["course_heading_patterns"]


def COURSE_MAP() -> dict:
    return _tax()["course_map"]


def DOMAIN_MAP() -> dict:
    return _tax()["domain_map"]


def FOLDER_COURSE_MAP() -> dict:
    return _tax()["folder_course_map"]


def COURSE_KEYWORDS() -> list[tuple[str, str]]:
    return [tuple(x) for x in _tax()["course_keywords"]]


def COURSE_CODE_REGEX() -> str:
    return _tax()["course_code_regex"]


def SKIP_HEADING_PATTERNS() -> list[str]:
    return _tax()["skip_heading_patterns"]


def ABBREVIATIONS() -> list[tuple[str, str]]:
    return [tuple(x) for x in _tax()["abbreviations"]]


def DAILY_NOTE_DIRS() -> list[str]:
    return _tax()["daily_note_dirs"]


# Back-compat: keep the bare constants as DEPRECATED tuples / empty dicts so
# older imports don't NameError. New code should call the functions above.
# (These are intentionally minimal — they reflect "no opinion" for users who
# never configured anything.)
COURSE_HEADING_PATTERNS_CONST: tuple = ()
COURSE_MAP_CONST: dict = {}
DOMAIN_MAP_CONST: dict = {}
FOLDER_COURSE_MAP_CONST: dict = {}
COURSE_KEYWORDS_CONST: tuple = ()


def detect_course_from_path(parts: list[str]) -> dict:
    """Detect course from path components. Single source of truth used by the
    PDF and notebook loaders (and the recalibration pass).

    Order:
      1. Exact FOLDER_COURSE_MAP (root->leaf). Only leaf course folders are
         mapped, so organizational parents (e.g. "Programming for Data Science",
         which holds ML/AI/TSF) do NOT clobber their children.
      2. Course-code regex (default: CS|DS|ENGS|BSDS|ECON ###; configurable).
      3. Keyword substring fallback (leaf->root, most-specific-first) for the
         many human folder-name variants exact matching can't enumerate.

    Returns {course_code, course_name, domain}; unknown/unknown/general if none.
    """
    fcm = FOLDER_COURSE_MAP()
    dmap = DOMAIN_MAP()
    cmap = COURSE_MAP()
    ckw = COURSE_KEYWORDS()
    code_re = COURSE_CODE_REGEX()
    # 1. exact folder-name match
    for part in parts:
        name = fcm.get(part.lower().strip())
        if name:
            return {"course_code": name, "course_name": name,
                    "domain": dmap.get(name, "general")}
    # 2. course-code regex
    for part in parts:
        m = re.search(code_re, part, re.IGNORECASE)
        if m:
            code = re.sub(r"\s+", " ",
                          re.sub(r"(\D)(\d)", r"\1 \2", m.group(0).upper())).strip()
            name = cmap.get(code, code)
            return {"course_code": code, "course_name": name,
                    "domain": dmap.get(name, "general")}
    # 3. keyword fallback, deepest folder first
    for part in reversed(parts):
        low = part.lower()
        for kw, name in ckw:
            if kw in low:
                return {"course_code": name, "course_name": name,
                        "domain": dmap.get(name, "general")}
    return {"course_code": "unknown", "course_name": "unknown", "domain": "general"}

# Chunk size constraints (in characters)
MIN_CHUNK_SIZE = 200       # Skip fragments smaller than this
MAX_CHUNK_SIZE = 3000      # Split sections larger than this
OVERLAP_SIZE = 150         # Overlap between split chunks


# ─────────────────────────────────────────────────────────────────────
# Frontmatter parser
# ─────────────────────────────────────────────────────────────────────

def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and return (metadata_dict, remaining_content)."""
    if not content.startswith("---"):
        return {}, content
    end = content.find("---", 3)
    if end == -1:
        return {}, content
    try:
        fm = yaml.safe_load(content[3:end])
        if not isinstance(fm, dict):
            fm = {}
    except yaml.YAMLError:
        fm = {}
    body = content[end + 3:].lstrip("\n")
    return fm, body


# ─────────────────────────────────────────────────────────────────────
# Heading tree builder
# ─────────────────────────────────────────────────────────────────────

@dataclass
class HeadingNode:
    """A node in the heading tree — represents one section of a markdown file."""
    level: int
    title: str
    content: str  # text directly under this heading (before any child heading)
    children: list = field(default_factory=list)
    line_start: int = 0


def build_heading_tree(body: str) -> list[HeadingNode]:
    """
    Parse markdown body into a tree of HeadingNode.
    Top-level content (before any heading) becomes a level-0 node.
    """
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    lines = body.split("\n")
    # Find all heading positions
    headings = []
    for i, line in enumerate(lines):
        m = heading_re.match(line)
        if m:
            headings.append((i, len(m.group(1)), m.group(2).strip()))

    if not headings:
        # No headings — entire body is one chunk
        return [HeadingNode(level=0, title="(untitled)", content=body.strip(), line_start=0)]

    nodes = []
    # Content before first heading
    if headings[0][0] > 0:
        pre = "\n".join(lines[:headings[0][0]]).strip()
        if pre:
            nodes.append(HeadingNode(level=0, title="(preamble)", content=pre, line_start=0))

    for idx, (line_num, level, title) in enumerate(headings):
        # Content runs from the line after this heading to the line before the next heading
        start = line_num + 1
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        content = "\n".join(lines[start:end]).strip()
        nodes.append(HeadingNode(level=level, title=title, content=content, line_start=line_num))

    return nodes


# ─────────────────────────────────────────────────────────────────────
# Course detection from headings
# ─────────────────────────────────────────────────────────────────────

def detect_course_from_heading(heading: str) -> Optional[str]:
    """Try to extract a course code or name from a heading string.
    Returns course code/name, or '__SKIP__' for non-course headings,
    or None if unrecognized. All matching dictionaries come from the
    active course taxonomy (parser.course_taxonomy in config.yaml) —
    see SKIP_HEADING_PATTERNS() and ABBREVIATIONS() for the schema.
    """
    h_stripped = heading.strip()
    h_lower = h_stripped.lower()

    # ── SKIP PATTERNS (non-academic headings) ──
    # Checked first to prevent false matches (e.g., "Self S" matching "stat")
    for pat in SKIP_HEADING_PATTERNS():
        if re.match(pat, h_lower):
            return "__SKIP__"

    # ── COURSE CODE MATCH (configurable regex, e.g. "CS 251", "DS 110", "ECON 101") ──
    code_match = re.search(COURSE_CODE_REGEX(), heading, re.IGNORECASE)
    if code_match:
        code = code_match.group(0).upper()
        # Normalize to exactly one space between letters and digits
        code = re.sub(r"\s+", " ", re.sub(r"(\D)(\d)", r"\1 \2", code)).strip()
        return code

    # ── ABBREVIATION / SHORT-NAME MATCH ──
    # Ordered: most-specific first to prevent partial matches.
    # Each entry: (pattern, canonical course name) — see config taxonomy.
    for pattern, course_name in ABBREVIATIONS():
        if re.match(pattern, h_lower):
            return course_name

    return None


def resolve_course_metadata(course_id: Optional[str]) -> dict:
    """Given a course code or name, return enriched metadata. Resolves
    course codes and canonical course names against the active taxonomy."""
    if not course_id:
        return {"course_code": "unknown", "course_name": "unknown", "domain": "general"}

    dmap = DOMAIN_MAP()
    cmap = COURSE_MAP()

    # If it's already a canonical name (present in DOMAIN_MAP), use it
    if course_id in dmap:
        return {
            "course_code": course_id,
            "course_name": course_id,
            "domain": dmap[course_id],
        }

    # Otherwise look up the code
    name = cmap.get(course_id, course_id)
    domain = dmap.get(name, "general")
    return {"course_code": course_id, "course_name": name, "domain": domain}


# ─────────────────────────────────────────────────────────────────────
# Daily note detection
# ─────────────────────────────────────────────────────────────────────

DAILY_NOTE_PATTERNS = [
    # "28.05.2026" or "28-05-2026" (DD.MM.YYYY — the user's format)
    r"^\d{2}[-_.]\d{2}[-_.]\d{4}$",
    # "2026-05-28" or "2026.05.28" (YYYY-MM-DD — standard ISO)
    r"^\d{4}[-_.]\d{2}[-_.]\d{2}$",
    # "May 28, 2026" or "28 May 2026"
    r"^\w+\s+\d{1,2},?\s+\d{4}$",
    r"^\d{1,2}\s+\w+\s+\d{4}$",
]


def is_daily_note(filepath: Path) -> bool:
    """Detect if a file is a daily note based on its filename."""
    stem = filepath.stem
    for pattern in DAILY_NOTE_PATTERNS:
        if re.match(pattern, stem):
            return True

    # Also check if it's inside a folder named in DAILY_NOTE_DIRS (configurable)
    parts = [p.lower() for p in filepath.parts]
    return any(d in parts for d in DAILY_NOTE_DIRS())

def extract_date_from_filename(filepath: Path) -> Optional[str]:
    """Try to extract an ISO date from the filename."""
    stem = filepath.stem
    # DD.MM.YYYY format (the user's vault)
    ddmmyyyy = re.search(r"(\d{2})[-_.](\d{2})[-_.](\d{4})", stem)
    if ddmmyyyy:
        dd, mm, yyyy = ddmmyyyy.group(1), ddmmyyyy.group(2), ddmmyyyy.group(3)
        return f"{yyyy}-{mm}-{dd}"
    # YYYY-MM-DD format (ISO)
    yyyymmdd = re.search(r"(\d{4})[-_.](\d{2})[-_.](\d{2})", stem)
    if yyyymmdd:
        return f"{yyyymmdd.group(1)}-{yyyymmdd.group(2)}-{yyyymmdd.group(3)}"
    return None


# ─────────────────────────────────────────────────────────────────────
# Wikilink + tag extraction
# ─────────────────────────────────────────────────────────────────────

def extract_wikilinks(text: str) -> list[str]:
    """Extract [[wikilinks]] and [[wikilinks|display text]]."""
    return re.findall(r"\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]", text)


def extract_tags(text: str) -> list[str]:
    """Extract #tags from text (not inside code blocks)."""
    # Simple approach — skip lines starting with ```
    tags = set()
    in_code = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            tags.update(re.findall(r"(?:^|\s)#([a-zA-Z][\w/-]*)", line))
    return sorted(tags)


# ─────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean markdown text for embedding — preserve structure, remove noise."""
    # Remove image embeds but keep alt text
    text = re.sub(r"!\[\[([^\]]*)\]\]", r"[image: \1]", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"[image: \1]", text)

    # Convert wikilinks to plain text
    text = re.sub(r"\[\[([^\]|]+?)\|([^\]]+?)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+?)\]\]", r"\1", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    return text.strip()


# ─────────────────────────────────────────────────────────────────────
# Chunk splitter (for oversized sections)
# ─────────────────────────────────────────────────────────────────────

def split_large_chunk(text: str, max_size: int = MAX_CHUNK_SIZE,
                      overlap: int = OVERLAP_SIZE) -> list[str]:
    """Split text that exceeds max_size into overlapping chunks.
    Splits on paragraph boundaries when possible, falls back to sentences."""
    if len(text) <= max_size:
        return [text]

    chunks = []
    paragraphs = text.split("\n\n")
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 > max_size:
            if current:
                chunks.append(current.strip())
                # Overlap: keep the last `overlap` chars
                current = current[-overlap:] + "\n\n" + para
            else:
                # Single paragraph exceeds max — split by sentences
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sent in sentences:
                    if len(current) + len(sent) + 1 > max_size:
                        if current:
                            chunks.append(current.strip())
                            current = current[-overlap:] + " " + sent
                        elif len(sent) > max_size:
                            # A single "sentence" longer than max_size (OCR
                            # walls, tables, minified text): window it — the
                            # old sent[:max_size] silently DROPPED the rest.
                            chunks.extend(fixed_window_chunks(sent, max_size, overlap))
                            current = ""
                        else:
                            current = sent
                    else:
                        current = current + " " + sent if current else sent
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ─────────────────────────────────────────────────────────────────────
# Selectable chunking strategies (for oversized sections ONLY)
# ─────────────────────────────────────────────────────────────────────
# Strategy selection changes how a section that exceeds max_chunk_size is
# cut into sub-chunks — nothing else. Heading detection, context headers,
# metadata, and the doc_ids of sections that fit in one chunk are identical
# under every strategy, so re-ingesting with a different strategy re-chunks
# (new text → new doc_ids → swap playbook) only the long sections.

CHUNKING_STRATEGIES = ("heading", "fixed")


def fixed_window_chunks(text: str, max_size: int, overlap: int) -> list[str]:
    """Strict sliding window with whitespace-snapped cuts.

    Every chunk is ~max_size chars; the window advances by (end - overlap),
    cutting at the last newline/space in the final 40% of the window so words
    survive intact. Predictable chunk size/count regardless of the text's
    paragraph structure — for text where that structure is noise (OCR output,
    minified exports, wall-of-text notes) and the default 'heading' packing
    produces wildly uneven chunks. Always makes forward progress.
    """
    text = text.strip()
    if len(text) <= max_size:
        return [text] if text else []
    overlap = max(0, min(overlap, max_size // 2))   # sane stride whatever the config says
    chunks = []
    i, n = 0, len(text)
    while i < n:
        end = min(i + max_size, n)
        if end < n:
            floor = i + int(max_size * 0.6)
            cut = max(text.rfind("\n", floor, end), text.rfind(" ", floor, end))
            if cut > i:
                end = cut
        piece = text[i:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return chunks


def split_section(text: str, max_size: int, overlap: int,
                  strategy: str = "heading") -> list[str]:
    """Split one oversized section using the named strategy.

    'heading' (default) = split_large_chunk, the historical
    paragraph/sentence-boundary packing. 'fixed' = fixed_window_chunks.
    Both loaders (markdown + PDF) route through here; `--chunking fixed`
    on the CLI or the console's Ingest tab selects per run.
    """
    strategy = (strategy or "heading").lower()
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(f"unknown chunking strategy {strategy!r}; "
                         f"expected one of {CHUNKING_STRATEGIES}")
    if strategy == "fixed":
        return fixed_window_chunks(text, max_size, overlap)
    return split_large_chunk(text, max_size, overlap)


# ─────────────────────────────────────────────────────────────────────
# Context header builder (hierarchical heading inheritance)
# ─────────────────────────────────────────────────────────────────────

def build_context_header(heading_stack: list[str], file_meta: dict) -> str:
    """
    Build a context prefix that gets prepended to every chunk.
    This is the KEY insight from NVIDIA's retrieval research:
    chunks without context lose meaning. A section titled "Results"
    is meaningless without knowing it's from "CS 362 > ARIMA Models > Results".

    The heading stack provides hierarchical context inheritance.
    """
    parts = []

    if file_meta.get("course_name") and file_meta["course_name"] != "unknown":
        parts.append(f"Course: {file_meta['course_name']}")
    if file_meta.get("date"):
        parts.append(f"Date: {file_meta['date']}")
    if heading_stack:
        parts.append("Section: " + " > ".join(heading_stack))

    if not parts:
        return ""
    return "[" + " | ".join(parts) + "]\n"


# ─────────────────────────────────────────────────────────────────────
# Parent sections (E2 small-to-big: match small chunks, generate from
# the enclosing section). Children carry parent_id in METADATA ONLY —
# chunk text is untouched, so doc_ids are unaffected by this feature.
# Parents live in a sidecar JSONL and are NEVER embedded.
# ─────────────────────────────────────────────────────────────────────

def section_spans(nodes: list) -> list[tuple[int, int]]:
    """
    For each node i return (start, end): the slice of `nodes` making up the
    FULL section rooted at i — i itself plus every following node with a
    deeper heading level (its subsections). Level-0 nodes (preamble) are
    standalone. The flat node list from build_heading_tree is depth-ordered,
    so a section ends at the next node whose level is <= its own.
    """
    spans = []
    for i, n in enumerate(nodes):
        if n.level == 0:
            spans.append((i, i + 1))
            continue
        j = i + 1
        while j < len(nodes) and (nodes[j].level == 0 or nodes[j].level > n.level):
            j += 1
        spans.append((i, j))
    return spans


def section_ancestors(nodes: list, i: int) -> list[int]:
    """Indices of enclosing-section roots for node i, nearest first."""
    out = []
    lvl = nodes[i].level
    j = i - 1
    while j >= 0 and lvl > 1:
        if 0 < nodes[j].level < lvl:
            out.append(j)
            lvl = nodes[j].level
        j -= 1
    return out


# ─────────────────────────────────────────────────────────────────────
# MAIN PARSER
# ─────────────────────────────────────────────────────────────────────

class ObsidianParser:
    """
    Parse an Obsidian vault into Document chunks for RAG ingestion.

    Architecture decision: heading-level splitting with context inheritance.
    Each H2/H3 section becomes its own chunk. The heading hierarchy is
    prepended as a context header so that "Results" in isolation becomes
    "[Course: Time Series | Section: ARIMA Models > Results]" — giving
    the embedding model the context it needs.
    """

    def __init__(self, vault_path: str, config: Optional[dict] = None):
        self.vault_path = Path(vault_path)
        if not self.vault_path.exists():
            raise FileNotFoundError(f"Vault not found: {vault_path}")

        self.config = config or {}
        self.min_chunk = self.config.get("min_chunk_size", MIN_CHUNK_SIZE)
        self.max_chunk = self.config.get("max_chunk_size", MAX_CHUNK_SIZE)
        self.overlap = self.config.get("overlap_size", OVERLAP_SIZE)
        # How OVERSIZED sections are split (see split_section): 'heading'
        # keeps the historical paragraph packing, 'fixed' = sliding window.
        self.chunking = str(self.config.get("chunking", "heading")).lower()
        if self.chunking not in CHUNKING_STRATEGIES:
            raise ValueError(f"parser.chunking must be one of "
                             f"{CHUNKING_STRATEGIES}, got {self.chunking!r}")
        # E2 parent-child: cap on a parent section's text. Parents above the
        # cap are not emitted (their children stand alone) so a child's text
        # is always CONTAINED in its parent — swapping never loses evidence.
        self.parent_max_chars = self.config.get("parent_max_chars", 5000)
        self.parents: dict[str, dict] = {}      # parent_id -> sidecar record

        # Directories to skip (any path depth). _Backups = timestamped
        # duplicate copies of agent skills (E1 verdict: never index backups —
        # near-duplicate floods aimed straight at fusion).
        self.skip_dirs = {".obsidian", ".trash", ".git", "node_modules",
                          ".smart-connections", ".obsidian-git", "_Backups"}
        self.skip_dirs.update(set(self.config.get("skip_dirs", [])))
        # Top-level vault trees to exclude (exact match on the FIRST path
        # component only — safer than skip_dirs for names like "Documents"
        # that could legitimately appear deeper inside course folders).
        # Session 8: the agent-project trees (Workspace1/Workspace2/Workspace3/...)
        # ballooned an unscoped parse 7K -> 88.7K chunks; the markdown corpus
        # stays scoped to coursework until the user decides otherwise.
        self.skip_roots = set(self.config.get("skip_roots", []))

        # Stats
        self.stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "daily_notes": 0,
            "course_notes": 0,
            "other_notes": 0,
            "total_chunks": 0,
            "skipped_sections": 0,
            "chunks_by_domain": {},
            "chunks_by_course": {},
        }

    # Filename stems/patterns that identify non-content files to skip entirely.
    # These are checked in discover_files before any parsing happens.
    SKIP_FILE_STEMS = {
        # Excalidraw files export as .md but contain only JSON drawing data
        # (they show up as "Excalidraw Data" headings — 129 junk chunks)
    }
    SKIP_FILE_SUFFIXES = (".excalidraw",)
    SKIP_FILE_STEM_PATTERNS = [
        r"\.excalidraw$",          # "MyDiagram.excalidraw.md"
        r"excalidraw",             # any file with excalidraw in the name
    ]

    def discover_files(self) -> list[Path]:
        """Find all markdown files in the vault, respecting skip rules."""
        files = []
        for f in self.vault_path.rglob("*.md"):
            # Skip hidden / system directories
            if any(part in self.skip_dirs for part in f.parts):
                continue
            # Skip excluded top-level trees (agent projects etc.)
            rel_parts = f.relative_to(self.vault_path).parts
            if rel_parts and rel_parts[0] in self.skip_roots:
                self.stats["files_skipped"] += 1
                continue
            # Skip empty files
            if f.stat().st_size < 50:
                self.stats["files_skipped"] += 1
                continue
            # Skip Excalidraw drawing files (JSON masquerading as .md)
            stem_lower = f.stem.lower()
            if any(re.search(pat, stem_lower) for pat in self.SKIP_FILE_STEM_PATTERNS):
                self.stats["files_skipped"] += 1
                continue
            files.append(f)
        return sorted(files)

    def parse_file(self, filepath: Path) -> list[Document]:
        """Parse a single markdown file into Document chunks."""
        content = filepath.read_text(encoding="utf-8", errors="replace")
        frontmatter, body = parse_frontmatter(content)
        rel_path = str(filepath.relative_to(self.vault_path))

        # Base metadata from frontmatter + file info
        base_meta = {
            "source_file": rel_path,
            "filename": filepath.stem,
            "file_type": "daily_note" if is_daily_note(filepath) else "note",
            "vault_path": str(self.vault_path),
            "tags": extract_tags(body),
            "wikilinks": extract_wikilinks(body)[:20],  # cap for metadata size
        }

        # Merge frontmatter (user's YAML takes priority)
        if "tags" in frontmatter:
            fm_tags = frontmatter["tags"]
            if isinstance(fm_tags, list):
                base_meta["tags"] = list(set(base_meta["tags"] + fm_tags))
        if "course" in frontmatter:
            base_meta["frontmatter_course"] = frontmatter["course"]
        if "date" in frontmatter:
            base_meta["date"] = str(frontmatter["date"])
        elif is_daily_note(filepath):
            date = extract_date_from_filename(filepath)
            if date:
                base_meta["date"] = date

        # Route to appropriate parser
        if is_daily_note(filepath):
            self.stats["daily_notes"] += 1
            return self._parse_daily_note(body, base_meta)
        else:
            # Detect if it's a course-specific note from path or frontmatter
            course = self._detect_course_from_path(filepath, frontmatter)
            if course:
                self.stats["course_notes"] += 1
                base_meta.update(resolve_course_metadata(course))
            else:
                self.stats["other_notes"] += 1
            return self._parse_standard_note(body, base_meta)

    def _detect_course_from_path(self, filepath: Path, frontmatter: dict) -> Optional[str]:
        """Try to detect course from file path or frontmatter.

        Priority order:
          1. Frontmatter 'course' key
          2. Heading-pattern match on directory/filename parts
          3. FOLDER_COURSE_MAP exact lookup (catches full names with & / spaces)
        """
        # 1. Frontmatter wins
        if "course" in frontmatter:
            return str(frontmatter["course"])

        # 2. Try heading-pattern detection on each path part
        for part in filepath.parts:
            course = detect_course_from_heading(part)
            if course and course != "__SKIP__":
                return course

        # 3. Folder-name exact lookup (case-insensitive) — catches full-title
        #    folder names like "Statistics & Inference" that have special chars
        for part in filepath.parts:
            match = FOLDER_COURSE_MAP().get(part.lower().strip())
            if match:
                return match

        # 4. Check filename via heading detection
        course = detect_course_from_heading(filepath.stem)
        if course == "__SKIP__":
            return None
        return course

    # ---- E2 parent-child helpers ----

    def _section_text(self, nodes: list, s: int, e: int) -> str:
        """Full markdown of the section nodes[s:e]: headings + cleaned content.
        May include content of sub-headings that were skip-listed as chunks
        (todo lists etc.) — capped and harmless as generation context."""
        parts = []
        for n in nodes[s:e]:
            if n.level > 0:
                parts.append("#" * n.level + " " + n.title)
            if n.content and n.content.strip():
                parts.append(clean_text(n.content))
        return "\n\n".join(parts)

    def _parent_candidate(self, nodes: list, spans: list, i: int,
                          stacks: list, chunk_meta: dict):
        """
        For the chunk(s) emitted from node i, pick the OUTERMOST enclosing
        section whose full text fits parent_max_chars (section text only grows
        going outward, so the first overflow stops the walk). Returns
        (parent_id, text_len, sidecar_record) or None. The caller links it
        per-child and records it into self.parents only when actually used.
        """
        best = None
        for idx in [i] + section_ancestors(nodes, i):
            s, e = spans[idx]
            text = self._section_text(nodes, s, e)
            if len(text) > self.parent_max_chars:
                break
            best = (idx, text)
        if best is None:
            return None
        idx, text = best
        header = build_context_header(stacks[idx], chunk_meta)
        pid = hashlib.sha256(
            f"{chunk_meta.get('source_file', '')}::P{idx}::{text[:200]}".encode()
        ).hexdigest()[:16]
        record = {
            "parent_id": pid,
            "source_file": chunk_meta.get("source_file", ""),
            "heading": nodes[idx].title,
            "heading_path": " > ".join(stacks[idx]) if stacks[idx] else "",
            "text": header + text,
        }
        return pid, len(text), record

    def _maybe_link_parent(self, meta: dict, pinfo, child_text: str) -> None:
        """Attach parent_id when the parent is meaningfully bigger than the
        child (>=1.3x) — swapping a chunk for a same-sized parent is noise."""
        if pinfo and pinfo[1] >= int(1.3 * len(child_text)):
            meta["parent_id"] = pinfo[0]
            self.parents.setdefault(pinfo[0], pinfo[2])

    def _parse_daily_note(self, body: str, base_meta: dict) -> list[Document]:
        """
        Parse a daily note with multiple course lectures.

        Strategy: walk through heading nodes. When a heading matches a course
        pattern, start a new "lecture segment". All content under that heading
        (including sub-headings) belongs to that lecture until the next
        course-level heading.

        This handles a common daily-note format:
            # CS 251
            ## Topic: Policy Gradient
            content...
            ---
            # CS 246
            ## Attention Mechanism
            content...
        """
        nodes = build_heading_tree(body)
        spans = section_spans(nodes)
        stacks: list[list[str]] = []   # per-node heading-stack snapshots (E2)
        documents = []
        current_course = None
        current_course_meta = {}
        heading_stack = []  # for context inheritance
        skip_mode = False   # skip non-academic sections (see SKIP_HEADING_PATTERNS)

        for i, node in enumerate(nodes):
            # Check if this heading starts a new course lecture
            detected = detect_course_from_heading(node.title) if node.level >= 1 else None

            if detected == "__SKIP__":
                # Non-academic section (matches SKIP_HEADING_PATTERNS)
                skip_mode = True
                current_course = None
                current_course_meta = {}
                heading_stack = [node.title]
                stacks.append(list(heading_stack))
                continue

            if detected and detected != "__SKIP__" and node.level <= 3:
                # New course segment — exit skip mode
                skip_mode = False
                current_course = detected
                current_course_meta = resolve_course_metadata(detected)
                heading_stack = [node.title]
            elif node.level > 0:
                # Sub-heading within current course (or skipped section)
                while heading_stack and len(heading_stack) >= node.level:
                    heading_stack.pop()
                heading_stack.append(node.title)
            stacks.append(list(heading_stack))

            # Skip content in non-academic sections
            if skip_mode:
                self.stats["skipped_sections"] += 1
                continue

            if not node.content or len(node.content.strip()) < self.min_chunk:
                continue

            # Build chunk metadata
            chunk_meta = {**base_meta}
            chunk_meta["heading"] = node.title
            chunk_meta["heading_level"] = node.level
            chunk_meta["heading_path"] = " > ".join(heading_stack)
            if current_course_meta:
                chunk_meta.update(current_course_meta)

            # Build the chunk text with context header
            context = build_context_header(heading_stack, chunk_meta)
            cleaned = clean_text(node.content)

            # Split if too large
            sub_chunks = split_section(cleaned, self.max_chunk, self.overlap,
                                       self.chunking)
            pinfo = self._parent_candidate(nodes, spans, i, stacks, chunk_meta)

            for pi, chunk_text in enumerate(sub_chunks):
                if len(chunk_text.strip()) < self.min_chunk:
                    continue

                full_text = context + chunk_text
                meta = {**chunk_meta}
                if len(sub_chunks) > 1:
                    meta["chunk_part"] = f"{pi + 1}/{len(sub_chunks)}"
                self._maybe_link_parent(meta, pinfo, chunk_text)

                doc = Document(text=full_text, metadata=meta)
                documents.append(doc)
                self._update_stats(meta)

        return documents

    def _parse_standard_note(self, body: str, base_meta: dict) -> list[Document]:
        """
        Parse a standard (non-daily) note.
        Splits on H2/H3 boundaries with context inheritance.
        """
        nodes = build_heading_tree(body)
        spans = section_spans(nodes)
        stacks: list[list[str]] = []   # per-node heading-stack snapshots (E2)
        documents = []
        heading_stack = []

        for i, node in enumerate(nodes):
            # Maintain heading stack for context
            if node.level > 0:
                while heading_stack and len(heading_stack) >= node.level:
                    heading_stack.pop()
                heading_stack.append(node.title)
            stacks.append(list(heading_stack))

            if not node.content or len(node.content.strip()) < self.min_chunk:
                continue

            # If no course detected yet, try from this heading
            if base_meta.get("course_code") is None or base_meta.get("course_code") == "unknown":
                detected = detect_course_from_heading(node.title)
                if detected:
                    base_meta.update(resolve_course_metadata(detected))

            chunk_meta = {**base_meta}
            chunk_meta["heading"] = node.title
            chunk_meta["heading_level"] = node.level
            chunk_meta["heading_path"] = " > ".join(heading_stack)

            context = build_context_header(heading_stack, chunk_meta)
            cleaned = clean_text(node.content)

            sub_chunks = split_section(cleaned, self.max_chunk, self.overlap,
                                       self.chunking)
            pinfo = self._parent_candidate(nodes, spans, i, stacks, chunk_meta)

            for pi, chunk_text in enumerate(sub_chunks):
                if len(chunk_text.strip()) < self.min_chunk:
                    continue
                full_text = context + chunk_text
                meta = {**chunk_meta}
                if len(sub_chunks) > 1:
                    meta["chunk_part"] = f"{pi + 1}/{len(sub_chunks)}"
                self._maybe_link_parent(meta, pinfo, chunk_text)

                doc = Document(text=full_text, metadata=meta)
                documents.append(doc)
                self._update_stats(meta)

        return documents

    def _update_stats(self, meta: dict):
        """Track parsing statistics."""
        self.stats["total_chunks"] += 1
        domain = meta.get("domain", "general")
        self.stats["chunks_by_domain"][domain] = \
            self.stats["chunks_by_domain"].get(domain, 0) + 1
        course = meta.get("course_code", "unknown")
        self.stats["chunks_by_course"][course] = \
            self.stats["chunks_by_course"].get(course, 0) + 1

    def parse_all(self, verbose: bool = True) -> list[Document]:
        """
        Parse the entire vault. Returns all documents.

        This is the main entry point.
        """
        files = self.discover_files()
        if verbose:
            print(f"[parser] Found {len(files)} markdown files in {self.vault_path}")

        all_docs = []
        for filepath in files:
            try:
                docs = self.parse_file(filepath)
                all_docs.extend(docs)
                self.stats["files_processed"] += 1
            except Exception as e:
                print(f"[parser] ERROR processing {filepath}: {e}", file=sys.stderr)
                self.stats["files_skipped"] += 1

        if verbose:
            self._print_stats()

        return all_docs

    def _print_stats(self):
        """Print a summary of parsing results."""
        s = self.stats
        print(f"\n{'=' * 60}")
        print(f"  OBSIDIAN VAULT PARSING COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Files processed:  {s['files_processed']}")
        print(f"  Files skipped:    {s['files_skipped']}")
        print(f"  Daily notes:      {s['daily_notes']}")
        print(f"  Course notes:     {s['course_notes']}")
        print(f"  Other notes:      {s['other_notes']}")
        print(f"  Total chunks:     {s['total_chunks']}")
        print(f"  Skipped sections: {s['skipped_sections']} (matched SKIP_HEADING_PATTERNS)")
        print(f"\n  Chunks by domain:")
        for domain, count in sorted(s["chunks_by_domain"].items(),
                                     key=lambda x: -x[1]):
            print(f"    {domain:.<30} {count}")
        print(f"\n  Chunks by course:")
        for course, count in sorted(s["chunks_by_course"].items(),
                                     key=lambda x: -x[1]):
            print(f"    {course:.<30} {count}")
        print(f"{'=' * 60}\n")

    def export_jsonl(self, documents: list[Document], output_path: str):
        """Export documents as JSONL for downstream processing."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
        print(f"[parser] Exported {len(documents)} documents to {path}")

    def export_parents_jsonl(self, output_path: str):
        """Export the E2 parent-section sidecar (NOT embedded, NOT indexed —
        consumed only by retrieval.parent_context at generation time)."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in self.parents.values():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[parser] Exported {len(self.parents)} parent sections to {path}")


# ─────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Parse Obsidian vault for RAG ingestion")
    ap.add_argument("vault_path", help="Path to Obsidian vault root")
    ap.add_argument("-o", "--output", default="chunks.jsonl",
                    help="Output JSONL file (default: chunks.jsonl)")
    ap.add_argument("--max-chunk", type=int, default=MAX_CHUNK_SIZE,
                    help=f"Max chunk size in chars (default: {MAX_CHUNK_SIZE})")
    ap.add_argument("--min-chunk", type=int, default=MIN_CHUNK_SIZE,
                    help=f"Min chunk size in chars (default: {MIN_CHUNK_SIZE})")
    ap.add_argument("--overlap", type=int, default=OVERLAP_SIZE,
                    help=f"Overlap between split chunks (default: {OVERLAP_SIZE})")
    ap.add_argument("--chunking", default="heading", choices=CHUNKING_STRATEGIES,
                    help="How oversized sections are split: 'heading' = "
                         "paragraph packing (default), 'fixed' = sliding window")
    ap.add_argument("--preview", type=int, default=0,
                    help="Print N sample chunks to stdout")
    ap.add_argument("--parents-out", default="data/parents_md.jsonl",
                    help="Sidecar JSONL for E2 parent sections "
                         "(default: data/parents_md.jsonl; '' disables)")
    ap.add_argument("--skip-roots", default="",
                    help="Comma-separated TOP-LEVEL vault folders to exclude "
                         "(e.g. \"Workspace1,Workspace2\" — agent trees, not coursework)")

    args = ap.parse_args()

    config = {
        "max_chunk_size": args.max_chunk,
        "min_chunk_size": args.min_chunk,
        "overlap_size": args.overlap,
        "chunking": args.chunking,
        "skip_roots": [s.strip() for s in args.skip_roots.split(",") if s.strip()],
    }

    parser = ObsidianParser(args.vault_path, config=config)
    docs = parser.parse_all()

    if args.preview > 0:
        print(f"\n--- SAMPLE CHUNKS (first {args.preview}) ---\n")
        for doc in docs[:args.preview]:
            print(f"[{doc.doc_id}] {doc.metadata.get('source_file', '?')}")
            print(f"  course: {doc.metadata.get('course_name', '?')}")
            print(f"  domain: {doc.metadata.get('domain', '?')}")
            print(f"  heading: {doc.metadata.get('heading_path', '?')}")
            print(f"  text: {doc.text[:200]}...")
            print()

    parser.export_jsonl(docs, args.output)
    if args.parents_out:
        parser.export_parents_jsonl(args.parents_out)


if __name__ == "__main__":
    main()
