"""
sample.py — draw the seed chunks the question drafts are written from.

Drafting 400 questions out of a ~160k-chunk corpus needs a sample that is spread
over the corpus instead of clumped in its biggest course, is the same sample
when asked twice, and never hands out the same seed again. A pack is one JSONL
of seed records per suite; a drafter reads a seed, writes at most one question
about it, and records the seed's doc_id in provenance.seed_chunks.

How a pack is drawn (deterministic per seed; no random module):
  1. the suite's selector keeps the chunks that suit it, after a skip applied
     to everything (too short, a contents page, an unassigned course);
  2. candidates are grouped into strata (a domain, a course, a book, ...) and
     each stratum is ordered by sha256(f"{seed}:{doc_id}");
  3. strata are visited round-robin, one chunk each per round, so a small suite
     draws from many strata instead of the biggest; at most 2 seeds per file
     (3 per canvas) so one file cannot dominate a pack.
Strata are visited in a seeded hash order as well, so a pack smaller than the
number of strata is not biased toward the start of the alphabet.

Memory: the corpus is read in two streaming passes — one keeps a small tuple
per candidate, the other fetches only the chunks chosen. Holding every
candidate's text is the pattern that MemoryError'd the index builder here.

The selectors are regexes over text, not parsers, and each accepts some chunks a
human would not (a snake_case token inside a URL, a sentence that opens "The
model", "value(s)" read as a call). That is acceptable: a drafter skips a seed
that cannot carry a vault-specific question, and the CLI over-draws to leave
room for it.

Packs contain vault text: local data, never synced.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Callable, Iterator, NamedTuple

from eval.bench.questions import load_sets
from eval.bench.relevance import norm_file
from src.embeddings.embedder import iter_jsonl_records

# Rule constants (spec §3.1). They define what the selectors MEAN, so they live
# beside the rules rather than in config.yaml, like the suite quotas.
SEED_TEXT_CHARS = 1600      # how much of a chunk a drafter is shown
MIN_CHARS = 300             # a shorter chunk cannot carry a question of its own...
CANVAS_MIN_CHARS = 150      # ...except a canvas node, whose connections carry the rest
TOC_DOTS = 5                # ". . ." leader runs that mark a contents page
FILE_CAP = 2                # seeds from one file in one pack...
CANVAS_FILE_CAP = 3         # ...a canvas file is the unit, so it may give three
ALPHA_SHARE = 0.6           # paraphrase wants prose, not tables or formulae
BOOKS_OCR_MIN = 6           # books seeds that must come from the OCR lanes, when that many exist

OCR_LANES = frozenset({"ocr_busstats_chunks.jsonl", "ocr_bertsekas_chunks.jsonl"})
BOOK_LANES = frozenset({"pdf_chunks.jsonl", "tech_chunks.jsonl",
                        "ai_engineering_pdf_chunks.jsonl"}) | OCR_LANES
CODE_TYPES = frozenset({"ipynb", "py", "r", "rmd", "code"})
PERSONAL_MARKERS = ("05 - Strategies", "03 - Ideas", "09 - Daily_Study_Notes",
                    "0 — Current Workflow", "DEAL LATER", "Workspace1")
_NO_COURSE = frozenset({"", "-", "unknown", "__skip__"})    # compared casefolded

_SNAKE = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")
_CAMEL = re.compile(r"\b[a-z]+[A-Z][A-Za-z0-9]+\b")
_CALL = re.compile(r"\b[A-Za-z_]\w*\(")
_ACRONYM = re.compile(r"\b([A-Z]{3,6})s?\b")
_ACRONYM_STOP = frozenset("PDF THE AND FOR NOT AUA DS CS ASCII".split())
# "Newton's method", "Kolmogorov–Smirnov test": a Capitalised name, then a noun
# (the noun alone is case-insensitive).
_EPONYM = re.compile(
    r"\b[A-Z][a-z]+(?:[-–][A-Z][a-z]+)?(?:'s)? "
    r"(?i:theorem|test|method|distribution|lemma|criterion|algorithm|inequality|formula|rule|law|model)\b")
_CODE_WORD = re.compile(r"\b(?:def|import|class|library|function)\b")
_TERM_JUNK = re.compile(r"[\W\d_]+")


# --- reading the corpus ---------------------------------------------------------

def _meta(rec: dict) -> dict:
    return rec.get("metadata") or {}


def _text(rec: dict) -> str:
    """Chunk text with whitespace collapsed: what the length and content rules
    judge, and what a drafter reads."""
    return " ".join((rec.get("text") or "").split())


def _did(rec: dict) -> str:
    return str(rec["doc_id"])


def iter_chunks(data_dir: Path) -> Iterator[tuple[str, dict]]:
    """(lane file name, record) for every record of every *chunks.jsonl in
    `data_dir`, files in sorted name order. The lane is the file a chunk lives in
    (`books` selects on it). Streams with the index builders' own reader, which
    splits on b"\\n" only — chunk text contains U+2028 and friends."""
    for path in sorted(Path(data_dir).glob("*chunks.jsonl")):
        for rec in iter_jsonl_records(path):
            if not rec.get("doc_id"):
                raise ValueError(f"{path.name}: a chunk record has no doc_id, "
                                 f"so it cannot be named as a seed")
            yield path.name, rec


def seed_record(lane: str, rec: dict) -> dict:
    """What a drafter sees of one chunk: where it is, and its text (whitespace
    collapsed, cut to SEED_TEXT_CHARS)."""
    meta = _meta(rec)
    return {
        "doc_id": _did(rec), "lane": lane,
        "source_file": meta.get("source_file"), "file_type": meta.get("file_type"),
        "page_start": meta.get("page_start"), "page_end": meta.get("page_end"),
        "heading_path": meta.get("heading_path"), "course_name": meta.get("course_name"),
        "domain": meta.get("domain"), "text": _text(rec)[:SEED_TEXT_CHARS],
    }


# --- selectors ------------------------------------------------------------------

def _last_part(meta: dict) -> str:
    return (meta.get("heading_path") or "").split(">")[-1].strip()


def _skipped(meta: dict, text: str) -> bool:
    """The skip every suite shares: nothing to ask about, or not real content."""
    if len(text) < (CANVAS_MIN_CHARS if meta.get("file_type") == "canvas" else MIN_CHARS):
        return True
    # a contents page: dot leaders, or a heading whose last part is "Contents"
    if text.count(". . .") >= TOC_DOTS or _last_part(meta).casefold() == "contents":
        return True
    return meta.get("course_name") == "__SKIP__"


def _selector(rule: Callable[[str, dict, str], bool]) -> Callable[[str, dict], bool]:
    """Turn a suite rule over (lane, metadata, collapsed text) into a selector
    over (lane, record) that applies the common skip first; the text is
    collapsed once and shared by both."""
    def select(lane: str, rec: dict) -> bool:
        meta, text = _meta(rec), _text(rec)
        return not _skipped(meta, text) and rule(lane, meta, text)
    return select


@_selector
def _lexical(lane, meta, text):
    # The exact-term suite: a chunk with an identifier, a call, an acronym or a
    # named theorem in it, so BM25 has something specific to match.
    return (any(rx.search(text) for rx in (_SNAKE, _CAMEL, _CALL, _EPONYM))
            or any(m.group(1) not in _ACRONYM_STOP for m in _ACRONYM.finditer(text)))


@_selector
def _paraphrase(lane, meta, text):
    return (meta.get("file_type") in ("pdf", "note") and "```" not in text
            and sum(ch.isalpha() for ch in text) / len(text) >= ALPHA_SHARE)


@_selector
def _code(lane, meta, text):
    return (meta.get("file_type") in CODE_TYPES
            and ("```" in text or bool(_CODE_WORD.search(text)) or "<-" in text))


@_selector
def _scoped(lane, meta, text):
    return (meta.get("course_name") or "").strip().casefold() not in _NO_COURSE


@_selector
def _books(lane, meta, text):
    return (meta.get("file_type") == "pdf" and meta.get("page_start") is not None
            and lane in BOOK_LANES)


@_selector
def _canvas(lane, meta, text):
    return meta.get("file_type") == "canvas" and bool(meta.get("canvas_edges"))


@_selector
def _personal(lane, meta, text):
    source = meta.get("source_file") or ""
    return meta.get("file_type") == "daily_note" or any(m in source for m in PERSONAL_MARKERS)


SELECTORS: dict[str, Callable[[str, dict], bool]] = {
    "lexical": _lexical, "paraphrase": _paraphrase, "code": _code, "scoped": _scoped,
    "books": _books, "canvas": _canvas, "personal": _personal,
}


def _top_folder(meta: dict) -> str:
    return (meta.get("source_file") or "").replace("\\", "/").split("/")[0]


# Suite -> the stratum a chunk belongs to, from its metadata. Works on a seed
# record as well (it carries the same keys), so the CLI can count strata in a pack.
STRATA: dict[str, Callable[[dict], str]] = {
    "lexical": lambda m: m.get("domain") or "",
    "paraphrase": lambda m: m.get("domain") or "",
    "code": lambda m: m.get("course_name") or "",
    "scoped": lambda m: m.get("course_name") or "",
    "books": lambda m: norm_file(m.get("source_file") or ""),
    "canvas": lambda m: norm_file(m.get("source_file") or ""),
    "personal": _top_folder,
}


# --- drawing a pack -------------------------------------------------------------

def _hash(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()


class _Cand(NamedTuple):
    key: str        # sha256 order within a stratum
    doc_id: str
    stratum: str
    file: str       # norm_file: the unit of the per-file cap
    lane: str


def _candidates(suite: str, data_dir: Path, seed: int, exclude_ids) -> list[_Cand]:
    select, stratum = SELECTORS[suite], STRATA[suite]
    seen: set[str] = set()
    out: list[_Cand] = []
    for lane, rec in iter_chunks(data_dir):
        did = _did(rec)
        # First occurrence wins, as in the index: chunks.jsonl repeats some doc_ids.
        if did in exclude_ids or did in seen or not select(lane, rec):
            continue
        seen.add(did)
        meta = _meta(rec)
        out.append(_Cand(_hash(seed, did), did, stratum(meta),
                         norm_file(meta.get("source_file") or ""), lane))
    return out


def _round_robin(cands: list[_Cand], n: int, cap: int, seed: int, per_file: Counter) -> list[_Cand]:
    """Up to n candidates, one per stratum per round. `per_file` is shared with
    the caller so a second call continues the same cap."""
    strata: dict[str, list[_Cand]] = defaultdict(list)
    for c in sorted(cands, key=lambda c: c.key):
        strata[c.stratum].append(c)
    queues = [deque(strata[s]) for s in sorted(strata, key=lambda s: _hash(seed, f"stratum:{s}"))]
    out: list[_Cand] = []
    while queues and len(out) < n:
        for q in queues:
            while q:
                c = q.popleft()
                if per_file[c.file] < cap:      # a capped file's chunk is dropped for good
                    per_file[c.file] += 1
                    out.append(c)
                    break
            if len(out) == n:
                break
        queues = [q for q in queues if q]
    return out


def _fetch(data_dir: Path, wanted: set[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    """(lane, doc_id) -> record for the chunks chosen, in a second streaming pass."""
    found: dict[tuple[str, str], dict] = {}
    if not wanted:
        return found
    for lane, rec in iter_chunks(data_dir):
        key = (lane, _did(rec))
        if key in wanted and key not in found:
            found[key] = rec
            if len(found) == len(wanted):
                break
    if len(found) != len(wanted):
        raise RuntimeError(f"{len(wanted) - len(found)} chosen chunk(s) are gone from {data_dir} "
                           f"since the first pass: the corpus changed mid-run; run again")
    return found


def sample_suite(suite: str, data_dir: Path, n: int, seed: int = 0,
                 exclude_ids=frozenset()) -> list[dict]:
    """Up to n seed records for `suite`, none of them in `exclude_ids`. Fewer than
    n when the corpus (after the skips and the per-file cap) has fewer to give."""
    if suite not in SELECTORS:
        raise ValueError(f"no seed selector for suite {suite!r}; have {', '.join(SELECTORS)} "
                         f"(multihop is multihop_pairs, unanswerable is topic_neighbourhoods)")
    cands = _candidates(suite, Path(data_dir), seed, exclude_ids)
    cap = CANVAS_FILE_CAP if suite == "canvas" else FILE_CAP
    per_file: Counter = Counter()
    picked: list[_Cand] = []
    if suite == "books":
        # The OCR lanes are a small share of the book rows but the hardest text
        # for retrieval (scans), so reserve a minimum before the general draw.
        picked = _round_robin([c for c in cands if c.lane in OCR_LANES],
                              min(BOOKS_OCR_MIN, n), cap, seed, per_file)
        taken = {c.doc_id for c in picked}
        cands = [c for c in cands if c.doc_id not in taken]
    picked += _round_robin(cands, n - len(picked), cap, seed, per_file)
    found = _fetch(Path(data_dir), {(c.lane, c.doc_id) for c in picked})
    return [seed_record(c.lane, found[(c.lane, c.doc_id)]) for c in picked]


# --- multihop and unanswerable --------------------------------------------------

# Structural words that open a heading without naming its topic ("Chapter 4 Text
# Classification", "Lecture 5 Gradient Descent"): stripped from the front. Kept
# deliberately short — "topic", "part" and "unit" also START real terms ("topic
# modeling", "part of speech", "unit root test").
_TERM_PREFIX = frozenset("chapter ch lecture lec week module".split())
# Headings every book and course has: two files sharing one says nothing about
# their content, so they make worthless multi-hop pairs. Measured on the real
# corpus (2026-10-06): about a third of the first unfiltered pack was these.
_GENERIC_TERMS = frozenset({
    "copyright", "chapter", "key takeaways", "takeaways", "executive summary",
    "summary", "introduction", "intro", "overview", "conclusion", "conclusions",
    "contents", "table of contents", "preface", "foreword", "dedication",
    "acknowledgments", "acknowledgements", "about packt", "about the author",
    "about the authors", "about the cover illustration", "references",
    "bibliography", "index", "appendix", "glossary", "further reading",
    "exercises", "problems", "problem points", "questions", "answers",
    "solutions", "notes", "models", "testing", "examples", "example", "review",
    "practice", "homework", "quiz", "exam", "group work response", "discussion",
    "results", "methods", "abstract", "objectives", "learning objectives",
    "agenda", "outline", "recap", "untitled", "slide", "slides",
    "technical requirements", "back cover", "appendices", "differences",
})


def _heading_term(meta: dict) -> str | None:
    """The last heading part as a comparable term: casefolded, digits and
    punctuation replaced by spaces (so "3.2 Bisection Method!" and "Bisection
    method" are one term), leading structural words ("chapter", "lecture", …)
    removed, generic headings ("copyright", "key takeaways", …) rejected, and
    kept if it is 2-6 words or one word of 5+ letters."""
    words = _TERM_JUNK.sub(" ", _last_part(meta).casefold()).split()
    while words and words[0] in _TERM_PREFIX:
        words = words[1:]
    if " ".join(words) in _GENERIC_TERMS:
        return None
    if 2 <= len(words) <= 6 or (len(words) == 1 and len(words[0]) >= 5):
        return " ".join(words)
    return None


class _Hop(NamedTuple):
    key: str
    doc_id: str
    file: str
    course: str
    lane: str


def multihop_pairs(data_dir: Path, n: int, seed: int = 0, exclude_ids=frozenset()) -> list[dict]:
    """Up to n {term, chunks: [seed, seed(, seed)]}: a heading term that names
    sections of different files, with one chunk from each of two (three for every
    fourth item, for the T3/T4 drafts). Terms that span courses or lanes come
    first. A term needs two files that are neither excluded nor at the per-file cap."""
    by_term: dict[str, list[_Hop]] = defaultdict(list)
    seen: set[str] = set()
    for lane, rec in iter_chunks(data_dir):
        did = _did(rec)
        if did in exclude_ids or did in seen:
            continue
        meta = _meta(rec)
        term = _heading_term(meta)
        if term is None or _skipped(meta, _text(rec)):
            continue
        seen.add(did)
        by_term[term].append(_Hop(_hash(seed, did), did, norm_file(meta.get("source_file") or ""),
                                  meta.get("course_name") or "", lane))
    for hops in by_term.values():
        hops.sort(key=lambda h: h.key)

    def spans(hops: list[_Hop]) -> bool:
        return len({h.course for h in hops}) >= 2 or len({h.lane for h in hops}) >= 2

    per_file: Counter = Counter()
    items: list[tuple[str, list[_Hop]]] = []
    for term in sorted(by_term, key=lambda t: (not spans(by_term[t]), _hash(seed, f"term:{t}"))):
        if len(items) >= n:
            break
        want = 3 if len(items) % 4 == 3 else 2
        picks: list[_Hop] = []
        for h in by_term[term]:
            if len(picks) < want and per_file[h.file] < FILE_CAP and h.file not in {p.file for p in picks}:
                picks.append(h)
        if len(picks) < 2:
            continue
        per_file.update(h.file for h in picks)
        items.append((term, picks))
    found = _fetch(Path(data_dir), {(h.lane, h.doc_id) for _, picks in items for h in picks})
    return [{"term": term, "chunks": [seed_record(h.lane, found[(h.lane, h.doc_id)]) for h in picks]}
            for term, picks in items]


def topic_neighbourhoods(data_dir: Path, n_courses: int = 15, n_headings: int = 40) -> list[dict]:
    """For the `unanswerable` drafts: the n_courses biggest courses, each with
    {course, n_chunks, headings} — its n_headings most common last heading parts,
    i.e. what the vault covers, so a drafter can ask about what it does not. Counts
    only chunks that pass the common skip; blank and `unknown` courses are not
    courses. Ties break alphabetically."""
    sizes: Counter = Counter()
    headings: dict[str, Counter] = defaultdict(Counter)
    for _, rec in iter_chunks(data_dir):
        meta = _meta(rec)
        course = (meta.get("course_name") or "").strip()
        if course.casefold() in _NO_COURSE or _skipped(meta, _text(rec)):
            continue
        sizes[course] += 1
        if _last_part(meta):
            headings[course][_last_part(meta)] += 1
    ranked = sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))[:n_courses]
    return [{"course": course, "n_chunks": count,
             "headings": [h for h, _ in sorted(headings[course].items(),
                                               key=lambda kv: (-kv[1], kv[0]))[:n_headings]]}
            for course, count in ranked]


# --- bookkeeping ----------------------------------------------------------------

def used_seed_ids(sets_dir) -> frozenset[str]:
    """Every provenance.seed_chunks id in the existing sets, so a re-run after
    drafting never hands a seed out twice. Rejected drafts count: their seed was
    used up. A sets directory that does not exist yet has none."""
    return frozenset(str(s) for q in load_sets(sets_dir, include_rejected=True)
                     for s in q.provenance.get("seed_chunks") or [])


def write_pack(path, records: list[dict]) -> None:
    """One JSON record per line, UTF-8 (not \\u escapes), LF endings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
