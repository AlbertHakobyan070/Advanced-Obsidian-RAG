"""route_strata.py — the route strata and the blind set they are drawn on (route-then-cite §5.8).

The strata (Task 15): a question's id prefix says whether a pointer covers its gold — rtc, some
required gold overlaps a sidecar target; rtu, none does. runner.py and report.py import
STRATA_PREFIXES.

The blind set (Task 16), eval/route_sets: questions drafted from primary pages only, never from
generated content, so the gate measures route on questions a map did not write.
  SCOPE_PREFIXES  vault-relative prefixes, matched on whole path segments after norm_file. The
                  spellings are copied from the live JSONLs (en and em dashes, 'Al', mojibake).
  sample_seeds    the seed pack the drafter reads: in scope, not generated, not skipped; hash
                  order; FILE_CAP per (file, chapter), or per file for a row with no chapter;
                  n // 3 reserved for the two books (R10); raises rather than come up short.
  classify        qid -> covered | uncovered against a pointer sidecar, over the required gold
                  and its alternatives.
  with_twins      a deck gold plus its same-stem .pdf <-> .md twin as an alternative (R11).
  locators_overlap  two gold locators some chunk could satisfy both (the dev/test guard, part C).
  NAMED           the phrasing denylist: a cold agent names no book, author, course or chapter.
eval/route_sets/README.md records the seed, n and how to regenerate the pack and the sidecar.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from eval.bench import sample
from eval.bench.questions import GoldSource
from eval.bench.relevance import _heading_prefix, norm_file
from src.ingestion.canvas_loader import norm_vault_path
from src.utils.provenance import provenance_of

STRATA_PREFIXES = ("rtc", "rtu")

# The two pilot books, where slice E measured the crowding: n // 3 of a pack is theirs (R10).
BOOK_PREFIXES = (
    "00 – AUA_DS/Other/Tech Books/AI Engineering & LLMs/AI Engineering - Chip Huyen (2025).pdf",
    "00 – AUA_DS/Other/Tech Books/AI Engineering & LLMs/Hands-On RAG for Production - Mendelevitch, Bao (2026).pdf",
)
# One prefix per AI Engineering deck folder: slide PDFs and their .md copies (R11 pairs them).
DECK_PREFIXES = (
    "00 – AUA_DS/Current Courses/AI Engineering/Building Al Applications with Pinecone/",
    "00 – AUA_DS/Current Courses/AI Engineering/Developing LLM Applications with LangChain/",
    "00 – AUA_DS/Current Courses/AI Engineering/Introduction to Embeddings with the OpenAI API/",
    "00 – AUA_DS/Current Courses/AI Engineering/Introduction to MCP/",
    "00 – AUA_DS/Current Courses/AI Engineering/LLM Application Evaluation with LangSmith/",
    "00 – AUA_DS/Current Courses/AI Engineering/LLMOps Concepts/",
    "00 – AUA_DS/Current Courses/AI Engineering/Prompt Engineering with the OpenAI/",
    "00 – AUA_DS/Current Courses/AI Engineering/Software Engineering Principles in Python/",
    "00 – AUA_DS/Current Courses/AI Engineering/Working with HuggingFace/",
    "00 – AUA_DS/Current Courses/AI Engineering/Working with the OpenAI API/",
    "00 – AUA_DS/Current Courses/AI Engineering/Working with the OpenAI Responses API/",
    "00 – AUA_DS/Current Courses/AI Engineering/Other/Building Agentic Workflows with LlamaIndex/",
    "00 – AUA_DS/Current Courses/AI Engineering/Other/Deploying AI into Production with FastAPI/",
    "00 – AUA_DS/Current Courses/AI Engineering/Other/Graph RAG with LangChain and Neo4j/",
    "00 – AUA_DS/Current Courses/AI Engineering/Other/Multi-Agent Systems with LangGraph/",
    "00 – AUA_DS/Current Courses/AI Engineering/Other/Software Development with Claude Code/",
    "00 – AUA_DS/Current Courses/AI Engineering/Other/Building Marketing Workflows with n8n/",
    "00 – AUA_DS/Current Courses/AI Engineering/Other/Intermediate n8n/",
    "00 – AUA_DS/Current Courses/AI Engineering/Other/Introduction to n8n/",
    "00 – AUA_DS/Current Courses/AI Engineering/Other/Introduction to Python API/",
)
SCOPE_PREFIXES = BOOK_PREFIXES + DECK_PREFIXES + (
    # the author's additions (2026-10-29): the course-root PDFs...
    "00 – AUA_DS/Current Courses/AI Engineering/Ai Engineering Expert 27Jul2026.pdf",
    "00 – AUA_DS/Current Courses/AI Engineering/Building Effective AI Agents (Anthropic).pdf",
    "00 – AUA_DS/Current Courses/AI Engineering/Claude+Certified+Architect+ΓÇô+Foundations+Exam+Guide.pdf",
    # ...and the project guides in Other/ (not Other/Datacamp ML)...
    "00 – AUA_DS/Current Courses/AI Engineering/Other/BASWE_Build_These_Six_Projects.pdf",
    "00 – AUA_DS/Current Courses/AI Engineering/Other/BASWE__15_AI_Engineering_Projects_Guide.docx.pdf",
    "00 – AUA_DS/Current Courses/AI Engineering/Other/Baswe-six-project-build-guide.pdf",
    "00 – AUA_DS/Current Courses/AI Engineering/Other/Final_projects.pdf",
    # ...and his own notes: the DataCamp canvases, the AI days of the daily notes, his NLP course
    # work (the HW5 Telegram project's .md docs, MD Notes/) and his Coursera NLP .md notes.
    "10 - CANVASes/AI Engineering (DataCamp)/",
    "09 - Daily_Study_Notes/16.05.2026.md",
    "09 - Daily_Study_Notes/15.05.2026.md",
    "09 - Daily_Study_Notes/31.01.2026.md",
    "09 - Daily_Study_Notes/04.02.2026.md",
    "09 - Daily_Study_Notes/02.05.2026.md",
    "09 - Daily_Study_Notes/20.05.2026.md",
    "09 - Daily_Study_Notes/21.05.2026.md",
    "09 - Daily_Study_Notes/23.05.2026.md",
    "00 – AUA_DS/Current Courses/NLP/Assignments/HW5/NLP Telegram Project/docs/design_report.md",
    "00 – AUA_DS/Current Courses/NLP/Assignments/HW5/NLP Telegram Project/README.md",
    "00 – AUA_DS/Current Courses/NLP/MD Notes/",
    "13 — Independent learning/Coursera Courses/NLP Full/Course 1/NLP C1 L1.md",
    "13 — Independent learning/Coursera Courses/NLP Full/Course 1/NLP C1 L2.md",
    "13 — Independent learning/Coursera Courses/NLP Full/Course 1/NLP C1 L3.md",
    "13 — Independent learning/Coursera Courses/NLP Full/Course 1/NLP C1 L4.md",
    "13 — Independent learning/Coursera Courses/NLP Full/Course 2/NLP C2 L1.md",
    "13 — Independent learning/Coursera Courses/NLP Full/Course 2/NLP C2 L2.md",
    "13 — Independent learning/Coursera Courses/NLP Full/Course 2/NLP C2 L3.md",
    "13 — Independent learning/Coursera Courses/NLP Full/Course 2/NLP C2 L4.md",
    "13 — Independent learning/Coursera Courses/NLP Full/Course 3/NLP C3 L1.md",
    "13 — Independent learning/Coursera Courses/NLP Full/Course 3/NLP C3 L2.md",
    "13 — Independent learning/Coursera Courses/NLP Full/Course 3/NLP C3 L3.md",
    "13 — Independent learning/Coursera Courses/NLP Full/Course 4/NLP C4 L1.md",
    "13 — Independent learning/Coursera Courses/NLP Full/Course 4/NLP C4 L2.md",
    "13 — Independent learning/Coursera Courses/NLP Full/Course 4/NLP C4 L3.md",
)


def _under(p: str, prefixes) -> str | None:
    """The first normalised prefix that normalised path p sits under, on whole path segments."""
    return next((pre for pre in prefixes if p == pre or p.startswith(pre + "/")), None)


def in_scope(source_file) -> bool:
    """source_file sits under a SCOPE_PREFIXES entry (norm_file on both sides, whole segments)."""
    return _under(norm_file(source_file), [norm_file(p) for p in SCOPE_PREFIXES]) is not None


def sample_seeds(data_dir, rule, n: int, seed: int) -> list[dict]:
    """n seed records (sample.seed_record) for the blind set's drafter.

    Keeps the rows that are in scope, that `rule` does not call generated, and that sample's
    common skip passes; a doc_id repeated across the JSONLs counts once, as its first
    occurrence (the index's rule). Rows are taken in sha256(seed:doc_id) order, at most
    sample.FILE_CAP per unit: (file, chapter) for a row with a chapter, else the file. n // 3
    come from BOOK_PREFIXES, the rest from the other prefixes. Raises ValueError when a scope
    prefix has no row to keep (a dead or mistyped prefix) or when the pack would be short."""
    scope = [norm_file(p) for p in SCOPE_PREFIXES]
    books = {norm_file(p) for p in BOOK_PREFIXES}
    kept: Counter = Counter()
    seen: set[str] = set()
    cands: list[tuple[str, bool, tuple, dict]] = []
    for lane, rec in sample.iter_chunks(Path(data_dir)):
        did = sample._did(rec)
        if did in seen:
            continue
        seen.add(did)
        meta = sample._meta(rec)
        f = norm_file(meta.get("source_file") or "")
        pre = _under(f, scope)
        if pre is None or provenance_of(meta, rule) == "generated" or sample._skipped(meta, sample._text(rec)):
            continue
        kept[pre] += 1
        chapter = meta.get("chapter")
        unit = (f, None if chapter in (None, "") else str(chapter))
        cands.append((sample._hash(seed, did), pre in books, unit, sample.seed_record(lane, rec)))
    dead = [p for p, q in zip(SCOPE_PREFIXES, scope) if not kept[q]]
    if dead:
        raise ValueError(f"{len(dead)} scope prefix(es) hold no primary, unskipped row under "
                         f"{data_dir}: {dead}; fix SCOPE_PREFIXES in eval/bench/route_strata.py")
    cands.sort(key=lambda c: c[0])
    per_unit: Counter = Counter()

    def draw(book: bool, k: int) -> list[dict]:
        out: list[dict] = []
        for _, is_book, unit, rec in cands:
            if len(out) == k:
                break
            if is_book == book and per_unit[unit] < sample.FILE_CAP:
                per_unit[unit] += 1
                out.append(rec)
        return out

    want = n // 3
    picked_books, picked_rest = draw(True, want), draw(False, n - want)
    if len(picked_books) < want or len(picked_rest) < n - want:
        raise ValueError(f"the scope gives fewer seeds than n={n} under the per-unit cap: "
                         f"{len(picked_books)} of {want} from the books, {len(picked_rest)} of "
                         f"{n - want} from the rest")
    return picked_books + picked_rest


def _overlap(pages, span) -> bool:
    """bench _overlaps: None on either side overlaps; else the ranges share a page."""
    return pages is None or span is None or not (pages[1] < span[0] or span[1] < pages[0])


def classify(questions, pointers) -> dict[str, str]:
    """qid -> "covered" when a required gold entry or one of its alternatives overlaps a target of
    `pointers` (a RoutePointers; same file by norm_vault_path, pages by _overlap), else "uncovered".
    The questions are not edited."""
    spans: dict[str, list] = defaultdict(list)
    for t in pointers.all_targets():
        a, b = t["page_start"], t["page_end"]
        spans[norm_vault_path(t["source_file"])].append(None if a is None else (a, a if b is None else b))
    out: dict[str, str] = {}
    for q in questions:
        if q.id in out:
            raise ValueError(f"{q.id}: two questions share this id; classify keys on it")
        hit = any(_overlap(s.pages, span) for g in q.gold if g.required for s in g.sources()
                  for span in spans.get(norm_vault_path(s.file), ()))
        out[q.id] = "covered" if hit else "uncovered"
    return out


def locators_overlap(a: GoldSource, b: GoldSource) -> bool:
    """Some chunk could satisfy both locators, as relevance._locates judges one: the same file,
    pages that overlap (None is the whole file), and, when both name a heading, one heading a
    prefix of the other. An empty heading constrains nothing, as in _locates. Two locators with
    neither pages nor heading are the whole file: they overlap."""
    return (norm_file(a.file) == norm_file(b.file) and _overlap(a.pages, b.pages)
            and (not a.heading or not b.heading
                 or _heading_prefix(a.heading, b.heading) or _heading_prefix(b.heading, a.heading)))


# The phrasing rule (eval/route_sets/README.md, "Phrasing"). "AI Engineering" is matched in its
# title form only (the book, the course), so the concept "AI engineering" stays askable; the rest
# in any case.
NAMED = (re.compile(r"\bAI Engineering\b"),
         re.compile(r"\b(?:Hands[- ]On RAG|Huyen|Mendelevitch|Bao|DataCamp|Coursera|lectures?|chapters?"
                    r"|Chapter ?\d+|Ch\.? ?\d+)\b", re.IGNORECASE))


def named_source(question: str) -> str | None:
    """The first text in `question` that NAMED forbids, else None."""
    return next((m.group(0) for rx in NAMED if (m := rx.search(question))), None)


# --- deck twins (R11) ---------------------------------------------------------------------

# A slide's page, as the .md copies mark headings. Only this marker pairs a twin: a digest's
# (p. N–M) or a practice set's (lesson N) points at slides that do not hold the passage itself.
_PAGE_MARK = re.compile(r"\[p\. ?(\d+)\]")


def _page(heading) -> int | None:
    """The page a heading path sits on: its last [p. N] marker."""
    marks = _PAGE_MARK.findall(heading or "")
    return int(marks[-1]) if marks else None


def deck_files(data_dir) -> dict[str, tuple[str, tuple[str, ...], tuple[tuple[int, int], ...]]]:
    """norm_file -> (stored source_file, distinct heading paths in first-seen order, the sorted
    distinct (page_start, page_end) spans of its chunks) for every corpus file under DECK_PREFIXES:
    what with_twins pairs against. A repeated doc_id counts once."""
    decks = [norm_file(p) for p in DECK_PREFIXES]
    out: dict[str, tuple[str, list[str], set]] = {}
    seen: set[str] = set()
    for _, rec in sample.iter_chunks(Path(data_dir)):
        did = sample._did(rec)
        if did in seen:
            continue
        seen.add(did)
        meta = sample._meta(rec)
        source = meta.get("source_file") or ""
        if _under(norm_file(source), decks) is None:
            continue
        _, heads, spans = out.setdefault(norm_file(source), (source, [], set()))
        if meta.get("heading_path") and meta["heading_path"] not in heads:
            heads.append(meta["heading_path"])
        if meta.get("page_start") is not None:        # page_end defaults to page_start, as in _locates
            a = int(meta["page_start"])
            spans.add((a, int(meta["page_end"]) if meta.get("page_end") is not None else a))
    return {f: (source, tuple(heads), tuple(sorted(spans))) for f, (source, heads, spans) in out.items()}


def _md_twins(twin: tuple, pages) -> tuple[GoldSource, ...]:
    """The .md copy of a .pdf gold: at each heading whose page falls inside the gold pages (one
    covering another's subtree is enough), else the whole file."""
    source, heads, _ = twin
    if pages is not None:
        hit = sorted((h for h in heads if _page(h) is not None and pages[0] <= _page(h) <= pages[1]),
                     key=lambda h: (_page(h), h))
        hit = [h for h in hit if not any(o != h and _heading_prefix(h, o) for o in hit)]
        if hit:
            return tuple(GoldSource(source, heading=h) for h in hit)
    return (GoldSource(source),)


def _pdf_twin(twin: tuple, md_heads: tuple[str, ...], heading) -> tuple[GoldSource, ...]:
    """The .pdf of an .md gold: min..max of the slide pages ([p. N]) its heading's subtree marks,
    when a .pdf chunk covers them; the whole file for a file-level gold. None for a heading no slide
    marks (a digest's (p. N–M), a practice set's (lesson N): not a slide's passage), or whose pages
    no .pdf chunk covers (that slide is not in the corpus; a twin there would be an empty locator)."""
    source, _, spans = twin
    if not heading:
        return (GoldSource(source),)
    pages = {_page(h) for h in (*md_heads, heading) if _heading_prefix(h, heading)} - {None}
    if not pages or not any(_overlap((min(pages), max(pages)), s) for s in spans):
        return ()
    return (GoldSource(source, pages=(min(pages), max(pages))),)


def with_twins(gold: GoldSource, decks) -> GoldSource:
    """`gold` with its same-stem twin (.pdf <-> .md, same folder) added as an alternative when the
    gold is a deck file, the twin is in `decks` (deck_files' output) and is not listed yet;
    otherwise `gold` itself. A deck gold missing from `decks` raises ValueError."""
    f = norm_file(gold.file)
    if _under(f, [norm_file(p) for p in DECK_PREFIXES]) is None:
        return gold
    if f not in decks:
        raise ValueError(f"{gold.file}: a deck gold that is not in the corpus")
    stem, _, ext = f.rpartition(".")
    twin = decks.get(f"{stem}.md" if ext == "pdf" else f"{stem}.pdf") if ext in ("pdf", "md") else None
    if twin is None:
        return gold
    new = _md_twins(twin, gold.pages) if ext == "pdf" else _pdf_twin(twin, decks[f][1], gold.heading)
    have = {(norm_file(a.file), a.pages, a.heading) for a in gold.alternatives}
    add = tuple(a for a in new if (norm_file(a.file), a.pages, a.heading) not in have)
    return replace(gold, alternatives=gold.alternatives + add) if add else gold
