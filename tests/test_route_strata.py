"""eval/bench/route_strata.py: the blind route set's scope, seed draw, strata and deck twins
(route-then-cite Task 16), on synthetic data. The live sets are guarded in test_route_sets.py."""
import hashlib
import json
import unicodedata
from collections import Counter

import pytest

from eval.bench import route_strata as rs
from eval.bench import sample
from eval.bench.questions import GoldSource, Question
from eval.bench.relevance import norm_file
from src.retrieval.route import RoutePointers
from src.utils.provenance import ProvenanceRule

BOOK = "Books/Book A.pdf"
DECK, DECK2 = "Course/Deck/", "Course/Deck2/"
RULE = ProvenanceRule(by_file_type={"pdf": "source"}, generated_paths=("course/maps",))
TEXT = "word " * (sample.MIN_CHARS // 5 + 1)        # passes the common skip's length rule
# One live path per area left out of the scope (by the author's choice or by the rule), copied from
# the stored source_file (the Exam workspace, which has no row, from the vault on disk).
EXCLUDED = (
    "00 – AUA_DS\\Current Courses\\AI Engineering\\langgraph-101-main\\agents\\deep_agent\\AGENTS.md",   # langgraph-101-main
    "00 – AUA_DS\\Current Courses\\AI Engineering\\_Blueprints — AI Engineering\\00 — START HERE (AGENT GUIDE).md",   # _Blueprints
    "00 – AUA_DS\\Current Courses\\AI Engineering\\AI Engineering Track - Module Notes\\01 - Working with the OpenAI API.md",   # Module Notes
    "00 – AUA_DS\\Current Courses\\AI Engineering\\Other\\Datacamp ML\\Developing Machine Learning Models for Production\\Ch1 - Adopting an MLOps mindset.md",   # Other/Datacamp ML
    "00 – AUA_DS\\Current Courses\\AI Engineering\\Other\\Datacamp SQL\\Analyzing Business Data in SQL\\Ch1 - Introduction and revenue.md",   # Other/Datacamp SQL
    "00 – AUA_DS/Current Courses/AI Engineering/AI Engineering Exam workspace/chatbot_responses.json",   # Exam workspace
    "00 – AUA_DS/Current Courses/AI Engineering",   # course root
    "00 – AUA_DS/Current Courses/AI Engineering/AI Engineering Track - Graph (generated).canvas",   # course-root generated canvas
    "09 - Daily_Study_Notes\\13.11.2024.md",   # non-AI daily note
    "13 — Independent learning\\Coursera Courses\\NLP Full\\Course 1\\W1\\C1_W1_Assignment.ipynb",   # Coursera ipynb
    "00 – AUA_DS\\Current Courses\\NLP\\Assignments\\HW5\\NLP Telegram Project\\bot.py",   # Telegram bot py
)


def _row(did, source_file, text=TEXT, **meta):
    return {"doc_id": did, "text": text, "metadata": {"source_file": source_file, **meta}}


def _data(tmp_path, **lanes):
    """One <lane>_chunks.jsonl per keyword, holding its rows."""
    d = tmp_path / "data"
    d.mkdir(parents=True)
    for lane, rows in lanes.items():
        (d / f"{lane}_chunks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return d


@pytest.fixture
def scope(monkeypatch):
    """Point the module at a synthetic scope; returns a setter for SCOPE_PREFIXES."""
    monkeypatch.setattr(rs, "BOOK_PREFIXES", (BOOK,))
    monkeypatch.setattr(rs, "DECK_PREFIXES", (DECK, DECK2))

    def set_scope(*prefixes):
        monkeypatch.setattr(rs, "SCOPE_PREFIXES", prefixes)
    set_scope(BOOK, "Course/")
    return set_scope


# ---- in_scope ---------------------------------------------------------------------------

def test_in_scope_matches_whole_path_segments(scope):
    scope("Course/Deck One/", "Notes/day.md")
    assert rs.in_scope("Course\\Deck One\\a.pdf")                  # stored backslashes
    assert rs.in_scope("course/DECK ONE/sub/b.md")                  # casefolded, any depth
    assert rs.in_scope(unicodedata.normalize("NFD", "Notes/day.md"))
    assert rs.in_scope("Notes/day.md")                              # a file prefix is the file itself
    assert not rs.in_scope("Course/Deck One Extra/a.pdf")          # a longer folder name is not under it
    assert not rs.in_scope("Notes/day.md.bak")
    assert not rs.in_scope("Course/Deck")                           # nor is a parent
    assert not rs.in_scope("Elsewhere/Course/Deck One/a.pdf")      # a prefix anchors at the vault root


def test_the_live_scope_spellings_are_the_stored_ones():
    """Copied from the live JSONLs (U+2013 in 00 – AUA_DS, U+2014 in 13 — Independent learning,
    'Al' with a lowercase L, the exam guide's mojibake), and the books sit inside the scope."""
    assert set(rs.BOOK_PREFIXES) | set(rs.DECK_PREFIXES) <= set(rs.SCOPE_PREFIXES)
    assert len(set(map(norm_file, rs.SCOPE_PREFIXES))) == len(rs.SCOPE_PREFIXES)
    joined = "\n".join(rs.SCOPE_PREFIXES)
    assert "00 – AUA_DS/" in joined and "13 — Independent learning/" in joined
    assert "Building Al Applications with Pinecone/" in joined
    assert "Claude+Certified+Architect+ΓÇô+Foundations+Exam+Guide.pdf" in joined
    assert not [p for p in rs.SCOPE_PREFIXES if "\\" in p or p != unicodedata.normalize("NFC", p)]
    assert not [p for p in rs.SCOPE_PREFIXES if "Datacamp ML" in p or "_AI Engineering Maps" in p]
    assert not [p for p in EXCLUDED if rs.in_scope(p)]      # an over-broad prefix would take one in


@pytest.mark.parametrize("text", [
    "What does AI Engineering say about evals?", "the Hands-On RAG way", "hands on rag for production",
    "As Huyen puts it", "per Mendelevitch", "Bao's example", "the DataCamp track", "a coursera quiz",
    "in lecture 4", "Lectures cover it", "the chapter on drift", "two chapters", "Chapter3", "chapter 3",
    "Ch3", "ch 3", "Ch.2", "ch. 2"])
def test_named_source_rejects_titles_authors_platforms_lectures_and_chapters(text):
    assert rs.named_source(text)


@pytest.mark.parametrize("text", [
    "How does AI engineering differ from ML engineering?", "What do AI-engineering teams log?",
    "Why teach 2 retrievers?", "Is Chapman's approach 3 sound?", "Why does a baobab of indexes help?"])
def test_named_source_allows_the_concepts(text):
    assert rs.named_source(text) is None


# ---- sample_seeds ----------------------------------------------------------------------

def test_sample_seeds_keeps_the_scope_and_drops_generated_skipped_and_repeated_rows(tmp_path, scope):
    """Every file here holds at most FILE_CAP keepers, so n = the keepers draws them all and n + 1
    raises: a wrongly kept row would make n + 1 succeed."""
    data = _data(
        tmp_path,
        a=[_row("dup", "Elsewhere/x.pdf"),                         # first occurrence: out of scope
           _row("bk1", BOOK, file_type="pdf"), _row("bk2", BOOK, file_type="pdf"),
           _row("k1", "Course\\Deck\\a.pdf", file_type="pdf"),     # stored backslashes
           _row("k2", "Course/Deck/a.pdf", file_type="pdf"),
           _row("k3", "Course/Deck/b.md", file_type="note"), _row("k4", "Course/Deck/b.md", file_type="note"),
           _row("g1", "Course/Maps/m.canvas", file_type="canvas"),        # generated by the path rule
           _row("g2", "Course/Deck/c.md", file_type="note", provenance="generated"),
           _row("s1", "Course/Deck/d.md", text="too short", file_type="note"),
           _row("o1", "Elsewhere/e.pdf", file_type="pdf")],
        b=[_row("dup", "Course/Deck/e.md", file_type="note")])     # a later copy does not win
    got = rs.sample_seeds(data, RULE, 6, 0)
    assert {r["doc_id"] for r in got} == {"bk1", "bk2", "k1", "k2", "k3", "k4"}
    assert all(r["lane"] == "a_chunks.jsonl" and r["text"] for r in got)
    with pytest.raises(ValueError, match="fewer"):
        rs.sample_seeds(data, RULE, 7, 0)


def _cap_world(tmp_path):
    """One book of five chapters (and chapterless rows), three course files of 10, 2 and 2 rows."""
    book = [_row(f"b{c}{i}", BOOK, file_type="pdf", chapter=str(c)) for c in range(5) for i in range(3)]
    book += [_row(f"bn{i}", BOOK, file_type="pdf") for i in range(3)]
    course = [_row(f"x{i}", "Course/Deck/x.pdf", file_type="pdf") for i in range(10)]
    course += [_row(f"{f}{i}", f"Course/Deck/{f}.pdf", file_type="pdf") for f in "yz" for i in range(2)]
    return _data(tmp_path, a=book + course)


def test_sample_seeds_caps_per_file_and_chapter_and_reserves_a_third_for_the_books(tmp_path, scope):
    got = rs.sample_seeds(_cap_world(tmp_path), RULE, 9, 0)
    assert len(got) == 9 and len({r["doc_id"] for r in got}) == 9
    books = [r for r in got if r["source_file"] == BOOK]
    assert len(books) == 9 // 3 > sample.FILE_CAP               # one file gives more than FILE_CAP...
    ids = {r["doc_id"] for r in got}
    per_chapter = Counter(i[:2] for i in ids if i.startswith("b"))
    assert max(per_chapter.values()) <= sample.FILE_CAP          # ...because the unit is (file, chapter)
    assert Counter(r["source_file"] for r in got if r["source_file"] != BOOK) == {
        "Course/Deck/x.pdf": 2, "Course/Deck/y.pdf": 2, "Course/Deck/z.pdf": 2}   # chapterless: per file


def test_sample_seeds_is_deterministic_per_seed(tmp_path, scope):
    data = _cap_world(tmp_path)
    first = [r["doc_id"] for r in rs.sample_seeds(data, RULE, 9, 0)]
    assert first == [r["doc_id"] for r in rs.sample_seeds(data, RULE, 9, 0)]
    assert first != [r["doc_id"] for r in rs.sample_seeds(data, RULE, 9, 1)]


def test_sample_seeds_takes_rows_in_the_documented_sha256_order(tmp_path, scope):
    """One row per unit, so the cap never binds: books first, each part the first k of its pool
    by sha256(f"{seed}:{doc_id}") (README, "Seeds")."""
    books, course = [f"b{i}" for i in range(6)], [f"c{i}" for i in range(12)]
    data = _data(tmp_path, a=[_row(d, BOOK, file_type="pdf", chapter=d) for d in books]
                 + [_row(d, f"Course/Deck/{d}.pdf", file_type="pdf") for d in course])

    def by_hash(ids, seed, k):
        return sorted(ids, key=lambda d: hashlib.sha256(f"{seed}:{d}".encode("utf-8")).hexdigest())[:k]

    assert by_hash(course, 0, 6) != course[:6] != by_hash(course, 1, 6)
    for seed in (0, 1):
        got = [r["doc_id"] for r in rs.sample_seeds(data, RULE, 9, seed)]
        assert got == by_hash(books, seed, 3) + by_hash(course, seed, 6), seed


def test_sample_seeds_raises_rather_than_return_a_short_pack(tmp_path, scope):
    with pytest.raises(ValueError, match="fewer"):
        rs.sample_seeds(_cap_world(tmp_path), RULE, 12, 0)   # the course gives 6 under the cap, not 8
    course = [_row(f"x{i}", "Course/Deck/x.pdf", file_type="pdf") for i in range(10)]
    course += [_row(f"{f}{i}", f"Course/Deck/{f}.pdf", file_type="pdf") for f in "yz" for i in range(2)]
    thin = _data(tmp_path / "thin", a=[_row("b1", BOOK, file_type="pdf"), _row("b2", BOOK, file_type="pdf")] + course)
    with pytest.raises(ValueError, match="fewer"):
        rs.sample_seeds(thin, RULE, 9, 0)        # the book gives 2, not 9 // 3; the course has its 6


def test_a_scope_prefix_with_no_row_to_keep_raises_naming_it(tmp_path, scope):
    scope(BOOK, "Course/Deck/", "Nowhere/", "Course/Maps/")
    data = _data(tmp_path, a=[_row("b1", BOOK, file_type="pdf"), _row("c1", "Course/Deck/x.pdf"),
                              _row("g1", "Course/Maps/m.canvas", file_type="canvas")])
    with pytest.raises(ValueError, match="Nowhere/") as e:
        rs.sample_seeds(data, RULE, 1, 0)
    assert "Course/Maps/" in str(e.value)        # only generated rows under it: nothing to keep


# ---- classify --------------------------------------------------------------------------

def _q(qid, *gold):
    return Question(id=qid, question="q?", suite="books", tier="T1", split=None, answerable=True,
                    gold=list(gold), nuggets=[], expect_course=None, provenance={"status": "draft"})


def _pointers(*targets):
    return RoutePointers({"pointers": {"m1": [{"source_file": f, "page_start": a, "page_end": b}
                                              for f, a, b in targets]}})


def test_classify_covered_when_a_required_gold_overlaps_a_target():
    ptr = _pointers(("Course\\Deck\\a.pdf", 6, 6), ("Café/Book.pdf", None, None), ("Course/Deck/c.pdf", 3, 9))
    got = rs.classify([
        _q("rtq-0001", GoldSource("Course/Deck/a.pdf", pages=(5, 7))),                    # overlap
        _q("rtq-0002", GoldSource("Course/Deck/a.pdf", pages=(1, 3))),                    # disjoint pages
        _q("rtq-0003", GoldSource("Café/Book.pdf", pages=(40, 41))),                      # pageless target
        _q("rtq-0004", GoldSource(unicodedata.normalize("NFD", "CAFÉ\\book.pdf"), pages=(2, 2))),  # NFD, '\', case
        _q("rtq-0005", GoldSource("Course/Deck/a.pdf")),                                  # pageless gold
        _q("rtq-0006", GoldSource("Course/Deck/b.pdf", pages=(6, 6),
                                  alternatives=(GoldSource("Course/Deck/a.md"), GoldSource("Course/Deck/a.pdf", pages=(6, 6))))),
        _q("rtq-0007", GoldSource("Course/Deck/b.pdf", pages=(6, 6))),                    # another file
        _q("rtq-0008", GoldSource("Course/Deck/b.pdf", pages=(6, 6)),
           GoldSource("Course/Deck/a.pdf", pages=(6, 6), required=False)),               # optional gold
        _q("rtq-0009", GoldSource("Course/Deck/c.pdf", pages=(9, 12))),                  # touches at an edge
        _q("rtq-0010", GoldSource("Course/Deck/b.pdf", pages=(6, 6), hop=1),             # multihop: hop 2's
           GoldSource("Course/Deck/a.pdf", pages=(6, 6), hop=2)),                        # gold is enough
        _q("rtq-0011", GoldSource("Course/Deck/b.pdf", pages=(6, 6), hop=1),
           GoldSource("Course/Deck/a.pdf", pages=(1, 3), hop=2)),                        # neither hop
    ], ptr)
    assert got == {"rtq-0001": "covered", "rtq-0002": "uncovered", "rtq-0003": "covered",
                   "rtq-0004": "covered", "rtq-0005": "covered", "rtq-0006": "covered",
                   "rtq-0007": "uncovered", "rtq-0008": "uncovered", "rtq-0009": "covered",
                   "rtq-0010": "covered", "rtq-0011": "uncovered"}


def test_classify_refuses_a_repeated_id():
    with pytest.raises(ValueError, match="rtq-0001"):
        rs.classify([_q("rtq-0001", GoldSource("a.pdf")), _q("rtq-0001", GoldSource("b.pdf"))], _pointers())


# ---- deck twins (R11) ------------------------------------------------------------------

PDF, MD = DECK + "Ch2 - Vectors.pdf", DECK + "Ch2 - Vectors.md"
QUERY = "Ch2 - Vectors > 3. Querying  <sub>[p. 9]</sub>"
H11, H12, H13 = (f"{QUERY} > The .query() method <sub>[p. {p}]</sub>" for p in (11, 12, 13))
PRACTICE = "Ch2 - Vectors > Practice set > Fetch or query <sub>(lessons 1, 3)</sub>"
DIGEST = "Ch2 - Vectors > Combining <sub>(p. 24–26)</sub>"                 # digest markers: en dash...
DIGEST_CHILD = f"{DIGEST} > Wrapping it <sub>(p. 27-30)</sub>"             # ...hyphen...
DIGEST_LIST = "Ch2 - Vectors > What not to store <sub>(p. 4, 20–23)</sub>"  # ...a list
HEADS = ("Ch2 - Vectors", QUERY, H11, H12, H13, PRACTICE, DIGEST, DIGEST_CHILD, DIGEST_LIST)
DOTTED, DOTTED_H = DECK + "Ch3 - v1.2.pdf", "Ch3 - v1.2 > Intro <sub>[p. 4]</sub>"
DECK2_H = "Ch2 - Vectors > Intro <sub>[p. 4]</sub>"     # same name as Deck's .md (whose .pdf covers p. 4)
PDF_SPANS = ((28, 28), (4, 4), (12, 12), (24, 24), (12, 12), (30, 32))
SPANNED = "Ch2 - Vectors > Late <sub>[p. 31]</sub>"         # p. 31 sits inside the 30-32 chunk


def _decks(tmp_path):
    rows = [_row(f"p{a}", PDF.replace("/", "\\"), file_type="pdf", page_start=a, page_end=b)
            for a, b in PDF_SPANS]                               # the .pdf's chunks: no page 11 or 13
    rows += [_row("p40", PDF.replace("/", "\\"), file_type="pdf", page_start=40)]   # no page_end: p. 40 only
    rows += [_row(f"m{i}", MD.replace("/", "\\"), file_type="note", heading_path=h)
             for i, h in enumerate(HEADS[:4] + (H12,) + HEADS[4:])]
    rows += [_row("o1", DECK + "Only.md", file_type="note", heading_path="Only"),
             _row("d2", DECK2 + "Ch2 - Vectors.md", file_type="note", heading_path=DECK2_H),
             _row("v1", DOTTED, file_type="pdf", page_start=4, page_end=4),
             _row("v2", DOTTED[:-4] + ".md", file_type="note", heading_path=DOTTED_H),
             _row("bk", BOOK, file_type="pdf", page_start=1, page_end=1)]
    later = [_row("m3", MD, file_type="note", heading_path="Ch2 - Vectors > Stray <sub>[p. 99]</sub>")]
    return rs.deck_files(_data(tmp_path, a=rows, b=later))     # a repeated doc_id counts once


def _alts(g):
    return [(a.file, a.pages, a.heading) for a in g.alternatives]


def test_deck_files_holds_the_deck_files_with_their_stored_spelling_and_headings(tmp_path, scope):
    decks = _decks(tmp_path)
    assert set(decks) == set(map(norm_file, (PDF, MD, DECK + "Only.md", DECK2 + "Ch2 - Vectors.md",
                                             DOTTED, DOTTED[:-4] + ".md")))     # no book
    assert decks[norm_file(MD)] == (MD.replace("/", "\\"), HEADS, ())           # not the repeat's heading
    assert decks[norm_file(PDF)] == (PDF.replace("/", "\\"), (),
                                     ((4, 4), (12, 12), (24, 24), (28, 28), (30, 32), (40, 40)))


def test_a_pdf_gold_gets_its_md_twin_at_the_headings_its_pages_hold(tmp_path, scope):
    decks, stored = _decks(tmp_path), MD.replace("/", "\\")
    assert _alts(rs.with_twins(GoldSource(PDF, pages=(12, 12)), decks)) == [(stored, None, H12)]
    assert _alts(rs.with_twins(GoldSource(PDF, pages=(11, 12)), decks)) == [(stored, None, H11), (stored, None, H12)]
    assert _alts(rs.with_twins(GoldSource(PDF, pages=(9, 12)), decks)) == [(stored, None, QUERY)]   # holds 11, 12
    assert _alts(rs.with_twins(GoldSource(PDF, pages=(27, 27)), decks)) == [(stored, None, None)]   # no heading there
    assert _alts(rs.with_twins(GoldSource(PDF), decks)) == [(stored, None, None)]


def test_an_md_gold_gets_its_pdf_twin_at_the_pages_its_headings_mark(tmp_path, scope):
    decks, stored = _decks(tmp_path), PDF.replace("/", "\\")
    assert _alts(rs.with_twins(GoldSource(MD, heading=H12), decks)) == [(stored, (12, 12), None)]
    assert _alts(rs.with_twins(GoldSource(MD, heading=QUERY), decks)) == [(stored, (9, 13), None)]
    assert _alts(rs.with_twins(GoldSource(MD), decks)) == [(stored, None, None)]
    assert _alts(rs.with_twins(GoldSource(MD, heading=""), decks)) == [(stored, None, None)]   # no constraint
    gold = GoldSource(MD, heading=PRACTICE)                       # no page marker: not a slide's passage
    assert rs.with_twins(gold, decks) is gold
    gold = GoldSource(MD, heading=H13)                            # marks p. 13, which no .pdf chunk covers:
    assert rs.with_twins(gold, decks) is gold                     # a twin there would be an empty locator
    assert _alts(rs.with_twins(GoldSource(MD, heading=SPANNED), decks)) == [(stored, (31, 31), None)]


def test_with_twins_leaves_what_has_no_twin_or_lists_it_already_and_refuses_an_unknown_deck_file(tmp_path, scope):
    decks = _decks(tmp_path)
    for gold in (GoldSource(DECK + "Only.md"),                      # no same-stem twin
                 GoldSource(BOOK, pages=(1, 1)),                    # not a deck gold
                 GoldSource(PDF, pages=(12, 12), alternatives=(GoldSource(MD.replace("/", "\\"), heading=H12),))):
        assert rs.with_twins(gold, decks) is gold
    once = rs.with_twins(GoldSource(unicodedata.normalize("NFD", PDF.upper()), pages=(12, 12)), decks)
    assert _alts(once) == [(MD.replace("/", "\\"), None, H12)] and rs.with_twins(once, decks) is once
    with pytest.raises(ValueError, match="not in the corpus"):
        rs.with_twins(GoldSource(DECK + "Gone.pdf", pages=(1, 1)), decks)


def test_with_twins_pairs_inside_one_folder_keeps_a_dotted_stem_and_knows_a_listed_twin_in_any_spelling(
        tmp_path, scope):
    decks = _decks(tmp_path)
    other = GoldSource(DECK2 + "Ch2 - Vectors.md", heading=DECK2_H)
    assert rs.with_twins(other, decks) is other              # Deck2 has no .pdf; Deck's is not its twin
    assert _alts(rs.with_twins(GoldSource(DOTTED, pages=(4, 4)), decks)) == [(DOTTED[:-4] + ".md", None, DOTTED_H)]
    for spelling in (MD, MD.upper()):                        # the stored spelling is MD with backslashes
        listed = GoldSource(PDF, pages=(12, 12), alternatives=(GoldSource(spelling, heading=H12),))
        assert rs.with_twins(listed, decks) is listed, spelling


def test_an_md_gold_under_a_digest_heading_gets_no_twin(tmp_path, scope):
    """A digest's (p. N–M, K) is not a slide-page claim: the slides it names do not hold the passage
    (part C review F2, reversing F13). So no twin, although .pdf chunks cover pp. 4, 24 and 28."""
    decks, md = _decks(tmp_path), MD.replace("/", "\\")
    for heading in (DIGEST_CHILD, DIGEST, DIGEST_LIST):          # hyphen, en dash, a list
        gold = GoldSource(MD, heading=heading)
        assert rs.with_twins(gold, decks) is gold, heading
    assert _alts(rs.with_twins(GoldSource(PDF, pages=(25, 25)), decks)) == [(md, None, None)]


def test_locators_overlap_as_the_relevance_rule_locates():
    """The dev/test guard's rule: same file, pages overlap (None = the whole file), headings related
    by prefix when both name one."""
    G, pdf = GoldSource, "Course/Deck/a.pdf"
    assert not rs.locators_overlap(G(MD, heading=H11), G(MD, heading=H12))     # one .md, two slides
    assert rs.locators_overlap(G(MD, heading=QUERY), G(MD.replace("/", "\\"), heading=H12))   # a heading, its child
    assert rs.locators_overlap(G(MD, heading=H12), G(MD, heading=QUERY))
    assert rs.locators_overlap(G(MD), G(MD, heading=H12))                       # the whole file holds the heading
    assert rs.locators_overlap(G(MD, heading=""), G(MD, heading=H12))           # "" constrains nothing (_locates)
    assert rs.locators_overlap(G(MD, heading=H11), G(MD, heading=""))
    assert rs.locators_overlap(G("Course/m.canvas"), G("course\\M.canvas"))     # two whole-file locators
    assert not rs.locators_overlap(G(pdf, pages=(5, 6)), G(pdf, pages=(7, 9)))
    assert rs.locators_overlap(G(pdf, pages=(5, 7)), G(pdf, pages=(7, 9)))
    assert rs.locators_overlap(G(pdf, pages=(5, 5)), G(pdf))
    assert not rs.locators_overlap(G(MD, heading=H12), G(PDF, pages=(12, 12)))  # twins are listed, not inferred
