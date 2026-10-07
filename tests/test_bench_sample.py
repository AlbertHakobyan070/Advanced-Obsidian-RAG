import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.bench.questions import question_from_dict, save_suite
from eval.bench.sample import (
    SELECTORS, STRATA, iter_chunks, multihop_pairs, sample_suite, seed_record,
    topic_neighbourhoods, used_seed_ids, write_pack,
)

# One sentence that trips none of the lexical triggers (no snake_case, camelCase,
# call, acronym or "Capitalised <noun>" eponym), repeated past the 300-char floor.
SENT = "The bisection method halves the bracketing interval at every step until the tolerance is met. "
PROSE = SENT * 4                                  # ~375 chars once whitespace is collapsed
CODE = "def step(f, a, b):\n    m = (a + b) / 2\n    return m\n" * 8
NODE = "Node text. " * 20                          # 219 chars: over the canvas floor, under the 300 one
LANE = "x_chunks.jsonl"


def _rec(doc_id, text=PROSE, **meta):
    meta.setdefault("source_file", f"notes\\{doc_id}.md")
    meta.setdefault("file_type", "note")
    meta.setdefault("course_name", "Numerical Methods")
    meta.setdefault("domain", "math")
    return {"doc_id": doc_id, "text": text, "metadata": meta}


def _write_lane(data_dir, name, recs):
    (Path(data_dir) / name).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")


def _ok(suite, rec, lane=LANE):
    return SELECTORS[suite](lane, rec)


def _ids(records):
    return [r["doc_id"] for r in records]


# --- selectors: each accepts its positive example and rejects near-misses ----

LEXICAL_TRIGGERS = {
    "snake_case": " Call scipy_optimize here.",
    "camelCase": " Read the bracketWidth value.",
    "call": " The routine bisect(f, a, b) returns the root.",
    "acronym": " The ARIMA model is fitted next.",
    "acronym-plural": " The RMSEs are reported next.",
    "eponym": " We use Newton's method next.",
    "eponym-hyphenated": " The Kolmogorov–Smirnov test follows.",
}


@pytest.mark.parametrize("trigger", LEXICAL_TRIGGERS.values(), ids=LEXICAL_TRIGGERS.keys())
def test_lexical_accepts_each_branch(trigger):
    assert _ok("lexical", _rec("a", PROSE + trigger))


@pytest.mark.parametrize("tail", [
    "",                                               # plain prose
    " Export the PDF and read the ASCII text.",       # acronyms on the stoplist
    " Open the PDFs.",                                # stoplisted, with the plural s
    " The ABCDEFG label.",                            # seven capitals is not an acronym
    " We use the Test now.",                          # eponym needs a Capitalised name first
])
def test_lexical_rejects_near_misses(tail):
    assert not _ok("lexical", _rec("a", PROSE + tail))


def test_snake_case_inside_a_url_is_still_lexical():
    # Known imprecision, accepted: the rule is a regex over text, not a parser.
    # A drafter skips a seed that cannot carry a question about the term.
    assert _ok("lexical", _rec("a", PROSE + " See https://example.com/my_page for details."))


def test_paraphrase_takes_prose_from_pdfs_and_notes():
    assert _ok("paraphrase", _rec("p", file_type="pdf"))
    assert _ok("paraphrase", _rec("p", file_type="note"))


@pytest.mark.parametrize("rec", [
    _rec("p", file_type="ipynb"),                                    # wrong kind of file
    _rec("p", PROSE + " ```python\nx = 1\n```"),                     # carries a code fence
    _rec("p", "1.5 2.5 3.5 4.5 " * 30),                              # numbers, not prose
])
def test_paraphrase_rejects_near_misses(rec):
    assert not _ok("paraphrase", rec)


def test_paraphrase_alphabetic_share_boundary_is_sixty_percent():
    assert _ok("paraphrase", _rec("p", "a" * 240 + "1" * 160))       # 240/400 = 60%
    assert not _ok("paraphrase", _rec("p", "a" * 239 + "1" * 161))


@pytest.mark.parametrize("rec", [
    _rec("c", CODE, file_type="py"),                                  # def
    _rec("c", PROSE + " ```python\nx = 1\n```", file_type="ipynb"),   # fence
    _rec("c", "x <- c(1, 2, 3)\n" * 20, file_type="r"),               # R assignment
    _rec("c", "library(stats)\n" * 25, file_type="rmd"),              # library
    _rec("c", "import numpy as np\n" * 20, file_type="code"),
])
def test_code_accepts_code_in_code_files(rec):
    assert _ok("code", rec)


@pytest.mark.parametrize("rec", [
    _rec("c", CODE, file_type="note"),               # code, but in a note: the code lane is for code files
    _rec("c", PROSE, file_type="py"),                # prose in a code file
    _rec("c", PROSE, file_type="ipynb"),
])
def test_code_rejects_near_misses(rec):
    assert not _ok("code", rec)


def test_scoped_needs_a_real_course():
    assert _ok("scoped", _rec("s", course_name="Numerical Methods"))
    for course in ("", "-", "unknown", "Unknown", "__SKIP__", None):
        assert not _ok("scoped", _rec("s", course_name=course)), course


def test_books_need_a_page_and_a_book_lane():
    pdf = _rec("b", file_type="pdf", page_start=3, page_end=3)
    assert _ok("books", pdf, lane="pdf_chunks.jsonl")
    assert _ok("books", pdf, lane="ocr_bertsekas_chunks.jsonl")
    assert not _ok("books", pdf, lane="random_chunks.jsonl")                       # not a book lane
    assert not _ok("books", _rec("b", file_type="pdf"), lane="pdf_chunks.jsonl")   # no page
    assert not _ok("books", _rec("b", page_start=3), lane="pdf_chunks.jsonl")      # not a pdf


def test_canvas_needs_edges_and_may_be_shorter_than_other_chunks():
    assert _ok("canvas", _rec("c", NODE, file_type="canvas", canvas_edges=["n2|out|label"]))
    for edges in ([], None):
        assert not _ok("canvas", _rec("c", NODE, file_type="canvas", canvas_edges=edges))
    assert not _ok("canvas", _rec("c", NODE, canvas_edges=["e"]))                    # not a canvas chunk
    assert not _ok("canvas", _rec("c", "x " * 74, file_type="canvas", canvas_edges=["e"]))   # 147 < 150
    assert not _ok("scoped", _rec("n", NODE))        # the same text is too short for everything else


@pytest.mark.parametrize("meta", [
    {"file_type": "daily_note", "source_file": "2026-01-01.md"},
    {"source_file": "05 - Strategies\\plan.md"},
    {"source_file": "03 - Ideas/maybe.md"},
    {"source_file": "09 - Daily_Study_Notes\\week.md"},
    {"source_file": "0 — Current Workflow\\today.md"},
    {"source_file": "x\\DEAL LATER\\idea.md"},
    {"source_file": "Workspace1\\notes.md"},
])
def test_personal_accepts_daily_notes_and_personal_folders(meta):
    assert _ok("personal", _rec("p", **meta))


def test_personal_rejects_course_notes():
    assert not _ok("personal", _rec("p", source_file="Courses\\Stats\\notes.md"))


# --- the common skip, whatever the suite --------------------------------------

@pytest.mark.parametrize("why, defect", [
    ("short", {"text": "Too short to seed anything."}),
    ("collapsed-short", {"text": "ab\n\n\n\n\n\n" * 60}),            # 480 raw chars, 179 once collapsed
    ("toc-dots", {"text": PROSE + " . . ." * 30}),
    ("toc-heading", {"heading_path": "Some Book > Contents"}),
    ("skip-course", {"course_name": "__SKIP__"}),
])
def test_common_skips_apply_to_every_selector(why, defect):
    text = defect.get("text", PROSE)
    meta = {k: v for k, v in defect.items() if k != "text"}
    for suite in ("scoped", "paraphrase"):
        assert _ok(suite, _rec("a"))                                  # the control: no defect, taken
        assert not _ok(suite, _rec("a", text, **meta)), (suite, why)


def test_toc_dot_rule_is_five_leader_runs_and_a_heading_must_be_exactly_contents():
    # Three spaced dots make one leader run; an ellipsis in prose stays under five.
    assert _ok("scoped", _rec("a", PROSE + " . . ." * 4))
    assert not _ok("scoped", _rec("a", PROSE + " . . ." * 5))
    assert _ok("scoped", _rec("a", heading_path="Notes > Contents of a stack frame"))


# --- strata ----------------------------------------------------------------------

def test_strata_follow_the_suite_rules():
    meta = {"domain": "math", "course_name": "Numerical Methods", "source_file": "00 – A\\B\\Book.pdf"}
    assert STRATA["lexical"](meta) == STRATA["paraphrase"](meta) == "math"
    assert STRATA["code"](meta) == STRATA["scoped"](meta) == "Numerical Methods"
    assert STRATA["books"](meta) == STRATA["canvas"](meta) == "00 – a/b/book.pdf"
    assert STRATA["personal"]({"source_file": "05 - Strategies\\a\\b.md"}) == "05 - Strategies"
    assert STRATA["personal"]({"source_file": "05 - Strategies/a/b.md"}) == "05 - Strategies"


# --- iter_chunks / seed_record / write_pack -----------------------------------

def test_iter_chunks_is_sorted_by_file_name_and_skips_other_files(tmp_path):
    _write_lane(tmp_path, "chunks.jsonl", [_rec("c1")])
    _write_lane(tmp_path, "b_chunks.jsonl", [_rec("b1"), _rec("b2")])
    _write_lane(tmp_path, "a_chunks.jsonl", [_rec("a1")])
    _write_lane(tmp_path, "notes.jsonl", [_rec("n1")])
    assert [(lane, rec["doc_id"]) for lane, rec in iter_chunks(tmp_path)] == [
        ("a_chunks.jsonl", "a1"), ("b_chunks.jsonl", "b1"), ("b_chunks.jsonl", "b2"),
        ("chunks.jsonl", "c1")]


def test_iter_chunks_keeps_a_record_whole_when_its_text_contains_u2028(tmp_path):
    # Chunk text carries U+2028 / U+2029 / U+0085, which str.splitlines() would cut at.
    _write_lane(tmp_path, "x_chunks.jsonl", [_rec("a", "one two three\x85four")])
    [(_, rec)] = list(iter_chunks(tmp_path))
    assert rec["text"] == "one two three\x85four"


def test_a_record_without_a_doc_id_is_an_error_naming_the_file(tmp_path):
    _write_lane(tmp_path, "x_chunks.jsonl", [{"text": "t", "metadata": {}}])
    with pytest.raises(ValueError, match=r"x_chunks\.jsonl.*doc_id"):
        list(iter_chunks(tmp_path))


def test_seed_record_fields_collapse_and_cut():
    rec = _rec("abc", "word  \n  " * 1000, file_type="pdf", page_start=4, page_end=5,
               heading_path="Book > Ch 1", course_name="C", domain="d", source_file="B\\b.pdf")
    s = seed_record("pdf_chunks.jsonl", rec)
    assert list(s) == ["doc_id", "lane", "source_file", "file_type", "page_start", "page_end",
                       "heading_path", "course_name", "domain", "text"]
    assert (s["doc_id"], s["lane"], s["source_file"], s["page_start"], s["page_end"]) == (
        "abc", "pdf_chunks.jsonl", "B\\b.pdf", 4, 5)
    assert len(s["text"]) == 1600 and "  " not in s["text"] and "\n" not in s["text"]
    assert seed_record("x", _rec("n"))["page_start"] is None        # absent stays absent


def test_write_pack_is_utf8_jsonl_with_lf_endings(tmp_path):
    out = tmp_path / "deep" / "seeds" / "lexical.jsonl"             # parent does not exist yet
    write_pack(out, [{"doc_id": "a", "text": "διχοτόμηση"}, {"doc_id": "b", "text": "x"}])
    raw = out.read_bytes()
    assert b"\r" not in raw and "διχοτόμηση".encode("utf-8") in raw   # not \u escapes
    assert [json.loads(l)["doc_id"] for l in raw.decode("utf-8").splitlines()] == ["a", "b"]


# --- sample_suite -----------------------------------------------------------------

def _scoped_corpus(data_dir, n_files=12, per_file=5, courses=("A", "B", "C")):
    _write_lane(data_dir, "chunks.jsonl", [
        _rec(f"d{f}-{i}", source_file=f"{courses[f % 3]}\\f{f}.md", course_name=courses[f % 3])
        for f in range(n_files) for i in range(per_file)])


def test_same_seed_same_pack_different_seed_different_pack(tmp_path):
    _scoped_corpus(tmp_path)
    a = sample_suite("scoped", tmp_path, 8, seed=0)
    assert len(a) == 8 and a == sample_suite("scoped", tmp_path, 8, seed=0)
    assert _ids(a) != _ids(sample_suite("scoped", tmp_path, 8, seed=1))


def test_within_a_stratum_candidates_are_ordered_by_the_documented_hash(tmp_path):
    # One stratum, one chunk per file: the pack is exactly the first n by sha256(f"{seed}:{doc_id}").
    ids = [f"w{i}" for i in range(30)]
    _write_lane(tmp_path, "chunks.jsonl", [
        _rec(d, source_file=f"N\\{d}.md", course_name="Only") for d in ids])

    def by_hash(seed):
        return sorted(ids, key=lambda d: hashlib.sha256(f"{seed}:{d}".encode("utf-8")).hexdigest())[:5]

    assert _ids(sample_suite("scoped", tmp_path, 5, seed=0)) == by_hash(0)
    assert _ids(sample_suite("scoped", tmp_path, 5, seed=1)) == by_hash(1)
    assert by_hash(0) != ids[:5] != by_hash(1)       # the seed really changes the draw


def test_a_small_pack_is_not_biased_toward_the_start_of_the_alphabet(tmp_path):
    courses = [chr(ord("a") + i) for i in range(26)]
    _write_lane(tmp_path, "chunks.jsonl", [
        _rec(f"c{i}", source_file=f"{c}\\f.md", course_name=c) for i, c in enumerate(courses)])
    packs = {tuple(sorted(r["course_name"] for r in sample_suite("scoped", tmp_path, 3, seed=s)))
             for s in range(8)}
    assert len(packs) > 1                      # which strata are visited depends on the seed...
    assert packs != {("a", "b", "c")}          # ...and is not simply the first three


def test_per_file_cap_is_two_and_three_for_canvas(tmp_path):
    _scoped_corpus(tmp_path)                                         # 12 files x 5 chunks
    out = sample_suite("scoped", tmp_path, 100)
    assert len(out) == 24 and set(Counter(r["source_file"] for r in out).values()) == {2}
    _write_lane(tmp_path, "canvas_chunks.jsonl", [
        _rec(f"cv{f}-{i}", NODE, file_type="canvas", canvas_edges=["e"], source_file=f"cv{f}.canvas")
        for f in range(3) for i in range(6)])
    out = sample_suite("canvas", tmp_path, 100)
    assert len(out) == 9 and set(Counter(r["source_file"] for r in out).values()) == {3}


def test_cap_counts_slash_and_backslash_spellings_as_one_file(tmp_path):
    _write_lane(tmp_path, "chunks.jsonl", [_rec(f"w{i}", source_file="N\\a.md") for i in range(3)])
    _write_lane(tmp_path, "canvas_chunks.jsonl", [_rec(f"s{i}", source_file="N/a.md") for i in range(3)])
    assert len(sample_suite("scoped", tmp_path, 100)) == 2


def test_a_repeated_doc_id_is_returned_once(tmp_path):
    dup = _rec("dup", source_file="N\\a.md")
    _write_lane(tmp_path, "chunks.jsonl", [dup, dup, _rec("other", source_file="N\\b.md")])
    _write_lane(tmp_path, "pdf_chunks.jsonl", [dup])
    assert sorted(_ids(sample_suite("scoped", tmp_path, 10))) == ["dup", "other"]


def test_exclude_ids_are_never_returned(tmp_path):
    _scoped_corpus(tmp_path)
    first = sample_suite("scoped", tmp_path, 100)
    excluded = frozenset(_ids(first)[:5])
    second = sample_suite("scoped", tmp_path, 100, exclude_ids=excluded)
    assert excluded.isdisjoint(_ids(second)) and len(second) == len(first)   # replaced from the same files


def test_used_seed_ids_keep_a_rerun_from_handing_seeds_out_again(tmp_path):
    data, sets = tmp_path / "data", tmp_path / "sets"
    data.mkdir(), sets.mkdir()
    _scoped_corpus(data)
    drafted = _ids(sample_suite("scoped", data, 6))

    def q(qid, seeds, status):
        return question_from_dict({
            "id": qid, "question": f"question {qid}", "suite": "scoped", "tier": "T1", "split": None,
            "answerable": False, "gold": [], "nuggets": [], "expect_course": None,
            "provenance": {"author": "draft", "status": status, "seed_chunks": seeds}}, "t")

    save_suite(sets / "scoped.yaml", [q("scope-0001", drafted[:3], "draft"),
                                       q("scope-0002", drafted[3:], "rejected")])    # a rejected draft used its seeds too
    used = used_seed_ids(sets)
    assert used == frozenset(drafted)
    again = sample_suite("scoped", data, 100, exclude_ids=used)
    assert used.isdisjoint(_ids(again)) and again


def test_used_seed_ids_with_no_sets_yet_is_empty(tmp_path):
    assert used_seed_ids(tmp_path / "does-not-exist") == frozenset()


@pytest.mark.parametrize("suite", ["multihop", "unanswerable", "nope"])
def test_suites_without_a_selector_are_refused(tmp_path, suite):
    with pytest.raises(ValueError, match=suite):
        sample_suite(suite, tmp_path, 5)


def _books_corpus(data_dir, n_ocr_files):
    _write_lane(data_dir, "pdf_chunks.jsonl", [
        _rec(f"p{f}-{i}", file_type="pdf", page_start=i + 1, page_end=i + 1, source_file=f"books\\b{f}.pdf")
        for f in range(40) for i in range(2)])
    for lane, name in (("ocr_busstats_chunks.jsonl", "bus"), ("ocr_bertsekas_chunks.jsonl", "bert")):
        _write_lane(data_dir, lane, [
            _rec(f"{name}{f}-{i}", file_type="pdf", page_start=i + 1, page_end=i + 1,
                 source_file=f"ocr\\{name}{f}.pdf")
            for f in range(n_ocr_files) for i in range(2)])


def test_books_meet_the_ocr_minimum_when_the_corpus_offers_it(tmp_path):
    _books_corpus(tmp_path, n_ocr_files=4)                           # 16 OCR rows among 96 book rows
    out = sample_suite("books", tmp_path, 20)
    assert len(out) == 20 and len(set(_ids(out))) == 20
    assert sum(r["lane"].startswith("ocr_") for r in out) >= 6
    assert max(Counter(r["source_file"] for r in out).values()) <= 2   # the cap still holds


def test_books_take_every_ocr_row_there_is_when_fewer_than_the_minimum(tmp_path):
    _books_corpus(tmp_path, n_ocr_files=1)                           # 4 OCR rows
    out = sample_suite("books", tmp_path, 10)
    assert len(out) == 10 and sum(r["lane"].startswith("ocr_") for r in out) == 4


def test_books_ocr_minimum_never_exceeds_n(tmp_path):
    _books_corpus(tmp_path, n_ocr_files=4)
    assert len(sample_suite("books", tmp_path, 3)) == 3


# --- multihop ---------------------------------------------------------------------

def _hop_corpus(data_dir):
    """Terms: 'bisection method' (3 files, 2 courses; file A holds five of its chunks),
    'gradient descent' and 'secant method' (2 files each, one course and lane, the second
    reachable only once digits are stripped), plus rejects: a 4-letter word, a 7-word
    phrase, a heading in one file only."""
    recs = []
    for fname, course, heading in [
        ("A\\notes.md", "Numerical Methods", "Unit 1 > 3.2 Bisection Method"),
        ("A\\notes.md", "Numerical Methods", "Unit 1 > Bisection method"),     # same file, more chunks
        ("A\\notes.md", "Numerical Methods", "Unit 2 > Bisection method"),
        ("A\\notes.md", "Numerical Methods", "Unit 3 > Bisection method"),
        ("A\\notes.md", "Numerical Methods", "Unit 4 > Bisection method"),
        ("B\\slides.md", "Optimization", "Week 2 > Bisection method"),
        ("C\\lab.md", "Numerical Methods", "Lab > Bisection Method!"),
        ("G\\g.md", "Optimization", "Part > 4.1 Secant Method"),
        ("H\\h.md", "Optimization", "Part > Secant method"),
        ("D\\x.md", "Optimization", "Part > Gradient Descent"),
        ("E\\y.md", "Optimization", "Part > Gradient descent"),
        ("D\\x.md", "Optimization", "Part > Tips"),
        ("E\\y.md", "Optimization", "Part > Tips"),
        ("D\\x.md", "Optimization", "Part > one two three four five six seven"),
        ("E\\y.md", "Optimization", "Part > one two three four five six seven"),
        ("F\\z.md", "Optimization", "Part > Lonely heading"),
    ]:
        recs.append(_rec(f"h{len(recs):02d}", source_file=fname, course_name=course, heading_path=heading))
    _write_lane(data_dir, "chunks.jsonl", recs)


def test_multihop_terms_are_normalised_and_need_two_files(tmp_path):
    _hop_corpus(tmp_path)
    out = multihop_pairs(tmp_path, 10)
    assert sorted(item["term"] for item in out) == ["bisection method", "gradient descent",
                                                    "secant method"]


def test_multihop_chunks_come_from_different_files(tmp_path):
    _hop_corpus(tmp_path)
    for seed in range(10):                    # file A alone holds most of the bisection chunks
        for item in multihop_pairs(tmp_path, 10, seed=seed):
            files = [c["source_file"] for c in item["chunks"]]
            assert len(files) == len(set(files)) >= 2, (seed, item["term"])


def test_multihop_prefers_terms_that_span_courses_or_lanes(tmp_path):
    _hop_corpus(tmp_path)
    for seed in range(5):
        [item] = multihop_pairs(tmp_path, 1, seed=seed)
        assert item["term"] == "bisection method"                    # the other term stays in one course


def test_multihop_every_fourth_item_has_three_chunks(tmp_path):
    _write_lane(tmp_path, "chunks.jsonl", [
        _rec(f"t{t}f{f}", source_file=f"T{t}\\f{f}.md", course_name=f"C{f}",
             heading_path=f"Part > Topic{'x' * t} idea")
        for t in range(8) for f in range(3)])
    out = multihop_pairs(tmp_path, 8)
    assert len(out) == 8 and [len(i["chunks"]) for i in out] == [2, 2, 2, 3, 2, 2, 2, 3]


def test_multihop_is_deterministic_per_seed(tmp_path):
    _write_lane(tmp_path, "chunks.jsonl", [
        _rec(f"t{t}f{f}", source_file=f"T{t}\\f{f}.md", course_name=f"C{f}",
             heading_path=f"Part > Topic{'x' * t} idea")
        for t in range(8) for f in range(3)])
    a = multihop_pairs(tmp_path, 3, seed=0)
    assert a == multihop_pairs(tmp_path, 3, seed=0)
    assert [i["term"] for i in a] != [i["term"] for i in multihop_pairs(tmp_path, 3, seed=1)]


def test_multihop_exclude_ids_are_never_returned(tmp_path):
    _hop_corpus(tmp_path)
    before = multihop_pairs(tmp_path, 10)
    assert {i["term"] for i in before} == {"bisection method", "gradient descent", "secant method"}
    gradient = [c["doc_id"] for i in before if i["term"] == "gradient descent" for c in i["chunks"]]
    out = multihop_pairs(tmp_path, 10, exclude_ids=frozenset(gradient[:1]))
    assert {i["term"] for i in out} == {"bisection method", "secant method"}   # one file left: no longer multihop
    assert not set(gradient) & {c["doc_id"] for i in out for c in i["chunks"]}


def test_multihop_respects_the_per_file_cap(tmp_path):
    # Five terms all living in the same two files; the cap of 2 allows only two items.
    _write_lane(tmp_path, "chunks.jsonl", [
        _rec(f"k{t}{f}", source_file=f"F{f}.md", heading_path=f"Part > Topic{'y' * t} idea")
        for t in range(5) for f in range(2)])
    out = multihop_pairs(tmp_path, 10)
    assert len(out) == 2
    assert set(Counter(c["source_file"] for i in out for c in i["chunks"]).values()) == {2}


# --- topic neighbourhoods -----------------------------------------------------------

def test_topic_neighbourhoods_rank_courses_by_size_and_headings_by_frequency(tmp_path):
    recs = []

    def add(course, heading, n):
        for _ in range(n):
            recs.append(_rec(f"t{len(recs):03d}", course_name=course, heading_path=f"Book > {heading}"))

    add("Alpha", "Roots", 3), add("Alpha", "Series", 2), add("Alpha", "Limits", 2)
    add("Beta", "Graphs", 3)
    add("Gamma", "Lonely", 1)
    for blank in ("unknown", "__SKIP__", "-", ""):
        add(blank, "Ignored", 6)                                     # not courses, however big
    _write_lane(tmp_path, "chunks.jsonl", recs)
    out = topic_neighbourhoods(tmp_path, n_courses=2, n_headings=2)
    assert out == [{"course": "Alpha", "n_chunks": 7, "headings": ["Roots", "Limits"]},   # tie broken alphabetically
                   {"course": "Beta", "n_chunks": 3, "headings": ["Graphs"]}]


def test_topic_neighbourhoods_apply_the_common_skip(tmp_path):
    _write_lane(tmp_path, "chunks.jsonl", [
        _rec("a", course_name="Alpha", heading_path="Book > Roots"),
        _rec("b", "too short", course_name="Alpha", heading_path="Book > Tiny"),
        _rec("c", course_name="Alpha", heading_path="Book > Contents"),
    ])
    assert topic_neighbourhoods(tmp_path) == [{"course": "Alpha", "n_chunks": 1, "headings": ["Roots"]}]


# ---- multihop term filter (added 2026-10-06 after the first real-corpus pack) ----

@pytest.mark.parametrize("heading, term", [
    ("Book > Chapter 4 Text Classification with spaCy", "text classification with spacy"),
    ("Lecture 5 Gradient Descent", "gradient descent"),
    ("Topic Modeling", "topic modeling"),      # "topic" starts real terms; not stripped
    ("Singular Value Decomposition", "singular value decomposition"),
    ("Copyright", None),
    ("Key Takeaways", None),
    ("About the cover illustration", None),
    ("Chapter 7", None),                       # nothing left once the prefix goes
])
def test_heading_term_drops_structure_and_generic_headings(heading, term):
    from eval.bench.sample import _heading_term
    assert _heading_term({"heading_path": heading}) == term
