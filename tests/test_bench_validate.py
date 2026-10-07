import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.bench.questions import SUITES, question_from_dict
from eval.bench.relevance import norm_file
from eval.bench.validate import (
    Finding, load_gold_chunks, quota_table, validate, write_review_cache,
)

PDF_TEXT = "The bisection method halves the bracketing interval each step until tolerance"
BOOK = "B\\book.pdf"
COPY = "the bisection method halves the bracketing interval"


def _write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


@pytest.fixture
def corpus(tmp_path):
    """Two chunk files, plus a decoy that is not a chunk file and must never be read."""
    _write(tmp_path / "x_chunks.jsonl", [
        {"doc_id": "p1", "text": PDF_TEXT,
         "metadata": {"source_file": BOOK, "page_start": 5, "page_end": 5}},
        {"doc_id": "m1", "text": "Newton's method converges quadratically near a simple root",
         "metadata": {"source_file": "N\\newton.md", "heading_path": "Ch 4 > Newton"}},
    ])
    _write(tmp_path / "y_chunks.jsonl", [
        {"doc_id": "z1", "text": "Unrelated text about something else entirely",
         "metadata": {"source_file": "Z\\other.md"}},
    ])
    _write(tmp_path / "notes.jsonl", [
        {"doc_id": "n1", "text": "decoy", "metadata": {"source_file": BOOK}},
    ])
    return tmp_path


def _q(**over):
    d = {"id": "par-0001",
         "question": "How does each iteration of the interval-halving root finder shrink the bracket?",
         "suite": "paraphrase", "tier": "T1", "split": None, "answerable": True,
         "gold": [{"file": "B/book.pdf", "pages": 5}],
         "nuggets": ["halves the bracketing interval"], "expect_course": None,
         "provenance": {"author": "draft", "status": "draft", "seed_chunks": ["p1"]},
         "notes": ""}
    d.update(over)
    return question_from_dict(d, "t")


def _run(corpus, *qs):
    files = {g.file for q in qs for g in q.gold}
    return validate(list(qs), load_gold_chunks(corpus, files))


def _codes(findings):
    return [f.code for f in findings]


def test_clean_question_has_no_findings(corpus):
    assert _run(corpus, _q()) == []


def test_page_outside_the_chunks_is_locator_empty(corpus):
    f = _run(corpus, _q(gold=[{"file": "B/book.pdf", "pages": 9}]))
    # One error and nothing else: the nugget is "unsupported" only because there
    # is no gold text left to judge it against, which would bury the real problem.
    assert [(x.level, x.code, x.qid) for x in f] == [("error", "locator-empty", "par-0001")]
    assert isinstance(f[0], Finding) and "9" in f[0].message


def test_heading_that_matches_no_chunk_is_locator_empty(corpus):
    f = _run(corpus, _q(gold=[{"file": "N/newton.md", "heading": "Ch 9"}], nuggets=["quadratically"]))
    assert _codes(f) == ["locator-empty"]


def test_absent_file_is_gold_file_missing(corpus):
    f = _run(corpus, _q(gold=[{"file": "C/none.pdf"}]))
    assert [(x.level, x.code) for x in f] == [("error", "gold-file-missing")]
    assert "C/none.pdf" in f[0].message


def test_unsupported_nugget_is_a_warning(corpus):
    f = _run(corpus, _q(nuggets=["uses Newton steps"]))
    assert [(x.level, x.code) for x in f] == [("warn", "nugget-unsupported")]
    assert "uses Newton steps" in f[0].message


def test_nugget_support_threshold_is_sixty_percent(corpus):
    # Five content words each; the chunk has bisection/method/halves but not the rest.
    assert _run(corpus, _q(nuggets=["bisection method halves alpha beta"])) == []      # 3/5 = 60%
    assert _codes(_run(corpus, _q(nuggets=["bisection method alpha beta gamma"]))) == [
        "nugget-unsupported"]                                                         # 2/5


def test_support_is_judged_against_the_union_of_all_gold_chunks(corpus):
    q = _q(gold=[{"file": "B/book.pdf", "pages": 5}, {"file": "N/newton.md"}],
           nuggets=["halves the bracketing interval", "converges quadratically"])
    assert _run(corpus, q) == []


def test_nuggets_with_no_content_words_are_not_judged(corpus):
    # Every token is shorter than three characters, so there is nothing lexical to check.
    found = _run(corpus, _q(nuggets=["k1 = 1.2", "42"]))
    assert not [f for f in found if f.code == "nugget-unsupported"]


def test_a_number_no_gold_chunk_states_is_flagged(tmp_path):
    _write(tmp_path / "s_chunks.jsonl", [
        {"doc_id": "s1", "text": "The terms sum to 273, with a discount of 0.20 on orders over 1000. "
                                 "Section 2.1Visualize (OCR glued) reports p > .05.",
         "metadata": {"source_file": "S\\sum.md"}}])
    gold = [{"file": "S/sum.md"}]
    stated = _q(id="par-0002", gold=gold, nuggets=["the terms sum to 273", "discount 0.2 over 1,000",
                                                   "section 2.1 reports p above 0.05"])
    invented = _q(id="par-0003", gold=gold, nuggets=["the terms sum to 92"])
    names = _q(id="par-0004", gold=gold, nuggets=["terms sum by k1 and bge-v2"])   # digits in names
    found = [f for f in validate([stated, invented, names], load_gold_chunks(tmp_path, {"S/sum.md"}))
             if f.code == "nugget-number-unsupported"]
    assert [f.qid for f in found] == ["par-0003"] and "92" in found[0].message


def test_unicode_nuggets_are_judged_not_dropped(tmp_path):
    _write(tmp_path / "g_chunks.jsonl", [
        {"doc_id": "g1", "text": "Η μέθοδος της διχοτόμησης υποδιπλασιάζει το διάστημα",
         "metadata": {"source_file": "G\\gr.md"}}])
    gold = [{"file": "G/gr.md"}]
    supported = _q(gold=gold, nuggets=["μέθοδος διχοτόμησης υποδιπλασιάζει"])
    unsupported = _q(gold=gold, nuggets=["μέθοδος Νεύτωνα"])      # 1 of 2 Greek words is in the chunk
    assert _run(tmp_path, supported) == []
    # An ASCII-only tokeniser would see no words in this nugget and wave it through.
    assert _codes(_run(tmp_path, unsupported)) == ["nugget-unsupported"]


def test_copied_phrasing_is_flagged_except_in_lexical(corpus):
    para = _run(corpus, _q(question=COPY))
    assert [(x.level, x.code) for x in para] == [("warn", "copied-phrasing")]
    assert _run(corpus, _q(id="lex-0001", suite="lexical", question=COPY)) == []


def test_a_four_word_overlap_is_not_copying(corpus):
    assert _run(corpus, _q(question="Explain why the bisection method halves everything each round?")) == []


def test_near_identical_questions_flag_once(corpus):
    a = _q(id="par-0001", question="What does the bisection method guarantee about the root bracket each iteration?")
    b = _q(id="par-0002", question="what does the bisection method guarantee about the root bracket each iteration")
    c = _q(id="par-0003", question="Which stopping rule ends a Newton iteration early when the step is tiny?")
    f = _run(corpus, a, b, c)
    assert [(x.level, x.code, x.qid) for x in f] == [("warn", "near-duplicate", "par-0002")]
    assert "par-0001" in f[0].message


def test_near_duplicate_threshold_is_point_eight(corpus):
    base = "alpha beta gamma delta epsilon"
    at = _run(corpus, _q(id="par-0001", question=base), _q(id="par-0002", question="alpha beta gamma delta"))
    below = _run(corpus, _q(id="par-0001", question=base), _q(id="par-0002", question="alpha beta gamma"))
    assert _codes(at) == ["near-duplicate"]        # 4/5 = 0.8
    assert below == []                             # 3/5


def test_unanswerable_question_has_nothing_to_check_against():
    q = question_from_dict({
        "id": "none-0001", "question": "Which consensus protocol does the vault never mention?",
        "suite": "unanswerable", "tier": "T2", "split": None, "answerable": False, "gold": [],
        "nuggets": [], "expect_course": None,
        "provenance": {"author": "draft", "status": "draft"}}, "t")
    assert validate([q], {}) == []


def test_load_gold_chunks_keeps_only_gold_files(corpus):
    out = load_gold_chunks(corpus, {"B/book.pdf"})
    assert set(out) == {norm_file(BOOK)}
    chunks = out[norm_file(BOOK)]
    assert [c["text"] for c in chunks] == [PDF_TEXT]          # the decoy .jsonl is not a chunk file
    assert chunks[0]["meta"]["page_start"] == 5
    assert load_gold_chunks(corpus, set()) == {}


def test_nothing_wanted_means_nothing_is_streamed(corpus, monkeypatch):
    # A set made only of unanswerable questions names no gold file; the real
    # corpus is hundreds of MB, so it must not be read to find nothing.
    def refuse(path, **kw):
        raise AssertionError(f"streamed {path} for an empty gold set")

    monkeypatch.setattr("eval.bench.validate.iter_jsonl_records", refuse)
    assert load_gold_chunks(corpus, set()) == {}


# Review Focus 1: the same file written two ways must be FOUND; a different file must not.

@pytest.mark.parametrize("spelling", ["B/book.pdf", "b\\book.pdf", "B/BOOK.PDF"])
def test_gold_path_differing_in_separator_or_case_is_found(corpus, spelling):
    assert _run(corpus, _q(gold=[{"file": spelling, "pages": 5}])) == []


def test_composed_and_decomposed_accents_are_the_same_file(tmp_path):
    _write(tmp_path / "c_chunks.jsonl", [
        {"doc_id": "c1", "text": "x " * 5, "metadata": {"source_file": "café\\notes.md"}}])
    assert _run(tmp_path, _q(gold=[{"file": "café/notes.md"}], nuggets=["x"])) == []


def test_gold_path_differing_in_dash_is_gold_file_missing(tmp_path):
    _write(tmp_path / "d_chunks.jsonl", [
        {"doc_id": "d1", "text": PDF_TEXT,
         "metadata": {"source_file": "00 – Courses\\book.pdf", "page_start": 5, "page_end": 5}}])
    en_dash, hyphen = "00 – Courses/book.pdf", "00 - Courses/book.pdf"
    assert _run(tmp_path, _q(gold=[{"file": en_dash, "pages": 5}])) == []
    # A hyphen is not an en dash: that is a different path, and it must say so.
    assert _codes(_run(tmp_path, _q(gold=[{"file": hyphen, "pages": 5}]))) == ["gold-file-missing"]


def test_quota_table_lists_every_suite_against_its_target():
    qs = [_q(id=f"par-{i:04d}", question=f"distinct question number {i}") for i in range(1, 4)]
    table = quota_table(qs)
    assert [row[0] for row in table] == list(SUITES)
    assert ("paraphrase", 3, SUITES["paraphrase"]) in table
    assert ("lexical", 0, SUITES["lexical"]) in table


# --- review cache (Phase 2, Task 11) ---------------------------------------

def _unanswerable():
    return question_from_dict({
        "id": "none-0001", "question": "Which consensus protocol does the vault never mention?",
        "suite": "unanswerable", "tier": "T2", "split": None, "answerable": False, "gold": [],
        "nuggets": [], "expect_course": None,
        "provenance": {"author": "draft", "status": "draft"}}, "t")


def test_review_cache_round_trip(corpus, tmp_path):
    q = _q(nuggets=["uses Newton steps"])               # carries exactly one finding
    chunks = load_gold_chunks(corpus, {"B/book.pdf"})
    findings = validate([q], chunks)
    out = tmp_path / "review_cache.json"
    write_review_cache(out, [q, _unanswerable()], chunks, findings)
    cache = json.loads(out.read_text(encoding="utf-8"))
    assert cache["par-0001"] == {
        "gold_texts": [{"file": "B/book.pdf", "pages": [5, 5], "heading": None, "text": PDF_TEXT}],
        "findings": [{"level": "warn", "code": "nugget-unsupported", "message": findings[0].message}],
    }
    assert cache["none-0001"] == {"gold_texts": [], "findings": []}


def test_review_cache_caps_chunks_and_text_and_keeps_unicode(tmp_path):
    rows = [{"doc_id": f"n{i}", "text": "διχοτόμηση " * 400,
             "metadata": {"source_file": "N\\big.md", "heading_path": "H"}} for i in range(5)]
    _write(tmp_path / "n_chunks.jsonl", rows)
    q = _q(gold=[{"file": "N/big.md"}, {"file": "N/big.md", "heading": "H"}], nuggets=["διχοτόμηση"])
    chunks = load_gold_chunks(tmp_path, {"N/big.md"})
    out = tmp_path / "c.json"
    write_review_cache(out, [q], chunks, [])
    raw = out.read_text(encoding="utf-8")
    texts = json.loads(raw)["par-0001"]["gold_texts"]
    assert len(texts) == 6                                           # 3 chunks for each of 2 gold entries
    assert all(len(t["text"]) == 1500 for t in texts)
    assert texts[0]["heading"] is None and texts[3]["heading"] == "H"
    assert "διχοτόμηση" in raw                                        # written as UTF-8, not \u escapes


def test_review_cache_is_replaced_atomically_from_the_same_directory(tmp_path, monkeypatch):
    out = tmp_path / "review_cache.json"
    out.write_text("OLD", encoding="utf-8")
    seen = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append((Path(src), Path(dst), Path(dst).read_text(encoding="utf-8")))
        real_replace(src, dst)

    monkeypatch.setattr("eval.bench.validate.os.replace", spy)
    write_review_cache(out, [_unanswerable()], {}, [])
    [(src, dst, target_when_called)] = seen
    assert src.parent == dst.parent == tmp_path       # same directory: the rename cannot cross volumes
    assert target_when_called == "OLD"                # the target is untouched until the swap
    assert json.loads(out.read_text(encoding="utf-8")) == {"none-0001": {"gold_texts": [], "findings": []}}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["review_cache.json"]    # no temp file left


def test_review_cache_failure_leaves_the_old_file_and_no_temp(tmp_path, monkeypatch):
    out = tmp_path / "review_cache.json"
    out.write_text("OLD", encoding="utf-8")

    def boom(src, dst):
        raise OSError("disk says no")

    monkeypatch.setattr("eval.bench.validate.os.replace", boom)
    with pytest.raises(OSError, match="disk says no"):
        write_review_cache(out, [_unanswerable()], {}, [])
    assert out.read_text(encoding="utf-8") == "OLD"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["review_cache.json"]


def test_a_missing_alternative_is_reported_as_an_alternative():
    from eval.bench.questions import question_from_dict
    from eval.bench.relevance import norm_file
    q = question_from_dict({
        "id": "book-0001", "question": "How does bisection shrink the bracket?", "suite": "books",
        "tier": "T1", "split": None, "answerable": True,
        "gold": [{"file": "a.md", "alternatives": [{"file": "gone.md"}]}],
        "nuggets": ["bisection halves the interval"], "expect_course": None,
        "provenance": {"author": "draft", "status": "draft"}}, "t")
    gold_chunks = {norm_file("a.md"): [{"meta": {"source_file": "a.md"},
                                        "text": "bisection halves the interval each step"}]}
    found = [(f.code, f.message) for f in validate([q], gold_chunks)]
    assert ("gold-file-missing", "alternative file 'gone.md' is not in the corpus") in found
