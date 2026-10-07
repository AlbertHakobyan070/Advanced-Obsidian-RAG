import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.bench.questions import GoldSource
from eval.bench.relevance import chunk_matches, doc_matches, judge, norm_file

BOOK = "00 – Courses\\Deep Learning\\Textbooks\\Some Book.pdf"


def test_norm_file_unifies_separators_case_and_dashes_survive():
    assert norm_file(BOOK) == norm_file("00 – Courses/deep learning/Textbooks/Some Book.pdf")
    assert "–" in norm_file(BOOK)                     # the en dash is content, not a separator
    assert norm_file("/a/b/") == "a/b"


def test_page_overlap_rules():
    g = GoldSource(file=BOOK, pages=(212, 213))
    m = lambda ps, pe: {"source_file": BOOK, "page_start": ps, "page_end": pe}
    assert chunk_matches(m(213, 214), g)
    assert chunk_matches(m(200, 212), g)
    assert not chunk_matches(m(214, 220), g)
    assert not chunk_matches({"source_file": BOOK}, g)       # pageless chunk vs paged gold


def test_heading_prefix_is_component_wise():
    g = GoldSource(file="n.md", heading="Ch 4 > Bisection")
    hp = lambda h: {"source_file": "n.md", "heading_path": h}
    assert chunk_matches(hp("Ch 4 > Bisection > Stopping rule"), g)
    assert chunk_matches(hp("ch 4 >  bisection"), g)
    assert not chunk_matches(hp("Ch 4 > Bisection method"), g)   # not a prefix match on words
    assert not chunk_matches(hp("Ch 4"), g)


def test_file_only_gold_and_doc_level():
    g = GoldSource(file="a.md")
    assert chunk_matches({"source_file": "A.md", "heading_path": "x"}, g)
    assert doc_matches({"source_file": "a.md", "page_start": 1}, GoldSource(file="a.md", pages=(9, 9)))
    assert not doc_matches({"source_file": "b.md"}, g)


def test_judge_records_which_gold_each_rank_satisfies():
    gold = [GoldSource(file="a.md"), GoldSource(file="b.pdf", pages=(3, 3))]
    metas = [{"source_file": "x.md"},
             {"source_file": "b.pdf", "page_start": 3, "page_end": 3},
             {"source_file": "a.md"},
             {"source_file": "b.pdf", "page_start": 9, "page_end": 9}]
    j = judge(metas, gold)
    assert j.matches == [frozenset(), frozenset({1}), frozenset({0}), frozenset()]
    assert j.doc_matches == [frozenset(), frozenset({1}), frozenset({0}), frozenset({1})]


def test_a_twin_file_satisfies_the_gold_entry_with_its_own_locator():
    g = GoldSource(file="deck.pdf", pages=(3, 3),
                   alternatives=(GoldSource(file="deck.md", heading="Slide 3"),))
    assert chunk_matches({"source_file": "deck.md", "heading_path": "Slide 3 > Notes"}, g)
    assert not chunk_matches({"source_file": "deck.md", "heading_path": "Slide 4"}, g)
    assert doc_matches({"source_file": "deck.md"}, g)
    j = judge([{"source_file": "x.md"}, {"source_file": "deck.md", "heading_path": "Slide 3"}], [g])
    assert j.matches == [frozenset(), frozenset({0})]
