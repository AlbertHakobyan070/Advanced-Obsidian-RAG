"""knowledge_map/locate.py tests — the quote normaliser and page locator.

Run:  python -m pytest tests/ -q          (project venv, not the the local agent one)

The locator is the trust core of the knowledge map: the LLM returns verbatim
quotes and Python derives the pages. These pin that the match is exact after
normalising (no fuzzy matching), that the section is searched before the
chapter, and that a quote on more than one page fails as ambiguous.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.knowledge_map.locate import (
    Located,
    PageText,
    format_pages,
    locate,
    norm,
    page_ranges,
)

FILLER = "Unrelated filler prose about something else entirely."
SENTENCE = "Replication keeps a copy of the same data on several machines."


def test_norm_ligature_quotes_hyphen():
    assert norm("The ﬁle’s “key-value”  store") == norm("the file's \"keyvalue\" store")
    assert norm("The ﬁle’s “key-value”  store") == "the file's \"keyvalue\" store"


def test_linebreak_hyphen_joins():
    text = PageText({5: "A relational data-\nbase stores rows in tables for later use."})
    got = locate("A relational database stores rows in tables", text, (5, 5), (5, 5))
    assert got == Located("located", (5,))


def test_unicode_linebreak_hyphen_joins():
    # O'Reilly PDFs break words with U+2010 HYPHEN, not ASCII '-' (FSA ch. 19 demo)
    text = PageText({5: "a good general under‐\nstanding of the major aspects of the domain."})
    got = locate("a good general understanding of the major aspects", text, (5, 5), (5, 5))
    assert got == Located("located", (5,))


def test_dash_and_whitespace():
    text = PageText({3: "Before that, A—B  and\n\nC were joined into one."})
    got = locate("A — B and C were joined", text, (3, 3), (3, 3))
    assert got == Located("located", (3,))


def test_wrong_page_offset():
    text = PageText({11: FILLER, 12: SENTENCE, 13: FILLER, 14: FILLER})
    got = locate(SENTENCE, text, section=(13, 14), chapter=(10, 20))
    assert got == Located("located", (12,))


def test_crosses_page_break():
    text = PageText({
        11: FILLER + " The leader writes every change to",
        12: "its local log before replying. " + FILLER,
    })
    got = locate("The leader writes every change to its local log", text, (11, 12), (11, 12))
    assert got == Located("located", (11, 12))


def test_three_page_span_not_located():
    text = PageText({
        1: FILLER + " The leader writes every change",
        2: "to its",
        3: "local log before replying. " + FILLER,
    })
    got = locate("The leader writes every change to its local log", text, (1, 3), (1, 3))
    assert got == Located("not_found", ())


def test_paraphrase_fails():
    text = PageText({11: SENTENCE})
    got = locate("Replication stores the same data on many machines.", text, (11, 11), (10, 20))
    assert got == Located("not_found", ())


def test_ambiguous_across_pages():
    text = PageText({11: SENTENCE, 12: FILLER, 13: SENTENCE})
    got = locate(SENTENCE, text, (11, 13), (10, 20))
    assert got.status == "ambiguous"


def test_same_page_twice_is_located():
    text = PageText({11: SENTENCE + " " + FILLER + " " + SENTENCE})
    got = locate(SENTENCE, text, (11, 11), (10, 20))
    assert got == Located("located", (11,))


def test_section_first():
    text = PageText({11: SENTENCE, 18: SENTENCE})
    got = locate(SENTENCE, text, section=(10, 12), chapter=(10, 20))
    assert got == Located("located", (11,))


def test_bad_length():
    text = PageText({11: SENTENCE})
    assert locate("Replication keeps a", text, (11, 11), (10, 20)).status == "bad_length"
    assert locate(" ".join(["word"] * 31), text, (11, 11), (10, 20)).status == "bad_length"


def test_format_pages():
    assert format_pages(page_ranges([153, 151, 152, 160])) == "151-153, 160"


def test_context_is_normalised_window_around_quote():
    text = PageText({11: FILLER + " " + SENTENCE + " " + FILLER})
    ctx = text.context("keeps a copy of the same data", 11, chars=10)
    assert "keeps a copy of the same data" in ctx
    assert len(ctx) == len("keeps a copy of the same data") + 20
