"""Chunking-strategy regression tests (session 11).

Run:  python -m pytest tests/ -q          (project venv)

Guards the two strategies (heading | fixed) and the session-11 fix for the
silent-text-loss bug: a single "sentence" longer than max_chunk used to be
truncated to sent[:max_size] with the remainder DROPPED.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.ingestion.obsidian_parser import (
    CHUNKING_STRATEGIES,
    document_element_chunks,
    fixed_window_chunks,
    split_large_chunk,
    split_section,
)

PARA = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 8).strip()
STRUCTURED = "\n\n".join(PARA for _ in range(12))          # ~5.6 KB, real paragraphs
WALL = ("x" * 120 + " ") * 80                              # one 9.6 KB "sentence"
NO_WS = "y" * 9000                                          # no whitespace at all


def probes(text, width=180, step=1500):
    """Evenly spaced substrings — all must survive into some chunk."""
    return [text[i:i + width] for i in range(0, max(len(text) - width, 1), step)
            if text[i:i + width].strip()]


def assert_full_coverage(text, chunks):
    for p in probes(text):
        assert any(p in c for c in chunks), f"lost content: {p[:60]!r}…"


# ---- fixed_window_chunks ----

def test_fixed_empty_and_small():
    assert fixed_window_chunks("", 3000, 150) == []
    assert fixed_window_chunks("   ", 3000, 150) == []
    assert fixed_window_chunks(PARA, 3000, 150) == [PARA]


def test_fixed_exact_boundary():
    text = "a" * 3000
    assert fixed_window_chunks(text, 3000, 150) == [text]


def test_fixed_overlap_clamped():
    # overlap >= max_size must not stall the window
    chunks = fixed_window_chunks(NO_WS, 1000, 5000)
    assert sum(len(c) for c in chunks) >= len(NO_WS)
    assert len(chunks) < 100          # sane count, not one-char steps


def test_fixed_forward_progress_no_whitespace():
    chunks = fixed_window_chunks(NO_WS, 3000, 150)
    assert len(chunks) == 4
    assert "".join(chunks) .startswith("y" * 3000)
    assert_full_coverage(NO_WS, chunks)


def test_fixed_whitespace_snapped():
    chunks = fixed_window_chunks(STRUCTURED, 3000, 150)
    # cuts land on whitespace, so no chunk starts/ends mid-word
    for c in chunks:
        assert not c[0].isspace() and not c[-1].isspace()
    assert_full_coverage(STRUCTURED, chunks)


def test_fixed_unicode_safe():
    text = ("данные и модели 📊 " * 400).strip()      # cyrillic + emoji, ~8 KB
    chunks = fixed_window_chunks(text, 3000, 150)
    assert_full_coverage(text, chunks)


# ---- split_large_chunk (heading strategy) ----

def test_heading_small_passthrough():
    assert split_large_chunk(PARA, 3000, 150) == [PARA]


def test_heading_paragraph_packing():
    chunks = split_large_chunk(STRUCTURED, 3000, 150)
    assert all(len(c) <= 3000 + 200 for c in chunks)   # overlap joins allowed
    assert_full_coverage(STRUCTURED, chunks)


def test_heading_giant_sentence_no_data_loss():
    """Regression: WALL used to come back as ONE 3000-char chunk (6.6 KB lost)."""
    chunks = split_large_chunk(WALL, 3000, 150)
    assert len(chunks) > 1
    assert_full_coverage(WALL, chunks)


def test_heading_no_ws_no_data_loss():
    chunks = split_large_chunk(NO_WS, 3000, 150)
    assert sum(len(c) for c in chunks) >= len(NO_WS)


# ---- split_section dispatcher ----

def test_dispatch_strategies():
    for strat in CHUNKING_STRATEGIES:      # registry is asserted further down
        assert split_section(PARA, 3000, 150, strat) == [PARA]


def test_dispatch_case_and_none():
    assert split_section(PARA, 3000, 150, "FIXED") == [PARA]
    assert split_section(PARA, 3000, 150, None) == [PARA]   # None -> heading


def test_dispatch_unknown_raises():
    with pytest.raises(ValueError):
        split_section(PARA, 3000, 150, "semantic")


def test_small_sections_identical_across_strategies():
    """The scoping contract: sections under max_size chunk identically, so
    doc_ids only change for oversized sections when the strategy changes."""
    for text in (PARA, "one line", "a\n\nb\n\nc"):
        assert (split_section(text, 3000, 150, "heading")
                == split_section(text, 3000, 150, "fixed"))


# ---- document_element_chunks (session 14: element-aware mode) ----

CODE_FENCE = "```python\n" + "\n".join(f"def f{i}(): return {i}" for i in range(40)) + "\n```"
TABLE = "| col1 | col2 |\n|------|------|\n" + "\n".join(
    f"| a{i} | b{i} |" for i in range(30))
LISTY = "\n".join(f"- item {i} with some words" for i in range(25))
DOC = "\n\n".join([PARA, CODE_FENCE, PARA, TABLE, LISTY, PARA, PARA, PARA])


def test_document_mode_small_passthrough():
    assert document_element_chunks(PARA, 3000, 150) == [PARA]
    assert document_element_chunks("", 3000, 150) == []


def test_document_mode_never_cuts_elements():
    chunks = document_element_chunks(DOC, 1400, 150)
    assert len(chunks) > 1
    # the code fence and the table each live WHOLE inside exactly one chunk
    assert sum(1 for c in chunks if CODE_FENCE in c) == 1
    assert sum(1 for c in chunks if TABLE in c) == 1
    assert sum(1 for c in chunks if LISTY in c) == 1


def test_document_mode_full_coverage():
    assert_full_coverage(DOC, document_element_chunks(DOC, 1400, 150))


def test_document_mode_oversized_element_falls_back():
    # a single element bigger than max_size still gets split (fixed windows)
    chunks = document_element_chunks(WALL, 3000, 150)
    assert len(chunks) > 1
    assert all(len(c) <= 3000 for c in chunks)


def test_none_mode_keeps_section_whole():
    assert split_section(STRUCTURED, 3000, 150, "none") == [STRUCTURED]
    assert split_section("  ", 3000, 150, "none") == []


def test_strategy_registry_and_dispatch():
    assert set(CHUNKING_STRATEGIES) == {"heading", "fixed", "document", "none"}
    with pytest.raises(ValueError):
        split_section(PARA, 3000, 150, "semantic")
    assert split_section(DOC, 1400, 150, "document") == \
        document_element_chunks(DOC, 1400, 150)
