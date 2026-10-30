"""Coverage (src/knowledge_map/coverage.py): heading spans and the headings short of nodes."""
import pytest

from src.knowledge_map.coverage import Gap, HeadingSpan, gaps, heading_spans, names
from src.knowledge_map.graph import Node
from src.knowledge_map.locate import PageText
from src.knowledge_map.scout import Heading

PAGES = {
    1: "Ledgers\nA ledger records every transaction in order. It is kept by the branch clerk each day.",
    2: "Controls\nDouble entry means every debit has a credit. The trial balance checks that totals agree.",
}


def node(title, quote, page):
    return Node(f"concept-{title.lower()}", "concept", title, "", "sec-1", "u", 3, [quote],
                located=[(quote, (page,))])


def test_spans_run_from_heading_to_heading():
    text = PageText(PAGES)
    spans, unplaced = heading_spans([Heading("Ledgers", 1), Heading("Controls", 2), Heading("Missing", 2)], text, 10)
    assert [s.title for s in spans] == ["Ledgers", "Controls"] and unplaced == ["Missing"]
    assert spans[0].end == spans[1].start and spans[1].end == len(text.text)
    assert text.text[spans[1].start:].startswith("controls")


def test_short_heading_is_not_checked():
    spans, _ = heading_spans([Heading("Ledgers", 1), Heading("Controls", 2)], PageText(PAGES), 200)
    assert spans == []


def test_gaps_count_nodes_against_length():
    text = PageText(PAGES)
    spans, _ = heading_spans([Heading("Ledgers", 1), Heading("Controls", 2)], text, 10)
    n = [node("Ledger", "A ledger records every transaction in order", 1)]
    short = gaps(spans, n, text, chars_per_node=50)          # each span needs 2 nodes
    assert [(g.span.title, g.have, g.need, g.named) for g in short] == [
        ("Ledgers", ["Ledger"], 2, True), ("Controls", [], 2, False)]
    assert gaps(spans, n, text, chars_per_node=10_000) == [Gap(spans[1], [], 1, False)]   # one node is enough


def test_unnamed_heading_is_short_even_when_long_enough_covered():
    text = PageText(PAGES)
    spans, _ = heading_spans([Heading("Ledgers", 1), Heading("Controls", 2)], text, 10)
    n = [node("Double Entry", "Double entry means every debit has a credit", 2)]
    short = gaps(spans, n, text, chars_per_node=10_000)      # Controls has its one node, not named after it
    assert [(g.span.title, g.named, g.target) for g in short] == [("Ledgers", False, 1), ("Controls", False, 1)]


@pytest.mark.parametrize("heading, title, ok", [
    ("Incremental Extract", "Incremental extraction technique", True),       # stems, not exact words
    ("Extracting from Files", "Extracting data from a flat file", True),     # "files" ~ "file"; "from" is skipped
    ("Whole Table Every Time", "Common decode table", False),               # an example, not the idea
    ("ETL Approaches and Architecture", "ETL Approaches (Staging vs. In-Memory)", False),
    ("Summary", "anything", True),                                           # names no idea
    ("Introduction to ETL", "ETL (Extract, Transform, Load)", True),
])
def test_names(heading, title, ok):
    assert names(heading, title) is ok


def test_gap_line_names_what_is_there():
    s = HeadingSpan("Ledgers", 1, 0, 10)
    assert Gap(s, ["Ledger", "Clerk"], 3, True).line() == "Ledgers (already in the map: Ledger; Clerk)"
    assert Gap(s, [], 1, True).line() == "Ledgers (nothing in the map yet)"
    assert Gap(s, ["Clerk"], 1, False).line() == \
        "Ledgers (already in the map: Clerk; still missing: a node for this heading's own idea)"
    assert Gap(s, ["Clerk"], 1, False).target == 1 and Gap(s, ["A", "B"], 3, True).target == 1
