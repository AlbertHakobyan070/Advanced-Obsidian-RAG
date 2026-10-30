import dataclasses
from pathlib import Path

import fitz
import pytest

from src.knowledge_map.config import KMConfig
from src.knowledge_map.scout import ScoutError, inventory, scout

BODY = "Body text of the page. " * 5   # > scanned_threshold chars


def kcfg(vault: Path, **over) -> KMConfig:
    base = KMConfig(
        staging_dir=vault.parent / "staging", vault_root=vault, min_grounded_ratio=0.9,
        max_span_pages=2, node_cap=40, fanout_max=8, quote_words=(6, 12),
        max_wall_minutes=10, max_llm_calls=50, retries=1, verify_batch=5,
        max_unit_chars=10_000, window_pages=3, max_scanned_ratio=0.3, scanned_threshold=50,
        heading_min_ratio=1.15, min_section_chars=400, chars_per_node=2000,
        rate_limit_sleep_s=0.0, request_timeout_s=60.0, generated_markers=("(generated)",), roles={})
    return dataclasses.replace(base, **over)


def make_pdf(path: Path, pages: list[str], toc=None, title=None, images: dict[int, int] | None = None) -> Path:
    doc = fitz.open()
    for i, text in enumerate(pages, start=1):
        page = doc.new_page()
        if text:
            y = 72
            for line in text.split("\n"):
                page.insert_text((72, y), line)
                y += 14
        for n in range((images or {}).get(i, 0)):
            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8 + n, 8))   # distinct sizes -> distinct images
            pix.clear_with(200)
            page.insert_image(fitz.Rect(100, 400 + 60 * n, 150, 450 + 60 * n), pixmap=pix)
    if toc:
        doc.set_toc(toc)
    if title:
        doc.set_metadata({"title": title})
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    return v


def book(vault):
    toc = [[1, "1. Intro", 1], [2, "Why", 1], [1, "3. Data Models", 3], [2, "Relational", 3], [2, "Graph", 5],
           [1, "4. Storage", 7]]
    return make_pdf(vault / "ddia.pdf", [f"page {i} {BODY}" for i in range(1, 9)], toc=toc)


def test_chapter_range_from_toc(vault):
    s = scout(book(vault), kcfg(vault), chapter="3. Data Models")
    assert (s.page_start, s.page_end) == (3, 6)
    assert sorted(s.pages) == [3, 4, 5, 6]
    assert s.chapter_title == "3. Data Models"
    assert s.toc_used and s.vault_rel == "ddia.pdf" and len(s.sha256) == 64
    assert "page 3" in s.pages[3]


def test_last_chapter_runs_to_last_page(vault):
    assert scout(book(vault), kcfg(vault), chapter="Storage").page_end == 8


def test_chapter_prefix_ignored(vault):
    assert scout(book(vault), kcfg(vault), chapter="Data Models").page_start == 3
    assert scout(book(vault), kcfg(vault), chapter="  chapter 3.  DATA   models ").page_start == 3


def test_chapter_not_found(vault):
    with pytest.raises(ScoutError, match="Nope"):
        scout(book(vault), kcfg(vault), chapter="Nope")


def test_sections_one_level_below(vault):
    s = scout(book(vault), kcfg(vault), chapter="Data Models")
    assert [(x.id, x.title, x.page_start, x.page_end) for x in s.sections] == [
        ("sec-1", "Relational", 3, 4), ("sec-2", "Graph", 5, 6)]


def test_no_toc_windows(vault):
    pdf = make_pdf(vault / "plain.pdf", [f"p{i} {BODY}" for i in range(1, 8)])
    s = scout(pdf, kcfg(vault), pages=(2, 7))
    assert not s.toc_used
    assert [(x.title, x.page_start, x.page_end) for x in s.sections] == [
        ("Pages 2-4", 2, 4), ("Pages 5-7", 5, 7)]
    assert s.chapter_title == "Pages 2-7" and (s.page_start, s.page_end) == (2, 7)
    assert s.book_title == "plain"


def test_no_toc_chapter_errors(vault):
    pdf = make_pdf(vault / "plain.pdf", [BODY] * 3)
    with pytest.raises(ScoutError, match="--pages"):
        scout(pdf, kcfg(vault), chapter="Anything")


def test_unit_split_under_cap(vault):
    pdf = make_pdf(vault / "u.pdf", [f"p{i} {BODY}" for i in range(1, 5)])
    one = len(scout(pdf, kcfg(vault), pages=(1, 1)).pages[1])
    s = scout(pdf, kcfg(vault, window_pages=4, max_unit_chars=2 * one + 5), pages=(1, 4))
    assert [(u.id, u.section_id, u.page_start, u.page_end) for u in s.units] == [
        ("sec-1-u1", "sec-1", 1, 2), ("sec-1-u2", "sec-1", 3, 4)]
    assert all(u.chars == len(u.text) <= 2 * one + 5 for u in s.units)
    # a single page over the cap is its own unit
    s = scout(pdf, kcfg(vault, window_pages=4, max_unit_chars=10), pages=(1, 4))
    assert [(u.page_start, u.page_end) for u in s.units] == [(1, 1), (2, 2), (3, 3), (4, 4)]


def test_captions_and_images(vault):
    pdf = make_pdf(vault / "c.pdf", [f"Figure 2-1. A thing\n{BODY}\nTable 3: Rows\nFig. 4 small\nnot a Figure 9 caption", BODY],
                   images={1: 2})
    s = scout(pdf, kcfg(vault), pages=(1, 2))
    assert s.captions[1] == ["Figure 2-1. A thing", "Table 3: Rows", "Fig. 4 small"]
    assert 2 not in s.captions
    assert s.image_counts == {1: 2, 2: 0}


def test_headings_from_font_size(vault):
    """No outline needed: large lines are headings, a wrapped one is joined, a large line on
    three pages is a running header, and a glyph-only line is not a heading."""
    doc = fitz.open()
    for i, head in enumerate([["Data Models and", "Query Languages"], ["Graph Models"], []], start=1):
        page = doc.new_page()
        page.insert_text((72, 40), "Running Head", fontsize=16)
        y = 80
        for line in head:
            page.insert_text((72, y), line, fontsize=20)
            y += 24
        if i == 2:
            page.insert_text((72, y), "**", fontsize=20)
            y += 24
        for _ in range(6):
            page.insert_text((72, y), BODY[:80])
            y += 14
    doc.save(vault / "h.pdf")
    doc.close()
    s = scout(vault / "h.pdf", kcfg(vault), pages=(1, 3))
    assert [(h.title, h.page) for h in s.headings] == [("Data Models and Query Languages", 1), ("Graph Models", 2)]


def test_scanned_ratio_errors(vault):
    pdf = make_pdf(vault / "s.pdf", [BODY, "", "", BODY])
    assert scout(pdf, kcfg(vault, max_scanned_ratio=0.5), pages=(1, 4)).scanned_pages == [2, 3]
    with pytest.raises(ScoutError, match="OCR"):
        scout(pdf, kcfg(vault, max_scanned_ratio=0.3), pages=(1, 4))


def test_pdf_outside_vault(vault, tmp_path):
    pdf = make_pdf(tmp_path / "elsewhere.pdf", [BODY])
    with pytest.raises(ScoutError, match="parser.vault_path"):
        scout(pdf, kcfg(vault), pages=(1, 1))


def test_relative_pdf_resolves_against_the_vault(vault):
    """I3: a non-absolute path is vault-relative (what the documents API hands an agent)."""
    (vault / "Books").mkdir()
    make_pdf(vault / "Books" / "b.pdf", [BODY])
    s = scout(Path("Books/b.pdf"), kcfg(vault), pages=(1, 1))
    assert s.vault_rel == "Books/b.pdf" and s.pdf_path == vault / "Books" / "b.pdf"


def test_missing_or_non_pdf_file(vault):
    """M2: a ScoutError (CLI: ERROR, exit 2), not a fitz traceback or a map that routes nothing."""
    with pytest.raises(ScoutError, match="PDF not found"):
        scout(vault / "nope.pdf", kcfg(vault), pages=(1, 1))
    with pytest.raises(ScoutError, match="PDF not found"):
        scout(Path("nope.pdf"), kcfg(vault), pages=(1, 1))
    (vault / "notes.txt").write_text(BODY, encoding="utf-8")
    with pytest.raises(ScoutError, match="not a .pdf file"):
        scout(vault / "notes.txt", kcfg(vault), pages=(1, 1))
    upper = make_pdf(vault / "UP.PDF", [BODY])            # the suffix check is case-folded
    assert scout(upper, kcfg(vault), pages=(1, 1)).page_end == 1


def test_pages_and_chapter_exclusive(vault):
    pdf = book(vault)
    with pytest.raises(ScoutError, match="exactly one"):
        scout(pdf, kcfg(vault))
    with pytest.raises(ScoutError, match="exactly one"):
        scout(pdf, kcfg(vault), chapter="Storage", pages=(1, 2))


def test_pages_out_of_range(vault):
    with pytest.raises(ScoutError, match="pages"):
        scout(book(vault), kcfg(vault), pages=(5, 99))


def test_book_title_from_metadata(vault):
    pdf = make_pdf(vault / "m.pdf", [BODY], title="  A   Very\nLong " + "x" * 80)
    t = scout(pdf, kcfg(vault), pages=(1, 1)).book_title
    assert t.startswith("A Very Long x") and len(t) <= 60


def test_inventory_counts(vault):
    book(vault)
    sub = vault / "sub"
    sub.mkdir()
    make_pdf(sub / "scan.pdf", [BODY, "", f"Figure 1. x\n{BODY}"], images={2: 1})
    (vault / "notes.txt").write_text("ignored")
    inv = inventory(vault, kcfg(vault))
    assert [r["file"] for r in inv] == ["ddia.pdf", "sub/scan.pdf"]
    assert inv[0] == {"file": "ddia.pdf", "pages": 8, "text_pages": 8, "scanned_pages": 0,
                      "toc_chapters": 3, "captions": 0, "images": 0}
    assert inv[1] == {"file": "sub/scan.pdf", "pages": 3, "text_pages": 2, "scanned_pages": 1,
                      "toc_chapters": 0, "captions": 1, "images": 1}
