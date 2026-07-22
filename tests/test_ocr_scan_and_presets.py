"""OCR page-scanner + VLM preset registry tests (session 16).

Run:  python -m pytest tests/ -q

The scanner's job is to save a human from scrolling a 700-page book, so the
thing worth guarding is that its page numbers are 1-BASED (what a viewer shows
and what --pages expects) and that its range compaction is exact — an
off-by-one here wastes a whole OCR pass on the wrong pages.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.ingestion.ocr_scan import _ranges, scan_pdf
from src.ingestion.ocr_vlm import VLMOCR
from src.utils.config_loader import Config

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------ range packing

@pytest.mark.parametrize("pages,expected", [
    ([], ""),
    ([5], "5"),
    ([1, 2, 3], "1-3"),
    ([1, 2, 3, 7, 9, 10], "1-3,7,9-10"),
    ([2, 4, 6], "2,4,6"),
    ([10, 11, 12, 14], "10-12,14"),
])
def test_range_compaction(pages, expected):
    assert _ranges(pages) == expected


# ------------------------------------------------------------ pdf scanning

def _make_pdf(path: Path, page_texts: list[str]):
    """Build a tiny real PDF; empty string -> a page with no text (a 'scan').

    Text is laid out line by line: a single long insert_text() runs off the
    page edge and the overflow is never extractable, which silently produces
    far less text than the input suggests.
    """
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for txt in page_texts:
        page = doc.new_page()
        if txt:
            y = 72
            for line in txt.split("\n"):
                page.insert_text((72, y), line[:80], fontsize=11)
                y += 16
    doc.save(str(path))
    doc.close()


def _lines(sentence: str, n: int) -> str:
    return "\n".join(sentence for _ in range(n))


def test_scan_flags_textless_pages_one_based(tmp_path):
    pdf = tmp_path / "mixed.pdf"
    body = _lines("Gradient clipping rescales the gradient norm.", 8)
    # pages 1,2 = text; page 3 = empty (scan); page 4 = text; page 5 = empty
    _make_pdf(pdf, [body, body, "", body, ""])
    rep = scan_pdf(pdf, threshold=50)

    assert rep["page_count"] == 5
    assert rep["needs_ocr"] == [3, 5]          # 1-BASED, not [2, 4]
    assert rep["needs_ocr_spec"] == "3,5"
    assert rep["kind"] == "mixed"
    assert rep["n_needs_ocr"] == 2
    assert rep["pct_needs_ocr"] == 40.0


def test_born_digital_pdf_needs_nothing(tmp_path):
    pdf = tmp_path / "digital.pdf"
    body = _lines("The bias-variance tradeoff splits generalization error.", 8)
    _make_pdf(pdf, [body, body, body])
    rep = scan_pdf(pdf, threshold=50)
    assert rep["needs_ocr"] == []
    assert rep["needs_ocr_spec"] == ""
    assert rep["kind"] == "born_digital"


def test_fully_scanned_pdf_is_labelled_as_such(tmp_path):
    pdf = tmp_path / "photographed.pdf"
    _make_pdf(pdf, ["", "", ""])
    rep = scan_pdf(pdf, threshold=50)
    assert rep["needs_ocr"] == [1, 2, 3]
    assert rep["kind"] == "fully_scanned"
    assert rep["pct_needs_ocr"] == 100.0


def test_scan_reports_a_sample_for_eyeballing(tmp_path):
    pdf = tmp_path / "s.pdf"
    _make_pdf(pdf, [_lines("Chapter 7 Numerical Methods and Root Finding", 8)])
    rep = scan_pdf(pdf, threshold=50)
    assert "Numerical Methods" in rep["pages"][0]["sample"]
    assert rep["pages"][0]["verdict"] == "ok"


def test_sparse_pages_are_flagged_separately_from_scans(tmp_path):
    """A caption-only plate page is neither 'ok' nor worth an OCR pass.

    Between `threshold` and `threshold * SPARSE_MULTIPLIER` the page has SOME
    text — usually a figure caption or chapter divider. Calling those
    needs_ocr would send a book's every plate page through the VLM for almost
    no gain, so they get their own verdict to eyeball instead.
    """
    pdf = tmp_path / "plate.pdf"
    _make_pdf(pdf, ["Figure 4.2 Convergence of the bisection method."])  # ~47
    rep = scan_pdf(pdf, threshold=20)          # 20 < 47 < 80 -> sparse
    assert rep["pages"][0]["verdict"] == "sparse"
    assert rep["needs_ocr"] == []
    assert rep["sparse"] == [1]


# ------------------------------------------------------- vlm preset registry

PRESETS = {
    "deepseek": {"model": "DeepSeek-OCR", "prompt": "<__media__>\nmd", "dpi": 150},
    "paddleocr_vl": {"model": "PaddleOCR-VL", "prompt": "md", "dpi": 200},
}


def cfg_of(data):
    return Config(data, ROOT)


def test_preset_supplies_model_prompt_and_dpi():
    v = VLMOCR.from_config(cfg_of({"pdf": {
        "vlm_ocr_presets": PRESETS,
        "vlm_ocr": {"preset": "paddleocr_vl", "base_url": "http://h/v1"},
    }}))
    assert v.model == "PaddleOCR-VL"
    assert v.dpi == 200
    assert v.prompt == "md"


def test_explicit_keys_override_the_preset():
    """An existing config must keep behaving exactly as before."""
    v = VLMOCR.from_config(cfg_of({"pdf": {
        "vlm_ocr_presets": PRESETS,
        "vlm_ocr": {"preset": "paddleocr_vl", "model": "MyFineTune", "dpi": 111},
    }}))
    assert v.model == "MyFineTune"
    assert v.dpi == 111
    assert v.prompt == "md"          # not overridden -> still from the preset


def test_no_preset_keeps_the_pre_registry_behaviour():
    v = VLMOCR.from_config(cfg_of({"pdf": {"vlm_ocr": {
        "model": "DeepSeek-OCR", "dpi": 150, "prompt": "<__media__>\nx"}}}))
    assert (v.model, v.dpi) == ("DeepSeek-OCR", 150)


def test_unknown_preset_names_the_known_ones():
    with pytest.raises(ValueError, match="not in pdf.vlm_ocr_presets"):
        VLMOCR.from_config(cfg_of({"pdf": {
            "vlm_ocr_presets": PRESETS, "vlm_ocr": {"preset": "typo"}}}))


def test_shipped_config_presets_are_well_formed():
    from conftest import shipped_config
    presets = shipped_config().get("pdf.vlm_ocr_presets", {}) or {}
    assert presets, "the shipped config lost its vlm_ocr_presets block"
    for name, spec in presets.items():
        assert spec.get("model"), f"{name} has no model"
        assert spec.get("prompt"), f"{name} has no prompt"
        assert isinstance(spec.get("dpi"), int), f"{name} has no integer dpi"
