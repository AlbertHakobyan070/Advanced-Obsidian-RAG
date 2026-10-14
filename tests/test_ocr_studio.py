"""OCR Studio backend (src/ocr_studio/ + the /api/ocrstudio/* routes).

The page images are generated here with PyMuPDF, so nothing reads the vault. The
real-Tesseract test runs only where tessdata is installed; the sidecar and VLM
lanes are exercised against scripted responses, because what matters there is
the contract: their output is mapped faithfully and a failure is an error on the
region, never a quiet switch to another engine.
"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from PIL import Image, ImageDraw

from src.ocr_studio import engines, render
from src.ocr_studio.ingest import build_chunks, output_name, write_chunks
from src.ocr_studio.pages import (Store, assemble_page, doc_state, levenshtein, quality,
                                  validate_page, validate_region)
from src.utils.config_loader import Config


def _pdf(path: Path, lines: list[str], scanned: bool = False) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=300)
    y = 50
    for ln in lines:
        page.insert_text((30, y), ln, fontsize=14)
        y += 24
    if scanned:                                   # replace the page by a picture of itself
        pix = page.get_pixmap(dpi=200)
        doc2 = pymupdf.open()
        p2 = doc2.new_page(width=420, height=300)
        p2.insert_image(p2.rect, pixmap=pix)
        doc2.save(path)
        return path
    doc.save(path)
    return path


def _region(**kw):
    base = {"id": "r1", "bbox": [10, 10, 200, 60], "type": "text", "order": 1,
            "engine": "tesseract", "lang": "eng"}
    base.update(kw)
    return base


def _cfg(tmp_path: Path, **extra) -> Config:
    data = {"paths": {"chunks_file": str(tmp_path / "data" / "chunks.jsonl")},
            "pdf": {"figures_dir": str(tmp_path / "figures"), "min_chunk_size": 20}}
    data.update(extra)
    return Config(data, tmp_path)


# ---- render ----------------------------------------------------------------

def test_ops_are_validated_against_one_table():
    assert render.validate_ops([{"op": "contrast", "factor": 1.5}]) == [
        {"op": "contrast", "factor": 1.5}]
    assert render.validate_ops([{"op": "deskew", "angle": None}]) == [
        {"op": "deskew", "angle": None}]
    for bad in ([{"op": "sharpen"}], [{"op": "contrast", "factor": 9}],
                [{"op": "denoise", "size": 4}], [{"op": "threshold", "level": "x"}]):
        with pytest.raises(ValueError):
            render.validate_ops(bad)


def test_detect_skew_finds_a_known_rotation():
    img = Image.new("L", (900, 700), 255)
    d = ImageDraw.Draw(img)
    for y in range(60, 660, 30):
        d.rectangle([60, y, 840, y + 8], fill=0)  # text-like bars
    tilted = img.rotate(-2.0, resample=Image.BICUBIC, fillcolor=255)
    assert abs(render.detect_skew(tilted.convert("RGB")) - 2.0) <= 0.25


def test_heal_paints_over_the_masked_blob_only():
    img = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(img).ellipse([40, 40, 60, 60], fill="black")
    ImageDraw.Draw(img).rectangle([0, 0, 10, 10], fill="black")      # NOT masked
    mask = Image.new("L", (50, 50), 0)                                # half size: resized
    ImageDraw.Draw(mask).ellipse([18, 18, 32, 32], fill=255)
    out = np.asarray(render.heal(img, mask))
    assert out[50, 50].min() > 200                                    # the blob is gone
    assert out[5, 5].max() < 50                                       # the rest is untouched


def test_deskew_bbox_follows_what_the_deskew_op_does_to_the_pixels():
    # a black block on the raw page; where apply_ops' deskew puts its pixels is
    # where a proposal for it must land (1 px = 1 pt here)
    img = Image.new("RGB", (400, 300), "white")
    ImageDraw.Draw(img).rectangle([40, 30, 160, 70], fill="black")
    ops = [{"op": "deskew", "angle": 6.0}]
    ys, xs = np.nonzero(np.asarray(render.apply_ops(img, ops).convert("L")) < 128)
    got = render.deskew_bbox([40, 30, 161, 71], ops, (400, 300))
    want = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    assert all(abs(g - w) <= 2 for g, w in zip(got, want)), (got, want)
    assert render.deskew_bbox([1, 2, 3, 4], [{"op": "contrast", "factor": 2}], (9, 9)) == [1, 2, 3, 4]


def test_crop_pads_clamps_and_reports_its_origin():
    img = Image.new("RGB", (400, 300), "white")
    piece, origin = render.crop(img, 2.0, [5, 5, 50, 40], pad=12)
    assert origin == (0, 0)                       # 10-12 clamps at 0
    assert piece.size == (112, 92)
    with pytest.raises(ValueError):
        render.crop(img, 2.0, [500, 500, 600, 600], pad=0)


def test_render_and_page_size_reject_a_page_out_of_range(tmp_path):
    pdf = _pdf(tmp_path / "a.pdf", ["x"])
    with pytest.raises(ValueError):
        render.page_size(pdf, 0)                  # 0 would silently mean "the last page"
    with pytest.raises(ValueError):
        render.render(pdf, 2, 72)


# ---- engines ---------------------------------------------------------------

def test_unknown_engine_is_an_error_not_a_fallback():
    with pytest.raises(engines.EngineError):
        engines.run(Config({}, Path(".")), "magic", b"")


def test_tesseract_names_the_missing_language(monkeypatch, tmp_path):
    (tmp_path / "eng.traineddata").write_bytes(b"x")
    monkeypatch.setattr(engines, "tessdata_dir", lambda: tmp_path)
    with pytest.raises(engines.EngineError, match="hye"):
        engines.ocr_tesseract(b"", "eng+hye")
    assert engines.tesseract_langs() == ["eng"]


@pytest.mark.skipif(not engines.tesseract_langs(), reason="Tesseract tessdata not installed")
def test_tesseract_reads_a_rendered_crop(tmp_path):
    pdf = _pdf(tmp_path / "scan.pdf", ["Hello OCR Studio 2026", "second line here"], True)
    img, scale = render.render(pdf, 1, 300)
    piece, _ = render.crop(img, scale, [20, 30, 400, 90], 12)
    out = engines.run(Config({}, tmp_path), "tesseract", render.to_png(piece), "eng")
    assert "Hello OCR Studio" in out["text"]
    assert out["lines"] and all(len(ln["bbox"]) == 4 for ln in out["lines"])
    assert out["format"] == "plain" and out["confidence"] is None


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.content = json.dumps(body).encode()
        self.text = self.content.decode()

    def json(self):
        return self._body

    def raise_for_status(self):
        import requests
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)


def _png(w=100, h=40):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_paddle_boxes_and_scores_reach_the_region(monkeypatch, tmp_path):
    sent = {}

    def post(url, json=None, timeout=None):
        sent.update(json)
        return _Resp(200, {"text": "a\nb", "boxes": [
            {"text": "a", "score": 0.9, "box": [0, 0, 10, 10]},
            {"text": "b", "score": 0.7, "box": [0, 20, 10, 30]}]})
    monkeypatch.setattr("requests.post", post)
    cfg = Config({"pdf": {"paddle_ocr": {"base_url": "http://sidecar:8103"}}}, tmp_path)
    out = engines.run(cfg, "paddle", _png(), "ru")
    assert sent["lang"] == "ru" and sent["pipeline"] == "ocr" and sent["return_boxes"] is True
    assert out["confidence"] == 0.8 and [ln["conf"] for ln in out["lines"]] == [0.9, 0.7]


def test_paddle_down_is_an_error_on_the_region(monkeypatch, tmp_path):
    import requests

    def post(*a, **k):
        raise requests.exceptions.ConnectionError("refused")
    monkeypatch.setattr("requests.post", post)
    cfg = Config({"pdf": {"paddle_ocr": {"base_url": "http://sidecar:8103"}}}, tmp_path)
    with pytest.raises(engines.EngineError, match="unreachable"):
        engines.run(cfg, "paddle-vl", _png(), "")


def test_unconfigured_remote_engines_say_so(tmp_path):
    for eng in ("paddle", "vlm"):
        with pytest.raises(engines.EngineError, match="not configured|no VLM"):
            engines.run(Config({}, tmp_path), eng, _png(), "")


def test_status_reports_every_engine_and_why_not(monkeypatch, tmp_path):
    import requests

    def get(url, timeout=None, headers=None):
        if "8103" in url:
            return _Resp(200, {"engine_importable": True, "langs": ["en", "ru"],
                               "pipelines": {"vl": {"available": False, "reason": "2.x image"}}})
        raise requests.exceptions.ConnectionError("refused")
    monkeypatch.setattr("requests.get", get)
    cfg = Config({"pdf": {"paddle_ocr": {"base_url": "http://127.0.0.1:8103"},
                          "vlm_ocr": {"base_url": "http://127.0.0.1:8100/v1"}}}, tmp_path)
    st = engines.status(cfg)
    assert st["paddle"]["available"] and st["paddle"]["langs"] == ["en", "ru"]
    assert not st["paddle-vl"]["available"] and st["paddle-vl"]["error"] == "2.x image"
    assert not st["vlm"]["available"] and "unreachable" in st["vlm"]["error"]
    assert set(st) == set(engines.ENGINES)


def test_layout_proposals_cover_text_and_pictures(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 80), "Chapter 1: Introduction", fontsize=20)
    for i in range(6):
        page.insert_text((72, 120 + 16 * i), f"Body text line number {i} with some words.",
                         fontsize=11)
    page.draw_rect(pymupdf.Rect(72, 300, 300, 450), color=(0, 0, 0), fill=(0.8, 0.8, 0.8))
    doc.save(tmp_path / "l.pdf")
    regions = engines.propose(Config({}, tmp_path), tmp_path / "l.pdf", 1, "layout", False)
    types = {r["type"] for r in regions}
    assert {"text", "figure"} <= types
    assert all(len(r["bbox"]) == 4 for r in regions)


def test_paddle_lines_merge_into_blocks():
    lines = [{"bbox": [0, 0, 100, 10]}, {"bbox": [0, 12, 90, 22]},      # one paragraph
             {"bbox": [0, 60, 100, 70]}]                                 # gap: a new one
    assert engines._merge_lines(lines) == [[0, 0, 100, 22], [0, 60, 100, 70]]


# ---- pages: validation, store, assembly, CER --------------------------------

def test_regions_are_validated():
    assert validate_region(_region())["engine"] == "tesseract"
    for bad in (_region(id="a b"), _region(bbox=[5, 5, 1, 9]), _region(type="poem"),
                _region(engine="ocr-magic"), _region(lang="en;rm")):
        with pytest.raises(ValueError):
            validate_region(bad)
    with pytest.raises(ValueError, match="unique"):
        validate_page({"regions": [_region(), _region()]})


def test_store_roundtrip_stays_outside_the_vault(tmp_path):
    store = Store(tmp_path / "data" / "ocr_studio")
    saved = store.save("Books/a.pdf", 3, {"regions": [_region(text="hi")], "status": "done"})
    assert saved["updated_at"] and store.load("Books/a.pdf", 3)["regions"][0]["text"] == "hi"
    assert store.pages("Books/a.pdf") == {3: store.load("Books/a.pdf", 3)}
    assert store.docs() == ["Books/a.pdf"]
    assert not list(store.doc_dir("Books/a.pdf").glob("*.tmp"))


def test_assembly_follows_reading_order_and_region_types():
    page = {"regions": [
        _region(id="b", order=2, type="formula", text="E = mc^2"),
        _region(id="a", order=1, text="Intro text"),
        _region(id="c", order=3, type="ignore", text="running header"),
        _region(id="d", order=4, type="figure", text="A plot", figure="C:/my figs/p1.png"),
        _region(id="e", order=5, type="table", text="| a | b |\n|---|---|\n| 1 | 2 |"),
        _region(id="f", order=6, type="formula", text="$$x$$")]}
    md = assemble_page(page)
    assert md.index("Intro text") < md.index("$$\nE = mc^2\n$$")
    assert "running header" not in md
    assert "![A plot](<C:/my figs/p1.png>)" in md and "| 1 | 2 |" in md
    # the PDF lane's chunker still reduces it to a caption (the path rides in metadata)
    from src.ingestion.pdf_loader import PDFLoader
    assert "[image: A plot]" in PDFLoader._clean_preserving_code(md)
    assert "$$x$$" in md and "$$\n$$x$$" not in md


def test_cer_compares_raw_engine_output_to_kept_text(tmp_path):
    assert levenshtein("kitten", "sitting") == 3
    store = Store(tmp_path)
    store.save("a.pdf", 1, {"status": "done", "regions": [
        _region(id="r1", text="hello world", raw="hel1o world", engine_used="tesseract"),
        _region(id="r2", text="same", raw="same", engine_used="tesseract")]})
    store.save("a.pdf", 2, {"status": "todo", "regions": [   # not hand-checked: left out
        _region(id="r1", text="x", raw="yyyy", engine_used="tesseract")]})
    rep = quality(store)
    assert rep["pages"] == 1
    (eng,) = rep["engines"]
    assert eng["engine"] == "tesseract" and eng["errors"] == 1 and eng["edited"] == 1
    assert eng["cer"] == round(1 / 15, 4)


def _book(path: Path, n: int) -> Path:
    """An n-page text PDF; page k says 'Page k words'."""
    doc = pymupdf.open()
    for k in range(1, n + 1):
        doc.new_page(width=300, height=200).insert_text((30, 50), f"Page {k} words", fontsize=14)
    doc.save(path)
    return path


def test_modified_means_saved_work_not_a_saved_record(tmp_path):
    store = Store(tmp_path)
    assert doc_state(store, "a.pdf")["modified"] is False
    store.save("a.pdf", 1, {})                        # Save pressed on an untouched page
    assert doc_state(store, "a.pdf")["modified"] is False
    store.save("a.pdf", 2, {"ops": [{"op": "contrast", "factor": 1.5}]})
    assert doc_state(store, "a.pdf")["modified"] and doc_state(store, "a.pdf")["pages_worked"] == 1
    store.save_mask("b.pdf", 3, _png())               # a heal mask alone counts
    assert doc_state(store, "b.pdf")["modified"]
    store.save("c.pdf", 1, {"status": "done"})        # so does a page marked done
    assert doc_state(store, "c.pdf")["modified"]
    store.save("d.pdf", 1, {"regions": [
        _region(id="r1", status="done", text="kept", raw="kpet", engine_used="tesseract"),
        _region(id="r2", status="done", text="same", raw="same", engine_used="tesseract"),
        _region(id="r3", text="drawn, not OCR'd yet")]})
    st = doc_state(store, "d.pdf")
    assert (st["regions"], st["ocr_done"], st["edited"], st["pages_done"]) == (3, 2, 1, 0)


def test_the_page_set_lives_in_the_doc_record_and_ingest_honours_it(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _book(vault / "b.pdf", 3)
    store = Store(tmp_path / "data" / "ocr_studio")
    store.set_page_set("b.pdf", None)                 # "all pages" on a fresh doc writes nothing
    assert not store.doc_dir("b.pdf").exists() and store.page_set("b.pdf") is None
    for n in (1, 3):
        store.save("b.pdf", n, {"status": "done", "regions": [
            _region(text=f"Text that is long enough to make one chunk on its own, page {n}.")]})
    store.set_page_set("b.pdf", [3, 3])
    assert store.page_set("b.pdf") == [3]
    meta = json.loads((store.doc_dir("b.pdf") / "doc.json").read_text(encoding="utf-8"))
    assert meta == {"source": "b.pdf", "pages": [3]}
    store.save("b.pdf", 3, store.load("b.pdf", 3))    # a page save keeps the set
    assert store.page_set("b.pdf") == [3]
    chunks, info = build_chunks(_cfg(tmp_path), vault, "b.pdf", "done", store)
    assert {c.metadata["page_start"] for c in chunks} == {3}     # the REAL page number
    assert info["pages_selected"] == 1
    store.set_page_set("b.pdf", None)
    assert store.page_set("b.pdf") is None
    assert build_chunks(_cfg(tmp_path), vault, "b.pdf", "done", store)[1]["pages_selected"] == 2


# ---- ingest -----------------------------------------------------------------

def test_ingest_builds_pdf_lane_chunks_from_done_pages(tmp_path):
    vault = tmp_path / "vault"
    (vault / "Books").mkdir(parents=True)
    _pdf(vault / "Books" / "scan.pdf", ["x"], scanned=True)
    cfg = _cfg(tmp_path)
    store = Store(tmp_path / "data" / "ocr_studio")
    text = "The corrected text of the scanned page, long enough to be a chunk."
    store.save("Books/scan.pdf", 1, {"status": "done", "regions": [
        _region(text=text), _region(id="f", order=2, type="formula", text="\\frac{a}{b}")]})
    chunks, info = build_chunks(cfg, vault, "Books/scan.pdf", "done")
    assert info["chunks"] == len(chunks) >= 1
    m = chunks[0].metadata
    assert m["page_start"] == m["page_end"] == 1 and m["ocr_source"] == "ocr_studio"
    assert m["source_file"] == str(Path("Books/scan.pdf"))      # the PDF lane's OS form
    assert m.get("has_formula") is True and text in chunks[0].text
    out = write_chunks(chunks, tmp_path / "data" / output_name("Books/scan.pdf"))
    assert out.name.startswith("ocrstudio_scan_") and out.name.endswith("_chunks.jsonl")
    assert len(out.read_text(encoding="utf-8").splitlines()) == len(chunks)


def test_ingest_skips_pages_that_are_not_done_unless_asked(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _pdf(vault / "a.pdf", ["x"])
    cfg = _cfg(tmp_path)
    Store(tmp_path / "data" / "ocr_studio").save("a.pdf", 1, {"regions": [
        _region(text="Text that is long enough to make one chunk on its own.")]})
    assert build_chunks(cfg, vault, "a.pdf", "done")[1]["chunks"] == 0
    assert build_chunks(cfg, vault, "a.pdf", "all")[1]["chunks"] == 1


# ---- the console routes -----------------------------------------------------

@pytest.fixture
def studio(management_module, monkeypatch, tmp_path):
    m = management_module
    vault = tmp_path / "vault"
    (vault / "Books").mkdir(parents=True)
    _pdf(vault / "Books" / "doc.pdf", ["Some words on a page"])
    store = Store(tmp_path / "studio")
    monkeypatch.setattr(m, "_vault_root", lambda: vault)
    monkeypatch.setattr(m, "_studio_store", lambda: store)
    monkeypatch.setattr("src.ocr_studio.pages.figures_root", lambda cfg: tmp_path / "figs")
    monkeypatch.setattr(m, "load_config", lambda *a, **k: Config({}, tmp_path))
    return m, vault, store


def test_studio_refuses_paths_outside_the_vault_and_non_documents(studio, tmp_path):
    m, vault, _ = studio
    (vault / "note.md").write_text("x")
    for path, code in (("../secret.pdf", 400), ("note.md", 400), ("Books/missing.pdf", 404)):
        r = m.ocrstudio_page(path, 1)
        assert r.status_code == code


def test_an_engine_failure_is_the_regions_error_not_a_fallback(studio, monkeypatch):
    m, _, store = studio
    called = []

    def boom(cfg, engine, png, lang):
        called.append(engine)
        raise engines.EngineError("the PaddleOCR sidecar is unreachable")
    monkeypatch.setattr(engines, "run", boom)
    r = m.ocrstudio_ocr(m.StudioOcrIn(path="Books/doc.pdf", page=1,
                                      region=_region(engine="paddle", lang="en")))
    assert called == ["paddle"]
    assert r["region"]["status"] == "error" and "unreachable" in r["region"]["error"]
    assert store.load("Books/doc.pdf", 1)["regions"][0]["status"] == "error"


def test_ocr_maps_line_boxes_back_to_page_points(studio, monkeypatch):
    m, _, store = studio
    monkeypatch.setattr(engines, "run", lambda cfg, e, png, lang: {
        "text": "Some words", "format": "plain", "confidence": 0.9, "ms": 3,
        "lines": [{"text": "Some words", "conf": 0.9, "bbox": [0, 0, 100, 20]}]})
    r = m.ocrstudio_ocr(m.StudioOcrIn(path="Books/doc.pdf", page=1, region=_region()))
    reg = r["region"]
    assert reg["status"] == "done" and reg["raw"] == reg["text"] == "Some words"
    assert reg["engine_used"] == "tesseract" and r["markdown"] == "Some words"
    x0, y0 = reg["lines"][0]["bbox"][:2]
    assert 0 <= x0 <= 10 and 0 <= y0 <= 10       # crop origin (bbox - pad) added back


def test_a_figure_region_is_saved_as_png_outside_the_vault(studio, tmp_path):
    m, vault, _ = studio
    r = m.ocrstudio_ocr(m.StudioOcrIn(path="Books/doc.pdf", page=1,
                                      region=_region(type="figure", engine="")))
    fig = Path(r["region"]["figure"])
    assert fig.is_file() and tmp_path / "figs" in fig.parents
    assert vault not in fig.parents


def test_saving_a_page_resolves_auto_deskew(studio):
    m, _, _ = studio
    r = m.ocrstudio_page_save(m.StudioPageIn(path="Books/doc.pdf", page=1, data={
        "regions": [], "ops": [{"op": "deskew", "angle": None}]}))
    assert isinstance(r["page"]["ops"][0]["angle"], float)


def test_append_over_an_indexed_document_is_refused(studio, monkeypatch):
    m, _, store = studio
    store.save("Books/doc.pdf", 1, {"status": "done", "regions": [
        _region(text="Corrected text that is long enough to become a chunk here.")]})
    monkeypatch.setattr(m, "_indexed_as", lambda rel: {
        "Books\\doc.pdf": {"chunks": 4, "jsonls": ["pdf_chunks.jsonl"]}})
    monkeypatch.setattr(m, "CFG", Config({"pdf": {"min_chunk_size": 20},
                                          "paths": {"chunks_file": "data/chunks.jsonl"}},
                                         Path(".")))
    queued = []
    monkeypatch.setattr(m, "enqueue", lambda kind, prm: queued.append((kind, prm)))
    r = m.ocrstudio_ingest(m.StudioIngestIn(path="Books/doc.pdf", mode="append"))
    assert r.status_code == 409 and "replace" in json.loads(r.body)["error"]
    assert queued == []


def test_replace_removes_the_old_chunks_from_both_indexes_first(studio, monkeypatch):
    m, _, store = studio
    store.save("Books/doc.pdf", 1, {"status": "done", "regions": [
        _region(text="Corrected text that is long enough to become a chunk here.")]})
    monkeypatch.setattr(m, "_indexed_as", lambda rel: {
        "Books\\doc.pdf": {"chunks": 4, "jsonls": ["pdf_chunks.jsonl"]}})
    monkeypatch.setattr(m, "CFG", Config({"pdf": {"min_chunk_size": 20},
                                          "paths": {"chunks_file": "data/chunks.jsonl"}},
                                         Path(".")))
    calls = []
    monkeypatch.setattr(m, "_chroma_delete_by_source", lambda s: calls.append(("chroma", s)) or 4)
    monkeypatch.setattr(m, "_jsonl_remove_sources",
                        lambda s, names: calls.append(("jsonl", s, names)) or {"pdf_chunks.jsonl": 4})
    monkeypatch.setattr(m, "build_manifest", lambda force=False: {})

    class J:
        def __init__(self, kind, prm):
            self.kind, self.prm = kind, prm
            calls.append(("job", kind, prm))

        def public(self):
            return {"kind": self.kind}
    monkeypatch.setattr(m, "enqueue", J)
    r = m.ocrstudio_ingest(m.StudioIngestIn(path="Books/doc.pdf", mode="replace"))
    assert r["ok"] and [c[0] for c in calls] == ["chroma", "jsonl", "job", "job"]
    assert calls[1][2] == {"pdf_chunks.jsonl"}
    assert calls[2][1] == "ocrstudio_ingest" and calls[3][1] == "index_append"
    assert calls[2][2]["output"] == calls[3][2]["file"] == r["output"]
    assert Path(r["output"]).name.startswith("ocrstudio_")


def test_the_job_only_ever_writes_its_own_chunk_file(management_module):
    m = management_module
    argv = m._build_argv("ocrstudio_ingest", {"doc": "Books/a.pdf",
                                              "output": "data/ocrstudio_a_1234_chunks.jsonl"})
    assert argv[2] == "ocr-studio-ingest" and argv[-1] == "done"
    for out in ("data/pdf_chunks.jsonl", "data/chunks.jsonl", "data/ocrstudio_a.jsonl"):
        with pytest.raises(ValueError):
            m._build_argv("ocrstudio_ingest", {"doc": "Books/a.pdf", "output": out})
    with pytest.raises(ValueError):
        m._build_argv("ocrstudio_ingest", {"doc": "../x.pdf",
                                           "output": "data/ocrstudio_a_1_chunks.jsonl"})


def test_schema_lists_the_studio_routes_and_job(management_module):
    schema = management_module.api_schema()
    eps = schema["endpoints"]
    for route in ("GET /api/ocrstudio/engines", "POST /api/ocrstudio/ocr",
                  "POST /api/ocrstudio/ingest", "GET /api/ocrstudio/quality"):
        assert route in eps
    assert "REMOVES" in eps["POST /api/ocrstudio/ingest"]["purpose"]
    assert schema["job_kinds"]["ocrstudio_ingest"]["permission"] == "mutating"
    assert eps["POST /api/ocrstudio/upload"]["permission"] == "mutating"
    assert eps["POST /api/ocrstudio/pageset"]["permission"] == "mutating"
    assert eps["POST /api/ocrstudio/export"]["permission"] == "read"


# ---- import, working page set, export (through the HTTP layer) -------------

@pytest.fixture
def client(studio, monkeypatch, tmp_path):
    """The routes over HTTP. CFG is pinned because manage_api is imported once
    per run: an earlier test may have loaded the machine's config.yaml, whose
    upload_dir / inbox_dir are not the shipped ones. _indexed_as is stubbed so
    the doc route never reads a real corpus's manifest."""
    from fastapi.testclient import TestClient
    m, vault, store = studio
    monkeypatch.setattr(m, "CFG", Config({
        "ocr_studio": {"upload_dir": "OCR Studio", "upload_max_mb": 500},
        "webui": {"inbox_dir": "Inbox"}}, tmp_path))
    monkeypatch.setattr(m, "_indexed_as", lambda rel: {})
    return TestClient(m.app), m, vault, store


def _upload(c, name, data, pages=None):
    return c.post("/api/ocrstudio/upload", files={"file": (name, data, "application/octet-stream")},
                  data={} if pages is None else {"pages": pages})


def test_upload_keeps_the_whole_file_and_stores_the_typed_page_set(client, tmp_path):
    c, m, vault, store = client
    src = _book(tmp_path / "book.pdf", 5).read_bytes()
    r = _upload(c, "book.pdf", src, " 2-3, 5")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["path"] == "OCR Studio/book.pdf" and j["page_set"] == "2-3,5" and j["page_count"] == 5
    assert (vault / "OCR Studio" / "book.pdf").read_bytes() == src      # the whole file, as sent
    assert store.page_set("OCR Studio/book.pdf") == [2, 3, 5]
    # never overwrites; no range (or every page) = all pages, stored as no set
    j2 = _upload(c, "book.pdf", src).json()
    assert j2["path"] == "OCR Studio/book (2).pdf" and j2["page_set"] is None and j2["renamed"]
    assert _upload(c, "x.pdf", src, "1-5").json()["page_set"] is None
    # a name with studio state left over from a deleted file is not reused either
    store.save("OCR Studio/old.pdf", 1, {"status": "done"})
    assert _upload(c, "old.pdf", src).json()["path"] == "OCR Studio/old (2).pdf"
    # images import too; their one page is all there is
    assert _upload(c, "shot.png", _png()).json()["path"] == "OCR Studio/shot.png"
    assert not [f for f in (vault / "OCR Studio").iterdir() if f.name.startswith(".")]


def test_upload_refusals_are_explicit_and_leave_nothing_behind(client, tmp_path, monkeypatch):
    c, m, vault, _ = client
    pdf = _book(tmp_path / "b.pdf", 3).read_bytes()
    up = vault / "OCR Studio"
    for name, data, pages, words in (
            ("notes.md", b"# hi", None, "PDFs and images"),
            ("b.pdf", pdf, "2-x", "bad page range"),
            ("b.pdf", pdf, "2,7", "past the end"),
            ("shot.png", _png(), "2", "past the end"),
            ("b.pdf", b"not a pdf at all", None, "not a readable")):
        r = _upload(c, name, data, pages)
        assert r.status_code == 400 and words in r.json()["error"], (name, pages, r.text)
        assert not up.exists(), name                 # a failed first upload creates nothing
    opt = m._studio_opt
    monkeypatch.setattr(m, "_studio_opt", lambda k, d: 0.0001 if k == "upload_max_mb" else opt(k, d))
    r = _upload(c, "b.pdf", pdf)
    assert r.status_code == 413 and "upload_max_mb" in r.json()["error"]
    assert not up.exists()


def test_an_upload_dir_inside_the_inbox_is_refused(client, tmp_path, monkeypatch):
    c, m, vault, _ = client
    opt = m._studio_opt
    monkeypatch.setattr(m, "_studio_opt", lambda k, d: "Inbox/OCR" if k == "upload_dir" else opt(k, d))
    r = _upload(c, "b.pdf", _book(tmp_path / "b.pdf", 1).read_bytes())
    assert r.status_code == 400 and "inbox" in r.json()["error"]
    assert not (vault / "Inbox").exists()
    assert "inbox" in c.get("/api/ocrstudio/files").json()["uploads_error"]


def test_files_lists_imports_with_their_state_and_page_set(client, tmp_path):
    c, m, vault, store = client
    j = c.get("/api/ocrstudio/files").json()
    assert j["uploads"] == [] and j["upload_dir"] == "OCR Studio"
    assert not (vault / "OCR Studio").exists()       # listing never creates the folder
    a = _upload(c, "a.pdf", _book(tmp_path / "a.pdf", 4).read_bytes(), "3-4").json()["path"]
    _upload(c, "b.pdf", _book(tmp_path / "b.pdf", 2).read_bytes())
    store.save(a, 3, {"regions": [_region()]})
    store.save("Books/doc.pdf", 1, {"status": "done"})
    j = c.get("/api/ocrstudio/files").json()
    rows = {r["name"]: r for r in j["uploads"]}
    assert rows["a.pdf"]["state"]["modified"] and rows["a.pdf"]["page_set"] == "3-4"
    assert not rows["b.pdf"]["state"]["modified"] and rows["b.pdf"]["page_set"] is None
    # "worked on" carries state too, and leaves out what "imported" already shows
    assert [(r["path"], r["state"]["modified"]) for r in j["studied"]] == [("Books/doc.pdf", True)]


def test_the_page_set_is_editable_and_guards_the_page_routes(client, tmp_path):
    c, m, vault, store = client
    path = _upload(c, "a.pdf", _book(tmp_path / "a.pdf", 4).read_bytes(), "2-3").json()["path"]
    d = c.get("/api/ocrstudio/doc", params={"path": path}).json()
    assert d["page_set"] == "2-3" and [p["page"] for p in d["pages"] if p["in_set"]] == [2, 3]
    # saves, masks and OCR outside the set are refused, naming the set
    for route, body in (("page", {"page": 4, "data": {"regions": [_region()]}}),
                        ("mask", {"page": 4, "png_b64": None}),
                        ("ocr", {"page": 1, "region": _region()})):
        r = c.post(f"/api/ocrstudio/{route}", json={"path": path, **body})
        assert r.status_code == 400 and "outside the working pages 2-3" in r.json()["error"], route
    r = c.post("/api/ocrstudio/page", json={"path": path, "page": 3, "data": {"regions": [_region()]}})
    assert r.status_code == 200
    assert c.get("/api/ocrstudio/doc", params={"path": path}).json()["pages"][2]["worked"]
    # narrowing the set past saved work is refused; widening and clearing are not
    r = c.post("/api/ocrstudio/pageset", json={"path": path, "pages": "2"})
    assert r.status_code == 409 and "3" in r.json()["error"] and store.page_set(path) == [2, 3]
    assert c.post("/api/ocrstudio/pageset", json={"path": path, "pages": "1-3"}).json()["page_set"] == "1-3"
    assert c.post("/api/ocrstudio/pageset", json={"path": path, "pages": ""}).json()["page_set"] is None
    assert store.page_set(path) is None
    for bad in ("9", "x"):
        assert c.post("/api/ocrstudio/pageset", json={"path": path, "pages": bad}).status_code == 400
    # any vault document can carry a set, not only imports
    assert c.post("/api/ocrstudio/pageset",
                  json={"path": "Books/doc.pdf", "pages": "1"}).json()["page_set"] is None


def test_export_zips_markdown_with_real_page_numbers_and_figures(client, tmp_path):
    c, m, vault, store = client
    path = _upload(c, "book.pdf", _book(tmp_path / "book.pdf", 5).read_bytes(), "4-5").json()["path"]
    fig = tmp_path / "figs" / "k" / "p0005_rf.png"
    fig.parent.mkdir(parents=True)
    fig.write_bytes(_png())
    store.save(path, 1, {"status": "done", "regions": [_region(text="First page text")]})
    store.save(path, 4, {"status": "done", "regions": [_region(text="Fourth page text")]})
    store.save(path, 5, {"regions": [_region(text="Fifth page text"),
                                     _region(id="f", order=2, type="figure", text="A plot",
                                             figure=str(fig))]})
    r = c.post("/api/ocrstudio/export", json={"paths": [path], "pages": "done"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert "attachment" in r.headers["content-disposition"]
    md = zipfile.ZipFile(io.BytesIO(r.content)).read("book.md").decode("utf-8")
    assert "<!-- page 4 -->" in md and "Fourth page text" in md
    assert "Fifth" not in md and "First" not in md              # not done / outside the set
    z = zipfile.ZipFile(io.BytesIO(c.post("/api/ocrstudio/export",
                                          json={"paths": [path], "pages": "all"}).content))
    md = z.read("book.md").decode("utf-8")
    assert md.index("<!-- page 4 -->") < md.index("<!-- page 5 -->") and "First" not in md
    assert "![A plot](<book_figures/p0005_rf.png>)" in md
    assert z.read("book_figures/p0005_rf.png") == fig.read_bytes()
    assert md.startswith(f"<!-- source: {path} · pages: all · working pages: 4-5 -->")


def test_export_refuses_what_it_cannot_export_by_name(client, tmp_path):
    c, m, vault, store = client
    a = _upload(c, "a.pdf", _book(tmp_path / "a.pdf", 1).read_bytes()).json()["path"]
    b = _upload(c, "b.pdf", _book(tmp_path / "b.pdf", 1).read_bytes()).json()["path"]
    store.save(a, 1, {"status": "done", "regions": [_region(text="words")]})
    r = c.post("/api/ocrstudio/export", json={"paths": [a, b]})
    assert r.status_code == 400 and "b.pdf" in r.json()["error"] and "a.pdf" not in r.json()["error"]
    store.save(a, 1, {"status": "done", "regions": [
        _region(type="figure", text="x", figure=str(tmp_path / "gone.png"))]})
    r = c.post("/api/ocrstudio/export", json={"paths": [a]})
    assert r.status_code == 400 and "missing" in r.json()["error"] and "page 1" in r.json()["error"]
    for body in ({"paths": []}, {"paths": [a], "pages": "some"}, {"paths": ["../x.pdf"]}):
        assert c.post("/api/ocrstudio/export", json=body).status_code == 400, body
    assert c.post("/api/ocrstudio/export", json={"paths": ["Books/none.pdf"]}).status_code == 404
