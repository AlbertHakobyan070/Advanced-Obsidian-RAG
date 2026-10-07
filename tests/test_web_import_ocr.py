"""convert-files --ocr-pages: OCR goes through PyMuPDF's Tesseract hook.

The first implementation imported `pytesseract`, which is not in
requirements.txt and not in the runtime venv, so every `--ocr-pages` run died
with a RuntimeError before it read a page. PyMuPDF already carries Tesseract
(it is what the ingest lane's pymupdf4llm drives), so the replacement needs no
new dependency.

Run:  python -m pytest tests/test_web_import_ocr.py -q

Most tests swap in a fake `fitz` so the CONTRACT is pinned without Tesseract.
The two that use real PyMuPDF prove the contract against the actual library.
"""
import ast
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.ingestion import web_import

ROOT = Path(__file__).resolve().parents[1]
CANARY = "NOETRIX OCR CANARY 4711"

needs_tessdata = pytest.mark.skipif(
    not (os.environ.get("TESSDATA_PREFIX") and Path(os.environ["TESSDATA_PREFIX"]).is_dir()),
    reason="set TESSDATA_PREFIX to a tessdata folder to OCR for real",
)


# ------------------------------------------------------------------- doubles

class _FakePage:
    def __init__(self, doc, index):
        self.doc, self.index = doc, index

    # Same keyword names as pymupdf.utils.get_textpage_ocr, so a misspelt kwarg
    # in the implementation fails here and not only against the real library.
    def get_textpage_ocr(self, flags=0, language="eng", dpi=72, full=False, tessdata=None):
        self.doc.ocr_calls.append(
            {"page": self.index, "language": language, "dpi": dpi, "full": full})
        if self.doc.ocr_error is not None:
            raise self.doc.ocr_error
        return ("ocr-textpage", self.index)

    def get_text(self, option="text", *, textpage=None):
        # Real PyMuPDF quietly falls back to the page's OWN text layer when no
        # textpage is passed. For a scan that is empty: a silent blank.
        if textpage != ("ocr-textpage", self.index):
            return "WRONG SOURCE: the page's own text layer"
        return f"  ocr text p{self.index + 1} \n"


class _FakeDoc:
    def __init__(self, page_count, ocr_error=None):
        self.page_count = page_count
        self.ocr_error = ocr_error
        self.ocr_calls = []
        self.closed = False

    def __getitem__(self, index):
        return _FakePage(self, index)

    def close(self):
        self.closed = True


def _forbid_config_reads(monkeypatch):
    """The OCR helper takes its language from the caller (main.py's cmd_convert_files
    already holds the loaded config). Reaching for load_config() again re-reads and
    re-parses config.yaml, and a fresh clone has no config.yaml at all (see
    conftest): no test here may depend on one, and the helper must not ask."""
    def refuse(*args, **kwargs):
        raise AssertionError("the OCR helper must not call load_config(): "
                             "the caller passes the language down")
    monkeypatch.setattr("src.utils.config_loader.load_config", refuse)


@pytest.fixture
def fake_fitz(monkeypatch):
    """install(doc) -> doc; `import fitz` inside the helper now yields a fake."""
    pymupdf = pytest.importorskip("pymupdf")

    def install(doc):
        monkeypatch.setitem(sys.modules, "fitz", types.SimpleNamespace(
            open=lambda path: doc,
            mupdf=pymupdf.mupdf,       # the helper names mupdf's error class in an `except`
        ))
        _forbid_config_reads(monkeypatch)
        return doc

    return install


def _scanned_pdf(path):
    """One page whose text exists only as pixels, like a real scan. (A page with
    a text layer would let a broken OCR path pass by reading that layer.)"""
    pymupdf = pytest.importorskip("pymupdf")
    src = pymupdf.open()
    page = src.new_page()
    page.insert_text((72, 120), CANARY, fontsize=28)
    pix = page.get_pixmap(dpi=200)
    scan = pymupdf.open()
    blank = scan.new_page(width=page.rect.width, height=page.rect.height)
    blank.insert_image(blank.rect, pixmap=pix)
    scan.save(str(path))
    assert not scan[0].get_text().strip()
    scan.close()
    src.close()
    return path


# ------------------------------------------------------------------ contract

def test_pages_are_ocrd_ascending_and_formatted_as_markdown(fake_fitz):
    doc = fake_fitz(_FakeDoc(page_count=5))
    # parse_page_spec sorts, collapses the duplicate 2 and drops 99 (past the end)
    out = web_import._ocr_pdf_pages(Path("scan.pdf"), "4,2-3,2,99")

    assert [c["page"] for c in doc.ocr_calls] == [1, 2, 3]            # 0-based indices
    assert out == ("## Page 2\n\nocr text p2\n\n"                     # 1-based headers, stripped
                   "## Page 3\n\nocr text p3\n\n"
                   "## Page 4\n\nocr text p4")
    # the WHOLE page image is OCR'd (full=True) at a resolution Tesseract can
    # read - PyMuPDF's default of 72 dpi is not one
    assert all(c["full"] is True and c["dpi"] > 72 for c in doc.ocr_calls)
    assert doc.closed


@pytest.mark.parametrize("kwargs,expected", [
    ({"lang": "deu"}, "deu"),
    ({}, "eng"),                                  # not passed -> the documented default
])
def test_language_is_the_callers_and_defaults_to_eng(fake_fitz, kwargs, expected):
    doc = fake_fitz(_FakeDoc(page_count=1))
    web_import._ocr_pdf_pages(Path("scan.pdf"), "1", **kwargs)
    assert {c["language"] for c in doc.ocr_calls} == {expected}


def _convert_one_pdf(tmp_path, monkeypatch, **kwargs):
    """convert_files over one inbox PDF with markitdown and the OCR helper replaced;
    returns the (pdf, pages, lang) the helper was handed."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "scan.pdf").write_bytes(b"%PDF-1.4")
    converter = types.SimpleNamespace(
        convert=lambda path: types.SimpleNamespace(text_content="text layer"))
    monkeypatch.setattr(web_import, "_markitdown", lambda: converter)
    seen = []
    monkeypatch.setattr(web_import, "_ocr_pdf_pages",
                        lambda pdf, pages, lang="eng": seen.append((pdf.name, pages, lang)) or "ocr")
    out = web_import.convert_files(["scan.pdf"], inbox, tmp_path / "out", ocr_pages="1-2", **kwargs)
    assert out[0]["ok"], out
    return seen


def test_convert_files_hands_the_ocr_language_to_the_helper(tmp_path, monkeypatch):
    assert _convert_one_pdf(tmp_path, monkeypatch, ocr_language="deu") == [("scan.pdf", "1-2", "deu")]


def test_convert_files_defaults_the_ocr_language_to_eng(tmp_path, monkeypatch):
    assert _convert_one_pdf(tmp_path, monkeypatch) == [("scan.pdf", "1-2", "eng")]


def test_convert_files_command_passes_the_configured_language(tmp_path, monkeypatch):
    """main.py's cmd_convert_files already holds the loaded config, so it is the one
    that reads pdf.ocr_language and hands it down: the helper never re-loads it."""
    import main
    from src.utils.config_loader import Config
    cfg = Config({"pdf": {"vault_path": str(tmp_path), "ocr_language": "deu"}}, tmp_path)
    monkeypatch.setattr(main, "load_config", lambda path=None: cfg)
    seen = {}

    def fake_convert(files, inbox, dest, **kwargs):
        seen.update(kwargs)
        return [{"file": "scan.pdf", "ok": True, "output": "scan.md"}]

    monkeypatch.setattr(web_import, "convert_files", fake_convert)
    args = main.build_parser().parse_args(["convert-files", "--files", "scan.pdf", "--ocr-pages", "1"])
    args.func(args)
    assert seen == {"ocr_pages": "1", "ocr_language": "deu"}

    bare = Config({"pdf": {"vault_path": str(tmp_path)}}, tmp_path)       # key absent
    monkeypatch.setattr(main, "load_config", lambda path=None: bare)
    args.func(args)
    assert seen["ocr_language"] == "eng"


# ---------------------------------------------------------- missing tessdata

@pytest.mark.parametrize("make_error", [
    # PyMuPDF cannot find Tesseract / a tessdata folder at all
    lambda pymupdf: RuntimeError("No tessdata specified and Tesseract is not installed"),
    # tessdata exists but holds no usable language - NOT a RuntimeError subclass
    lambda pymupdf: pymupdf.mupdf.FzErrorLibrary("code=3: Tesseract language initialisation failed"),
], ids=["no-tesseract", "no-language-data"])
def test_missing_tessdata_raises_runtimeerror_naming_tessdata_prefix(fake_fitz, make_error):
    original = make_error(pytest.importorskip("pymupdf"))
    doc = fake_fitz(_FakeDoc(page_count=3, ocr_error=original))

    with pytest.raises(RuntimeError) as err:
        web_import._ocr_pdf_pages(Path("scan.pdf"), "1-3", lang="deu")

    assert "TESSDATA_PREFIX" in str(err.value)
    assert "deu" in str(err.value)                # the language it could not load
    assert err.value.__cause__ is original        # the real reason is chained, not swallowed
    assert doc.closed                             # and the document is closed on this path too


def test_real_pymupdf_with_unusable_tessdata_raises_readably(tmp_path, monkeypatch):
    """The real library's failure rather than my model of it. PyMuPDF bundles
    Tesseract, so this needs no system install and runs everywhere."""
    pdf = _scanned_pdf(tmp_path / "scan.pdf")
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path / "no-such-tessdata"))
    _forbid_config_reads(monkeypatch)

    with pytest.raises(RuntimeError) as err:
        web_import._ocr_pdf_pages(pdf, "1")

    assert "TESSDATA_PREFIX" in str(err.value)
    assert "eng" in str(err.value)


# ---------------------------------------------------------------------- live

@needs_tessdata
def test_live_ocr_reads_a_textless_scan(tmp_path, monkeypatch):
    pdf = _scanned_pdf(tmp_path / "scan.pdf")
    _forbid_config_reads(monkeypatch)

    out = web_import._ocr_pdf_pages(pdf, "1")

    assert out.startswith("## Page 1\n\n")
    assert all(word in out.upper() for word in CANARY.split())


# ------------------------------------------------------------------ the guard

def test_pytesseract_is_not_imported_anywhere_in_src():
    """It is not installed in the runtime venv, so any import of it is a
    RuntimeError waiting for the day that code path runs."""
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.split(".")[0] == "pytesseract" for name in names):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, f"pytesseract imported at: {offenders}"
