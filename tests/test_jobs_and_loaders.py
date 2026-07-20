"""Job-builder + loader guard tests (session 11).

Covers the console job argv builder (param validation is the API's contract
with agents), the PDF --pages spec parser, and the code-loader discovery
guard for directories named like files.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from manage_api import _build_argv, _retag_meta, _safe_rel
from src.ingestion.pdf_loader import parse_page_spec
from src.ingestion.code_loader import CodeLoader


# ---- _build_argv: the agent-facing job contract ----

def test_ingest_pdfs_chunking_flag():
    argv = _build_argv("ingest_pdfs", {"chunking": "fixed"})
    assert argv[-2:] == ["--chunking", "fixed"]
    argv = _build_argv("ingest_pdfs", {})
    assert "--chunking" not in argv


def test_ingest_pdfs_chunking_invalid():
    with pytest.raises(ValueError):
        _build_argv("ingest_pdfs", {"chunking": "semantic"})


def test_ingest_pdfs_ocr_invalid():
    with pytest.raises(ValueError):
        _build_argv("ingest_pdfs", {"ocr_engine": "gpt4v"})


def test_ingest_pdfs_pages_validated():
    argv = _build_argv("ingest_pdfs", {"pages": "1-50,60,70-80"})
    assert "--pages" in argv
    with pytest.raises(ValueError):
        _build_argv("ingest_pdfs", {"pages": "0-5"})       # 1-based


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        _build_argv("drop_all_tables", {})


def test_safe_rel_guards():
    assert _safe_rel("foo.jsonl") == "data/foo.jsonl"       # bare name -> data/
    assert _safe_rel("data/foo.jsonl") == "data/foo.jsonl"
    for bad in ("C:/windows/x.jsonl", "/etc/passwd", "../outside.jsonl",
                "data/../../x.jsonl", ""):
        with pytest.raises(ValueError):
            _safe_rel(bad)


# ---- _retag_meta: the metadata transform behind /api/documents/retag ----

def test_retag_meta_domain_course_tags():
    m = {"source_file": "x.sql", "domain": "general",
         "course_code": "unknown", "course_name": "unknown", "tags": ["old"]}
    out = _retag_meta(m, "db", "Databases & Data Engineering", ["sql"], {"old"})
    assert out["domain"] == "db"
    # course sets BOTH fields (manifest + eval read course_name first)
    assert out["course_name"] == "Databases & Data Engineering"
    assert out["course_code"] == "Databases & Data Engineering"
    assert out["tags"] == ["sql"]


def test_retag_meta_none_keeps_everything():
    m = {"domain": "biz", "course_name": "Business Intelligence & Analytics",
         "course_code": "DS 206"}
    out = _retag_meta(m, None, None, [], set())
    assert out == {"domain": "biz",
                   "course_name": "Business Intelligence & Analytics",
                   "course_code": "DS 206"}


# ---- parse_page_spec ----

def test_page_spec_ranges_and_singletons():
    # 1-based spec -> SORTED 0-BASED indices (what pymupdf4llm wants)
    assert parse_page_spec("1-3,5") == [0, 1, 2, 4]
    assert parse_page_spec("7") == [6]
    assert parse_page_spec("3, 1-2 ,3") == [0, 1, 2]        # whitespace + dups
    assert parse_page_spec("") == []                        # empty = no subset
    assert parse_page_spec("1-9999", page_count=3) == [0, 1, 2]   # clamped


def test_page_spec_rejects_garbage():
    for bad in ("0", "5-3", "a-b", "1-2-3", "-4"):
        with pytest.raises(ValueError):
            parse_page_spec(bad)


# ---- code loader: directories named like files ----

def test_discovery_skips_dir_named_like_sql(tmp_path):
    (tmp_path / "real.sql").write_text("SELECT 1;", encoding="utf-8")
    trap = tmp_path / "PSS2_Solutions.sql"                  # a real vault pattern
    trap.mkdir()
    (trap / "inner.sql").write_text("SELECT 2;", encoding="utf-8")
    loader = CodeLoader(vault_path=tmp_path, output_file=tmp_path / "out.jsonl")
    found = loader.discover_files()
    names = {f.relative_to(tmp_path).as_posix() for f in found}
    assert names == {"real.sql", "PSS2_Solutions.sql/inner.sql"}


# ---- session 14: new job kinds + chunking values + rerank modes ----

def test_chunking_document_none_accepted():
    for mode in ("document", "none"):
        argv = _build_argv("ingest_pdfs", {"chunking": mode})
        assert argv[-2:] == ["--chunking", mode]
    with pytest.raises(ValueError):
        _build_argv("ingest_pdfs", {"chunking": "semantic"})


def test_include_files_pass_through_and_guards():
    argv = _build_argv("ingest_pdfs", {"include_files": ["a.pdf", "b.pdf"]})
    assert "--include-files" in argv and "a.pdf,b.pdf" in argv
    argv = _build_argv("ingest_code", {"include_files": "x.sql"})
    assert "--include-files" in argv and "x.sql" in argv
    with pytest.raises(ValueError):
        _build_argv("ingest_pdfs", {"include_files": ["../evil.pdf"]})
    # empty list = no filter at all, not an error
    assert "--include-files" not in _build_argv("ingest_pdfs", {"include_files": []})


def test_ingest_md_guards():
    argv = _build_argv("ingest_md", {"include_path": "Inbox/x.md",
                                     "output": "data/inbox_md.jsonl",
                                     "chunking": "document"})
    assert "ingest-md" in argv and "--chunking" in argv
    with pytest.raises(ValueError):        # scoped parse may never hit chunks.jsonl
        _build_argv("ingest_md", {"include_path": "Inbox",
                                  "output": "data/chunks.jsonl"})
    with pytest.raises(ValueError):        # include filter is mandatory
        _build_argv("ingest_md", {"output": "data/x.jsonl"})


def test_fetch_web_and_convert_files_validation():
    argv = _build_argv("fetch_web", {"urls": ["https://a.io/x"], "backend": "auto"})
    assert "fetch-web" in argv
    with pytest.raises(ValueError):
        _build_argv("fetch_web", {"urls": ["ftp://a.io/x"]})
    with pytest.raises(ValueError):
        _build_argv("fetch_web", {"urls": [], "backend": "auto"})
    argv = _build_argv("convert_files", {"files": ["r.docx"], "ocr_pages": "1-3"})
    assert "convert-files" in argv and "--ocr-pages" in argv
    with pytest.raises(ValueError):
        _build_argv("convert_files", {"files": ["r.docx"], "ocr_pages": "0-3"})


def test_lexical_reranker_orders_by_term_coverage():
    from src.retrieval.reranker import Reranker
    from src.retrieval.retriever import RetrievedDoc

    def doc(text):
        return RetrievedDoc(id=text[:8], text=text, metadata={}, score=0.0)

    rr = Reranker(model_name="unused", top_k=2, mode="lexical")
    docs = [doc("nothing relevant here at all whatsoever"),
            doc("gradient descent updates weights via the gradient"),
            doc("the weather was nice")]
    out = rr.rerank("gradient descent weights", docs)
    assert out[0].text.startswith("gradient descent")
    assert len(out) == 2
    # none-mode just truncates in fused order, loading no model
    rr_none = Reranker(model_name="unused", top_k=2, mode="none")
    assert [d.text for d in rr_none.rerank("q", docs)] == \
        [docs[0].text, docs[1].text]


def test_persist_section_keys_section_aware(tmp_path):
    from manage_api import _persist_section_keys
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        'parser:\n  vault_path: "old/path"   # comment kept\n  chunking: heading\n'
        'generation:\n  model: auto\n  base_url: "http://localhost:3001/v1"\n'
        'retrieval:\n  rerank_mode: cross_encoder\n', encoding="utf-8")
    written = _persist_section_keys(cfg, {"parser.vault_path": "A:/new vault",
                                          "retrieval.rerank_mode": "lexical"})
    text = cfg.read_text(encoding="utf-8")
    assert set(written) == {"parser.vault_path", "retrieval.rerank_mode"}
    # value swapped in place, quoting + trailing comment preserved
    assert 'vault_path: "A:/new vault"   # comment kept' in text
    assert "rerank_mode: lexical" in text
    assert "model: auto" in text                      # other sections untouched
    with pytest.raises(ValueError):                   # unknown leaf -> refuse
        _persist_section_keys(cfg, {"generation.nope": "x"})


def test_write_sparse_meta_sidecar(tmp_path):
    from src.embeddings.embedder import write_sparse_meta
    import json
    pkl = tmp_path / "bm25_index.pkl"
    meta = write_sparse_meta(pkl, 173606)
    assert meta.name == "bm25_index.pkl.meta.json"
    data = json.loads(meta.read_text(encoding="utf-8"))
    assert data["count"] == 173606 and data["built_at"]
