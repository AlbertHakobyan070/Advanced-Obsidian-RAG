"""Job-builder + loader guard tests (session 11).

Covers the console job argv builder (param validation is the API's contract
with agents), the PDF --pages spec parser, and the code-loader discovery
guard for directories named like files.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

# manage_api builds its CFG at import time, so it needs a config.yaml — which a
# fresh clone does not have (it is gitignored; you copy config.example.yaml).
# Importing unconditionally made `pytest tests/` collapse at COLLECTION for
# anyone who had just cloned the repo, taking the loader tests below down with
# it even though they need no config at all.
try:
    from manage_api import _build_argv, _retag_meta, _safe_rel
    _HAVE_CONSOLE = True
except (FileNotFoundError, KeyError):
    _HAVE_CONSOLE = False

from src.ingestion.pdf_loader import parse_page_spec
from src.ingestion.code_loader import CodeLoader

pytestmark_console = pytest.mark.skipif(
    not _HAVE_CONSOLE,
    reason="no config.yaml — copy config.example.yaml to config.yaml")


# ---- _build_argv: the agent-facing job contract ----

@pytestmark_console
def test_ingest_pdfs_chunking_flag():
    argv = _build_argv("ingest_pdfs", {"chunking": "fixed"})
    assert argv[-2:] == ["--chunking", "fixed"]
    argv = _build_argv("ingest_pdfs", {})
    assert "--chunking" not in argv


@pytestmark_console
def test_ingest_pdfs_chunking_invalid():
    with pytest.raises(ValueError):
        _build_argv("ingest_pdfs", {"chunking": "semantic"})


@pytestmark_console
def test_ingest_pdfs_ocr_invalid():
    with pytest.raises(ValueError):
        _build_argv("ingest_pdfs", {"ocr_engine": "gpt4v"})


@pytestmark_console
def test_ingest_pdfs_pages_validated():
    # A page subset is a scoped run, so it names its own output — otherwise the
    # scoped-run guard would refuse it first and the 1-based check below would
    # pass for the wrong reason.
    own = "data/pages_chunks.jsonl"
    argv = _build_argv("ingest_pdfs", {"pages": "1-50,60,70-80", "output": own})
    assert "--pages" in argv
    with pytest.raises(ValueError):
        _build_argv("ingest_pdfs", {"pages": "0-5", "output": own})   # 1-based


@pytestmark_console
def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        _build_argv("drop_all_tables", {})


@pytestmark_console
def test_safe_rel_guards():
    assert _safe_rel("foo.jsonl") == "data/foo.jsonl"       # bare name -> data/
    assert _safe_rel("data/foo.jsonl") == "data/foo.jsonl"
    for bad in ("C:/windows/x.jsonl", "/etc/passwd", "../outside.jsonl",
                "data/../../x.jsonl", ""):
        with pytest.raises(ValueError):
            _safe_rel(bad)


# ---- _retag_meta: the metadata transform behind /api/documents/retag ----

@pytestmark_console
def test_retag_meta_domain_course_tags():
    m = {"source_file": "x.sql", "domain": "general",
         "course_code": "unknown", "course_name": "unknown", "tags": ["old"]}
    out = _retag_meta(m, "db", "Databases & Data Engineering", ["sql"], {"old"})
    assert out["domain"] == "db"
    # course sets BOTH fields (manifest + eval read course_name first)
    assert out["course_name"] == "Databases & Data Engineering"
    assert out["course_code"] == "Databases & Data Engineering"
    assert out["tags"] == ["sql"]


@pytestmark_console
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

@pytestmark_console
def test_chunking_document_none_accepted():
    for mode in ("document", "none"):
        argv = _build_argv("ingest_pdfs", {"chunking": mode})
        assert argv[-2:] == ["--chunking", mode]
    with pytest.raises(ValueError):
        _build_argv("ingest_pdfs", {"chunking": "semantic"})


@pytestmark_console
def test_include_files_pass_through_and_guards():
    # File-scoped, so each run names its own output (see the scoped-run guard
    # tests at the end of this module).
    own = "data/files_chunks.jsonl"
    argv = _build_argv("ingest_pdfs", {"include_files": ["a.pdf", "b.pdf"], "output": own})
    assert "--include-files" in argv and "a.pdf,b.pdf" in argv
    argv = _build_argv("ingest_code", {"include_files": "x.sql", "output": own})
    assert "--include-files" in argv and "x.sql" in argv
    with pytest.raises(ValueError):
        _build_argv("ingest_pdfs", {"include_files": ["../evil.pdf"], "output": own})
    # empty list = no filter at all, not an error
    assert "--include-files" not in _build_argv("ingest_pdfs", {"include_files": []})


@pytestmark_console
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


@pytestmark_console
def test_ingest_canvas_argv():
    # A scoped run gets its OWN output file. This test used to pair
    # include_path with the canonical canvas_chunks.jsonl, which the loader
    # would truncate to just that scope — see
    # test_a_scoped_canvas_run_may_not_clobber_the_canonical_file below.
    argv = _build_argv("ingest_canvas", {
        "output": "data/canvas_bayesian_chunks.jsonl",
        "include_path": "Bayesian",
        "force_domain": "stats",
        "force_tags": ["canvas", "bayesian"],
    })
    assert argv[:3] == [argv[0], "main.py", "ingest-canvas"]
    # _vault_data_path anchors "data/x.jsonl" to the active vault's data dir,
    # so only the basename is stable here (see test_safe_rel_guards).
    assert "--output" in argv and any(a.endswith("canvas_bayesian_chunks.jsonl") for a in argv)
    assert "--include-path" in argv and "Bayesian" in argv
    assert "--force-domain" in argv and "stats" in argv
    assert "--force-tags" in argv and "canvas,bayesian" in argv
    # no params at all -> just the bare subcommand
    assert _build_argv("ingest_canvas", {}) == [argv[0], "main.py", "ingest-canvas"]
    with pytest.raises(ValueError):
        _build_argv("ingest_canvass", {})


@pytestmark_console
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


@pytestmark_console
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


# ---- ingest_canvas: the graph lane's job contract ----

@pytestmark_console
def test_canvas_graph_hyperparameters_reach_the_cli():
    argv = _build_argv("ingest_canvas", {
        "include_path": "ArmHist EXAM", "output": "data/canvas_armhist_chunks.jsonl",
        "max_chunk_size": 1500, "chunking": "document", "context_depth": 1})
    assert argv[argv.index("--max-chunk-size") + 1] == "1500"
    assert argv[argv.index("--chunking") + 1] == "document"
    assert argv[argv.index("--context-depth") + 1] == "1"
    assert argv[argv.index("--include-path") + 1] == "ArmHist EXAM"


@pytestmark_console
def test_canvas_defaults_pass_no_chunking_flags_at_all():
    """Every canvas chunk in the index was produced with splitting off and
    depth 0; an omitted knob must stay omitted, not be materialised."""
    argv = _build_argv("ingest_canvas", {})
    for flag in ("--max-chunk-size", "--chunking", "--context-depth", "--output"):
        assert flag not in argv


@pytestmark_console
def test_context_depth_zero_is_passed_explicitly_not_dropped():
    """0 is a real value here — `if prm.get(...)` would silently discard it
    and let a configured depth of 1 leak into a run that asked for 0."""
    argv = _build_argv("ingest_canvas", {"context_depth": 0})
    assert argv[argv.index("--context-depth") + 1] == "0"


@pytestmark_console
def test_a_scoped_canvas_run_may_not_clobber_the_canonical_file():
    """The loader truncates its output file. A folder-scoped run pointed at
    canvas_chunks.jsonl would shrink it to that folder: the dense index keeps
    every chunk (append upserts) while build_sparse_union re-derives the
    sparse half from the JSONLs and loses the rest — silent drift from a job
    that reported success."""
    with pytest.raises(ValueError) as exc_info:
        _build_argv("ingest_canvas", {"include_path": "10 - CANVASes"})
    assert "canvas_chunks.jsonl" in str(exc_info.value)

    with pytest.raises(ValueError):
        _build_argv("ingest_canvas", {"include_path": "10 - CANVASes",
                                      "output": "data/canvas_chunks.jsonl"})

    # Scoped WITH its own output file is the supported shape.
    argv = _build_argv("ingest_canvas", {"include_path": "10 - CANVASes",
                                         "output": "data/canvas_maps_chunks.jsonl"})
    assert "--include-path" in argv and "--output" in argv


@pytestmark_console
def test_whole_vault_canvas_run_still_writes_the_canonical_file():
    argv = _build_argv("ingest_canvas", {"output": "data/canvas_chunks.jsonl"})
    assert argv[argv.index("--output") + 1].endswith("canvas_chunks.jsonl")


@pytestmark_console
def test_canvas_rejects_an_unknown_splitter_and_an_out_of_range_depth():
    with pytest.raises(ValueError):
        _build_argv("ingest_canvas", {"chunking": "sideways"})
    with pytest.raises(ValueError):
        _build_argv("ingest_canvas", {"context_depth": 3})


# ---- pdf / notebook / code: the scoped-run guard canvas and md already carry ----

_CANONICAL = {"ingest_pdfs": "pdf_chunks.jsonl",
              "ingest_notebooks": "ipynb_chunks.jsonl",
              "ingest_code": "code_chunks.jsonl"}

# Every param that narrows WHICH files or pages a lane reads. Options that only
# change how chunks are made (chunking, OCR engine, force_domain) are not here.
_NARROWING = [
    ("ingest_pdfs", "include_path", "Wackerly"),
    ("ingest_pdfs", "exclude_path", "Current Courses"),
    ("ingest_pdfs", "include_files", ["a.pdf"]),
    ("ingest_pdfs", "only_books", True),
    ("ingest_pdfs", "skip_books", True),
    ("ingest_pdfs", "max_pages", 5),
    ("ingest_pdfs", "pages", "1-50,60"),
    ("ingest_notebooks", "include_path", "Capstone"),
    ("ingest_notebooks", "include_files", ["a.ipynb"]),
    ("ingest_notebooks", "exts", ".ipynb"),
    ("ingest_code", "include_path", "Capstone"),
    ("ingest_code", "exclude_path", "node"),
    ("ingest_code", "include_files", ["x.sql"]),
    ("ingest_code", "exts", ".sql"),
]


@pytest.fixture
def default_lane_files(monkeypatch, tmp_path):
    """Pin each lane's output_file to its shipped default. The guard reads the
    canonical name from config, so without this these tests would depend on
    whatever output_file the machine's own config.yaml sets."""
    import manage_api
    from src.utils.config_loader import Config
    monkeypatch.setattr(manage_api, "CFG", Config({
        "pdf": {"output_file": "data/pdf_chunks.jsonl"},
        "notebooks": {"output_file": "data/ipynb_chunks.jsonl"},
        "code": {"output_file": "data/code_chunks.jsonl"},
    }, tmp_path))


@pytestmark_console
@pytest.mark.parametrize("kind,key,value", _NARROWING)
def test_a_scoped_lane_run_may_not_clobber_the_canonical_file(
        default_lane_files, kind, key, value):
    """The loaders open their output with "w". A run narrowed to one folder,
    file or page range, pointed at the lane's canonical JSONL, would shrink it
    to that scope: the dense index keeps every chunk (append upserts) while the
    sparse half is re-derived from the JSONLs and loses the rest — silent drift
    from a job that reported success. Same guard as canvas and md."""
    canonical = _CANONICAL[kind]
    scoped = {key: value}
    with pytest.raises(ValueError) as exc_info:       # blank output = the canonical default
        _build_argv(kind, scoped)
    assert canonical in str(exc_info.value)

    with pytest.raises(ValueError):                   # naming it is no way round
        _build_argv(kind, {**scoped, "output": f"data/{canonical}"})
    with pytest.raises(ValueError):                   # NTFS is case-insensitive: same file
        _build_argv(kind, {**scoped, "output": f"data/{canonical.upper()}"})

    # Scoped WITH its own output file is the supported shape.
    argv = _build_argv(kind, {**scoped, "output": "data/scoped_chunks.jsonl"})
    assert argv[argv.index("--output") + 1].endswith("scoped_chunks.jsonl")


@pytestmark_console
@pytest.mark.parametrize("kind", sorted(_CANONICAL))
def test_a_whole_lane_run_still_writes_the_canonical_file(default_lane_files, kind):
    """The guard is for NARROWED runs only: an unscoped run owns its lane's
    file, and a blank output still means the loader's own default."""
    canonical = _CANONICAL[kind]
    argv = _build_argv(kind, {"output": f"data/{canonical}"})
    assert argv[argv.index("--output") + 1].endswith(canonical)
    assert "--output" not in _build_argv(kind, {})
    assert "--output" not in _build_argv(kind, {"force_domain": "ml"})


@pytestmark_console
def test_the_canonical_file_is_the_one_config_names(monkeypatch, tmp_path):
    """A lane whose config.yaml points output_file elsewhere has a different
    canonical file; guarding the built-in name would protect the wrong one."""
    import manage_api
    from src.utils.config_loader import Config
    monkeypatch.setattr(manage_api, "CFG", Config(
        {"pdf": {"output_file": "data/books_chunks.jsonl"}}, tmp_path))
    with pytest.raises(ValueError):
        _build_argv("ingest_pdfs", {"include_path": "X",
                                    "output": "data/books_chunks.jsonl"})
    argv = _build_argv("ingest_pdfs", {"include_path": "X",
                                       "output": "data/pdf_chunks.jsonl"})
    assert argv[argv.index("--output") + 1].endswith("pdf_chunks.jsonl")
    # A lane config says nothing about falls back to the loader's built-in name.
    with pytest.raises(ValueError):
        _build_argv("ingest_notebooks", {"include_path": "X",
                                         "output": "data/ipynb_chunks.jsonl"})


# ---- no run may take ANOTHER lane's chunk file as its output ----------------
# The guard above keeps a SCOPED run off its own lane's file. Pointing ANY run, even
# a whole-lane one, at a different lane's file (or at chunks.jsonl) is the same
# clobber: the loader opens it with "w", so that lane's rows drop out of the sparse
# index while they stay in the dense one.

_LANE_FILE = {"pdf": "pdf_chunks.jsonl", "notebooks": "ipynb_chunks.jsonl",
              "code": "code_chunks.jsonl", "canvas": "canvas_chunks.jsonl",
              "markdown": "chunks.jsonl"}
_LANE_OF = {"ingest_pdfs": "pdf", "ingest_notebooks": "notebooks", "ingest_code": "code",
            "ingest_canvas": "canvas", "ingest_md": "markdown"}
# Otherwise-valid params for each kind; ingest_md refuses to run without a scope.
_RUN = {"ingest_pdfs": {}, "ingest_notebooks": {}, "ingest_code": {}, "ingest_canvas": {},
        "ingest_md": {"include_path": "Inbox"}}
# Every kind against every lane's file but its own. ingest_md is always scoped, so
# for it chunks.jsonl is refused too.
_FOREIGN = [(kind, file) for kind in _RUN for lane, file in _LANE_FILE.items()
            if lane != _LANE_OF[kind] or kind == "ingest_md"]


@pytestmark_console
@pytest.mark.parametrize("kind,file", _FOREIGN)
def test_a_run_may_not_write_another_lanes_chunk_file(default_lane_files, kind, file):
    with pytest.raises(ValueError):
        _build_argv(kind, {**_RUN[kind], "output": f"data/{file}"})
    with pytest.raises(ValueError):                   # NTFS is case-insensitive: same file
        _build_argv(kind, {**_RUN[kind], "output": f"data/{file.upper()}"})


@pytestmark_console
def test_the_refusal_names_the_file_and_the_lane_that_owns_it(default_lane_files):
    with pytest.raises(ValueError) as exc_info:
        _build_argv("ingest_notebooks", {"output": "data/pdf_chunks.jsonl"})
    message = str(exc_info.value)
    assert "pdf_chunks.jsonl" in message and "pdf" in message and "notebooks" in message


@pytestmark_console
@pytest.mark.parametrize("kind", sorted(_RUN))
def test_a_file_of_its_own_is_still_accepted(default_lane_files, kind):
    argv = _build_argv(kind, {**_RUN[kind], "output": "data/fresh_chunks.jsonl"})
    assert argv[argv.index("--output") + 1].endswith("fresh_chunks.jsonl")


@pytestmark_console
def test_the_protected_names_are_the_ones_config_gives_each_lane(monkeypatch, tmp_path):
    """A lane whose config.yaml points output_file elsewhere has a different
    canonical file; protecting the built-in name would guard the wrong one."""
    import manage_api
    from src.utils.config_loader import Config
    monkeypatch.setattr(manage_api, "CFG", Config(
        {"code": {"output_file": "data/src_code_chunks.jsonl"}}, tmp_path))
    with pytest.raises(ValueError):                   # code's file is the configured one now
        _build_argv("ingest_notebooks", {"output": "data/src_code_chunks.jsonl"})
    argv = _build_argv("ingest_notebooks", {"output": "data/code_chunks.jsonl"})   # nobody's now
    assert argv[argv.index("--output") + 1].endswith("code_chunks.jsonl")
