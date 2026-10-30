"""One exclusion list for every ingest lane, and one output guard for every writer.

TWO THINGS THIS PINS (slice B, tasks 1 and 2).

1. `parser.skip_dirs` was honoured by the md lane's library default only, and by
   nothing on `main.py ingest-md`; `parser.skip_roots` had no config key at all.
   Now every lane prunes `parser.skip_dirs` and skips `parser.skip_roots` unless a
   run scopes the root back in.

2. Every loader opens its output with "w". The console already refused a scoped
   run writing its lane's canonical file, any run writing ANOTHER lane's file,
   and the NTFS alias spellings of either; the CLI refused none of it. The guard
   now lives in src/utils/lane_guard.py and `main.py` and the loaders' `python -m`
   entry points call it too.
"""
import json
import sys
from pathlib import Path

import pytest
import yaml

from src.utils.config_loader import Config

ROOT = Path(__file__).resolve().parents[1]

# Over 200 characters, so the md parser keeps the section, and over 50 bytes, so
# discover_files does not drop the file.
BODY = ("# Notes\n\n"
        + "Enough body text to clear the minimum-size filters in the loaders. " * 5
        + "\n")


class _Reached(Exception):
    """Raised by a loader stub: the guard let the run through to the loader."""


def _put(path: Path, body: str = BODY) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _names(files) -> set[str]:
    return {f.name for f in files}


# ---------------------------------------------------------------- exclusions


def test_md_ingest_honours_parser_skip_dirs(tmp_path):
    import main
    vault = tmp_path / "vault"
    _put(vault / "scope" / "note.md")
    _put(vault / "scope" / "venv" / "junk.md")
    cfg = _write_config(tmp_path, {"parser": {
        "vault_path": str(vault), "chunking": "none", "skip_dirs": ["venv"]}})
    out = tmp_path / "scoped_chunks.jsonl"

    args = main.build_parser().parse_args(
        ["--config", str(cfg), "ingest-md", "--include-path", "scope",
         "--output", str(out)])
    args.func(args)

    sources = {json.loads(line)["metadata"]["source_file"]
               for line in out.read_text(encoding="utf-8").splitlines()}
    assert {s.replace("\\", "/") for s in sources} == {"scope/note.md"}, (
        "parser.skip_dirs did not reach ingest-md: a note under venv/ was parsed")


def test_md_skip_roots_ignore_case(tmp_path):
    """On NTFS, skip_roots: ["workspace1"] must skip Workspace1/ (the code and notebook
    lanes already compare lowercased)."""
    from src.ingestion.obsidian_parser import ObsidianParser
    vault = tmp_path / "vault"
    _put(vault / "workspace1" / "a.md")
    _put(vault / "Course" / "b.md")

    found = ObsidianParser(str(vault), {"skip_roots": ["Workspace1"]}).discover_files()

    assert _names(found) == {"b.md"}


def _pdf_stub_loader(root: Path):
    import collections
    from src.ingestion.pdf_loader import PDFLoader

    loader = PDFLoader.__new__(PDFLoader)       # as tests/test_vault_relative_discovery.py
    loader.vault_path = root
    loader.include_path = None
    loader.include_files = None
    loader.exclude_path = None
    loader.skip_set = set()
    loader.skip_folders = []
    loader.only_book_folders = False
    loader.skip_books = False
    loader.stats = collections.defaultdict(int)
    return loader


def test_pdf_lane_prunes_parser_skip_dirs(tmp_path):
    root = tmp_path / "vault"
    _put(root / "venv" / "x.pdf", "%PDF-1.4 stub")
    _put(root / "Course" / "p.pdf", "%PDF-1.4 stub")
    loader = _pdf_stub_loader(root)
    loader.extra_skip_dirs = frozenset({"venv"})

    assert _names(loader.discover_pdfs()) == {"p.pdf"}


def test_pdf_from_config_takes_parser_skip_dirs(tmp_path, monkeypatch):
    from src.ingestion import pdf_loader
    from src.ingestion.pdf_loader import PDFLoader
    monkeypatch.setattr(pdf_loader, "route_caches_to_disk", lambda p: None)
    cfg = Config({"parser": {"vault_path": str(tmp_path), "skip_dirs": ["venv", "_vendor"]},
                  "pdf": {"ocr_enabled": False}}, tmp_path)

    assert PDFLoader.from_config(cfg).extra_skip_dirs == frozenset(["venv", "_vendor"])


def test_pdf_lane_skips_parser_skip_roots_unless_scoped_in(tmp_path, monkeypatch):
    """REGRESSION: parser.skip_roots reached the md, notebook and code lanes only, so a
    whole-lane ingest-pdfs discovered the agent workspaces' PDFs (73 under Workspace1/)."""
    from src.ingestion import pdf_loader
    from src.ingestion.pdf_loader import PDFLoader
    monkeypatch.setattr(pdf_loader, "route_caches_to_disk", lambda p: None)
    vault = tmp_path / "vault"
    _put(vault / "workspace1" / "a.pdf", "%PDF-1.4 stub")          # another case than the config's
    _put(vault / "Course" / "b.pdf", "%PDF-1.4 stub")
    cfg = Config({"parser": {"vault_path": str(vault), "skip_roots": ["Workspace1"]},
                  "pdf": {"ocr_enabled": False}}, tmp_path)

    loader = PDFLoader.from_config(cfg)
    assert _names(loader.discover_pdfs()) == {"b.pdf"}

    loader.include_path = "workspace1"          # a scoped run is the user asking for that tree
    assert _names(loader.discover_pdfs()) == {"a.pdf"}


def test_canvas_lane_skips_parser_skip_roots_unless_scoped_in(tmp_path):
    """REGRESSION: as the pdf lane. canvas.skip_roots keeps its own rule (exact names,
    scoped or not); parser.skip_roots follows the other lanes'."""
    from src.ingestion.canvas_loader import CanvasLoader
    vault = tmp_path / "vault"
    _put(vault / "workspace1" / "x.canvas", "{}")
    _put(vault / "Course" / "ok.canvas", "{}")
    cfg = Config({"parser": {"vault_path": str(vault), "skip_roots": ["Workspace1"]}}, tmp_path)

    loader = CanvasLoader.from_config(cfg)
    assert _names(loader.discover_files()) == {"ok.canvas"}

    loader.include_path = "workspace1/x.canvas"     # the Graph tab's re-index scopes one canvas
    assert _names(loader.discover_files()) == {"x.canvas"}


def test_notebook_lane_skips_agent_roots_unless_scoped_in(tmp_path):
    from src.ingestion.ipynb_loader import NotebookLoader
    vault = tmp_path / "vault"
    _put(vault / "Workspace1" / "a.py", "x = 1\n")
    _put(vault / "Course" / "b.py", "x = 2\n")
    _put(vault / "Course" / "_vendor" / "c.py", "x = 3\n")
    cfg = Config({"parser": {"vault_path": str(vault), "skip_roots": ["Workspace1"],
                             "skip_dirs": ["_vendor"]}}, tmp_path)

    loader = NotebookLoader.from_config(cfg)
    assert _names(loader.discover_files()) == {"b.py"}

    loader.include_path = "workspace1"          # a scoped run is the user asking for that tree
    assert _names(loader.discover_files()) == {"a.py"}


def test_code_lane_null_roots_extend_with_parser_skip_roots(tmp_path):
    from src.ingestion.code_loader import CodeLoader, _DEFAULT_SKIP_ROOTS
    vault = tmp_path / "vault"
    _put(vault / "Course" / "a.js", "var a = 1;\n")
    _put(vault / "_vault_quarantine" / "x.js", "var x = 1;\n")
    _put(vault / "NewRoot" / "n.js", "var n = 1;\n")
    parser = {"vault_path": str(vault), "skip_roots": ["NewRoot"],
              "skip_dirs": ["_vault_quarantine"]}

    null_roots = CodeLoader.from_config(Config({"parser": parser}, tmp_path))
    assert null_roots.skip_roots == _DEFAULT_SKIP_ROOTS | {"newroot"}
    # n.js is pruned by the parser skip root alone (skip_dirs knows nothing of NewRoot):
    # the union must be APPLIED by discover_files, not only built.
    assert _names(null_roots.discover_files()) == {"a.js"}

    # An explicit code.skip_roots still overrides, as documented.
    explicit = CodeLoader.from_config(
        Config({"parser": parser, "code": {"skip_roots": []}}, tmp_path))
    assert explicit.skip_roots == set()
    assert _names(explicit.discover_files()) == {"a.js", "n.js"}      # n.js IS discoverable


def test_canvas_lane_adds_parser_skip_dirs(tmp_path):
    from src.ingestion.canvas_loader import CanvasLoader, _DEFAULT_SKIP_DIRS
    vault = tmp_path / "vault"
    _put(vault / "Course" / "ok.canvas", "{}")
    _put(vault / "venv" / "x.canvas", "{}")

    loader = CanvasLoader.from_config(
        Config({"parser": {"vault_path": str(vault), "skip_dirs": ["venv"]}}, tmp_path))

    assert loader.skip_dirs >= _DEFAULT_SKIP_DIRS | {"venv"}
    assert _names(loader.discover_files()) == {"ok.canvas"}


def test_without_the_parser_keys_every_lane_behaves_as_today(tmp_path, monkeypatch):
    from src.ingestion import pdf_loader
    from src.ingestion.canvas_loader import CanvasLoader, _DEFAULT_SKIP_DIRS
    from src.ingestion.code_loader import CodeLoader, _DEFAULT_SKIP_ROOTS
    from src.ingestion.ipynb_loader import NotebookLoader
    from src.ingestion.pdf_loader import PDFLoader
    monkeypatch.setattr(pdf_loader, "route_caches_to_disk", lambda p: None)
    cfg = Config({"parser": {"vault_path": str(tmp_path)}, "pdf": {"ocr_enabled": False}},
                 tmp_path)

    pdf = PDFLoader.from_config(cfg)
    assert pdf.extra_skip_dirs == frozenset() and pdf.skip_roots == frozenset()
    nb = NotebookLoader.from_config(cfg)
    assert nb.extra_skip_dirs == frozenset() and nb.skip_roots == frozenset()
    code = CodeLoader.from_config(cfg)
    assert code.extra_skip_dirs == frozenset() and code.skip_roots == _DEFAULT_SKIP_ROOTS
    canvas = CanvasLoader.from_config(cfg)
    assert canvas.skip_dirs == _DEFAULT_SKIP_DIRS and canvas.extra_skip_roots == frozenset()


# ---------------------------------------------------------------- the guard

# (command, narrowing flags, loader module, loader class, the lane's canonical file)
_SCOPED = [
    ("ingest-canvas", ["--include-path", "x"],
     "src.ingestion.canvas_loader", "CanvasLoader", "canvas_chunks.jsonl"),
    ("ingest-pdfs", ["--pages", "1-2"],
     "src.ingestion.pdf_loader", "PDFLoader", "pdf_chunks.jsonl"),
    ("ingest-notebooks", ["--exts", ".py"],
     "src.ingestion.ipynb_loader", "NotebookLoader", "ipynb_chunks.jsonl"),
    ("ingest-code", ["--include-files", "a.js"],
     "src.ingestion.code_loader", "CodeLoader", "code_chunks.jsonl"),
]


def _cli_args(tmp_path: Path, *argv: str):
    import main
    cfg = _write_config(tmp_path, {"parser": {"vault_path": str(tmp_path / "vault")}})
    return main.build_parser().parse_args(["--config", str(cfg), *argv])


def _stub(monkeypatch, target, name: str, exc: Exception) -> None:
    def raiser(*a, **k):
        raise exc
    monkeypatch.setattr(target, name, raiser)


def _lane_class(module: str, cls: str):
    import importlib
    return getattr(importlib.import_module(module), cls)


def _refused(args) -> int:
    with pytest.raises(SystemExit) as exc:
        args.func(args)
    return exc.value.code


@pytest.mark.parametrize("command,flags,module,cls,canonical", _SCOPED)
@pytest.mark.parametrize("output", [None, "UPPER"])
def test_a_scoped_cli_run_may_not_write_the_canonical_file(
        tmp_path, monkeypatch, command, flags, module, cls, canonical, output):
    """The loader opens its output with "w": a run narrowed to one folder, file
    or page range, pointed at the lane's canonical file (or left to default to
    it), truncates the whole-lane file to that scope."""
    out = [] if output is None else ["--output", f"data/{canonical.upper()}"]
    _stub(monkeypatch, _lane_class(module, cls), "from_config",
          AssertionError("the loader was reached"))

    assert _refused(_cli_args(tmp_path, command, *flags, *out)) == 2


@pytest.mark.parametrize("output", ["chunks.jsonl", "data/CHUNKS.JSONL"])
def test_a_scoped_md_cli_run_may_not_write_chunks_jsonl(tmp_path, monkeypatch, output):
    from src.ingestion import obsidian_parser
    _stub(monkeypatch, obsidian_parser, "ObsidianParser", AssertionError("the parser was reached"))

    args = _cli_args(tmp_path, "ingest-md", "--include-path", "x", "--output", output)
    assert _refused(args) == 2


def test_no_cli_run_may_write_another_lanes_chunk_file(tmp_path, monkeypatch):
    """Whole-lane runs too: the loader truncates whatever it is pointed at."""
    from src.ocr_studio import ingest as ocr_ingest
    monkeypatch.chdir(tmp_path)
    pdf_file = tmp_path / "data" / "pdf_chunks.jsonl"
    _put(pdf_file, '{"doc_id": "d1"}\n')
    before = pdf_file.read_bytes()
    boom = AssertionError("the loader was reached")
    for module, cls in (("src.ingestion.canvas_loader", "CanvasLoader"),
                        ("src.ingestion.code_loader", "CodeLoader"),
                        ("src.ingestion.pdf_loader", "PDFLoader")):
        _stub(monkeypatch, _lane_class(module, cls), "from_config", boom)
    _stub(monkeypatch, ocr_ingest, "build_chunks", boom)

    for argv in (["ingest-canvas", "--output", "data/pdf_chunks.jsonl"],
                 ["ingest-code", "--output", "data/PDF_CHUNKS.JSONL"],
                 ["ingest-pdfs", "--output", "data/chunks.jsonl"],
                 ["ocr-studio-ingest", "--doc", "x.pdf", "--output", "data/canvas_chunks.jsonl"]):
        assert _refused(_cli_args(tmp_path, *argv)) == 2, argv

    assert pdf_file.read_bytes() == before


@pytest.mark.parametrize("alias", [
    "data/chunks.jsonl.", "data/pdf_chunks.jsonl::$DATA", "./data/x_chunks.jsonl",
    "data./x_chunks.jsonl", "data//x_chunks.jsonl", "C:/tmp/canvas_chunks.jsonl.",
    # Win32 drops trailing spaces as it drops trailing dots
    "data/pdf_chunks.jsonl ", "data/chunks.jsonl ", "data/x_chunks.jsonl ",
    "data /x_chunks.jsonl"])
def test_no_windows_alias_gets_past_the_cli_guard(tmp_path, monkeypatch, alias):
    """NTFS opens a file under spellings the by-name checks miss."""
    from src.ingestion.code_loader import CodeLoader
    _stub(monkeypatch, CodeLoader, "from_config", AssertionError("the loader was reached"))

    assert _refused(_cli_args(tmp_path, "ingest-code", "--output", alias)) == 2


def test_refuse_alias_takes_plain_and_absolute_paths():
    from src.utils import lane_guard
    # A UNC prefix is a mount point like a drive (review B2-4: the console passes resolve()d
    # paths, and a data folder on a share or mapped drive resolves to \\server\share\...).
    for ok in ("data/x_chunks.jsonl", "C:/Users/x/out.jsonl",
               "D:\\runtime\\data\\x_chunks.jsonl",
               "\\\\nas\\share\\data\\inbox_x_chunks.jsonl", "//nas/share/data/x_chunks.jsonl"):
        lane_guard.refuse_alias(ok)
    for bad in ("x::$DATA", "a/chunks.jsonl.", "//server/x.jsonl", "../x.jsonl",
                "data/chunks.jsonl ", "data/x_chunks.jsonl ", "data /x.jsonl",
                "//nas/share//x_chunks.jsonl", "\\\\nas\\share\\data\\chunks.jsonl."):
        with pytest.raises(ValueError):
            lane_guard.refuse_alias(bad)


@pytest.mark.parametrize("lane", ["pdf", "notebooks", "code", "canvas", "markdown", "ocrstudio"])
@pytest.mark.parametrize("name", ["ocrstudio_bookA_1a2b3c4d_chunks.jsonl",
                                  "canvas_edit_m_1a2b3c4d_chunks.jsonl"])
def test_per_document_chunk_files_belong_to_their_lane(tmp_path, lane, name):
    """OCR Studio and the canvas re-index write one chunk file per document. They are
    not in LANE_CHUNK_FILES, but another lane pointed at one still truncates it."""
    from src.utils import lane_guard
    cfg = Config({}, tmp_path)
    owner = "ocrstudio" if name.startswith("ocrstudio_") else "canvas"

    if lane == owner:
        lane_guard.check_output(cfg, lane, f"data/{name}", scoped=False)
    else:
        with pytest.raises(ValueError, match=f"chunk file of the {owner} lane"):
            lane_guard.check_output(cfg, lane, f"data/{name.upper()}", scoped=False)


def test_a_cli_run_may_not_write_another_lanes_per_document_file(tmp_path, monkeypatch):
    from src.ingestion.pdf_loader import PDFLoader
    _stub(monkeypatch, PDFLoader, "from_config", AssertionError("the loader was reached"))

    args = _cli_args(tmp_path, "ingest-pdfs", "--output", "data/ocrstudio_bookA_1a2b3c4d_chunks.jsonl")
    assert _refused(args) == 2


@pytest.mark.parametrize("lane", ["pdf", "canvas", "markdown", "ocrstudio"])
@pytest.mark.parametrize("name", ["parents_md.jsonl", "HYPE_QUESTIONS.JSONL"])
def test_no_run_may_write_a_sidecar(tmp_path, lane, name):
    """REGRESSION (review B2-6): the parents and HyPE sidecars are stores a delete and a
    relocate rewrite; a lane's --output named like one truncated it to chunk rows."""
    from src.utils import lane_guard
    with pytest.raises(ValueError, match="sidecar"):
        lane_guard.check_output(Config({}, tmp_path), lane, f"data/{name}", scoped=False)


def test_a_cli_lane_run_may_not_write_the_parents_sidecar(tmp_path, monkeypatch):
    from src.ingestion.canvas_loader import CanvasLoader
    _stub(monkeypatch, CanvasLoader, "from_config", AssertionError("the loader was reached"))

    assert _refused(_cli_args(tmp_path, "ingest-canvas", "--output", "data/parents_md.jsonl")) == 2


@pytest.mark.parametrize("out,parents_out", [
    ("chunks.jsonl", "data/pdf_chunks.jsonl"),           # another lane's canonical file
    ("chunks.jsonl", "data/chunks.jsonl"),               # this run's own -o
    ("x_chunks.jsonl", "data/X_CHUNKS.JSONL"),
    ("chunks.jsonl", "data/hype_questions.jsonl"),
    ("chunks.jsonl", "data/mine_chunks.jsonl"),          # would join the sparse union
])
def test_parents_out_may_not_be_a_chunk_file_or_the_hype_sidecar(tmp_path, out, parents_out):
    """REGRESSION (review B2-PARENTS-OUT-CROSS-LANE): export_parents_jsonl opens --parents-out
    with "w", and only the alias and scoped-parents checks looked at it."""
    from src.utils import lane_guard
    with pytest.raises(ValueError, match="--parents-out"):
        lane_guard.check_output(Config({}, tmp_path), "markdown", out, scoped=False,
                                parents_out=parents_out)


def test_parents_out_still_takes_the_parents_sidecar_and_its_own_name(tmp_path):
    from src.utils import lane_guard
    cfg = Config({}, tmp_path)
    lane_guard.check_output(cfg, "markdown", "chunks.jsonl", scoped=False,
                            parents_out="data/parents_md.jsonl")
    lane_guard.check_output(cfg, "markdown", "x_chunks.jsonl", scoped=True,
                            parents_out="data/x_parents.jsonl")


@pytest.mark.parametrize("module,cls,flags,key,default", [
    ("src.ingestion.pdf_loader", "PDFLoader", ["--only-books"],
     "pdf", "pdf_chunks.jsonl"),
    ("src.ingestion.ipynb_loader", "NotebookLoader", ["--exts", ".py"],
     "notebooks", "ipynb_chunks.jsonl"),
    ("src.ingestion.code_loader", "CodeLoader", ["--include-path", "x"],
     "code", "code_chunks.jsonl"),
])
def test_a_standalone_loader_entry_point_refuses_a_scoped_canonical_run(
        tmp_path, monkeypatch, capsys, module, cls, flags, key, default):
    """`python -m src.ingestion.<loader>` writes through from_config's output_file."""
    import importlib
    canonical = tmp_path / "data" / default
    _put(canonical, '{"doc_id": "d1"}\n')
    before = canonical.read_bytes()
    cfg = _write_config(tmp_path, {"parser": {"vault_path": str(tmp_path / "vault")},
                                   key: {"output_file": str(canonical)}})
    mod = importlib.import_module(module)
    _stub(monkeypatch, getattr(mod, cls), "from_config", AssertionError("the loader was reached"))
    monkeypatch.setattr(sys, "argv", [module, "--config", str(cfg), *flags])

    with pytest.raises(SystemExit) as exc:
        mod.main()

    assert exc.value.code == 2
    assert "ERROR:" in capsys.readouterr().err
    assert canonical.read_bytes() == before


def test_a_scoped_cli_run_with_its_own_output_passes_the_guard(tmp_path, monkeypatch):
    from src.ingestion.canvas_loader import CanvasLoader
    _stub(monkeypatch, CanvasLoader, "from_config", _Reached())

    args = _cli_args(tmp_path, "ingest-canvas", "--include-path", "x",
                     "--output", "data/x_chunks.jsonl")
    with pytest.raises(_Reached):
        args.func(args)


def test_a_whole_lane_cli_run_still_writes_the_canonical_file(tmp_path, monkeypatch):
    from src.ingestion.canvas_loader import CanvasLoader
    _stub(monkeypatch, CanvasLoader, "from_config", _Reached())

    args = _cli_args(tmp_path, "ingest-canvas")
    with pytest.raises(_Reached):
        args.func(args)


# ------------------------------------------------------------ config files


@pytest.mark.parametrize("name", ["config.yaml", "config.example.yaml"])
def test_config_files_keep_their_line_endings_and_bom(name):
    """Uniform endings, never mixed: CRLF in this checkout (core.autocrlf), LF in an
    LF checkout. The example keeps its UTF-8 BOM. Edit them in place."""
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} is not present in this checkout")
    raw = path.read_bytes()

    assert raw.count(b"\r\n") in (0, raw.count(b"\n")), f"{name} mixes CRLF and LF"
    if name == "config.example.yaml":
        assert raw.startswith(b"\xef\xbb\xbf")
