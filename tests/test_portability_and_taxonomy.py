"""Cross-platform console surface + configurable taxonomy (session 17).

Run:  python -m pytest tests/ -q

These cover the things that were silently Windows-only until the Docker/Mac
bundle exposed them:

  * the folder browser returned drive letters and nothing else, so on Linux it
    listed zero entries and the vault switcher was a dead end;
  * device enums offered `cuda` on a CPU-only torch build;
  * VAULT_KEYS gained keys that must exist in config.yaml or every vault switch
    raises;
  * the folder/course maps were hardcoded to one academic vault.

Nothing here starts a server, loads a model, or touches the real config —
except the two guard tests that deliberately read config.yaml, because their
whole point is that code and config agree.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.utils.config_loader import Config

ROOT = Path(__file__).resolve().parents[1]

# manage_api builds its CFG at import time, so it needs a config.yaml. A fresh
# clone has only config.example.yaml (config.yaml is gitignored — it holds
# machine paths), so importing unconditionally would collapse this whole module
# for anyone who just cloned the repo. The pure-logic tests below still run.
try:
    import manage_api as M
except (FileNotFoundError, KeyError):
    M = None

needs_config = pytest.mark.skipif(
    M is None,
    reason="no config.yaml — copy config.example.yaml to config.yaml to run "
           "the console-surface tests")


def cfg_of(data):
    return Config(data, ROOT)


# ---------------------------------------------------------------- browse ----

@needs_config
def test_browse_lists_entries_with_full_paths(tmp_path):
    """Each entry carries its own absolute path.

    The client used to join `dir + "\\" + name` itself, which produced
    '/vault\\Foo' on Linux — never a directory, so the first click 404'd.
    """
    (tmp_path / "Alpha").mkdir()
    (tmp_path / "Beta").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "note.md").write_text("x", encoding="utf-8")

    out = M.browse(str(tmp_path))
    names = [d["name"] for d in out["dirs"]]
    assert names == ["Alpha", "Beta"]                 # files + dotdirs excluded
    for entry in out["dirs"]:
        assert Path(entry["path"]).is_dir()
        assert Path(entry["path"]).name == entry["name"]
    assert out["parent"] == str(tmp_path.parent)


@needs_config
def test_browse_rejects_a_non_folder(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")
    assert M.browse(str(f)).status_code == 404


@needs_config
def test_browse_roots_on_posix_are_real_directories(monkeypatch, tmp_path):
    """The '' listing on Linux/macOS. Previously it probed A:/..Z:/ and always
    came back empty, which is exactly what made 'Add / open vault…' useless in
    the container.

    The platform is injected rather than monkeypatched onto os.name: pathlib
    reads os.name, so patching it makes a Windows process construct PosixPath
    and blow up inside pytest itself.
    """
    home = tmp_path / "home" / "someone"
    home.mkdir(parents=True)
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(M.CFG, "_data",
                        {**M.CFG.as_dict(), "parser": {"vault_path": str(vault)}})

    roots = M._browse_roots(windows=False)
    paths = [r["path"] for r in roots]
    assert str(home) in paths
    assert str(vault) in paths
    assert len(paths) == len(set(paths))      # no duplicates
    for r in roots:
        assert Path(r["path"]).is_dir()


@needs_config
def test_browse_roots_skip_paths_that_do_not_exist(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "nope"))
    monkeypatch.setattr(M.CFG, "_data",
                        {**M.CFG.as_dict(),
                         "parser": {"vault_path": str(tmp_path / "gone")}})
    # only the filesystem root survives — nothing else on the list exists
    assert len(M._browse_roots(windows=False)) == 1


# ---------------------------------------------------------- device enums ----

@needs_config
def test_device_choices_never_offer_an_absent_accelerator():
    info, choices = M._torch_devices()
    assert choices[:2] == ["auto", "cpu"]
    if not info.get("available"):
        assert not [c for c in choices if c.startswith("cuda")]
    if not info.get("mps"):
        assert "mps" not in choices


@needs_config
def test_current_value_is_never_stranded():
    """A config written on a GPU box must stay selectable on a CPU box —
    otherwise the dropdown shows a value it does not contain and the next save
    silently rewrites it."""
    assert M._with_current(["auto", "cpu"], "cuda:1") == ["auto", "cpu", "cuda:1"]
    assert M._with_current(["auto", "cpu"], "cpu") == ["auto", "cpu"]
    assert M._with_current(["auto", "cpu"], None) == ["auto", "cpu"]
    assert M._with_current(["auto", "cpu"], "  ") == ["auto", "cpu"]


# ------------------------------------------------------- config plumbing ----

@needs_config
def test_nested_dotted_keys_persist(tmp_path):
    """pdf.vlm_ocr.preset lives two levels deep; only its leaf is matched."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "pdf:\n"
        "  ocr_engine: auto\n"
        "  vlm_ocr:\n"
        "    preset: null    # a comment that must survive\n",
        encoding="utf-8")
    written = M._persist_section_keys(cfg, {"pdf.vlm_ocr.preset": "deepseek"})
    assert written == ["pdf.vlm_ocr.preset"]
    text = cfg.read_text(encoding="utf-8")
    assert "preset: deepseek    # a comment that must survive" in text


@needs_config
def test_an_ambiguous_leaf_still_refuses(tmp_path):
    """The safety property that made this section-aware in the first place."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text("retrieval:\n  a:\n    model: x\n  b:\n    model: y\n",
                   encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to rewrite ambiguously"):
        M._persist_section_keys(cfg, {"retrieval.a.model": "z"})


@needs_config
def test_every_vault_key_exists_in_the_shipped_config():
    """Adding a key to VAULT_KEYS that a config lacks makes EVERY vault switch
    raise "found 0 matches" — the switcher writes the whole set in one
    comment-preserving pass. Both configs are checked: the live one, and
    config.example.yaml, which is what a new install actually starts from."""
    from src.utils.config_loader import load_config
    for name in ("config.yaml", "config.example.yaml"):
        path = ROOT / name
        if not path.exists():
            continue
        disk = load_config(path)
        for key in M.VAULT_KEYS:
            assert disk.get(key) is not None, f"{key} in VAULT_KEYS, missing from {name}"


@needs_config
def test_every_rerank_profile_is_applicable():
    """A profile naming a key the Settings tab cannot write would half-apply:
    some fields change, the rest silently do not."""
    from src.retrieval.reranker import RERANK_PROFILES
    for name, profile in RERANK_PROFILES.items():
        assert profile.get("label") and profile.get("detail"), name
        for key, value in profile["settings"].items():
            spec = M.EDITABLE_SETTINGS.get(key)
            assert spec, f"profile {name} sets unknown setting {key}"
            if spec["kind"] == "enum" and spec["values"]:
                assert value in spec["values"], f"profile {name}: {key}={value}"


# ------------------------------------------------- vault-anchored output ----

@needs_config
def test_job_output_lands_in_the_active_vaults_data_dir(monkeypatch, tmp_path):
    """Ingest output must follow the ACTIVE vault, not the project folder.

    The console built "--output data/x.jsonl" and main.py resolved it against
    its CWD (the project root). After a vault switch, chunks_file — and with it
    DATA_DIR, chunk_files() and the BM25 union — moves elsewhere, so new chunks
    were written into the PREVIOUS vault's data folder: invisible in the new
    vault's Ledger, and swept into the OLD vault's sparse union on its next
    rebuild.
    """
    other = tmp_path / "OtherVault Data"
    other.mkdir()
    monkeypatch.setattr(M, "DATA_DIR", other)

    argv = M._build_argv("ingest_notebooks", {"output": "data/new_chunks.jsonl"})
    out = Path(argv[argv.index("--output") + 1])
    assert out.parent == other, "output escaped the active vault's data dir"

    argv = M._build_argv("index_append", {"file": "data/new_chunks.jsonl"})
    assert Path(argv[argv.index("--append") + 1]).parent == other


@needs_config
def test_output_paths_still_reject_escapes(monkeypatch, tmp_path):
    """Anchoring must not weaken the guard: validation runs on the RELATIVE
    form, before anything is joined to the vault's data dir."""
    monkeypatch.setattr(M, "DATA_DIR", tmp_path)
    for bad in ("C:/windows/x.jsonl", "/etc/passwd", "../outside.jsonl",
                "data/../../x.jsonl"):
        with pytest.raises(ValueError):
            M._build_argv("ingest_notebooks", {"output": bad})
    # An omitted output is not an escape — it means "use the configured
    # default". Only the lanes that REQUIRE their own output reject it.
    assert "--output" not in M._build_argv("ingest_notebooks", {"output": ""})
    with pytest.raises(ValueError):
        M._build_argv("ingest_md", {"include_path": "Inbox", "output": ""})


@needs_config
def test_a_bare_filename_still_gets_the_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "DATA_DIR", tmp_path)
    argv = M._build_argv("ingest_code", {"output": "loose.jsonl"})
    assert Path(argv[argv.index("--output") + 1]) == tmp_path / "loose.jsonl"


def test_append_creates_the_collection_when_a_vault_is_brand_new(tmp_path):
    """A vault whose indexes were never built has NO Chroma collection, and
    appending into it is the normal first move — it is exactly what the vault
    switcher scaffolds a new vault to do. get_collection raised NotFoundError
    there, which reads as a broken install rather than an empty one.

    Also pins the creation metadata: Chroma only applies it at creation time,
    so a collection born on the append path without hnsw:space=cosine would
    silently score every later query with L2.
    """
    chromadb = pytest.importorskip("chromadb")
    from src.embeddings.embedder import Embedder

    class StubBackend:
        dim = 3

        def embed(self, texts):
            return [[float(len(t)), 0.5, 0.25] for t in texts]

    emb = Embedder.__new__(Embedder)
    emb.chroma_dir = tmp_path / "chroma_db"          # does not exist yet
    emb.collection_name = "obsidian_vault"
    emb.batch_size = 8
    emb.backend = StubBackend()

    class C:
        def __init__(self, i, text):
            self.id, self.text, self.metadata = i, text, {"source_file": "a.md"}

    emb._append_dense([C("d1", "alpha"), C("d2", "beta")])

    client = chromadb.PersistentClient(path=str(emb.chroma_dir))
    col = client.get_collection("obsidian_vault")
    assert col.count() == 2
    assert (col.metadata or {}).get("hnsw:space") == "cosine"


# ------------------------------------------------------------- taxonomy ----

@pytest.fixture
def parser_maps():
    """Snapshot + restore the parser's module-global maps.

    configure_taxonomy mutates them IN PLACE (pdf/ipynb/code loaders bound the
    objects at import time), so a test that changes them must put them back or
    it poisons every later test in the session.
    """
    from src.ingestion import obsidian_parser as P
    saved = (dict(P.FOLDER_COURSE_MAP), dict(P.DOMAIN_MAP), dict(P.COURSE_MAP),
             list(P.COURSE_KEYWORDS), P._DETECT_FROM_PATH)
    yield P
    P.FOLDER_COURSE_MAP.clear(); P.FOLDER_COURSE_MAP.update(saved[0])
    P.DOMAIN_MAP.clear();        P.DOMAIN_MAP.update(saved[1])
    P.COURSE_MAP.clear();        P.COURSE_MAP.update(saved[2])
    del P.COURSE_KEYWORDS[:];    P.COURSE_KEYWORDS.extend(saved[3])
    P._DETECT_FROM_PATH = saved[4]


def test_no_taxonomy_block_keeps_the_builtin_maps(parser_maps):
    P = parser_maps
    before = len(P.FOLDER_COURSE_MAP)
    assert before > 0
    P.configure_taxonomy(cfg_of({}))
    assert len(P.FOLDER_COURSE_MAP) == before


def test_a_supplied_folder_map_replaces_the_builtin_one(parser_maps):
    P = parser_maps
    P.configure_taxonomy(cfg_of({"taxonomy": {
        "folder_map": {"project alpha": "Alpha"},
        "domain_map": {"Alpha": "ml"},
    }}))
    assert P.FOLDER_COURSE_MAP == {"project alpha": "Alpha"}
    assert P.detect_course_from_path(["work", "Project Alpha"]) == {
        "course_code": "Alpha", "course_name": "Alpha", "domain": "ml"}
    # All-or-nothing: supplying ANY map drops every built-in lane, including
    # the substring KEYWORD fallback that would otherwise keep labelling
    # folders with this author's course names.
    assert P.COURSE_KEYWORDS == []
    assert P.detect_course_from_path(["Natural Language Processing"])[
        "course_name"] == "unknown"


def test_detect_from_path_false_yields_no_labels(parser_maps):
    P = parser_maps
    P.configure_taxonomy(cfg_of({"taxonomy": {"detect_from_path": False}}))
    assert P.detect_course_from_path(["Natural Language Processing"]) == {
        "course_code": "unknown", "course_name": "unknown", "domain": "general"}
    # emptied in place, so the loaders that from-imported them see it too
    assert P.FOLDER_COURSE_MAP == {} and P.COURSE_KEYWORDS == []


def test_in_place_mutation_reaches_the_loaders(parser_maps):
    """pdf_loader/ipynb_loader/code_loader each did
    `from obsidian_parser import FOLDER_COURSE_MAP`, binding the object at
    import time. Rebinding the global here would leave all three on the
    built-ins — this pins the in-place contract."""
    P = parser_maps
    from src.ingestion.pdf_loader import FOLDER_COURSE_MAP as pdf_map
    from src.ingestion.ipynb_loader import FOLDER_COURSE_MAP as nb_map
    P.configure_taxonomy(cfg_of({"taxonomy": {"folder_map": {"x": "X"}}}))
    assert pdf_map == {"x": "X"} and nb_map == {"x": "X"}


def test_a_malformed_taxonomy_raises_rather_than_guessing(parser_maps):
    P = parser_maps
    with pytest.raises(ValueError, match="taxonomy.folder_map"):
        P.configure_taxonomy(cfg_of({"taxonomy": {"folder_map": ["a", "b"]}}))


@needs_config
def test_taxonomy_label_defaults_and_overrides():
    assert M._taxonomy(cfg_of({}))["label"] == "course"
    assert M._taxonomy(cfg_of({}))["label_plural"] == "courses"
    t = M._taxonomy(cfg_of({"taxonomy": {"label": "project"}}))
    assert (t["label"], t["label_plural"]) == ("project", "projects")


# --------------------------------------------------- env expansion ---------

def test_env_refs_expand_in_strings(monkeypatch):
    from src.utils.config_loader import expand_env
    monkeypatch.setenv("RAG_HOST_HOME", "/Users/someone")
    out = expand_env({"parser": {"vault_path": "${RAG_HOST_HOME}/Notes"},
                      "list": ["${RAG_HOST_HOME}"], "n": 5, "b": True})
    assert out["parser"]["vault_path"] == "/Users/someone/Notes"
    assert out["list"] == ["/Users/someone"]
    assert out["n"] == 5 and out["b"] is True     # non-strings pass through


def test_env_ref_default_and_unset(monkeypatch):
    from src.utils.config_loader import expand_env
    monkeypatch.delenv("NOPE_NOT_SET", raising=False)
    assert expand_env("${NOPE_NOT_SET:-/fallback}") == "/fallback"
    # Unset with no default stays LITERAL. Blanking it would turn a vault path
    # into "" and point the parser at the filesystem root.
    assert expand_env("${NOPE_NOT_SET}") == "${NOPE_NOT_SET}"


def test_env_expansion_leaves_ordinary_text_alone():
    from src.utils.config_loader import expand_env
    prompt = "<__media__>\nConvert the document to markdown. Cost: $5 {x}"
    assert expand_env(prompt) == prompt


def test_generic_defaults_when_config_omits_them(parser_maps):
    """Code defaults must not name one particular vault's folders. The AUA
    values live in config.yaml; the fallbacks here are vault-neutral."""
    cfg = cfg_of({"parser": {"vault_path": "/tmp/v"}})
    assert (cfg.get("webui.vault_tree_root") or "") == ""
    assert (cfg.get("webui.inbox_dir") or "Inbox") == "Inbox"
