"""graph.embedding_provider: graph mode's seed index <collection>__graph, built by
src/embeddings/graph_twin.py on top of rebuild_into(role="graph").

Run:  python -m pytest tests/ -q

Real Chroma stores under tmp_path, a fake embedding backend (vectors depend on a
salt that stands for the model, and on the text), no model, no network, no GPU.
The FakeBackend/_vec pattern is copied from test_reembed.py (tests/ is not a
package).
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.embeddings.embedder import Embedder
from src.embeddings.graph_twin import (
    build_twin, drop_twin, graph_spec, twin_id_diff, twin_status)
from src.embeddings.reembed import ReembedError, rebuild_into
from src.embeddings.sidecar import (
    GRAPH_SUFFIX, ROLES, StampError, check_collection, delete_sidecar, iter_ids, make_sidecar,
    read_sidecar, role_for, verify_sample, write_sidecar)
from src.utils.chroma_client import persistent_client
from src.utils.config_loader import Config

pytest.importorskip("chromadb")

DIM = 16
G2 = {"kind": "openai", "base_url": "http://127.0.0.1:8104/v1", "model": "fake/g2",
      "dimensions": DIM, "query_prefix": "q: ", "doc_prefix": "d: "}
TWIN = "vault" + GRAPH_SUFFIX


@pytest.fixture(autouse=True)
def _release_chroma_stores():
    """A store's loaded collections hold C-runtime file streams (4 per HNSW
    segment, of 512 per process) until its System stops, and nothing here closes
    them. Stop the stores this test opened, so later fopen users in the same run
    (PyMuPDF in test_web_import_ocr.py) are not starved."""
    from chromadb.api.shared_system_client import SharedSystemClient
    before = set(SharedSystemClient._identifier_to_system)
    yield
    for key in set(SharedSystemClient._identifier_to_system) - before:
        SharedSystemClient._identifier_to_refcount.pop(key, None)
        SharedSystemClient._identifier_to_system.pop(key).stop()


def _vec(text):
    raw = np.frombuffer(hashlib.sha256(text.encode()).digest()[:DIM], dtype=np.uint8).astype(float) - 127.5
    return raw / np.linalg.norm(raw)


class FakeBackend:
    def __init__(self, salt="g2"):
        self.salt, self.calls = salt, []
        self.dimension = DIM

    def embed(self, texts):
        self.calls.append(list(texts))
        return [_vec(self.salt + t).tolist() for t in texts]


def _jsonl(path, rows, file_type):
    path.write_text("".join(json.dumps({"doc_id": i, "text": t, "metadata": {
        "source_file": f"x.{file_type}", "file_type": file_type}}) + "\n" for i, t in rows),
        encoding="utf-8")


def _config(root, *, provider="g2", main="local", providers=None, chroma_dir=None):
    return Config({
        "paths": {"chunks_file": str(root / "data" / "chunks.jsonl"),
                  "chroma_dir": str(chroma_dir or root / "chroma"),
                  "bm25_index": str(root / "bm25.pkl"), "collection_name": "vault"},
        "embedding": {"provider": main, "local_model": "fake/main", "batch_size": 3,
                      "providers": providers or {"g2": G2}},
        "graph": {"enabled": True, "file_type": "canvas", "embedding_provider": provider},
    }, root)


@pytest.fixture
def world(tmp_path):
    """A main collection of 10 markdown and 8 canvas chunks, with chunks.jsonl and
    canvas_chunks.jsonl holding their text (the union is the corpus)."""
    data = tmp_path / "data"
    data.mkdir()
    md = [(f"m{i:02d}", f"md text {i}") for i in range(10)]
    canvas = [(f"c{i:02d}", f"canvas text {i}") for i in range(8)]
    _jsonl(data / "chunks.jsonl", md, "md")
    _jsonl(data / "canvas_chunks.jsonl", canvas, "canvas")
    cfg = _config(tmp_path)
    client = persistent_client(cfg.path("paths.chroma_dir"))
    main = client.get_or_create_collection("vault", metadata={"hnsw:space": "cosine"})
    rows = [(i, t, "md") for i, t in md] + [(i, t, "canvas") for i, t in canvas]
    main.add(ids=[r[0] for r in rows], embeddings=[_vec("main" + r[1]).tolist() for r in rows],
             documents=[r[1] for r in rows],
             metadatas=[{"source_file": f"x.{r[2]}", "file_type": r[2], "n": n}
                        for n, r in enumerate(rows)])
    return SimpleNamespace(root=tmp_path, cfg=cfg, client=client, main=main, md=md, canvas=canvas,
                           data=data, chroma=cfg.path("paths.chroma_dir"))


def _embedder(cfg, backend=None):
    return Embedder(backend=backend or FakeBackend(), chunks_file=cfg.path("paths.chunks_file"),
                    chroma_dir=cfg.path("paths.chroma_dir"), bm25_index=cfg.path("paths.bm25_index"),
                    collection_name="vault", batch_size=cfg.get("embedding.batch_size"),
                    spec=graph_spec(cfg))


def _quiet(_):
    pass


def _build(w, cfg=None, **kw):
    return build_twin(cfg or w.cfg, kw.pop("embedder", None) or _embedder(cfg or w.cfg),
                      client=w.client, progress=_quiet, **kw)


def _names(w):
    return {c.name for c in w.client.list_collections()}


# ---------------------------------------------------------------- the build ----

def test_build_twin_holds_exactly_the_graph_lane_with_a_graph_fingerprint(world):
    jsonls = {p.name: p.read_bytes() for p in world.data.iterdir()}
    main_vec = world.main.get(ids=["c03"], include=["embeddings"])["embeddings"][0]
    emb = _embedder(world.cfg)
    sidecar = _build(world, embedder=emb)

    twin = world.client.get_collection(TWIN)
    got = twin.get(include=["documents", "metadatas"])
    assert sorted(got["ids"]) == [i for i, _ in world.canvas]                    # the graph lane only
    assert dict(zip(got["ids"], got["documents"]))["c03"] == "canvas text 3"      # JSONL text, no prefix
    assert dict(zip(got["ids"], got["metadatas"]))["c03"] == \
        world.main.get(ids=["c03"], include=["metadatas"])["metadatas"][0]       # main's metadata
    assert twin.metadata["hnsw:space"] == "cosine"
    assert np.allclose(twin.get(ids=["c03"], include=["embeddings"])["embeddings"][0],
                       _vec("g2d: canvas text 3"))                               # the graph embedder's vector

    assert (sidecar["role"], sidecar["status"], sidecar["count"], sidecar["expected"]) == \
        ("graph", "complete", 8, 8)
    assert (sidecar["provider"], sidecar["model"], sidecar["dimension"]) == ("g2", "fake/g2", DIM)
    assert (sidecar["hype_collection"], sidecar["hype"], sidecar["source_collection"]) == (None, None, "vault")
    assert read_sidecar(world.chroma, TWIN) == sidecar
    assert check_collection(emb, world.chroma, TWIN, role="graph") == sidecar

    # the main collection and the chunk files are untouched
    assert world.main.count() == 18
    assert np.allclose(world.main.get(ids=["c03"], include=["embeddings"])["embeddings"][0], main_vec)
    assert read_sidecar(world.chroma, "vault") is None
    assert {p.name: p.read_bytes() for p in world.data.iterdir()} == jsonls


def test_a_rebuild_starts_from_scratch_so_edited_text_is_re_embedded(world):
    _build(world)
    # A Graph-tab re-index: the node's text changes, its id does not.
    _jsonl(world.data / "canvas_chunks.jsonl",
           [(i, "EDITED " + t if i == "c03" else t) for i, t in world.canvas], "canvas")

    def c03():
        return world.client.get_collection(TWIN).get(ids=["c03"], include=["documents", "embeddings"])

    # what a top-up does: every id is present, so nothing is re-embedded
    topped = rebuild_into(world.cfg, TWIN, _embedder(world.cfg), client=world.client,
                          progress=_quiet, role="graph")
    assert topped["build"]["embedded"] == 0
    assert c03()["documents"] == ["canvas text 3"]
    assert np.allclose(c03()["embeddings"][0], _vec("g2d: canvas text 3"))       # the stale vector

    sidecar = _build(world)
    assert (sidecar["build"]["embedded"], sidecar["build"]["skipped_present"]) == (8, 0)
    assert c03()["documents"] == ["EDITED canvas text 3"]
    assert np.allclose(c03()["embeddings"][0], _vec("g2d: EDITED canvas text 3"))


# ----------------------------------------------------- what it maps and refuses ----

def test_graph_spec_maps_null_and_refuses_main_and_unknown(world):
    root = world.root
    assert graph_spec(_config(root, provider=None)) is None
    assert graph_spec(world.cfg).label() == "g2 · fake/g2"
    with pytest.raises(ValueError, match=r"'nope' is not in embedding\.providers.*Known providers: \['g2'\]"):
        graph_spec(_config(root, provider="nope"))
    with pytest.raises(ValueError, match="is the main collection's embedder"):
        graph_spec(_config(root, provider="local"))                              # main is local
    # the same embedder under another name and URL is still the main embedder
    alias = {"g2": G2, "g2_alias": {**G2, "base_url": "http://elsewhere:9/v1"}}
    with pytest.raises(ValueError, match="is the main collection's embedder"):
        graph_spec(_config(root, main="g2", provider="g2_alias", providers=alias))
    # local is a graph embedder when the main one is not local
    assert graph_spec(_config(root, main="g2", provider="local")).model == "fake/main"


def test_drop_refuses_a_collection_that_is_not_a_graph_twin(world):
    world.client.get_or_create_collection(TWIN)                                  # nobody's twin
    with pytest.raises(ReembedError, match="no embedding fingerprint"):
        drop_twin(world.chroma, TWIN, world.client)
    with pytest.raises(ReembedError, match="no embedding fingerprint"):
        _build(world)                                                            # the build drops first
    with pytest.raises(ReembedError, match="no embedding fingerprint"):
        drop_twin(world.chroma, "vault", world.client)                           # nor the main one
    assert {TWIN, "vault"} <= _names(world) and world.main.count() == 18

    write_sidecar(world.chroma, TWIN, make_sidecar(
        graph_spec(world.cfg), collection=TWIN, role="chunks", status="complete", dimension=DIM,
        count=0, written_by="test"))
    with pytest.raises(ReembedError, match="recorded as a 'chunks' collection"):
        drop_twin(world.chroma, TWIN, world.client)
    assert TWIN in _names(world) and read_sidecar(world.chroma, TWIN)["role"] == "chunks"

    # what graph_embed built, it drops: the collection, then the sidecar
    world.client.delete_collection(TWIN)
    delete_sidecar(world.chroma, TWIN)
    _build(world)
    assert drop_twin(world.chroma, TWIN, world.client) is True
    assert TWIN not in _names(world) and read_sidecar(world.chroma, TWIN) is None
    assert drop_twin(world.chroma, TWIN, world.client) is False


def test_a_chunks_reembed_may_not_take_the_graph_suffix(world):
    backend = FakeBackend()
    with pytest.raises(ReembedError, match="reserved for graph mode's seed index"):
        rebuild_into(world.cfg, TWIN, _embedder(world.cfg, backend), client=world.client,
                     progress=_quiet)
    assert backend.calls == [] and TWIN not in _names(world)


@pytest.mark.parametrize("target", ["other__graph", "vault_graph", "vault"])
def test_a_graph_build_may_only_target_the_derived_name(world, target):
    backend = FakeBackend()
    with pytest.raises(ReembedError, match=r"a graph index is always 'vault__graph'"):
        rebuild_into(world.cfg, target, _embedder(world.cfg, backend), client=world.client,
                     progress=_quiet, role="graph")
    assert backend.calls == [] and world.main.count() == 18


def test_a_graph_collection_is_verified_document_side(world):
    emb = _embedder(world.cfg)                       # G2: query_prefix "q: " != doc_prefix "d: "
    sidecar = _build(world, embedder=emb)
    assert sidecar["verified"]["min_cosine"] >= 0.99
    assert all(t.startswith("d: ") for t in emb.backend.calls[-1])               # the sample check
    col = world.client.get_collection(TWIN)
    assert verify_sample(emb, col, role="graph", sample_size=4, min_cosine=0.99)["sample"] == 4
    with pytest.raises(StampError, match="do not match their stored vectors"):
        verify_sample(emb, col, role="hype", sample_size=4, min_cosine=0.99)     # query-side would fail


def test_roles_and_role_for_know_graph(world):
    assert ROLES == ("chunks", "hype", "graph") and GRAPH_SUFFIX == "__graph"
    assert role_for(world.cfg, TWIN) == "graph"
    assert role_for(world.cfg, "vault") == "chunks"
    assert role_for(world.cfg, "vault__hype") == role_for(world.cfg, "hype_questions") == "hype"
    write_sidecar(world.chroma, TWIN, make_sidecar(
        graph_spec(world.cfg), collection=TWIN, role="graph", status="building", dimension=DIM,
        count=0, written_by="test"))
    assert read_sidecar(world.chroma, TWIN)["role"] == "graph"                   # a valid sidecar


class _Spy:
    """The real collection, with every get() recorded."""

    def __init__(self, col):
        self.col, self.calls = col, []

    def count(self):
        return self.col.count()

    def get(self, **kw):
        got = self.col.get(**kw)
        self.calls.append((kw, len(got["ids"])))
        return got


def test_iter_ids_with_where_pages_only_matching_rows(world):
    spy = _Spy(world.main)
    ids = list(iter_ids(spy, 3, where={"file_type": "canvas"}))
    assert sorted(ids) == [i for i, _ in world.canvas] and len(ids) == len(set(ids))
    assert [n for _, n in spy.calls] == [3, 3, 2]                                # until a short page
    assert all(kw == {"where": {"file_type": "canvas"}, "limit": 3, "offset": 3 * k, "include": []}
               for k, (kw, _) in enumerate(spy.calls))
    spy = _Spy(world.main)
    assert len(list(iter_ids(spy, 4, where={"file_type": "canvas"}))) == 8
    assert [n for _, n in spy.calls] == [4, 4, 0]                                # an exact multiple
    assert sorted(iter_ids(world.main, 3)) == sorted(i for i, _ in world.md + world.canvas)


# -------------------------------------------------------------- the verdict ----

def test_twin_status_walks_missing_building_mismatch_stale_ready(world):
    def status(cfg=None):
        return twin_status(cfg or world.cfg, client=world.client)

    off = status(_config(world.root, provider=None))
    assert (off["verdict"], off["twin"], off["main"]["chunks"]) == ("off", None, 8)
    s = status()
    assert (s["verdict"], s["twin"], s["collection"], s["configured"]) == ("missing", None, TWIN, "g2")
    assert "has not been built" in s["reason"] and "graph_embed" in s["reason"]
    assert s["main"] == {"collection": "vault", "embedder": "local · fake/main", "chunks": 8}

    _build(world, limit=3)
    s = status()
    assert s["verdict"] == "building" and (s["twin"]["count"], s["twin"]["expected"]) == (3, 8)

    _build(world)
    s = status()
    assert s["verdict"] == "ready" and s["twin"]["diff"] == [] and s["twin"]["status"] == "complete"
    assert (s["twin"]["only_main"], s["twin"]["only_twin"], s["twin"]["exists"]) == (0, 0, True)
    assert s["twin"]["embedder"] == "g2 · fake/g2" and s["twin"]["completed_at"]
    off = status(_config(world.root, provider=None))
    assert off["verdict"] == "off" and off["twin"]["exists"] is True             # off, twin or not

    changed = _config(world.root, providers={"g2": {**G2, "model": "fake/g2-v2"}})
    s = status(changed)
    assert s["verdict"] == "mismatch" and s["twin"]["diff"] == ["model"]

    world.main.delete(ids=["c00"])
    world.main.add(ids=["c99"], embeddings=[_vec("c99").tolist()], documents=["new"],
                   metadatas=[{"file_type": "canvas"}])
    s = status()
    assert s["verdict"] == "stale" and (s["twin"]["only_main"], s["twin"]["only_twin"]) == (1, 1)

    s = status(_config(world.root, provider="nope"))
    assert s["verdict"] == "error" and "not in embedding.providers" in s["reason"]
    delete_sidecar(world.chroma, TWIN)
    s = status()
    assert s["verdict"] == "error" and "no embedding fingerprint" in s["reason"]


def test_twin_status_never_creates_the_store(world):
    nowhere = world.root / "nowhere"
    s = twin_status(_config(world.root, chroma_dir=nowhere))
    assert (s["verdict"], s["main"]["chunks"], s["twin"]) == ("missing", 0, None)
    assert not nowhere.exists()


def test_twin_id_diff_is_both_directions(world):
    _build(world)
    twin = world.client.get_collection(TWIN)
    assert twin_id_diff(world.main, twin, "canvas") == (set(), set())
    twin.delete(ids=["c01"])
    world.main.delete(ids=["c02"])
    assert twin_id_diff(world.main, twin, "canvas") == ({"c01"}, {"c02"})


# ------------------------------------------------------------------ the CLI ----

def _cli(monkeypatch, cfg, *argv):
    import main
    monkeypatch.setattr(main, "load_config", lambda path=None: cfg)
    monkeypatch.setattr(main, "_bootstrap_logging", lambda c: None)
    monkeypatch.setattr(Embedder, "from_config", classmethod(lambda cls, c, spec=None: _embedder(c)))
    args = main.build_parser().parse_args(["graph-embed", *argv])
    args.func(args)


@pytest.mark.parametrize("provider, argv, fragment", [
    (None, [], "graph.embedding_provider is null"),
    ("", [], "graph.embedding_provider is null"),                  # empty = unset
    ("nope", [], "is not in embedding.providers"),
    ("local", [], "is the main collection's embedder"),
    ("g2", ["--limit", "0"], "--limit must be a positive integer"),
])
def test_the_cli_reports_bad_configs_as_error_lines_exit_2(world, monkeypatch, capsys,
                                                          provider, argv, fragment):
    with pytest.raises(SystemExit) as e:
        _cli(monkeypatch, _config(world.root, provider=provider), *argv)
    assert e.value.code == 2 and fragment in capsys.readouterr().err
    assert TWIN not in _names(world) and world.main.count() == 18


def test_the_cli_reports_a_missing_api_key_as_an_error_line_exit_2(world, monkeypatch, capsys):
    import main
    from src.embeddings import graph_twin
    keyed = _config(world.root, providers={"g2": {**G2, "api_key_env": "NOETRIX_TEST_GRAPH_KEY"}})
    monkeypatch.delenv("NOETRIX_TEST_GRAPH_KEY", raising=False)
    # the type the real build raises: Config.require_secret's plain RuntimeError
    with pytest.raises(RuntimeError, match="Missing required secret 'NOETRIX_TEST_GRAPH_KEY'"):
        Embedder.from_config(keyed, spec=graph_spec(keyed))

    monkeypatch.setattr(main, "load_config", lambda path=None: keyed)
    monkeypatch.setattr(main, "_bootstrap_logging", lambda c: None)
    args = main.build_parser().parse_args(["graph-embed"])
    with pytest.raises(SystemExit) as e:
        args.func(args)
    assert e.value.code == 2 and capsys.readouterr().err.startswith(
        "ERROR: g2 · fake/g2 needs its API key: Missing required secret 'NOETRIX_TEST_GRAPH_KEY'")
    assert TWIN not in _names(world) and world.main.count() == 18

    # with the key set, a RuntimeError from the build itself is still a traceback
    def failing_build(*args, **kwargs):
        raise RuntimeError("'vault__graph' holds 3 ids but 8 were expected")
    monkeypatch.setenv("NOETRIX_TEST_GRAPH_KEY", "test-key-value")
    monkeypatch.setattr(graph_twin, "build_twin", failing_build)
    with pytest.raises(RuntimeError, match="holds 3 ids but 8 were expected"):
        args.func(args)


def test_an_empty_embedder_name_reads_as_null(world):
    blank = _config(world.root, provider="")
    assert graph_spec(blank) is None
    s = twin_status(blank, client=world.client)
    assert (s["configured"], s["verdict"]) == (None, "off")


def test_the_cli_builds_and_says_ok_complete(world, monkeypatch, capsys):
    _cli(monkeypatch, world.cfg)
    out = capsys.readouterr().out
    assert "OK 'vault__graph': 8 of 8 chunks" in out and "status complete" in out
    assert "Restart :8051" in out
    _cli(monkeypatch, world.cfg, "--limit", "2")
    out = capsys.readouterr().out
    assert "PARTIAL 'vault__graph': 2 of 8 chunks" in out and "left 'building'" in out
