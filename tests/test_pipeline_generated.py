"""retrieval.generated_mode in the pipeline (route-then-cite S1, spec §5.1, §5.2).

The pipeline resolves the mode (config, default drop; or per call), hands it to
EVERY retrieve() call, echoes it, and stamps doc.debug["provenance"] on what it
returns. At build, under drop or route, it refuses to start while a stored row
the rule calls generated is not stamped generated: the dense lanes filter on the
stored key, so such a row would slip through them.

No model loads: the search tests use stand-ins; the build tests use the real
Embedder around a fake sentence_transformers module, the real HybridRetriever,
a pickled payload and a real Chroma collection under tmp_path.
"""
import pickle
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from fastapi.testclient import TestClient

import serve_api as S
from src.pipeline import RAGPipeline
from src.retrieval.decompose import Decomposer
from src.retrieval.retriever import RetrievedDoc
from src.utils.config_loader import Config
from src.utils.provenance import UnstampedProvenanceError, provenance_rule

chromadb = pytest.importorskip("chromadb")

ROOT = Path(__file__).resolve().parents[1]
DIM = 16
PRIMARY = {"source_file": "Notes\\p.md", "provenance": "authored", "file_type": "note"}
GENERATED = {"source_file": "Maps\\g.md", "provenance": "generated", "file_type": "note"}
UNSTAMPED = [{"source_file": "Maps\\a.md", "provenance": "authored", "file_type": "note"}]


# ---- search(): stand-ins (copies of tests/test_lane_selection.py's and
# tests/test_decompose.py's helpers: private test helpers are not imported across files)

class _Reranker:
    mode = "none"
    model_name = "test/reranker"
    max_length = 512
    instruction = None

    @staticmethod
    def rerank(question, docs, top_k=None, mode=None, instruction=None):
        return docs[:top_k]

    @staticmethod
    def scoring_query(query, mode, instruction=None):
        return query


class _Hyde:
    @staticmethod
    def expand_with_info(question, enabled=None):
        return question, "off"

    @staticmethod
    def code_intent_signal(question):
        return None


class _RecordingRetriever:
    """Records every retrieve() call's kwargs; returns fresh copies of `docs`."""
    dense_top_k = sparse_top_k = 20
    omnisearch = None
    hype_enabled = False
    lane_weights: dict = {}
    metadata_boost = True
    rrf_k = 60

    def __init__(self, docs=()):
        self.calls, self.docs = [], list(docs)

    @property
    def kwargs(self):
        return self.calls[-1]

    def retrieve(self, query, **kwargs):
        self.calls.append(kwargs)
        return [RetrievedDoc(d.id, d.text, dict(d.metadata)) for d in self.docs]

    def _get_collection(self):
        return None


class _LLM:
    model = "stub"

    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, user, temperature=None, max_tokens=None):
        return SimpleNamespace(text=self.reply)


def _pipeline(retriever, generator=None, **kw):
    return RAGPipeline(retriever=retriever, reranker=_Reranker(), hyde=_Hyde(),
                       generator=generator, rerank_top_k=3, **kw)


def test_search_passes_the_resolved_mode_and_echoes_it():
    retr = _RecordingRetriever()
    assert _pipeline(retr).generated_mode == "off"                # a direct construction's default
    _, info = _pipeline(retr, generated_mode="drop").search("q")
    assert retr.kwargs["generated_mode"] == "drop" and info["generated_mode"] == "drop"
    _, info = _pipeline(retr, generated_mode="drop").search("q", generated_mode="off")
    assert retr.kwargs["generated_mode"] == "off"                 # in-process only; the API refuses it
    assert info["generated_mode"] == "off"
    with pytest.raises(ValueError, match="generated_mode"):
        _pipeline(retr).search("q", generated_mode="bogus")


def test_every_branch_of_a_decomposed_search_passes_the_mode():
    """retrieve() defaults to off, so a branch that omitted the mode would let
    generated text back in as evidence."""
    retr = _RecordingRetriever([RetrievedDoc("a", "t", {"source_file": "Notes\\a.md"})])
    rag = _pipeline(retr, generated_mode="drop",
                    decomposer=Decomposer(_LLM('["part one", "part two"]'), enabled=True))
    _, info = rag.search("q")
    assert info["decompose"]["status"] == "split"
    assert [c["generated_mode"] for c in retr.calls] == ["drop", "drop", "drop"]


def test_query_forwards_the_mode():
    retr = _RecordingRetriever()
    generator = SimpleNamespace(
        generate=lambda q, docs, max_tokens=None: SimpleNamespace(retrieval=None))
    rag = _pipeline(retr, generator, generated_mode="drop")
    rag.query("q")
    assert retr.kwargs["generated_mode"] == "drop"
    answer = rag.query("q", generated_mode="off")
    assert retr.kwargs["generated_mode"] == "off" and answer.retrieval["generated_mode"] == "off"


def test_returned_docs_carry_their_provenance():
    """Stamped on the FINAL list: the parent swap here hands back new objects and
    the neighbour lane adds one, so a stamp made before either would be missing."""
    rule = provenance_rule(Config({"provenance": {"generated_paths": ["Maps/"]}}, ROOT))
    retr = _RecordingRetriever([
        RetrievedDoc("n1", "child", {"file_type": "note", "source_file": "Notes\\n.md", "parent_id": "N"}),
        RetrievedDoc("m1", "map", {"file_type": "note", "source_file": "Maps\\m.md",
                                   "provenance": "authored"})])
    parent = SimpleNamespace(enabled=True, apply=lambda docs: (
        [RetrievedDoc(d.id, "PARENT", dict(d.metadata)) for d in docs], len(docs), 0))
    neighbor = SimpleNamespace(enabled=True, apply=lambda docs, col: (
        docs + [RetrievedDoc("p2", "page", {"file_type": "pdf", "source_file": "Books\\b.pdf"})], 1))
    rag = _pipeline(retr, provenance_map=rule, parent_ctx=parent, neighbor_ctx=neighbor)
    top, info = rag.search("q")
    assert info["parent_swaps"] == 2 and info["neighbors_added"] == 1
    assert {d.id: d.debug["provenance"] for d in top} == {
        "n1": "authored", "m1": "generated", "p2": "source"}


# ---- the build: the real lifespan, the real retriever (copies of
# tests/test_embedding_guard.py's fake_st and _cfg)

@pytest.fixture
def fake_st(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, *args, **kwargs):
            return np.stack([np.ones(DIM) / np.sqrt(DIM) for _ in texts])

        def get_embedding_dimension(self):
            return DIM

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def _cfg(tmp_path, collection="obsidian_vault", **sections):
    data = {"paths": {"chunks_file": "chunks.jsonl", "chroma_dir": "chroma_db",
                      "bm25_index": "bm25.pkl", "collection_name": collection}}
    data.update(sections)
    return Config(data, tmp_path)


def _setup(tmp_path, monkeypatch, mode, metas=(), chroma_only=(), generated_paths=("Maps/",),
           bm25=True):
    """A config (no retrieval.generated_mode when mode is None), the stand-in
    builders, and with bm25 the pickled four-key payload plus a real collection
    holding the payload's rows and the Chroma-only ones."""
    import src.pipeline as P
    tmp_path.mkdir(parents=True, exist_ok=True)
    sections = {"embedding": {"provider": "local", "local_model": "fake/model-a"},
                "provenance": {"generated_paths": list(generated_paths)}}
    if mode is not None:
        sections["retrieval"] = {"generated_mode": mode}
    cfg = _cfg(tmp_path, **sections)
    if bm25:
        metas = list(metas)
        ids = [f"p{i}" for i in range(len(metas))]
        with open(cfg.path("paths.bm25_index"), "wb") as f:
            pickle.dump({"ids": ids, "documents": ["text"] * len(ids), "metadatas": metas,
                         "bm25": None}, f)
        rows = list(zip(ids, metas)) + list(chroma_only)
        col = chromadb.PersistentClient(path=str(cfg.path("paths.chroma_dir"))).get_or_create_collection(
            cfg.get("paths.collection_name"), metadata={"hnsw:space": "cosine"})
        col.add(ids=[i for i, _ in rows], documents=["text"] * len(rows),
                metadatas=[m for _, m in rows],
                embeddings=[[float(j == n % DIM) for j in range(DIM)] for n in range(len(rows))])
    stand_in = SimpleNamespace(from_config=lambda *args, **kwargs: SimpleNamespace())
    for name in ("LLMClient", "HyDE", "Generator", "GraphExpander"):
        monkeypatch.setattr(P, name, stand_in)
    monkeypatch.setattr(P, "configure_logging", lambda **kwargs: None)
    return cfg


def _boot_with_payload(tmp_path, monkeypatch, mode, **kw):
    cfg = _setup(tmp_path, monkeypatch, mode, **kw)
    return lambda: RAGPipeline.from_config(cfg)


def _client(tmp_path, monkeypatch, mode, **kw):
    cfg = _setup(tmp_path, monkeypatch, mode, **kw)
    monkeypatch.setattr(S, "load_config", lambda *args, **kwargs: cfg)
    return TestClient(S.app, base_url="http://127.0.0.1")    # the request guard refuses Host: testserver


def test_from_config_refuses_unstamped_generated_rows_under_drop(tmp_path, monkeypatch, fake_st):
    rag_boot = _boot_with_payload(tmp_path, monkeypatch, mode="drop",
                                  metas=[{"source_file": "Maps/a.md", "provenance": "authored"}],
                                  generated_paths=["Maps/"])
    with pytest.raises(UnstampedProvenanceError, match="stamp-provenance") as exc:
        rag_boot()
    msg = str(exc.value)
    assert "Maps/a.md (1)" in msg
    assert "--dry-run" in msg and "--yes" in msg and "python rebuild_bm25.py" in msg
    # the Chroma write keeps no .bak: the backup heads the fix
    assert msg.index("back up chroma_db and the bm25 index first") < msg.index("stamp-provenance")
    assert "documents/delete" not in msg and "remove exactly those ids" not in msg   # no Chroma-only row


def test_the_refusal_names_ten_files_then_the_fix(tmp_path, monkeypatch, fake_st):
    metas = [{"source_file": f"Maps/f{i:02}.md", "provenance": "authored"} for i in range(12)]
    with pytest.raises(UnstampedProvenanceError) as exc:
        _boot_with_payload(tmp_path, monkeypatch, mode="drop", metas=metas)()
    msg = str(exc.value)
    assert sum(f"Maps/f{i:02}.md" in msg for i in range(12)) == 10 and "2 more" in msg
    assert max(msg.index(f"Maps/f{i:02}.md") for i in range(10)) < msg.index("stamp-provenance")


def test_a_chroma_only_unstamped_generated_row_refuses_start(tmp_path, monkeypatch, fake_st):
    # the collection holds "o1"; the payload (every chunk JSONL) does not
    rag_boot = _boot_with_payload(tmp_path / "a", monkeypatch, mode="drop", metas=[PRIMARY],
                                  chroma_only=[("o1", {"source_file": "Maps\\o.md",
                                                       "provenance": "authored"})],
                                  generated_paths=["Maps/"])
    with pytest.raises(UnstampedProvenanceError, match="remove exactly those ids") as exc:
        rag_boot()
    msg = str(exc.value)
    assert "Maps\\o.md" in msg and "in Chroma but not in the bm25 index" in msg
    assert "o1 (Maps\\o.md)" in msg                       # the row id, with its file
    # a row not in the bm25 index may still be in a JSONL (an append with no rebuild), and
    # the JSONLs are primary data: the backup and the stamp come first, then the removal
    # of exactly those ids, never of their source
    assert msg.index("back up chroma_db and the bm25 index first") < msg.index("stamp-provenance --dry-run") \
        < msg.index("rebuild_bm25.py") < msg.index("still reported after the stamp") \
        < msg.index("back up chroma_db, then remove exactly those ids")
    assert "Do not delete their source" in msg and "documents/delete" not in msg
    # the same Chroma-only row under Notes\ is not generated by the rule: it starts ready
    rag = _boot_with_payload(tmp_path / "b", monkeypatch, mode="drop", metas=[PRIMARY, GENERATED],
                             chroma_only=[("o1", {"source_file": "Notes\\o.md",
                                                  "provenance": "authored"})])()
    assert (rag.generated_mode, rag.unstamped, rag.generated_rows) == ("drop", {}, 1)


def test_the_chroma_only_fix_names_ten_row_ids_then_the_rest():
    from src.pipeline import _unstamped_message
    rows = [(f"o{i:02}", "Maps\\o.md") for i in range(12)]
    msg = _unstamped_message({"Maps\\o.md [in Chroma but not in the bm25 index]": 12}, rows)
    assert sum(f"o{i:02} (Maps\\o.md)" in msg for i in range(12)) == 10 and "; and 2 more." in msg


def test_a_configured_off_keeps_serving_with_unstamped_rows():
    """The live state once this merges: config.yaml says off and the store is not
    stamped yet. :8051 must keep answering; only a per-call drop or route refuses."""
    retr = _RecordingRetriever()
    rag = _pipeline(retr, generated_mode="off")
    rag.unstamped = {"Maps\\a.md": 1}
    _, info = rag.search("q")
    assert info["generated_mode"] == "off" and retr.kwargs["generated_mode"] == "off"
    with pytest.raises(ValueError, match="stamp-provenance"):
        rag.search("q", generated_mode="drop")


def test_off_starts_and_a_per_call_drop_then_refuses(tmp_path, monkeypatch, fake_st):
    rag = _boot_with_payload(tmp_path, monkeypatch, mode="off", metas=UNSTAMPED)()
    assert rag.generated_mode == "off" and rag.unstamped == {"Maps\\a.md": 1}
    for mode in ("drop", "route"):
        # raised before HyDE: the stand-in here has no methods at all
        with pytest.raises(ValueError, match="stamp-provenance"):
            rag.search("q", generated_mode=mode)


def test_a_bare_yaml_off_is_refused_naming_the_quote(tmp_path, monkeypatch, fake_st):
    with pytest.raises(ValueError, match=r"retrieval\.generated_mode.*quote it"):
        _boot_with_payload(tmp_path, monkeypatch, mode=False, bm25=False)()


def test_a_fresh_install_starts_ready(tmp_path, monkeypatch, fake_st):     # Review Focus 2
    with _client(tmp_path, monkeypatch, mode=None, bm25=False) as client:   # key missing -> drop
        body = client.get("/health").json()
        rag = S._STATE["rag"]
    assert body["state"] == "ready"
    assert (rag.generated_mode, rag.unstamped, rag.generated_rows) == ("drop", {}, 0)


def test_the_service_comes_up_failed_naming_the_fix(tmp_path, monkeypatch, fake_st):
    with _client(tmp_path, monkeypatch, mode="drop", metas=UNSTAMPED) as client:
        body = client.get("/health").json()
        refused = client.post("/search", json={"q": "anything"})
    assert body["state"] == "failed" and "stamp-provenance" in body["error"]
    assert body["error"].startswith("UnstampedProvenanceError:")
    assert "stamp-provenance" in body["hint"] and "model ids" not in body["hint"]
    assert refused.status_code == 503
