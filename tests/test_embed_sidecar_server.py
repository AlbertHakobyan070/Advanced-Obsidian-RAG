"""The GPU embedding sidecar (embed-sidecar/server.py): an OpenAI-compatible
/v1/embeddings in front of EmbeddingGemma 2.

Run:  python -m pytest tests/ -q

No GPU and no model: the service wraps a fake whose encode() returns known
vectors and records what it was handed. Requests go over a real socket to the
real handler, so the wire format (including the base64 form the OpenAI SDK asks
for by default) is what is tested.
"""
import base64
import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("embed_sidecar_server", ROOT / "embed-sidecar" / "server.py")
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)

NATIVE = 8
MODEL = "google/embeddinggemma-2"


class FakeModel:
    """Unit-length rows whose first value tells which text it was for."""

    def __init__(self):
        self.calls = []

    def get_embedding_dimension(self):
        return NATIVE

    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar):
        self.calls.append({"texts": list(texts), "batch_size": batch_size,
                           "normalize": normalize_embeddings})
        rows = []
        for t in texts:
            v = np.arange(1, NATIVE + 1, dtype=np.float32) + len(t)
            rows.append(v / np.linalg.norm(v))
        return np.stack(rows)


@pytest.fixture
def live():
    """(model, call) with a server running on a free port; call(path, body) -> (status, json)."""
    model = FakeModel()
    service = server.EmbedService(model, MODEL, "cuda", max_seq_length=1024, truncate_dim=None,
                                  batch_size=16, versions={"torch": "x"})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(service))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(path, body=None):
        req = urllib.request.Request(base + path, method="GET" if body is None else "POST",
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    yield model, call
    httpd.shutdown()
    httpd.server_close()


def test_health_reports_the_cap_and_settings_it_embeds_with(live):
    _, call = live
    status, body = call("/health")
    assert status == 200
    assert (body["model"], body["max_seq_length"], body["dimension"], body["dtype"]) == (MODEL, 1024, NATIVE, "float32")
    assert body["batch_size"] == 16 and body["device"] == "cuda"


def test_embeddings_response_has_the_openai_shape_and_reports_the_cap(live):
    model, call = live
    status, body = call("/v1/embeddings", {"model": MODEL, "input": ["alpha", "bb"]})
    assert status == 200
    assert body["object"] == "list" and body["model"] == MODEL
    assert [d["index"] for d in body["data"]] == [0, 1]
    assert len(body["data"][0]["embedding"]) == NATIVE
    assert body["max_seq_length"] == 1024                     # the client checks this against its entry
    assert {"prompt_tokens", "total_tokens"} <= set(body["usage"])
    assert model.calls[0]["normalize"] is True and model.calls[0]["batch_size"] == 16


def test_the_text_reaches_the_model_unchanged_so_the_client_owns_the_prefixes(live):
    model, call = live
    call("/v1/embeddings", {"model": MODEL, "input": "task: search result | query: what is x"})
    assert model.calls[0]["texts"] == ["task: search result | query: what is x"]


def test_base64_encoding_round_trips_to_the_same_float32_values(live):
    _, call = live
    _, as_float = call("/v1/embeddings", {"model": MODEL, "input": ["alpha"]})
    _, as_b64 = call("/v1/embeddings", {"model": MODEL, "input": ["alpha"], "encoding_format": "base64"})
    decoded = np.frombuffer(base64.b64decode(as_b64["data"][0]["embedding"]), dtype="<f4")
    np.testing.assert_allclose(decoded, as_float["data"][0]["embedding"], rtol=1e-6)


def test_dimensions_truncates_and_renormalises(live):
    _, call = live
    _, full = call("/v1/embeddings", {"model": MODEL, "input": ["alpha"]})
    _, cut = call("/v1/embeddings", {"model": MODEL, "input": ["alpha"], "dimensions": 3})
    v = np.array(cut["data"][0]["embedding"])
    assert len(v) == 3 and abs(np.linalg.norm(v) - 1) < 1e-6
    head = np.array(full["data"][0]["embedding"])[:3]
    np.testing.assert_allclose(v, head / np.linalg.norm(head), rtol=1e-5)


def test_the_server_default_truncate_dim_applies_when_the_request_names_none():
    service = server.EmbedService(FakeModel(), MODEL, "cuda", 512, truncate_dim=4, batch_size=2)
    assert service.embed(["x"]).shape == (1, 4)
    assert service.embed(["x"], dimensions=6).shape == (1, 6)        # a request overrides it


@pytest.mark.parametrize("body, fragment", [
    ({"model": "some/other-model", "input": ["a"]}, "serves"),         # never substitute a model
    ({"model": MODEL, "input": []}, "non-empty list of strings"),
    ({"model": MODEL, "input": [[1, 2, 3]]}, "non-empty list of strings"),   # token arrays
    ({"model": MODEL, "input": ["a"], "dimensions": NATIVE + 1}, "between 1 and"),
    ({"model": MODEL, "input": ["a"], "dimensions": "512"}, "must be an integer"),
    ({"model": MODEL, "input": ["a"], "encoding_format": "int8"}, "encoding_format"),
])
def test_a_bad_request_is_a_400_naming_the_problem(live, body, fragment):
    model, call = live
    status, err = call("/v1/embeddings", body)
    assert status == 400 and fragment in err["error"]["message"]
    assert model.calls == []                                          # nothing was embedded


def test_unknown_routes_are_404_and_a_model_fault_is_a_500_not_a_silent_empty_answer(live):
    model, call = live
    assert call("/nope")[0] == 404
    model.encode = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("CUDA out of memory"))
    status, err = call("/v1/embeddings", {"model": MODEL, "input": ["a"]})
    assert status == 500 and "CUDA out of memory" in err["error"]["message"]


def test_the_cap_must_be_a_positive_integer_never_unbounded():
    with pytest.raises(ValueError, match="max_seq_length"):
        server.EmbedService(FakeModel(), MODEL, "cuda", 0, None, 16)
    with pytest.raises(ValueError, match="truncate_dim"):
        server.EmbedService(FakeModel(), MODEL, "cuda", 512, NATIVE + 1, 16)


def test_a_non_integer_environment_value_is_an_error(monkeypatch):
    monkeypatch.setenv("EMBED_MAX_SEQ_LENGTH", "lots")
    with pytest.raises(SystemExit, match="EMBED_MAX_SEQ_LENGTH"):
        server._env_int("EMBED_MAX_SEQ_LENGTH", 512)
    monkeypatch.delenv("EMBED_MAX_SEQ_LENGTH")
    assert server._env_int("EMBED_MAX_SEQ_LENGTH", 512) == 512         # the bounded default


# --- rerank mode (SIDECAR_TASK=rerank) ------------------------------------------------

RERANKER = "BAAI/bge-reranker-v2-m3"


class FakeCrossEncoder:
    """Scores each (query, document) pair by the document's length, and records the pairs."""

    def __init__(self):
        self.calls = []

    def predict(self, pairs, batch_size, show_progress_bar):
        self.calls.append({"pairs": list(pairs), "batch_size": batch_size})
        return np.array([float(len(d)) for _, d in pairs], dtype=np.float32)


@pytest.fixture
def live_rerank():
    """(model, base_url, call) with a rerank server on a free port."""
    model = FakeCrossEncoder()
    service = server.RerankService(model, RERANKER, "cuda", max_seq_length=512, batch_size=16)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(service))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(path, body=None):
        req = urllib.request.Request(base + path, method="GET" if body is None else "POST",
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    yield model, base, call
    httpd.shutdown()
    httpd.server_close()


def test_rerank_returns_every_document_best_first_with_its_index(live_rerank):
    model, _, call = live_rerank
    status, body = call("/v1/rerank", {"model": RERANKER, "query": "q", "documents": ["bb", "a", "cccc"]})
    assert status == 200 and body["model"] == RERANKER
    assert body["results"] == [{"index": 2, "relevance_score": 4.0}, {"index": 0, "relevance_score": 2.0},
                               {"index": 1, "relevance_score": 1.0}]
    assert model.calls[0] == {"pairs": [("q", "bb"), ("q", "a"), ("q", "cccc")], "batch_size": 16}
    assert call("/health")[1]["task"] == "rerank"


def test_rerank_top_n_keeps_the_best_n_and_the_bare_model_name_is_accepted(live_rerank):
    _, _, call = live_rerank
    status, body = call("/v1/rerank", {"model": "bge-reranker-v2-m3", "query": "q",
                                       "documents": ["bb", "a", "cccc"], "top_n": 1})
    assert status == 200 and [r["index"] for r in body["results"]] == [2]


@pytest.mark.parametrize("body, fragment", [
    ({"model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "query": "q", "documents": ["a"]}, "serves"),
    ({"model": RERANKER, "query": " ", "documents": ["a"]}, "query"),
    ({"model": RERANKER, "query": "q", "documents": []}, "non-empty list of strings"),
    ({"model": RERANKER, "query": "q", "documents": [3]}, "non-empty list of strings"),
    ({"model": RERANKER, "query": "q", "documents": ["a"], "top_n": 0}, "top_n"),
    ({"model": RERANKER, "query": "q", "documents": ["a"], "top_n": True}, "top_n"),
])
def test_a_bad_rerank_request_is_a_400_and_scores_nothing(live_rerank, body, fragment):
    model, _, call = live_rerank
    status, err = call("/v1/rerank", body)
    assert status == 400 and fragment in err["error"]["message"]
    assert model.calls == []


def test_a_rerank_server_has_no_embeddings_route_and_a_short_score_list_is_a_500(live_rerank):
    model, _, call = live_rerank
    assert call("/v1/embeddings", {"model": RERANKER, "input": ["a"]})[0] == 404
    model.predict = lambda pairs, **k: np.array([1.0])          # one score for two documents
    status, err = call("/v1/rerank", {"model": RERANKER, "query": "q", "documents": ["a", "b"]})
    assert status == 500 and "refusing to rank" in err["error"]["message"]


def test_the_projects_http_reranker_reads_this_server(live_rerank):
    """The contract end to end: rerank_mode http, configured as config.yaml names the model."""
    from src.retrieval.reranker import Reranker
    from src.retrieval.retriever import RetrievedDoc

    _, base, _ = live_rerank
    rr = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", mode="http",
                  http_url=base + "/v1", http_model="bge-reranker-v2-m3")
    docs = [RetrievedDoc(id=i, text=t, metadata={}) for i, t in (("a", "xx"), ("b", "x"), ("c", "xxx"))]
    top = rr.rerank("q", docs, top_k=2)
    assert [d.id for d in top] == ["c", "a"] and top[0].rerank_score == 3.0
