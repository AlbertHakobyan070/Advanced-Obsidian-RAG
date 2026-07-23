"""Warm API comparison/provenance contract tests.

These tests stay fully offline: no model, Chroma collection, or generation
backend is constructed. They protect the agent-facing query-tree primitives
and the explicit config-only baseline used for fair branch comparisons.
"""
from pathlib import Path
from types import SimpleNamespace

import serve_api
from src.llm.llm_client import LLMClient
from src.pipeline import RAGPipeline
from src.retrieval.reranker import RERANK_MODES, RerankerExecutionError
from src.utils.config_loader import Config

ROOT = Path(__file__).resolve().parents[1]


def test_retrieval_cache_key_reuses_evidence_for_provider_only_branches():
    first = serve_api.CompareBranchIn(
        id="minimax",
        label="MiniMax",
        preset="concept",
        provider="minimax",
        model="MiniMax-M3",
    )
    second = serve_api.CompareBranchIn(
        id="local",
        label="Local",
        preset="concept",
        provider="free_local",
        model="different-model",
    )
    changed_retrieval = serve_api.CompareBranchIn(
        id="lexical",
        preset="concept",
        provider="minimax",
        rerank="lexical",
    )

    assert (
        serve_api._retrieval_cache_key(first)
        == serve_api._retrieval_cache_key(second)
    )
    assert (
        serve_api._retrieval_cache_key(first)
        != serve_api._retrieval_cache_key(changed_retrieval)
    )


def test_comparison_summary_uses_ids_and_ranks_not_raw_scores():
    branches = [
        {
            "id": "base",
            "sources": [
                {"id": "s1", "score": 9.8},
                {"id": "s2", "score": -1.2},
                {"id": "s3", "score": 0.1},
            ],
            "retrieval_error": False,
        },
        {
            "id": "lexical",
            "sources": [
                {"id": "s2", "score": 0.75},
                {"id": "s1", "score": 0.5},
                {"id": "s4", "score": 0.25},
            ],
            "retrieval_error": False,
        },
        {
            # A generation failure must not discard successful retrieval from
            # the evidence comparison.
            "id": "other_llm",
            "sources": [
                {"id": "s1", "score": 100.0},
                {"id": "s2", "score": 50.0},
                {"id": "s5", "score": 1.0},
            ],
            "retrieval_error": False,
            "error": "Generation failed",
        },
        {
            "id": "broken_retrieval",
            "sources": [],
            "retrieval_error": True,
        },
    ]

    summary = serve_api._comparison_summary(branches)

    assert summary["successful_branches"] == ["base", "lexical", "other_llm"]
    assert summary["common_source_ids"] == ["s1", "s2"]
    assert summary["unique_source_ids"] == {
        "base": ["s3"],
        "lexical": ["s4"],
        "other_llm": ["s5"],
    }
    assert summary["membership"]["s1"] == ["base", "lexical", "other_llm"]
    assert summary["ranks"]["s1"] == {
        "base": 1,
        "lexical": 2,
        "other_llm": 1,
    }
    assert summary["rank_spread"]["s1"]["spread"] == 1
    assert summary["pairwise"][0] == {
        "left": "base",
        "right": "lexical",
        "depth": 3,
        "overlap": 2,
        "overlap_rate": 0.6667,
        "jaccard": 0.5,
    }
    assert "Raw scores are intentionally not compared" in summary["note"]


def test_effective_parent_evidence_is_distinct_but_origin_is_preserved():
    branches = [
        {
            "id": "child",
            "sources": [{"id": "chunk-1", "origin_id": "chunk-1"}],
            "retrieval_error": False,
        },
        {
            "id": "parent",
            "sources": [{"id": "parent:section-1", "origin_id": "chunk-1"}],
            "retrieval_error": False,
        },
    ]

    summary = serve_api._comparison_summary(branches)

    assert summary["common_source_ids"] == []
    assert summary["origin_membership"]["chunk-1"] == ["child", "parent"]
    assert summary["origin_ranks"]["chunk-1"] == {"child": 1, "parent": 1}


def test_source_identity_tracks_parent_and_marks_live_lookup_unavailable():
    parent_doc = SimpleNamespace(
        id="child-1",
        text="full parent text",
        metadata={},
        debug={"parent_swap": "section-1"},
        rerank_score=1.0,
        score=0.0,
        source_label="",
    )
    live_doc = SimpleNamespace(
        id="omni::Folder/Note.md",
        text="query-shaped live excerpt",
        metadata={"live": True, "source_file": "Folder/Note.md"},
        debug={},
        rerank_score=0.5,
        score=0.0,
        source_label="",
    )

    parent, live = serve_api._sources_out(
        [parent_doc, live_doc], [], include_text=100, cap=None
    )

    assert parent.id == "parent:section-1"
    assert parent.origin_id == "child-1"
    assert parent.lookup_available is True
    assert live.id.startswith("live:")
    assert "/" not in live.id
    assert live.origin_id == "omni::Folder/Note.md"
    assert live.lookup_available is False


def test_parent_evidence_id_is_dereferenceable(monkeypatch):
    parent_ctx = SimpleNamespace(
        _load=lambda: {
            "section-1": {
                "parent_id": "section-1",
                "source_file": "notes.md",
                "heading": "Overview",
                "text": "full parent section",
            }
        }
    )
    monkeypatch.setitem(
        serve_api._STATE,
        "rag",
        SimpleNamespace(parent_ctx=parent_ctx),
    )

    result = serve_api.chunk("parent:section-1", include_text=100)

    assert result["id"] == "parent:section-1"
    assert result["kind"] == "parent_section"
    assert result["text"] == "full parent section"
    assert result["metadata"] == {
        "source_file": "notes.md",
        "heading": "Overview",
    }


def test_provider_catalog_reports_key_type_without_exposing_secret(monkeypatch):
    cfg = Config(
        {
            "providers": {
                # Reserved legacy protocol names are not request-selectable
                # registry aliases and must not appear in the catalog.
                "openai": {
                    "kind": "openai",
                    "model": "legacy",
                },
                "minimax": {
                    "label": "MiniMax M3 Token Plan",
                    "description": "Subscription key",
                    "kind": "anthropic",
                    "base_url": "https://api.minimax.io/anthropic",
                    "model": "MiniMax-M3",
                    "api_key_env": "TEST_MINIMAX_TOKEN_PLAN_KEY",
                    "api_key_prefix": "sk-cp-",
                },
                "local_proxy": {
                    "kind": "openai",
                    "base_url": "http://127.0.0.1:3001/v1",
                    "model": "auto",
                    "api_key_env": None,
                },
                "optional_proxy": {
                    "kind": "openai",
                    "model": "auto",
                    "api_key_env": "TEST_OPTIONAL_PROXY_KEY",
                    "api_key_optional": True,
                },
                "optional_typed": {
                    "kind": "openai",
                    "model": "auto",
                    "api_key_env": "TEST_OPTIONAL_TYPED_KEY",
                    "api_key_optional": True,
                    "api_key_prefix": "typed-",
                },
            }
        },
        ROOT,
    )
    monkeypatch.setitem(serve_api._STATE, "cfg", cfg)
    monkeypatch.setenv("TEST_MINIMAX_TOKEN_PLAN_KEY", "sk-api-wrong-kind")
    monkeypatch.delenv("TEST_OPTIONAL_PROXY_KEY", raising=False)
    monkeypatch.setenv("TEST_OPTIONAL_TYPED_KEY", "wrong-present-key")

    rows = {row["name"]: row for row in serve_api._provider_catalog()}

    assert not (set(rows) & set(LLMClient.RESERVED_PROVIDERS))
    assert rows["minimax"]["key_present"] is True
    assert rows["minimax"]["key_compatible"] is False
    assert rows["minimax"]["available"] is False
    assert rows["local_proxy"]["available"] is True
    assert rows["optional_proxy"]["key_present"] is False
    assert rows["optional_proxy"]["available"] is True
    assert rows["optional_typed"]["key_present"] is True
    assert rows["optional_typed"]["key_compatible"] is False
    assert rows["optional_typed"]["available"] is False
    assert "sk-api-wrong-kind" not in repr(rows)

    monkeypatch.setenv("TEST_MINIMAX_TOKEN_PLAN_KEY", "sk-cp-correct-kind")
    compatible = {
        row["name"]: row for row in serve_api._provider_catalog()
    }["minimax"]
    assert compatible["key_compatible"] is True
    assert compatible["available"] is True
    assert "sk-cp-correct-kind" not in repr(compatible)


class _CodeAwareHyDE:
    enabled = True

    @staticmethod
    def code_intent_signal(question):
        return "traceback" if "traceback" in question else None

    @staticmethod
    def expand(question, enabled=None):
        return question


class _EmptyRetriever:
    dense_top_k = 20
    sparse_top_k = 20
    omnisearch = None
    hype_enabled = False

    def retrieve(self, *args, **kwargs):
        return []


class _NoopReranker:
    mode = "cross_encoder"
    model_name = "test/reranker"
    max_length = 512

    @staticmethod
    def rerank(question, docs, top_k=None, mode=None):
        return docs[:top_k]


def _pipeline_with_code_preset():
    return RAGPipeline(
        retriever=_EmptyRetriever(),
        reranker=_NoopReranker(),
        hyde=_CodeAwareHyDE(),
        generator=SimpleNamespace(),
        rerank_top_k=7,
        presets={
            "code": {
                "rerank_top_k": 11,
                "dense_top_k": 41,
                "use_hyde": False,
            }
        },
        scope_router=SimpleNamespace(detect=lambda question: None),
    )


def test_auto_preset_false_produces_config_only_baseline():
    rag = _pipeline_with_code_preset()

    _, automatic = rag.search("python traceback", auto_preset=True)
    _, baseline = rag.search("python traceback", auto_preset=False)
    _, explicit = rag.search(
        "python traceback", preset="code", auto_preset=False
    )

    assert automatic["preset"] == "code (auto)"
    assert automatic["rerank_top_k"] == 11
    assert automatic["dense_top_k"] == 41

    assert baseline["preset"] is None
    assert baseline["auto_preset"] is False
    assert baseline["rerank_top_k"] == 7
    assert baseline["dense_top_k"] == 20

    # auto_preset controls only inference; an explicit caller choice still wins.
    assert explicit["preset"] == "code"
    assert explicit["rerank_top_k"] == 11


def test_schema_advertises_comparison_provenance_and_live_choices(monkeypatch):
    monkeypatch.setitem(
        serve_api._STATE,
        "rag",
        SimpleNamespace(presets={"code": {"rerank_top_k": 11}}),
    )

    contract = serve_api.schema()
    endpoints = contract["endpoints"]

    assert "GET /providers" in endpoints
    assert "GET /chunks/{chunk_id}?include_text=" in endpoints
    assert "POST /compare" in endpoints
    assert endpoints["POST /search"]["body"]["auto_preset"].startswith("bool")
    assert endpoints["POST /query"]["body"]["provider"].startswith("str?")
    assert "generation" in endpoints["POST /query"]["response"]
    assert endpoints["POST /compare"]["body"]["branch"]["model"] == "str?"
    assert contract["rerank_modes"] == list(RERANK_MODES)
    assert contract["presets"] == {"code": {"rerank_top_k": 11}}
    assert "stable evidence ids" in contract["provenance"]["sources"]
    assert "origin_id" in contract["provenance"]["sources"]
    assert serve_api.ConfigIn().persist is False
    assert "persist defaults to false" in endpoints["POST /config"]


def test_compare_preserves_generation_provenance_on_both_failure_stages(
        monkeypatch):
    doc = SimpleNamespace(
        id="chunk-1",
        text="retrieved evidence",
        metadata={},
        debug={},
        rerank_score=1.0,
        score=0.0,
        source_label="",
    )
    default_llm = SimpleNamespace(
        backend="default",
        provider="openai",
        model="default-model",
    )
    rag = SimpleNamespace(
        generator=SimpleNamespace(llm=default_llm),
        search=lambda *args, **kwargs: (
            [doc],
            {"rerank_mode": "lexical", "rerank_top_k": 1},
        ),
    )
    monkeypatch.setitem(serve_api._STATE, "rag", rag)

    class RuntimeFailure:
        llm = SimpleNamespace(
            backend="runtime",
            provider="anthropic",
            model="runtime-model",
        )

        @staticmethod
        def generate(*args, **kwargs):
            raise RuntimeError("remote request failed")

    def generator_for(provider, model):
        if provider == "configfail":
            raise ValueError("wrong credential type")
        return RuntimeFailure()

    monkeypatch.setattr(serve_api, "_generator_for", generator_for)

    result = serve_api.compare(serve_api.CompareIn(
        q="question",
        mode="query",
        branches=[
            serve_api.CompareBranchIn(id="config", provider="configfail"),
            serve_api.CompareBranchIn(id="runtime", provider="runtime"),
        ],
    ))

    config_failure, runtime_failure = result["branches"]
    assert config_failure["generation"] == {
        "backend": "configfail",
        "model": None,
        "stage": "configuration",
        "error": "ValueError: wrong credential type",
    }
    assert config_failure["error"].startswith(
        "Generation configuration failed")
    assert runtime_failure["generation"]["backend"] == "runtime"
    assert runtime_failure["generation"]["protocol"] == "anthropic"
    assert runtime_failure["generation"]["model"] == "runtime-model"
    assert runtime_failure["generation"]["stage"] == "request"
    assert runtime_failure["generation"]["error"].startswith("RuntimeError:")
    assert runtime_failure["sources"][0]["id"] == "chunk-1"


class _BrokenRerankerPipeline:
    @staticmethod
    def search(*args, **kwargs):
        raise RerankerExecutionError(
            "BAAI/bge-reranker-base failed at max_length=512"
        )


def test_query_and_search_surface_typed_reranker_failures(monkeypatch):
    monkeypatch.setitem(serve_api._STATE, "rag", _BrokenRerankerPipeline())

    query_result = serve_api.query(
        serve_api.QueryIn(q="why did reranking fail?", retrieve_only=True)
    )
    search_result = serve_api.search(
        serve_api.SearchIn(q="why did reranking fail?")
    )

    assert query_result.confidence == "ERROR"
    assert query_result.answer.startswith("Reranking failed:")
    assert "BAAI/bge-reranker-base" in query_result.answer
    assert search_result["results"] == []
    assert search_result["error"].startswith("Reranking failed:")


def test_generation_runtime_failure_is_not_mislabeled_as_configuration(
        monkeypatch):
    doc = SimpleNamespace(
        id="chunk-1",
        text="retrieved evidence",
        metadata={},
        debug={},
        rerank_score=1.0,
        score=0.0,
        source_label="",
    )
    llm = SimpleNamespace(
        backend="minimax",
        provider="anthropic",
        model="MiniMax-M3",
    )

    class FailingGenerator:
        def __init__(self):
            self.llm = llm

        @staticmethod
        def generate(*args, **kwargs):
            raise RuntimeError("rate limit reached")

    rag = SimpleNamespace(
        generator=SimpleNamespace(llm=llm),
        search=lambda *args, **kwargs: ([doc], {"rerank_mode": "lexical"}),
    )
    monkeypatch.setitem(serve_api._STATE, "rag", rag)
    monkeypatch.setattr(
        serve_api, "_generator_for", lambda provider, model: FailingGenerator()
    )

    result = serve_api.query(serve_api.QueryIn(q="question", include_text=100))

    assert result.confidence == "ERROR"
    assert result.answer.startswith("Generation backend 'minimax'")
    assert "configuration failed" not in result.answer.lower()
    assert result.sources[0].id == "chunk-1"


def test_generator_construction_failure_is_configuration_error(monkeypatch):
    doc = SimpleNamespace(
        id="chunk-1",
        text="retrieved evidence",
        metadata={},
        debug={},
        rerank_score=1.0,
        score=0.0,
        source_label="",
    )
    llm = SimpleNamespace(
        backend="openai",
        provider="openai",
        model="auto",
    )
    rag = SimpleNamespace(
        generator=SimpleNamespace(llm=llm),
        search=lambda *args, **kwargs: ([doc], {"rerank_mode": "lexical"}),
    )
    monkeypatch.setitem(serve_api._STATE, "rag", rag)

    def fail_to_build(provider, model):
        raise RuntimeError("MINIMAX_API_KEY has the wrong credential type")

    monkeypatch.setattr(serve_api, "_generator_for", fail_to_build)

    result = serve_api.query(
        serve_api.QueryIn(q="question", provider="minimax")
    )

    assert result.confidence == "ERROR"
    assert result.answer.startswith("Generation configuration failed:")
