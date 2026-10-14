"""retrieval.max_per_file: the per-file cap on the final list.

The selection itself (cap, backfill, keyless chunks) and how search() feeds it:
with the cap on, the reranker must hand back its WHOLE scored pool, otherwise
there is nothing to backfill from. Offline: stubs only, nothing is loaded.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.bench.configs import BASELINES, diversity_ablation, resolve, routing_ablation
from src.pipeline import RAGPipeline
from src.retrieval.diversity import cap_per_file
from src.retrieval.retriever import RetrievedDoc

ROOT = Path(__file__).resolve().parents[1]


def _docs(*files):
    """One doc per entry, in rank order; None = a chunk with no source file."""
    return [RetrievedDoc(id=f"c{i}", text="t", metadata={} if f is None else {"source_file": f})
            for i, f in enumerate(files)]


def _ids(docs):
    return [d.id for d in docs]


# ---- the selection ----

def test_cap_keeps_best_first_order_and_at_most_n_per_file():
    docs = _docs("A", "A", "A", "B", "A", "C")
    assert _ids(cap_per_file(docs, 4, 2)) == ["c0", "c1", "c3", "c5"]


def test_cap_backfills_in_score_order_when_too_few_files():
    docs = _docs("A", "A", "A", "A", "B")
    # A, A, B pass the cap; the skipped A's fill the tail best first.
    assert _ids(cap_per_file(docs, 5, 2)) == ["c0", "c1", "c4", "c2", "c3"]


def test_cap_zero_is_a_plain_cut():
    docs = _docs("A", "A", "A")
    assert _ids(cap_per_file(docs, 2, 0)) == ["c0", "c1"]


def test_k_larger_than_the_pool_returns_the_whole_pool():
    docs = _docs("A", "A", "B")
    assert sorted(_ids(cap_per_file(docs, 10, 1))) == ["c0", "c1", "c2"]


def test_file_key_ignores_separator_and_case_and_keyless_chunks_are_never_capped():
    docs = _docs("Books\\Stats.pdf", "books/stats.PDF", None, None, "X")
    assert _ids(cap_per_file(docs, 4, 1)) == ["c0", "c2", "c3", "c4"]


# ---- search() ----

class _Reranker:
    mode = "cross_encoder"
    instruction = None
    max_length = 512
    model_name = "stub"

    def __init__(self):
        self.top_k = None

    def rerank(self, question, docs, top_k=None, mode=None, instruction=None):
        self.top_k = top_k
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


class _Retriever:
    dense_top_k = sparse_top_k = 20
    omnisearch = None
    hype_enabled = False
    lane_weights: dict = {}
    metadata_boost = False

    def retrieve(self, query, **kwargs):
        return _docs("A", "A", "A", "B", "C", "A")


def _pipeline(**kw):
    rr = _Reranker()
    return RAGPipeline(retriever=_Retriever(), reranker=rr, hyde=_Hyde(),
                       generator=None, rerank_top_k=3, **kw), rr


def test_search_uncapped_by_default_asks_the_reranker_for_top_k_only():
    rag, rr = _pipeline()
    top, info = rag.search("q")
    assert rr.top_k == 3 and _ids(top) == ["c0", "c1", "c2"] and info["max_per_file"] == 0


def test_search_capped_reranks_the_whole_pool_then_caps():
    rag, rr = _pipeline()
    top, info = rag.search("q", max_per_file=1)
    assert rr.top_k == 6                                  # the whole pool came back scored
    assert _ids(top) == ["c0", "c3", "c4"] and info["max_per_file"] == 1


def test_configured_cap_is_followed_and_a_call_can_switch_it_off():
    rag, _ = _pipeline(max_per_file=2)
    assert _ids(rag.search("q")[0]) == ["c0", "c1", "c3"]
    assert _ids(rag.search("q", max_per_file=0)[0]) == ["c0", "c1", "c2"]


def test_query_forwards_max_per_file_to_search():
    rag, _ = _pipeline()
    seen = {}
    rag.generator = SimpleNamespace(
        generate=lambda q, docs, max_tokens=None: seen.setdefault("ids", _ids(docs))
        and SimpleNamespace(retrieval=None))
    rag.query("q", max_per_file=1)
    assert seen["ids"] == ["c0", "c3", "c4"]


# ---- config, API, bench ----

def test_the_key_ships_in_both_configs_and_defaults_off():
    for name in ("config.yaml", "config.example.yaml"):
        path = ROOT / name
        if not path.exists():           # a fresh clone has no config.yaml
            continue
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["retrieval"]["max_per_file"] == 0, name


def test_the_api_passes_it_through():
    import serve_api
    assert "max_per_file" in serve_api._RETRIEVAL_FIELDS
    assert serve_api.SearchIn(q="x", max_per_file=2).max_per_file == 2


def test_diversity_spec_is_the_hyde_off_pipeline_at_three_caps():
    specs = dict(diversity_ablation())
    assert list(specs) == ["D-off", "D-cap2", "D-cap3"]
    assert [specs[n]["max_per_file"] for n in specs] == [0, 2, 3]
    # D-off is F-none of the routing ablation, plus the cap key at 0.
    f_none = dict(routing_ablation())["F-none"]
    assert {k: v for k, v in specs["D-off"].items() if k != "max_per_file"} == f_none
    assert BASELINES["D-cap2"] == BASELINES["D-cap3"] == "D-off"
    assert [n for n, _ in resolve("diversity", ROOT / "eval" / "configs.yaml")] == list(specs)
