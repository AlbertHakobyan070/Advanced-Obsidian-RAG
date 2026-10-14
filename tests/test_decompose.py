"""retrieval.decompose: query decomposition with per-branch reranking.

The reply parser (atomic, split, dedup, junk raises), the reciprocal-rank merge
(every branch gets a seat, agreement rises, the original wins ties), and how
search() runs the branches. Offline: stubs only, no model, no LLM.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.bench.configs import BASELINES, decompose_ablation, diversity_ablation, resolve
from src.pipeline import RAGPipeline
from src.retrieval.decompose import Decomposer, merge_branches, parse_subqueries
from src.retrieval.retriever import RetrievedDoc

ROOT = Path(__file__).resolve().parents[1]
Q = "Which chapters of my stats syllabus cover estimation, and what does All of Statistics say?"


# ---- the reply parser ----

def test_one_element_means_atomic():
    assert parse_subqueries('["what is a monad"]', "what is a monad", 3) == []


def test_split_keeps_distinct_parts_in_order_and_drops_the_question_itself():
    reply = f'["{Q}", "stats syllabus estimation weeks", "All of Statistics estimation chapter"]'
    assert parse_subqueries(reply, Q, 3) == ["stats syllabus estimation weeks",
                                             "All of Statistics estimation chapter"]


def test_duplicates_collapse_and_the_list_is_cut_to_max():
    reply = '["a  b", "A B", "c", "d", "e"]'
    assert parse_subqueries(reply, "q", 2) == ["a b", "c"]


def test_a_fenced_reply_with_prose_around_it_still_parses():
    reply = 'Sure:\n```json\n["x", "y"]\n```'
    assert parse_subqueries(reply, "q", 3) == ["x", "y"]


@pytest.mark.parametrize("reply", ["no array here", '["unclosed', '{"a": 1}', '[1, 2]', "[]x]"])
def test_junk_raises_instead_of_falling_back(reply):
    with pytest.raises(ValueError, match="decompose"):
        parse_subqueries(reply, "q", 3)


def test_max_subqueries_below_two_is_refused():
    with pytest.raises(ValueError):
        Decomposer(llm=None, max_subqueries=1)


# ---- the merge ----

def _d(i):
    return RetrievedDoc(id=i, text="t", metadata={"source_file": i})


def test_every_branch_gets_a_seat_and_the_original_wins_ties():
    original = [_d("o1"), _d("o2"), _d("o3")]
    branch = [_d("b1"), _d("b2")]
    assert [d.id for d in merge_branches([original, branch], 60)] == ["o1", "b1", "o2", "b2", "o3"]


def test_a_chunk_two_branches_agree_on_rises_and_appears_once():
    original = [_d("o1"), _d("x"), _d("o3")]
    branch = [_d("b1"), _d("x")]
    merged = [d.id for d in merge_branches([original, branch], 60)]
    assert merged[0] == "x" and merged.count("x") == 1


# ---- search() ----

class _Reranker:
    mode = "cross_encoder"
    instruction = None
    max_length = 512
    model_name = "stub"

    def __init__(self):
        self.calls = []

    def rerank(self, question, docs, top_k=None, mode=None, instruction=None):
        self.calls.append((question, top_k))
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
    rrf_k = 60

    def __init__(self):
        self.texts = []

    def retrieve(self, query, **kwargs):
        self.texts.append(query)
        tag = "o" if len(self.texts) == 1 else f"b{len(self.texts) - 1}_"
        return [_d(f"{tag}{n}") for n in range(1, 4)]


class _LLM:
    model = "stub"

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def complete(self, system, user, temperature=None, max_tokens=None):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return SimpleNamespace(text=self.reply)


def _pipeline(reply, enabled=False):
    retr, rr, llm = _Retriever(), _Reranker(), _LLM(reply)
    rag = RAGPipeline(retriever=retr, reranker=rr, hyde=_Hyde(), generator=None, rerank_top_k=4,
                      decomposer=Decomposer(llm, enabled=enabled))
    return rag, retr, rr, llm


def test_off_by_default_never_calls_the_llm():
    rag, retr, _, llm = _pipeline('["a", "b"]')
    _, info = rag.search(Q)
    assert llm.calls == 0 and retr.texts == [Q] and info["decompose"]["status"] == "off"


def test_an_atomic_question_runs_exactly_as_without_decomposition():
    rag, retr, rr, llm = _pipeline(f'["{Q}"]')
    top, info = rag.search(Q, decompose=True)
    assert llm.calls == 1 and retr.texts == [Q] and rr.calls == [(Q, 4)]
    assert info["decompose"] == {"status": "atomic", "subqueries": [], "branch_pools": [3]}
    assert [d.id for d in top] == ["o1", "o2", "o3"]


def test_a_split_question_retrieves_and_reranks_each_branch_on_its_own_text():
    rag, retr, rr, _ = _pipeline('["syllabus estimation", "All of Statistics estimation"]')
    top, info = rag.search(Q, decompose=True)
    assert retr.texts == [Q, "syllabus estimation", "All of Statistics estimation"]
    # every branch reranked against ITS text, whole pool back
    assert rr.calls == [(Q, 3), ("syllabus estimation", 3), ("All of Statistics estimation", 3)]
    assert [d.id for d in top] == ["o1", "b1_1", "b2_1", "o2"]      # round-robin by rank, original first
    assert info["decompose"]["status"] == "split" and info["decompose"]["branch_pools"] == [3, 3, 3]


def test_configured_on_is_followed_and_a_call_can_switch_it_off():
    rag, retr, _, llm = _pipeline('["a", "b"]', enabled=True)
    rag.search(Q)
    assert llm.calls == 1 and len(retr.texts) == 3
    rag.search(Q, decompose=False)
    assert llm.calls == 1


def test_an_llm_failure_raises_instead_of_searching_undecomposed():
    rag, retr, _, _ = _pipeline(RuntimeError("provider down"))
    with pytest.raises(RuntimeError, match="provider down"):
        rag.search(Q, decompose=True)
    assert retr.texts == []


def test_decompose_with_the_gate_is_refused():
    rag, _, _, _ = _pipeline('["a", "b"]')
    with pytest.raises(ValueError, match="relevance gate"):
        rag.search(Q, decompose=True, gate_threshold=0.0)


def test_the_cap_applies_after_the_merge():
    rag, _, _, _ = _pipeline('["a", "b"]')
    top, _ = rag.search(Q, decompose=True, max_per_file=1)
    assert len({d.metadata["source_file"] for d in top}) == len(top)


def test_lane_counts_add_up_across_branches_on_one_trace(monkeypatch):
    from src.retrieval.retriever import HybridRetriever
    from src.utils.timing import StageTrace
    r = HybridRetriever.__new__(HybridRetriever)
    r.embedder = type("E", (), {"embed_query": staticmethod(lambda t: [0.0])})()
    r.rrf_k, r.dense_top_k, r.sparse_top_k = 60, 20, 20
    r.metadata_boost, r.code_file_types, r.omnisearch = False, [], None
    r.hype_enabled, r.hype_top_k, r.lane_weights = False, 15, {}
    monkeypatch.setattr(r, "_dense_search", lambda qvec, k, where=None: [("d1", "t", {})])
    monkeypatch.setattr(r, "_sparse_search", lambda q, k, predicate=None: [("s1", "t", {}), ("s2", "t", {})])
    trace = StageTrace()
    r.retrieve("branch one", trace=trace)
    r.retrieve("branch two", trace=trace)
    assert trace.lanes == {"dense": 2, "sparse": 4}


# ---- config, API, bench ----

def test_the_block_ships_in_both_configs_and_defaults_off():
    for name in ("config.yaml", "config.example.yaml"):
        path = ROOT / name
        if not path.exists():           # a fresh clone has no config.yaml
            continue
        block = yaml.safe_load(path.read_text(encoding="utf-8"))["retrieval"]["decompose"]
        assert block == {"enabled": False, "max_subqueries": 3}, name


def test_the_api_passes_it_through():
    import serve_api
    assert "decompose" in serve_api._RETRIEVAL_FIELDS
    assert serve_api.SearchIn(q="x", decompose=True).decompose is True


def test_decompose_spec_is_d_off_with_the_switch():
    specs = dict(decompose_ablation())
    assert list(specs) == ["Q-off", "Q-on", "Q-on-cap2"]
    d_off = dict(diversity_ablation())["D-off"]
    assert specs["Q-off"] == {**d_off, "decompose": False}
    assert specs["Q-on-cap2"]["decompose"] is True and specs["Q-on-cap2"]["max_per_file"] == 2
    assert BASELINES["Q-on"] == BASELINES["Q-on-cap2"] == "Q-off"
    assert [n for n, _ in resolve("decompose", ROOT / "eval" / "configs.yaml")] == list(specs)
