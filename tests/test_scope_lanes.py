"""retrieval.scope_lanes: the scope lanes as a switch of their own.

`routing` used to be one lump: a detected scope opened dense_scope/sparse_scope
AND metadata_boost nudged chunks that matched the query. They are separate
mechanisms, so each has its own key (config + per call), and switching the scope
lanes off must leave DETECTION alone (the scope is still found and echoed).

Offline: retrievers are stubs, nothing is loaded.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.bench.configs import BASELINES, FACTORS, LADDER, hyde_ablation, resolve, routing_ablation
from eval.bench.report import render
from eval.bench.runner import summarise
from src.pipeline import RAGPipeline
from src.retrieval.scope import Scope, ScopeRouter

ROOT = Path(__file__).resolve().parents[1]


# ---- the pipeline ----

class _Reranker:
    mode = "none"
    instruction = None
    max_length = 512
    model_name = "stub"

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


class _Retriever:
    dense_top_k = sparse_top_k = 20
    omnisearch = None
    hype_enabled = False
    lane_weights: dict = {}
    metadata_boost = True

    def __init__(self):
        self.kwargs = None

    def retrieve(self, query, **kwargs):
        self.kwargs = kwargs
        return []


def _pipeline(retriever, **kw):
    router = ScopeRouter({"stats": ["statistics"]}, None)
    return RAGPipeline(retriever=retriever, reranker=_Reranker(), hyde=_Hyde(),
                       generator=None, rerank_top_k=3, scope_router=router, **kw)


Q = "bayes theorem in my statistics notes"


def test_scope_lanes_default_on_hands_the_detected_scope_to_the_retriever():
    retr = _Retriever()
    _, info = _pipeline(retr).search(Q)
    assert retr.kwargs["scope"] is not None and retr.kwargs["scope"].domains == ["stats"]
    assert info["scope_lanes"] is True and info["scope"] == ["domain:stats(statistics)"]


def test_scope_lanes_off_per_call_hides_the_scope_from_fusion_but_still_detects_it():
    retr = _Retriever()
    _, info = _pipeline(retr).search(Q, scope_lanes=False)
    assert retr.kwargs["scope"] is None                       # nothing for the retriever to open lanes on
    assert info["scope_lanes"] is False
    assert info["scope"] == ["domain:stats(statistics)"]      # detection is untouched
    assert retr.kwargs["metadata_boost"] is None              # the other half is not collateral


def test_configured_off_is_followed_and_a_call_can_turn_it_back_on():
    retr = _Retriever()
    rag = _pipeline(retr, scope_lanes=False)
    _, info = rag.search(Q)
    assert retr.kwargs["scope"] is None and info["scope_lanes"] is False
    _, info = rag.search(Q, scope_lanes=True)
    assert retr.kwargs["scope"] is not None and info["scope_lanes"] is True


def test_a_query_naming_nothing_is_unaffected_either_way():
    retr = _Retriever()
    for flag in (None, True, False):
        _pipeline(retr).search("what is a monad", scope_lanes=flag)
        assert retr.kwargs["scope"] is None


def test_query_forwards_scope_lanes_to_search():
    retr = _Retriever()
    generator = SimpleNamespace(
        generate=lambda q, docs, max_tokens=None: SimpleNamespace(retrieval=None))
    rag = _pipeline(retr)
    rag.generator = generator
    rag.query(Q, scope_lanes=False)
    assert retr.kwargs["scope"] is None


def test_the_two_switches_are_independent():
    retr = _Retriever()
    _, info = _pipeline(retr).search(Q, scope_lanes=False, metadata_boost=True)
    assert retr.kwargs["metadata_boost"] is True and info["metadata_boost"] is True
    _, info = _pipeline(retr).search(Q, scope_lanes=True, metadata_boost=False)
    assert retr.kwargs["scope"] is not None and info["metadata_boost"] is False


def test_the_real_retriever_runs_no_scope_lane_when_it_is_given_no_scope(monkeypatch):
    from src.retrieval.retriever import HybridRetriever
    r = HybridRetriever.__new__(HybridRetriever)
    r.embedder = type("E", (), {"embed_query": staticmethod(lambda t: [0.0])})()
    r.rrf_k, r.dense_top_k, r.sparse_top_k = 60, 20, 20
    r.metadata_boost, r.code_file_types, r.omnisearch = False, [], None
    r.hype_enabled, r.hype_top_k, r.lane_weights = False, 15, {}
    rows = lambda tag: [(f"{tag}1", "t", {"domain": "stats"})]
    ran = []
    monkeypatch.setattr(r, "_dense_search", lambda qvec, k, where=None: rows("d"))
    monkeypatch.setattr(r, "_sparse_search", lambda q, k, predicate=None: rows("s"))

    def dense_scope(qvec, k, scope):
        ran.append("dense_scope")
        return rows("ds")
    monkeypatch.setattr(r, "_dense_scope_search", dense_scope)
    rag = RAGPipeline(retriever=r, reranker=_Reranker(), hyde=_Hyde(), generator=None,
                      rerank_top_k=3, scope_router=ScopeRouter({"stats": ["statistics"]}, None))
    _, info = rag.search(Q, scope_lanes=False)
    assert ran == [] and "dense_scope" not in info["lanes_run"]
    _, info = rag.search(Q)
    assert ran == ["dense_scope"] and "dense_scope" in info["lanes_run"]


# ---- config ----

@pytest.mark.parametrize("name", ["config.yaml", "config.example.yaml"])
def test_both_keys_ship_in_the_config_and_default_off(name):
    # Off since the 2026-10-11 dev ablation: no effect after rerank, -0.040 nDCG@10 without it.
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} is not present")
    retrieval = yaml.safe_load(path.read_text(encoding="utf-8"))["retrieval"]
    assert retrieval["scope_lanes"] is False and retrieval["metadata_boost"] is False


# ---- the API ----

def test_search_query_and_compare_branches_accept_and_forward_scope_lanes(monkeypatch):
    import serve_api as S
    seen = []

    def record(q, **kwargs):
        seen.append(kwargs)
        return [], {"rerank_mode": "none"}

    monkeypatch.setitem(S._STATE, "rag", SimpleNamespace(search=record))
    S.search(S.SearchIn(q="x", scope_lanes=False))
    S.query(S.QueryIn(q="x", retrieve_only=True, scope_lanes=False))
    assert len(seen) == 2 and all(k["scope_lanes"] is False for k in seen)
    branch = S.CompareBranchIn(id="a", scope_lanes=False)
    assert S._retrieval_kwargs(branch)["scope_lanes"] is False
    # part of the retrieval key: branches differing only by it must not share evidence
    assert S._retrieval_cache_key(branch) != S._retrieval_cache_key(S.CompareBranchIn(id="a"))


def test_every_retrieval_field_is_declared_on_all_three_request_models():
    import serve_api as S
    for model in (S.QueryIn, S.SearchIn, S.CompareBranchIn):
        assert set(S._RETRIEVAL_FIELDS) <= set(model.model_fields), model.__name__


# ---- the Settings tab ----

def test_both_keys_are_editable_boolean_settings_with_a_one_line_description(management_module):
    for key in ("retrieval.scope_lanes", "retrieval.metadata_boost"):
        spec = management_module.EDITABLE_SETTINGS[key]
        assert spec["kind"] == "enum" and spec["values"] == ["true", "false"]
        assert spec["desc"] and "\n" not in spec["desc"]


def test_neither_key_is_an_experimental_feature(management_module):
    """They ship on and are tunables, not unproven features: the Experimental
    panel (and its data-exp-key checkbox) is the wrong home."""
    flagged = {f["key"] for f in management_module.EXPERIMENTAL_FEATURES}
    assert not flagged & {"retrieval.scope_lanes", "retrieval.metadata_boost"}


def test_settings_save_accepts_the_values_and_refuses_anything_else(management_module, monkeypatch):
    written = {}
    monkeypatch.setattr(management_module, "_persist_section_keys",
                        lambda path, changes: written.update(changes) or list(changes))
    ok = management_module.settings_update(management_module.SettingsIn(changes={
        "retrieval.scope_lanes": "false", "retrieval.metadata_boost": "true"}))
    assert ok["ok"] is True
    assert written == {"retrieval.scope_lanes": "false", "retrieval.metadata_boost": "true"}
    bad = management_module.settings_update(management_module.SettingsIn(
        changes={"retrieval.scope_lanes": "maybe"}))
    assert bad.status_code == 400 and "retrieval.scope_lanes" in json.loads(bad.body)["error"]


def test_settings_grid_prints_the_description_and_keeps_the_select_in_the_save_scan():
    html = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    row = html[html.index("const desc=f.desc"):][:400]
    assert "esc(f.desc)" in row and "${desc}" in row
    # a plain <select data-key>, as for every other enum: the value is "true"/"false",
    # which matches settingsOrig, so Save never posts an unknown setting
    assert 'f.kind==="enum"' in html and '<select id="${id}" data-key="${esc(key)}">' in html


# ---- the bench specs ----

def test_routing_spec_is_four_r5_configs_differing_only_in_the_two_keys():
    got = resolve("routing", ROOT / "absent.yaml")                 # generated: never read
    assert [n for n, _ in got] == ["F-full", "F-noscope", "F-noboost", "F-none"]
    by = dict(got)
    flags = {n: (c["scope_lanes"], c["metadata_boost"]) for n, c in by.items()}
    assert flags == {"F-full": (True, True), "F-noscope": (False, True),
                     "F-noboost": (True, False), "F-none": (False, False)}
    shared = {k: v for k, v in by["F-full"].items() if k not in ("scope_lanes", "metadata_boost")}
    for c in by.values():
        assert {k: v for k, v in c.items() if k not in ("scope_lanes", "metadata_boost")} == shared
    assert shared["hyde"] is False and shared["rerank"] == "cross_encoder" and shared["auto_preset"]
    assert shared["lanes"] == sorted(["dense", "sparse", "dense_code", "sparse_code",
                                      "dense_scope", "sparse_scope"])


def test_hyde_spec_is_r5_off_and_on():
    got = dict(resolve("hyde", ROOT / "absent.yaml"))
    assert list(got) == ["H-off", "H-on"]
    assert got["H-off"]["hyde"] is False and got["H-on"]["hyde"] is True
    assert {k: v for k, v in got["H-on"].items() if k != "hyde"} == \
           {k: v for k, v in got["H-off"].items() if k != "hyde"}
    assert got["H-off"]["lanes"] == dict(routing_ablation())["F-full"]["lanes"]
    assert set(hyde_ablation()[0][1]["lanes"]) <= {"dense", "sparse", "dense_scope", "sparse_scope",
                                                    "dense_code", "sparse_code"}


def test_ladder_pairs_are_unchanged_and_the_new_specs_pair_against_their_reference():
    assert BASELINES["R3-routing"] == "R2-hybrid" and BASELINES["R6-hyde"] == "R5-rerank"
    assert {b: a for b, a in BASELINES.items() if b.startswith("R")} == {
        b: a for (a, _), (b, _) in zip(LADDER, LADDER[1:])}
    assert {b: a for b, a in BASELINES.items() if b.startswith(("F-", "H-"))} == {
        "F-noscope": "F-full", "F-noboost": "F-full", "F-none": "F-full", "H-on": "H-off"}
    assert FACTORS                                               # the generated sets still exist


def _rows(ndcg_by_q):
    return [{"qid": q, "config": "c", "suite": "lexical", "tier": "T1", "split": "dev",
             "author": "owner", "answerable": True, "cold": False, "timings": {}, "lanes_run": {},
             "metrics": {"ndcg@10": v, "hit@5": 1.0 if v else 0.0}} for q, v in ndcg_by_q.items()]


def test_the_report_prints_each_routing_variant_paired_against_f_full():
    full = _rows({f"q{i}": 1.0 for i in range(10)})
    worse = _rows({f"q{i}": 0.5 for i in range(10)})
    summary = summarise({"F-full": full, "F-noscope": worse, "F-noboost": full, "F-none": worse})
    assert "paired" not in summary["F-full"]
    for name in ("F-noscope", "F-noboost", "F-none"):
        assert summary[name]["paired"]["vs"] == "F-full" and summary[name]["paired"]["n"] == 10
    assert summary["F-noscope"]["paired"]["ndcg@10"]["mean"] == pytest.approx(-0.5)
    meta = {"run_id": "r", "git": {"sha": "0" * 40, "dirty": False}, "config_digest": "d",
            "split": "dev", "n_questions": 10, "suites": ["lexical"], "test_ledger_count": 0}
    md = render(summary, meta)
    for name in ("F-noscope", "F-noboost", "F-none"):
        assert f"{name} vs F-full" in md
    assert "H-on vs H-off" not in md
