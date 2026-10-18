"""The engines map's reliability rule, on hand-built probe facts.

build_engines is pure (manage_api gathers the facts), so this is the one place
hand-built input is the right input. tests/test_engines_api.py drives the real
probes.
"""
import pytest

from src.engines_map import (
    COMPONENTS, SWITCH, Option, build_engines, switch_changes, switch_option,
)

STATUS_UP = {"available": True, "error": None}


def _probes():
    return {
        "generation": {
            "active": "freellmapi", "model": "auto", "reserved": ["openai"],
            "providers": [
                {"name": "freellmapi", "available": True, "api_key_env": None},
                {"name": "minimax", "available": False, "api_key_env": "MINIMAX_API_KEY",
                 "key_compatible": None},
                {"name": "cp", "available": False, "api_key_env": "CP_KEY",
                 "key_compatible": False},
            ]},
        "reranker": {
            "active": "cross_encoder", "model": "cross-encoder/x", "cached": True,
            "modes": ["cross_encoder", "http", "lexical", "none", "laya", "quantum"],
            "laya": {"enabled": False, "package": False, "checkpoint": False,
                     "model_dir": "models/laya-noetrix"}},
        "embeddings": {
            "active": "vault", "model": "local · bge-small", "provider": "local",
            "providers": ["local", "eg2", "spare"],
            "collections": [
                {"name": "vault", "provider": "local", "model": "bge-small",
                 "status": "complete", "diff": []},
                {"name": "vault_eg2", "provider": "eg2", "model": "gemma",
                 "status": "complete", "diff": ["kind", "model"]},
                {"name": "half", "provider": "local", "model": "bge-small",
                 "status": "building", "diff": []},
                {"name": "broken", "error": "broken.embedding.json is not readable JSON"},
            ]},
        "sparse": {"active": "bm25s", "backends": ["bm25s", "tantivy"],
                   "tantivy_package": True, "tantivy_index": False,
                   "tantivy_dir": "data/bm25_index.pkl.tantivy"},
        "ocr": {"active": "auto", "model": None, "pipeline": None,
                "engines": ["auto", "tesseract", "paddle", "vlm", "none"],
                "status": {"tesseract": STATUS_UP,
                           "paddle": {"available": False, "error": "sidecar unreachable"},
                           "paddle-vl": STATUS_UP,
                           "vlm": {"available": False, "error": "not configured (pdf.vlm_ocr)"}}},
        "fetch": {"active": "auto",
                  "backends": ["auto", "requests", "crawl4ai", "scrapling", "crawlee"],
                  "installed": {"crawl4ai": True, "scrapling": False, "crawlee": True}},
        "chunking": {"active": "heading", "values": ["heading", "fixed", "document", "none"]},
    }


def _opts(engines, component):
    return {o["engine"]: o for o in engines["components"][component]["options"]}


def test_every_component_is_built_and_every_option_obeys_the_rule():
    engines = build_engines(_probes())
    assert list(engines["components"]) == list(COMPONENTS)
    for comp in engines["components"].values():
        assert set(comp) == {"active", "setting", "model", "options"}
        for o in comp["options"]:
            assert set(o) == {"engine", "available", "reason", "swappable", "switch"}
            assert o["switch"] in SWITCH
            if not (o["available"] and o["swappable"]):
                assert o["reason"], o


def test_an_option_without_a_reason_or_with_an_unknown_switch_cannot_exist():
    with pytest.raises(ValueError, match="needs a reason"):
        Option("x", False, None, True, "restart")
    with pytest.raises(ValueError, match="not one of"):
        Option("x", True, None, True, "someday")


def test_only_present_components_are_built_and_unknown_ones_raise():
    assert list(build_engines({"chunking": _probes()["chunking"]})["components"]) == ["chunking"]
    with pytest.raises(ValueError, match="unknown component"):
        build_engines({"vision": {}})


def test_generation_lists_unavailable_and_legacy_providers_with_their_reasons():
    gen = _opts(build_engines(_probes()), "generation")
    assert gen["freellmapi"] == {"engine": "freellmapi", "available": True, "reason": None,
                                 "swappable": True, "switch": "per_call"}
    assert not gen["minimax"]["available"] and "MINIMAX_API_KEY is not set" in gen["minimax"]["reason"]
    assert "wrong credential type" in gen["cp"]["reason"]
    assert gen["openai"]["swappable"] is False and "providers: entry" in gen["openai"]["reason"]


def test_reranker_rules():
    probes = _probes()
    rr = _opts(build_engines(probes), "reranker")
    assert rr["cross_encoder"]["available"] and rr["cross_encoder"]["swappable"]
    assert rr["http"]["swappable"] is False and "per call" in rr["http"]["reason"]
    assert rr["lexical"]["available"] and rr["none"]["swappable"]
    assert "retrieval.laya.enabled" in rr["laya"]["reason"] and rr["laya"]["swappable"]
    assert rr["quantum"] == {"engine": "quantum", "available": False, "reason": rr["quantum"]["reason"],
                             "swappable": False, "switch": "per_call"}
    probes["reranker"]["cached"] = False
    ce = _opts(build_engines(probes), "reranker")["cross_encoder"]
    assert not ce["available"] and ce["swappable"] and "local model cache" in ce["reason"]


def test_embeddings_switch_only_to_a_complete_collection_of_the_configured_embedder():
    emb = _opts(build_engines(_probes()), "embeddings")
    assert emb["vault"] == {"engine": "vault", "available": True, "reason": None,
                            "swappable": True, "switch": "restart"}
    assert emb["vault_eg2"]["swappable"] is False
    assert "differ in kind, model" in emb["vault_eg2"]["reason"]
    assert emb["half"]["available"] is False and "unfinished re-embed" in emb["half"]["reason"]
    assert emb["broken"]["swappable"] is False and "not readable JSON" in emb["broken"]["reason"]
    # eg2 has a built collection, so only `spare` costs a re-embed; the configured
    # provider is never offered as one.
    assert emb["spare"]["switch"] == "reembed" and "--embedding spare" in emb["spare"]["reason"]
    assert "eg2" not in emb and "local" not in emb


def test_an_unstamped_active_collection_is_still_listed():
    probes = _probes()
    probes["embeddings"]["active"] = "legacy"
    legacy = _opts(build_engines(probes), "embeddings")["legacy"]
    assert legacy["available"] and not legacy["swappable"] and "main.py stamp" in legacy["reason"]


def test_tantivy_costs_a_build_until_its_index_exists():
    probes = _probes()
    tv = _opts(build_engines(probes), "sparse")["tantivy"]
    assert (tv["available"], tv["swappable"], tv["switch"]) == (False, True, "build")
    assert "build_tantivy" in tv["reason"]
    probes["sparse"]["tantivy_package"] = False
    assert "not installed" in _opts(build_engines(probes), "sparse")["tantivy"]["reason"]
    probes["sparse"].update(tantivy_package=True, tantivy_index=True)
    assert _opts(build_engines(probes), "sparse")["tantivy"]["switch"] == "restart"


def test_ocr_auto_is_a_chain_and_paddle_follows_its_pipeline():
    probes = _probes()
    ocr = _opts(build_engines(probes), "ocr")
    assert ocr["auto"]["swappable"] is False and "chain" in ocr["auto"]["reason"]
    assert "pdf.ocr_engine" in ocr["auto"]["reason"]    # Review Focus 5: the way back is named
    assert ocr["paddle"]["reason"] == "sidecar unreachable"
    assert ocr["vlm"]["available"] is False and ocr["none"]["available"]
    probes["ocr"]["pipeline"] = "vl"
    assert _opts(build_engines(probes), "ocr")["paddle"]["available"] is True


def test_fetch_auto_is_the_declared_fallback_and_has_no_setting():
    engines = build_engines(_probes())
    fetch = _opts(engines, "fetch")
    assert fetch["auto"]["swappable"] and fetch["requests"]["available"]
    assert "scrapling package is not installed" in fetch["scrapling"]["reason"]
    assert engines["components"]["fetch"]["setting"] is None


def test_switch_option_refuses_with_the_reason():
    engines = build_engines(_probes())
    assert switch_option(engines, "sparse", "tantivy")["engine"] == "tantivy"  # availability: caller's preflight
    for args, msg in [(("vision", "x"), "unknown component"),
                      (("fetch", "crawl4ai"), "chosen per request"),
                      (("sparse", "lucene"), "unknown sparse engine"),
                      (("embeddings", "vault_eg2"), "differ in kind, model")]:
        with pytest.raises(ValueError, match=msg):
            switch_option(engines, *args)


def test_switch_changes_writes_the_model_only_where_it_means_something():
    assert switch_changes("sparse", "tantivy", None) == {"retrieval.sparse_backend": "tantivy"}
    assert switch_changes("generation", "minimax", "MiniMax-M3") == {
        "generation.provider": "minimax", "generation.model": "MiniMax-M3"}
    assert switch_changes("reranker", "cross_encoder", "BAAI/bge-reranker-base") == {
        "retrieval.rerank_mode": "cross_encoder",
        "retrieval.cross_encoder_model": "BAAI/bge-reranker-base"}
    for component, engine in [("reranker", "lexical"), ("sparse", "bm25s")]:
        with pytest.raises(ValueError, match="model applies only"):
            switch_changes(component, engine, "some-model")
