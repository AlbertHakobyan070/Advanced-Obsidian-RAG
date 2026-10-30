"""Knowledge-map harness: config section, JSON contract, templates (src/knowledge_map)."""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
from pydantic import ValidationError

from src.knowledge_map.config import KMConfig, RoleSpec, load_km_config
from src.knowledge_map.schema import EdgeIn, Extraction, NodeIn
from src.knowledge_map.templates import check_extraction, load_templates
from src.utils.config_loader import Config, load_config

NAVIGATOR_TEN = ("is-a", "part-of", "example-of", "prerequisite-of", "leads-to",
                 "causes", "contrasts-with", "implements", "uses", "defines")


def _role(provider="freellmapi", model="auto"):
    return {"provider": provider, "model": model, "max_tokens": 100, "temperature": 0.0,
            "fallbacks": [{"provider": "freellmapi", "model": "gpt-oss-120b"}]}


def _data(tmp_path):
    return {
        "parser": {"vault_path": str(tmp_path / "vault")},
        "knowledge_map": {
            "staging_dir": str(tmp_path / "staging"), "min_grounded_ratio": 0.9,
            "max_span_pages": 3, "node_cap": 40, "fanout_max": 8, "quote_words": [5, 30],
            "max_wall_minutes": 30, "max_llm_calls": 150, "retries": 2, "verify_batch": 10,
            "rate_limit_sleep_s": 20, "request_timeout_s": 300, "vision": None,
            "classify": _role(), "link": _role(),
            "extract": {**_role(), "max_unit_chars": 24000},
            "verify": _role(model="gpt-oss-120b"),
            "reverify": {**_role(model="qwen3.8-27b"),
                         "fallbacks": [{"provider": "freellmapi", "model": "gemini-2.5-flash-lite"}]},
            "scout": {"window_pages": 8, "max_scanned_ratio": 0.2},
            "coverage": {"heading_min_ratio": 1.15, "min_section_chars": 400, "chars_per_node": 2000},
        },
    }


def _cfg(tmp_path, mutate=None):
    d = copy.deepcopy(_data(tmp_path))
    if mutate:
        mutate(d)
    return Config(d, tmp_path)


def test_shipped_config_loads():
    km = load_km_config(load_config(ROOT / "config.example.yaml"))
    assert isinstance(km, KMConfig)
    assert km.roles["verify"].model == "gpt-oss-120b"
    assert km.node_cap == 40
    assert km.quote_words == (5, 30)
    assert km.roles["extract"].fallbacks == (("freellmapi", "gpt-oss-120b"),)
    assert set(km.roles) == {"classify", "extract", "link", "verify", "reverify"}
    assert km.roles["reverify"].model == "qwen3.8-27b"
    assert isinstance(km.roles["verify"], RoleSpec)
    assert km.scanned_threshold == 50
    assert km.request_timeout_s > 0 and km.generated_markers == ("(generated)",)


def test_shipped_live_config_loads(shipped_cfg):
    load_km_config(shipped_cfg)


def test_adhoc_config_loads(tmp_path):
    km = load_km_config(_cfg(tmp_path))
    assert km.max_unit_chars == 24000 and km.window_pages == 8 and km.max_scanned_ratio == 0.2
    assert km.scanned_threshold == 50          # pdf.skip_scanned_threshold absent: default 50
    assert km.vault_root == tmp_path / "vault"


def test_verify_auto_refused(tmp_path):
    with pytest.raises(ValueError, match=r"knowledge_map\.verify\.model"):
        load_km_config(_cfg(tmp_path, lambda d: d["knowledge_map"]["verify"].update(model="auto")))


def test_verify_fallback_auto_refused(tmp_path):
    def m(d):
        d["knowledge_map"]["verify"]["fallbacks"] = [{"provider": "freellmapi", "model": "auto"}]
    with pytest.raises(ValueError, match=r"knowledge_map\.verify\.fallbacks\[0\]\.model"):
        load_km_config(_cfg(tmp_path, m))


def test_reverify_auto_refused(tmp_path):
    with pytest.raises(ValueError, match=r"knowledge_map\.reverify\.model"):
        load_km_config(_cfg(tmp_path, lambda d: d["knowledge_map"]["reverify"].update(model="auto")))
    def m(d):
        d["knowledge_map"]["reverify"]["fallbacks"] = [{"provider": "freellmapi", "model": "auto"}]
    with pytest.raises(ValueError, match=r"knowledge_map\.reverify\.fallbacks\[0\]\.model"):
        load_km_config(_cfg(tmp_path, m))


@pytest.mark.parametrize("where", ["model", "fallback"])
def test_reverify_sharing_a_verify_model_refused(tmp_path, where):
    # a trimmed summary is written by a verify model, so the re-verifier must be another model
    def m(d):
        if where == "model":
            d["knowledge_map"]["reverify"]["model"] = "gpt-oss-120b"
        else:
            d["knowledge_map"]["reverify"]["fallbacks"].append({"provider": "freellmapi", "model": "gpt-oss-120b"})
    with pytest.raises(ValueError, match=r"knowledge_map\.reverify.*gpt-oss-120b"):
        load_km_config(_cfg(tmp_path, m))


def test_staging_inside_vault_refused(tmp_path):
    def m(d):
        d["knowledge_map"]["staging_dir"] = str(tmp_path / "vault" / "maps")
    with pytest.raises(ValueError, match=r"knowledge_map\.staging_dir"):
        load_km_config(_cfg(tmp_path, m))


def test_missing_role_key_raises(tmp_path):
    with pytest.raises(ValueError, match=r"knowledge_map\.extract\.provider"):
        load_km_config(_cfg(tmp_path, lambda d: d["knowledge_map"]["extract"].pop("provider")))


def test_missing_section_raises(tmp_path):
    with pytest.raises(ValueError, match="knowledge_map"):
        load_km_config(_cfg(tmp_path, lambda d: d.pop("knowledge_map")))


def test_templates_load():
    ts = load_templates()
    assert set(ts) == {"concept", "comparison", "procedure", "outline"}
    assert ts["concept"].edge_types == NAVIGATOR_TEN
    assert ts["concept"].layer_edges == ("prerequisite-of", "leads-to")
    assert ts["comparison"].layer_edges == ()
    assert ts["procedure"].layer_edges == ("precedes",)
    assert ts["outline"].layer_edges == ("leads-to",)
    for t in ts.values():
        assert t.purpose and t.guidance and t.node_kinds
        assert set(t.layer_edges) <= set(t.edge_types)
    assert "caption" in ts["concept"].guidance          # figure rule (spec 3.7)


def _node(kind="concept", **kw):
    return NodeIn(**{"title": "T", "kind": kind, "summary": "s", "quotes": ["a b c d e"],
                     "importance": 3, **kw})


def test_check_extraction_flags_unknown_kind_and_type():
    t = load_templates()["concept"]
    ok = Extraction(nodes=[_node()], edges=[EdgeIn(source="a", target="b", type="uses")])
    assert check_extraction(ok, t) == []
    bad = Extraction(nodes=[_node("step")], edges=[EdgeIn(source="a", target="b", type="precedes")])
    errs = check_extraction(bad, t)
    assert len(errs) == 2
    assert any("step" in e for e in errs) and any("precedes" in e for e in errs)


def test_schema_limits():
    with pytest.raises(ValidationError):
        _node(quotes=["a"] * 4)
    with pytest.raises(ValidationError):
        _node(quotes=[])
    with pytest.raises(ValidationError):
        _node(importance=6)
    with pytest.raises(ValidationError):
        _node(title="x" * 81)
    with pytest.raises(ValidationError):
        _node(summary="x" * 601)
