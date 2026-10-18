"""GET /api/engines and POST /api/engines/set against the real probes.

The console is imported against the shipped example config (management_module),
in a temp working directory: every path the probes read resolves under tmp_path,
so no real index, cache or config.yaml is read or written.
"""
import subprocess
import sys
import types

import pytest
from fastapi.testclient import TestClient

from src.engines_map import COMPONENTS, SWITCH


@pytest.fixture
def client(management_module, monkeypatch):
    monkeypatch.delenv("RAG_API_TOKEN", raising=False)
    # A loopback Host: the request guard refuses TestClient's default "testserver".
    return TestClient(management_module.app, base_url="http://127.0.0.1")


class _Tripwire(types.ModuleType):
    """A module that records any attribute touched on it, then raises."""

    def __init__(self, name, reached):
        super().__init__(name)
        self._reached = reached

    def __getattr__(self, attr):
        self._reached.append(f"{self.__name__}.{attr}")
        raise AssertionError(f"{self.__name__}.{attr} was reached")


def _tripwires(monkeypatch, management_module, tmp_path):
    """Model loaders, Chroma, the hub, and /api/ocr/status's Tesseract
    subprocess all record and raise if reached; the sidecars refuse at once."""
    import huggingface_hub
    import requests
    import src.utils.chroma_client as chroma_client
    reached = []

    def trip(name):
        def _f(*a, **k):
            reached.append(name)
            raise AssertionError(f"{name} was reached")
        return _f

    for mod in ("torch", "sentence_transformers", "transformers", "chromadb"):
        monkeypatch.setitem(sys.modules, mod, _Tripwire(mod, reached))
    monkeypatch.setattr(huggingface_hub, "model_info", trip("huggingface_hub.model_info"))
    monkeypatch.setattr(subprocess, "run", trip("subprocess.run"))
    monkeypatch.setattr(chroma_client, "persistent_client", trip("persistent_client"))
    monkeypatch.setattr(management_module, "persistent_client", trip("persistent_client"))
    monkeypatch.setattr(management_module, "ocr_status", trip("/api/ocr/status"))

    def refused(*a, **k):
        raise requests.exceptions.ConnectionError("refused")
    monkeypatch.setattr(requests, "get", refused)
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path / "no-tessdata"))
    return reached


def test_engines_map_shape_with_fast_probes_only(client, management_module, monkeypatch, tmp_path):
    reached = _tripwires(monkeypatch, management_module, tmp_path)
    r = client.get("/api/engines")
    assert r.status_code == 200, r.text
    assert reached == []
    comps = r.json()["components"]
    assert list(comps) == list(COMPONENTS)
    for comp in comps.values():
        assert set(comp) == {"active", "setting", "model", "options"}
        for o in comp["options"]:
            assert o["switch"] in SWITCH
            if not (o["available"] and o["swappable"]):
                assert o["reason"], o
    assert comps["sparse"]["setting"] == "retrieval.sparse_backend"
    assert comps["fetch"]["setting"] is None
    ocr = {o["engine"]: o for o in comps["ocr"]["options"]}
    assert ocr["paddle"]["available"] is False and "unreachable" in ocr["paddle"]["reason"]
    emb = {o["engine"]: o for o in comps["embeddings"]["options"]}
    # The example config's store has no fingerprint, and eg2 has no collection.
    assert emb["obsidian_vault"]["swappable"] is False
    assert emb["eg2"]["switch"] == "reembed"


def test_engines_route_is_in_the_capability_map(management_module):
    entry = management_module.api_schema()["endpoints"]["GET /api/engines"]
    assert entry["permission"] == "read"


# ---- POST /api/engines/set ----------------------------------------------------

@pytest.fixture
def cfg_file(management_module, monkeypatch, tmp_path):
    """settings_update writes ROOT/config.yaml: point it at the temp copy the
    console loaded, and skip the torch import its device list costs."""
    monkeypatch.setattr(management_module, "ROOT", tmp_path)
    monkeypatch.setattr(management_module, "_torch_devices", lambda: ({}, ["auto", "cpu"]))
    return tmp_path / "config.yaml"


def _sidecar(tmp_path, name, provider):
    """A complete chunk-collection fingerprint, built by `provider` as the
    shipped config defines it."""
    from src.embeddings.registry import resolve_embedding_spec
    from src.embeddings.sidecar import make_sidecar, write_sidecar
    from src.utils.config_loader import load_config
    cfg = load_config()
    spec = resolve_embedding_spec(cfg, provider)
    write_sidecar(cfg.path("paths.chroma_dir"), name, make_sidecar(
        spec, collection=name, role="chunks", status="complete",
        dimension=spec.dimensions or 384, count=3, written_by="test"))


def test_a_refused_preflight_leaves_the_config_bytes_unchanged(client, cfg_file):
    before = cfg_file.read_bytes()
    r = client.post("/api/engines/set", json={"component": "sparse", "engine": "tantivy"})
    assert r.status_code == 400
    assert "tantivy" in r.json()["error"]          # no index here (or no package)
    assert cfg_file.read_bytes() == before


def test_an_unusable_cross_encoder_is_refused_before_anything_is_written(
        client, cfg_file, monkeypatch, tmp_path):
    import huggingface_hub
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(tmp_path / "no-st"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "no-hub"))

    def offline(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(huggingface_hub, "model_info", offline)
    before = cfg_file.read_bytes()
    r = client.post("/api/engines/set", json={"component": "reranker", "engine": "cross_encoder",
                                              "model": "cross-encoder/not-here"})
    assert r.status_code == 400 and "cannot be used" in r.json()["error"]
    assert cfg_file.read_bytes() == before


def test_a_switch_reports_what_it_wrote_and_how_it_takes_effect(
        client, cfg_file, monkeypatch):
    import requests

    def refused(*a, **k):                         # the OCR sidecars: down, at once
        raise requests.exceptions.ConnectionError("refused")
    monkeypatch.setattr(requests, "get", refused)

    r = client.post("/api/engines/set", json={"component": "sparse", "engine": "bm25s"})
    assert r.status_code == 200, r.text
    assert r.json() == {"written": {"retrieval.sparse_backend": "bm25s"},
                        "takes_effect": "after a restart of :8051",
                        "next_op": "POST /api/service/restart", "warnings": []}

    r = client.post("/api/engines/set", json={"component": "chunking", "engine": "fixed"})
    assert r.json()["written"] == {"parser.chunking": "fixed"}
    assert (r.json()["takes_effect"], r.json()["next_op"]) == ("after a restart of :8052", None)
    assert "  chunking: fixed              # heading | fixed" in cfg_file.read_text(encoding="utf-8")

    r = client.post("/api/engines/set", json={"component": "ocr", "engine": "none"})
    assert r.json()["written"] == {"pdf.ocr_engine": "none"}
    assert r.json()["next_op"] is None and "read per job" in r.json()["takes_effect"]


def test_an_embedding_switch_needs_the_configured_embedders_fingerprint(
        client, cfg_file, tmp_path):
    _sidecar(tmp_path, "vault_eg2", "eg2")
    _sidecar(tmp_path, "vault_copy", "local")
    before = cfg_file.read_bytes()
    r = client.post("/api/engines/set", json={"component": "embeddings", "engine": "vault_eg2"})
    assert r.status_code == 400 and "differ in" in r.json()["error"]
    assert cfg_file.read_bytes() == before

    r = client.post("/api/engines/set", json={"component": "embeddings", "engine": "vault_copy"})
    assert r.status_code == 200, r.text
    assert r.json()["written"] == {"paths.collection_name": "vault_copy"}
    assert r.json()["next_op"] == "POST /api/service/restart"
    # next_op restarts :8051 only; the console reads the collection once, at import
    assert any("restart :8052 by hand" in w for w in r.json()["warnings"])


def test_the_settings_writers_notes_reach_the_warnings(client, cfg_file):
    """A provider switch with no model carries the provider's default model along, and
    settings_update says so in a note: engines_set must pass that note on."""
    r = client.post("/api/engines/set", json={"component": "generation", "engine": "ollama"})
    assert r.status_code == 200, r.text
    assert r.json()["written"] == {"generation.provider": "ollama", "generation.model": "llama3.1:8b"}
    assert any("generation.model set to 'llama3.1:8b'" in w for w in r.json()["warnings"])


def test_tantivy_is_accepted_once_its_index_exists(client, cfg_file, tmp_path):
    pytest.importorskip("tantivy")
    index = tmp_path / "data" / "bm25_index.pkl.tantivy"
    index.mkdir(parents=True)
    (index / "noetrix_meta.json").write_text("{}", encoding="utf-8")
    r = client.post("/api/engines/set", json={"component": "sparse", "engine": "tantivy"})
    assert r.status_code == 200, r.text
    assert r.json()["written"] == {"retrieval.sparse_backend": "tantivy"}


@pytest.mark.parametrize("body, message", [
    ({"component": "vision", "engine": "x"}, "unknown component"),
    ({"component": "fetch", "engine": "crawl4ai"}, "chosen per request"),
    ({"component": "sparse", "engine": "bm25s", "model": "m"}, "model applies only"),
    ({"component": "reranker", "engine": "http"}, "per call"),
])
def test_unknown_or_unswappable_switches_are_refused(client, cfg_file, body, message):
    before = cfg_file.read_bytes()
    r = client.post("/api/engines/set", json=body)
    assert r.status_code == 400 and message in r.json()["error"]
    assert cfg_file.read_bytes() == before


def test_sparse_backend_is_an_editable_setting_and_set_is_mapped(management_module):
    spec = management_module.EDITABLE_SETTINGS["retrieval.sparse_backend"]
    assert (spec["kind"], spec["values"], spec["restart"]) == ("enum", ["bm25s", "tantivy"], ":8051")
    entry = management_module.api_schema()["endpoints"]["POST /api/engines/set"]
    assert entry["permission"] == "mutating"


def test_the_hub_check_has_an_explicit_timeout(management_module, monkeypatch, tmp_path):
    """model_info's own default is no timeout at all; set runs this check."""
    import huggingface_hub
    seen = []

    def record(model, **kw):
        seen.append(kw.get("timeout"))
        raise OSError("offline")
    monkeypatch.setattr(huggingface_hub, "model_info", record)
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(tmp_path / "no-st"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "no-hub"))
    management_module.rerank_check(management_module.RerankCheckIn(model="cross-encoder/x"))
    assert len(seen) == 2 and set(seen) == {management_module._HUB_TIMEOUT_S}


def test_the_engine_tools_are_built_from_the_live_console_schema(client):
    """Part 2's builder, over a catalog captured from the real console: the read
    tool carries readOnlyHint, the mutating one the prompt flag."""
    import json
    from src.mcp_bridge.catalog import build_routes, console_tiers
    from src.mcp_bridge.client import Catalog
    from src.mcp_bridge.tools import PROMPT_META, build_tool_list
    tiers, purposes, job_kind_tiers = console_tiers(client.get("/api/schema").json())
    routes = build_routes(client.get("/openapi.json").json(), tiers, purposes, "console")
    assert routes["GET /api/engines"].tier == "read"
    assert routes["POST /api/engines/set"].tier == "mutating"
    catalog = Catalog(routes=routes, job_kind_tiers=job_kind_tiers,
                      missing={"query": "not started in this test"})
    tools = {t.name: t for t in build_tool_list(catalog)}
    engines, set_engine = tools["noetrix_engines"], tools["noetrix_set_engine"]
    assert engines.annotations.read_only_hint is True
    assert not (engines.meta or {}).get("anthropic/requiresUserInteraction")
    assert not set_engine.annotations.read_only_hint
    assert not set_engine.annotations.destructive_hint
    assert PROMPT_META.items() <= (set_engine.meta or {}).items()
    schema = json.dumps(set_engine.input_schema)
    assert '"component"' in schema and '"engine"' in schema and "$ref" not in schema


def test_parallel_writers_write_config_yaml_one_at_a_time(management_module, cfg_file, monkeypatch):
    """Review Focus 2. Agents call noetrix_set_engine in parallel, the Settings tab's Save
    writes the same file, and FastAPI runs these sync handlers in a thread pool. Every
    in-process writer goes through _persist_section_keys's read-edit-write, so the lock is
    taken there. It is taken INSIDE that function, so the slow step goes inside it too:
    _yaml_key_index runs between the read and the write."""
    import threading
    import time

    from src.utils.config_loader import load_config

    m = management_module
    real, active, peak = m._yaml_key_index, [0], [0]

    def slow_index(lines):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        time.sleep(0.3)
        try:
            return real(lines)
        finally:
            active[0] -= 1

    monkeypatch.setattr(m, "_yaml_key_index", slow_index)
    writers = (lambda: m.engines_set(m.EngineSetIn(component="sparse", engine="bm25s")),
               lambda: m.engines_set(m.EngineSetIn(component="chunking", engine="fixed")),
               lambda: m.settings_update(m.SettingsIn(changes={"pdf.ocr_engine": "none"})))
    results = []
    threads = [threading.Thread(target=lambda w=w: results.append(w())) for w in writers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1
    assert len(results) == 3 and all(isinstance(r, dict) for r in results), results
    disk = load_config(cfg_file)
    assert (disk.get("parser.chunking"), disk.get("pdf.ocr_engine")) == ("fixed", "none")
    assert disk.get("retrieval.sparse_backend") == "bm25s"
