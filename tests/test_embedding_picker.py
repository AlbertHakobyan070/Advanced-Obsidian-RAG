"""Settings' embedding model picker: GET /api/settings lists the embedders by
name with the collections each built, and POST accepts the keys it fills.

Against the management_module fixture's own config.yaml (tmp_path); the Chroma
dir is a tmp folder holding one fingerprint sidecar, so no store is opened.
"""
import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _tracked_config_untouched():
    path = ROOT / "config.yaml"
    before = path.read_bytes() if path.exists() else None
    yield
    assert (path.read_bytes() if path.exists() else None) == before, "a test wrote the tracked config.yaml"


def _client(m, monkeypatch, tmp_path):
    monkeypatch.setattr(m, "ROOT", tmp_path)     # POST writes ROOT/config.yaml
    return TestClient(m.app, base_url="http://127.0.0.1")


def _sidecar(chroma, name, provider, model):
    (chroma / f"{name}.embedding.json").write_text(json.dumps({
        "sidecar_version": 1, "collection": name, "role": "chunks", "status": "complete",
        "provider": provider, "kind": "local", "model": model, "dimension": 768,
        "normalize": True, "query_prefix": "", "doc_prefix": ""}), encoding="utf-8")


def test_settings_lists_embedders_with_the_collections_they_built(management_module, monkeypatch, tmp_path):
    m = management_module
    chroma = tmp_path / "chroma"
    chroma.mkdir()
    _sidecar(chroma, "notes_base", "local", "BAAI/bge-base-en-v1.5")
    m._persist_section_keys(tmp_path / "config.yaml", {"paths.chroma_dir": chroma.as_posix()})
    s = _client(m, monkeypatch, tmp_path).get("/api/settings").json()
    by_model = {(e["provider"], e["model"]): e for e in s["embedders"]}
    assert by_model[("local", "BAAI/bge-base-en-v1.5")]["collections"] == ["notes_base"]
    assert by_model[("local", "BAAI/bge-small-en-v1.5")]["collections"] == []
    assert "local" in s["editable"]["embedding.provider"]["values"]
    # Collection -> the embedder that built it, prefixes included: the panel
    # switches the model to match a picked collection.
    [c] = s["embed_collections"]
    assert (c["name"], c["provider"], c["model"], c["query_prefix"], c["doc_prefix"]) == (
        "notes_base", "local", "BAAI/bge-base-en-v1.5", "", "")


def test_saving_a_pick_keeps_the_prefix_trailing_space(management_module, monkeypatch, tmp_path):
    m = management_module
    r = _client(m, monkeypatch, tmp_path).post("/api/settings", json={"changes": {
        "embedding.provider": "local", "embedding.local_model": "intfloat/multilingual-e5-base",
        "embedding.query_prefix": "query: ", "embedding.doc_prefix": "passage: "}})
    assert r.status_code == 200, r.text
    disk = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))["embedding"]
    assert (disk["local_model"], disk["query_prefix"], disk["doc_prefix"]) == (
        "intfloat/multilingual-e5-base", "query: ", "passage: ")


def test_an_unknown_provider_or_a_bad_prefix_is_refused(management_module, monkeypatch, tmp_path):
    c = _client(management_module, monkeypatch, tmp_path)
    assert c.post("/api/settings", json={"changes": {"embedding.provider": "nope"}}).status_code == 400
    assert c.post("/api/settings", json={"changes": {"embedding.query_prefix": "a # b"}}).status_code == 400
