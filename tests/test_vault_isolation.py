"""The vault switcher keeps each vault's files and embedder its own.

Every switch rewrites tmp_path/config.yaml (the fixture's copy of the example
config); the registry is pinned under tmp_path, so nothing touches the live one.
"""
import json

import yaml


def _setup(m, monkeypatch, tmp_path):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    registry = tmp_path / "data" / ".vault_registry.json"
    monkeypatch.setattr(m, "_VAULT_REGISTRY", registry)
    registry.parent.mkdir(exist_ok=True)
    return registry


def _disk(tmp_path, dotted):
    node = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    for part in dotted.split("."):
        node = node[part]
    return node


def test_a_fresh_vault_gets_its_own_files_and_no_vault_relative_folders(
        management_module, monkeypatch, tmp_path):
    m = management_module
    _setup(m, monkeypatch, tmp_path)
    model = _disk(tmp_path, "embedding.local_model")
    new = tmp_path / "Vault B"
    new.mkdir()
    assert m.vaults_switch(m.VaultSwitchIn(path=str(new)))["scaffolded"] is True
    own = "data/vaults/vault_b"
    for key, base in [("pdf.output_file", "pdf_chunks.jsonl"),
                      ("notebooks.output_file", "ipynb_chunks.jsonl"),
                      ("code.output_file", "code_chunks.jsonl"),
                      ("canvas.output_file", "canvas_chunks.jsonl"),
                      ("pdf.figures_dir", "figures"),
                      ("notebooks.figures_dir", "notebook_figures"),
                      ("canvas.backup_dir", "canvas_backups"),
                      ("knowledge_map.staging_dir", "canvas_maps/staging")]:
        assert _disk(tmp_path, key) == f"{own}/{base}", key
    assert _disk(tmp_path, "webui.inbox_dir") == "Inbox"
    assert _disk(tmp_path, "ocr_studio.upload_dir") == "OCR Studio"
    assert _disk(tmp_path, "graph.embedding_provider") is None
    assert _disk(tmp_path, "embedding.local_model") == model     # inherited


def test_each_vault_keeps_the_embedder_it_was_given(management_module, monkeypatch, tmp_path):
    m = management_module
    _setup(m, monkeypatch, tmp_path)
    vault_a = tmp_path / "Vault A"
    vault_b = tmp_path / "Vault B"
    vault_a.mkdir()
    vault_b.mkdir()
    m.vaults_switch(m.VaultSwitchIn(path=str(vault_a)))
    a_model = _disk(tmp_path, "embedding.local_model")
    m.vaults_switch(m.VaultSwitchIn(path=str(vault_b)))
    m._persist_section_keys(tmp_path / "config.yaml", {
        "embedding.local_model": "intfloat/multilingual-e5-base",
        "embedding.query_prefix": "query: ", "embedding.doc_prefix": "passage: "})

    m.vaults_switch(m.VaultSwitchIn(path=str(vault_a)))
    assert _disk(tmp_path, "embedding.local_model") == a_model
    assert _disk(tmp_path, "embedding.query_prefix") == ""
    assert _disk(tmp_path, "pdf.output_file") == "data/vaults/vault_a/pdf_chunks.jsonl"

    m.vaults_switch(m.VaultSwitchIn(path=str(vault_b)))
    assert _disk(tmp_path, "embedding.local_model") == "intfloat/multilingual-e5-base"
    assert _disk(tmp_path, "embedding.query_prefix") == "query: "
    assert _disk(tmp_path, "embedding.doc_prefix") == "passage: "
    assert _disk(tmp_path, "pdf.output_file") == "data/vaults/vault_b/pdf_chunks.jsonl"


def test_an_old_registry_entry_is_backfilled_from_its_own_dir_and_sidecar(
        management_module, monkeypatch, tmp_path):
    """Registered before these keys existed: lane files go beside ITS chunks, and
    the embedder is the one its collection's fingerprint names."""
    m = management_module
    registry = _setup(m, monkeypatch, tmp_path)
    old, chroma = tmp_path / "Old Vault", tmp_path / "old_chroma"
    old.mkdir()
    chroma.mkdir()
    (chroma / "notes.embedding.json").write_text(json.dumps({
        "sidecar_version": 1, "collection": "notes", "role": "chunks",
        "status": "complete", "provider": "local", "kind": "local",
        "model": "BAAI/bge-m3", "dimension": 1024, "normalize": True,
        "query_prefix": "", "doc_prefix": ""}), encoding="utf-8")
    registry.write_text(json.dumps({"vaults": [{"path": str(old), "label": "old", "settings": {
        "parser.vault_path": old.as_posix(),
        "paths.chunks_file": "data/vaults/old/chunks.jsonl",
        "paths.chroma_dir": chroma.as_posix(), "paths.collection_name": "notes"}}]}),
        encoding="utf-8")
    assert m.vaults_switch(m.VaultSwitchIn(path=str(old)))["scaffolded"] is False
    assert _disk(tmp_path, "embedding.local_model") == "BAAI/bge-m3"
    assert _disk(tmp_path, "pdf.output_file") == "data/vaults/old/pdf_chunks.jsonl"
    assert _disk(tmp_path, "canvas.output_file") == "data/vaults/old/canvas_chunks.jsonl"


def test_a_null_setting_survives_the_round_trip(management_module, monkeypatch, tmp_path):
    m = management_module
    _setup(m, monkeypatch, tmp_path)
    assert _disk(tmp_path, "graph.embedding_provider") is None
    vault_a, vault_b = tmp_path / "A", tmp_path / "B"
    vault_a.mkdir()
    vault_b.mkdir()
    m.vaults_switch(m.VaultSwitchIn(path=str(vault_a)))
    m.vaults_switch(m.VaultSwitchIn(path=str(vault_b)))
    m.vaults_switch(m.VaultSwitchIn(path=str(vault_a)))
    assert _disk(tmp_path, "graph.embedding_provider") is None
