"""GET /api/vaults and the registry file it reads and rewrites.

A fresh source clone has no data/ until something ingests, and the vault
registry lives at ROOT/data/.vault_registry.json. The first GET registers the
current vault, so it writes that file: with the folder missing, the console's
Settings > Vaults panel got a 500 on first run.

A registry that exists but cannot be read is a different case: reading it as
"no vaults yet" made that same GET save a registry holding only the current
vault, dropping every other vault and its saved settings.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_get_vaults_succeeds_before_the_data_folder_exists(
        management_module, monkeypatch, tmp_path):
    # The real registry sits beside manage_api.py, in the checkout's own data/:
    # point it at a temp root that has no data/ folder.
    registry = tmp_path / "data" / ".vault_registry.json"
    monkeypatch.setattr(management_module, "_VAULT_REGISTRY", registry)
    assert not registry.parent.exists()

    client = TestClient(management_module.app, base_url="http://127.0.0.1")
    r = client.get("/api/vaults")

    assert r.status_code == 200, r.text
    # The shipped config names a vault, so this GET self-registered it. Without
    # the write, the test would pass without ever reaching the missing folder.
    assert registry.exists()


@pytest.mark.parametrize("raw", [
    # An in-place write cut short: the other vault's entry is half on disk.
    pytest.param(b'{\n  "vaults": [\n    {\n      "path": "D:/Notes/Other vault",'
                 b'\n      "lab', id="truncated-json"),
    # Valid JSON, but no "vaults" list to read.
    pytest.param(b'{"vault": [{"path": "D:/Notes/Other vault", "label": "Other vault"}]}',
                 id="no-vaults-key"),
])
def test_get_vaults_leaves_a_registry_it_cannot_read_untouched(
        management_module, monkeypatch, tmp_path, raw):
    registry = tmp_path / "data" / ".vault_registry.json"
    registry.parent.mkdir()
    registry.write_bytes(raw)
    monkeypatch.setattr(management_module, "_VAULT_REGISTRY", registry)

    client = TestClient(management_module.app, base_url="http://127.0.0.1",
                        raise_server_exceptions=False)
    r = client.get("/api/vaults")

    # The shipped config names a vault this file does not list: taking the read
    # error for "no vaults yet" self-registers that vault and saves over the file.
    assert registry.read_bytes() == raw
    assert r.status_code == 500


def test_a_save_that_dies_midway_leaves_the_previous_registry_whole(
        management_module, monkeypatch, tmp_path):
    registry = tmp_path / "data" / ".vault_registry.json"
    registry.parent.mkdir()
    before = b'{"vaults": [{"path": "D:/Notes/Other vault", "label": "Other vault"}]}'
    registry.write_bytes(before)
    monkeypatch.setattr(management_module, "_VAULT_REGISTRY", registry)

    def dies_midway(real_write):
        def write(self, data, *args, **kwargs):
            real_write(self, data[:20], *args, **kwargs)    # half a file reaches the disk...
            raise OSError("power cut")                      # ...and then the process dies
        return write

    # Both write methods die, so the test holds whichever one the save uses.
    monkeypatch.setattr(Path, "write_text", dies_midway(Path.write_text))
    monkeypatch.setattr(Path, "write_bytes", dies_midway(Path.write_bytes))
    with pytest.raises(OSError, match="power cut"):
        management_module._save_vaults([{"path": "D:/Notes/New vault", "label": "New vault"}])

    assert registry.read_bytes() == before                                  # the old file is whole
    assert [p.name for p in registry.parent.iterdir()] == [registry.name]   # and no temp is left
