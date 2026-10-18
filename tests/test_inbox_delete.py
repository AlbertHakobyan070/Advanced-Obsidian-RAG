"""POST /api/inbox/delete: a permanent delete must reach exactly the file meant.

The inbox and its _converted staging folder can hold the same filename (a
conversion keeps its source's name). The delete takes `folder` to say which;
without it, a name both hold is refused before anything is deleted.
"""
import pytest
from fastapi.testclient import TestClient

from src.utils.config_loader import Config


@pytest.fixture
def inbox(management_module, monkeypatch, tmp_path):
    m = management_module
    vault = tmp_path / "vault"
    monkeypatch.setattr(m, "CFG", Config({"parser": {"vault_path": str(vault)},
                                          "webui": {"inbox_dir": "Inbox"}}, tmp_path))
    box = vault / "Inbox"
    (box / "_converted").mkdir(parents=True)
    return TestClient(m.app, base_url="http://127.0.0.1"), box


def _delete(c, **body):
    return c.post("/api/inbox/delete", json=body)


def _both(box, name="a.md"):
    for f in (box / name, box / "_converted" / name):
        f.write_text("x", encoding="utf-8")


def test_a_name_in_both_folders_is_refused_without_a_folder(inbox):
    c, box = inbox
    _both(box)
    r = _delete(c, names=["a.md"])
    assert r.status_code == 400 and "folder" in r.json()["error"], r.text
    assert (box / "a.md").is_file() and (box / "_converted" / "a.md").is_file()


def test_folder_picks_exactly_one_copy(inbox):
    c, box = inbox
    _both(box)
    assert _delete(c, names=["a.md"], folder="converted").json()["removed"] == ["a.md"]
    assert (box / "a.md").is_file() and not (box / "_converted" / "a.md").exists()
    _both(box)
    assert _delete(c, names=["a.md"], folder="inbox").json()["removed"] == ["a.md"]
    assert not (box / "a.md").exists() and (box / "_converted" / "a.md").is_file()


def test_without_a_folder_an_unambiguous_name_is_deleted_wherever_it_is(inbox):
    c, box = inbox
    (box / "only.md").write_text("x", encoding="utf-8")
    (box / "_converted" / "conv.md").write_text("x", encoding="utf-8")
    j = _delete(c, names=["only.md", "conv.md", "gone.md", "../x.md"]).json()
    assert (j["removed"], j["missing"]) == (["only.md", "conv.md"], ["gone.md", "../x.md"])
    assert not (box / "only.md").exists() and not (box / "_converted" / "conv.md").exists()


def test_a_folder_never_reaches_into_the_other(inbox):
    c, box = inbox
    (box / "_converted" / "conv.md").write_text("x", encoding="utf-8")
    j = _delete(c, names=["conv.md"], folder="inbox").json()
    assert (j["removed"], j["missing"]) == ([], ["conv.md"])
    assert (box / "_converted" / "conv.md").is_file()
    assert _delete(c, names=["conv.md"], folder="elsewhere").status_code == 422
