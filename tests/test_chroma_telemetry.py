"""Chroma's anonymised product telemetry stays OFF for every client this repo opens.

The corpus is personal (lecture notes, homework, notebooks), and Chroma reports
usage unless a client is built with Settings(anonymized_telemetry=False). Every
store is opened through src.utils.chroma_client.persistent_client, so the setting
lives in one place. It has to: Chroma caches a client system per store path and
raises ValueError ("An instance of Chroma already exists ... with different
settings") when a second client on that path disagrees, so one call site that
forgot the setting would turn the next client opened in the same process into a
crash, not just a telemetry leak.

tests/conftest.py sets ANONYMIZED_TELEMETRY=False for every test, so that a test's
own bare chromadb.PersistentClient(path) agrees with the code it is testing. That
would let a "the client has telemetry off" check pass even if the code forgot the
setting, so the retriever test looks at what the code PASSES to chromadb, and the
helper test turns the environment variable the other way first.
"""
import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.retriever import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "src" / "utils" / "chroma_client.py"

# Every way chromadb builds a client. The helper is the only caller allowed.
CLIENT_CONSTRUCTORS = {"PersistentClient", "Client", "EphemeralClient", "HttpClient",
                       "AsyncHttpClient", "CloudClient", "AdminClient"}


class _SpyClient:
    def get_collection(self, name):
        return f"collection:{name}"


def test_the_retriever_opens_its_store_with_telemetry_off(monkeypatch, tmp_path):
    pytest.importorskip("chromadb")
    opened = []

    def fake_persistent_client(path=None, settings=None, **kwargs):
        opened.append({"path": str(path), "settings": settings})
        return _SpyClient()

    monkeypatch.setattr("chromadb.PersistentClient", fake_persistent_client)
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.chroma_dir = tmp_path / "chroma"
    retriever.collection_name = "chunks_coll"
    retriever._collection = None

    assert retriever._get_collection() == "collection:chunks_coll"
    (client,) = opened
    assert client["path"] == str(tmp_path / "chroma")
    assert client["settings"] is not None, "the retriever passed no settings to chromadb"
    assert client["settings"].anonymized_telemetry is False


def test_the_helper_builds_a_working_client_with_telemetry_off_whatever_the_environment_says(
        monkeypatch, tmp_path):
    pytest.importorskip("chromadb")
    from src.utils.chroma_client import persistent_client
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "True")      # what Chroma's own default would read
    client = persistent_client(tmp_path / "store")
    assert client.get_settings().anonymized_telemetry is False
    assert client.get_or_create_collection("probe_collection").count() == 0
    again = persistent_client(str(tmp_path / "store"))      # a str path, a second client: no conflict
    assert again.get_settings() == client.get_settings()


def _source_files():
    yield from sorted(ROOT.glob("*.py"))
    for folder in ("src", "eval", "tools"):
        yield from sorted((ROOT / folder).rglob("*.py"))


def test_no_module_opens_a_chroma_store_except_through_the_helper():
    offenders = []
    for path in _source_files():
        if path == HELPER:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            direct = (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                      and node.func.attr in CLIENT_CONSTRUCTORS
                      and isinstance(node.func.value, ast.Name)
                      and node.func.value.id == "chromadb")
            imported = (isinstance(node, ast.ImportFrom) and node.module == "chromadb"
                        and any(alias.name in CLIENT_CONSTRUCTORS for alias in node.names))
            if direct or imported:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, (
        "open the store with src.utils.chroma_client.persistent_client (telemetry off, one "
        f"settings object for the whole process), not chromadb directly: {offenders}")
