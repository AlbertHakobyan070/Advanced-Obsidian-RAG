"""The embedding fingerprint sidecar: its file, its digest and its diff
(embedding-switch slice 2).

Run:  python -m pytest tests/ -q

A collection's vectors only mean something to the embedder that built them, so
every collection carries `<chroma_dir>/<collection>.embedding.json` and the
startup guard compares it with the configured embedder. This file covers the
pieces with no Chroma and no model in them; the guard, the write paths and
`stamp` are in test_embedding_guard.py and test_embedding_stamp.py.
"""
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.embeddings.registry import EmbeddingSpec
from src.embeddings.sidecar import (
    delete_sidecar, fingerprint_diff, ids_digest, make_sidecar, read_sidecar,
    sidecar_path, write_sidecar)

SPEC = EmbeddingSpec("local", "local", "BAAI/bge-small-en-v1.5")


def _sidecar(spec=SPEC, **fields):
    args = dict(collection="obsidian_vault", role="chunks", status="complete",
                dimension=384, count=10, written_by="stamp")
    args.update(fields)
    return make_sidecar(spec, **args)


# ------------------------------------------------------------------ file I/O ----

def test_sidecar_round_trips_and_leaves_no_temp_file(tmp_path):
    chroma = tmp_path / "not" / "yet" / "chroma_db"          # written before the store exists
    # A trailing space is part of a prefix, and a non-ASCII one must survive a
    # machine whose default text encoding is not UTF-8.
    spec = EmbeddingSpec("local", "local", "intfloat/multilingual-e5-base",
                         query_prefix="запрос: ", doc_prefix="passage: ")
    data = _sidecar(spec, dimension=768, source_digest=ids_digest(["a", "b"]),
                    hype_collection="hype_questions")

    path = write_sidecar(chroma, "obsidian_vault", data)

    assert path == chroma / "obsidian_vault.embedding.json" == sidecar_path(chroma, "obsidian_vault")
    assert read_sidecar(chroma, "obsidian_vault") == data
    assert json.loads(path.read_text(encoding="utf-8"))["doc_prefix"] == "passage: "
    assert [p.name for p in chroma.iterdir()] == ["obsidian_vault.embedding.json"]   # no .tmp left behind

    # An absent file is the one case that reads as None ...
    assert read_sidecar(chroma, "other_collection") is None
    # ... and a rewrite replaces in place.
    write_sidecar(chroma, "obsidian_vault", {**data, "count": 11})
    assert read_sidecar(chroma, "obsidian_vault")["count"] == 11
    assert [p.name for p in chroma.iterdir()] == ["obsidian_vault.embedding.json"]

    assert delete_sidecar(chroma, "obsidian_vault") is True
    assert delete_sidecar(chroma, "obsidian_vault") is False
    assert read_sidecar(chroma, "obsidian_vault") is None


def test_an_unreadable_or_incomplete_sidecar_is_an_error_naming_the_file(tmp_path):
    """Never treated as "no sidecar": that would let a half-written or hand-damaged
    file unlock a collection the guard exists to protect."""
    path = sidecar_path(tmp_path, "obsidian_vault")
    good = _sidecar()

    path.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(ValueError, match=r"obsidian_vault\.embedding\.json.*JSON"):
        read_sidecar(tmp_path, "obsidian_vault")

    path.write_text(json.dumps([good]), encoding="utf-8")
    with pytest.raises(ValueError, match=r"obsidian_vault\.embedding\.json.*JSON object"):
        read_sidecar(tmp_path, "obsidian_vault")

    for field in ("model", "doc_prefix", "dimension"):
        path.write_text(json.dumps({k: v for k, v in good.items() if k != field}), encoding="utf-8")
        with pytest.raises(ValueError, match=rf"obsidian_vault\.embedding\.json.*missing.*{field}"):
            read_sidecar(tmp_path, "obsidian_vault")

    # A file renamed or copied for another collection describes the wrong one.
    path.write_text(json.dumps({**good, "collection": "some_other_collection"}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"obsidian_vault\.embedding\.json.*some_other_collection"):
        read_sidecar(tmp_path, "obsidian_vault")

    # Values outside the vocabulary are refused too: a typo'd status must not read as complete.
    for bad in ({"status": "finished"}, {"role": "questions"}, {"sidecar_version": 2}):
        path.write_text(json.dumps({**good, **bad}), encoding="utf-8")
        with pytest.raises(ValueError, match=rf"obsidian_vault\.embedding\.json.*{next(iter(bad))}"):
            read_sidecar(tmp_path, "obsidian_vault")

    # Writing is held to the same standard, and nothing is left behind when it refuses.
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        write_sidecar(tmp_path, "obsidian_vault", {k: v for k, v in good.items() if k != "model"})
    with pytest.raises(ValueError, match="some_other_collection"):
        write_sidecar(tmp_path, "obsidian_vault", {**good, "collection": "some_other_collection"})
    assert list(tmp_path.iterdir()) == []

    with pytest.raises(ValueError, match="unknown field"):
        _sidecar(typo_field=1)


# ------------------------------------------------------------------- digest ----

def test_ids_digest_is_order_independent_and_membership_sensitive():
    expected = "sha256:" + hashlib.sha256("a\nb\nc".encode("utf-8")).hexdigest()
    assert ids_digest(["c", "a", "b"]) == ids_digest(["a", "b", "c"]) == expected
    assert ids_digest(iter(["b", "c", "a"])) == expected              # any iterable, a generator included
    assert ids_digest(["a", "b"]) != ids_digest(["a", "c"])           # a different member
    assert ids_digest(["a", "b"]) != expected                         # a missing member
    assert ids_digest(["a", "a", "b", "c"]) == expected               # a set: Chroma holds one row per id
    assert ids_digest([]).startswith("sha256:")


# --------------------------------------------------------------------- diff ----

def test_fingerprint_diff_names_each_differing_field():
    base = _sidecar()
    assert fingerprint_diff(SPEC, 384, base, role="chunks") == []

    assert fingerprint_diff(replace(SPEC, model="intfloat/multilingual-e5-base"), 384, base,
                            role="chunks") == ["model"]
    assert fingerprint_diff(replace(SPEC, doc_prefix="passage: "), 384, base, role="chunks") == ["doc_prefix"]
    assert fingerprint_diff(replace(SPEC, query_prefix="query: "), 384, base, role="chunks") == ["query_prefix"]
    assert fingerprint_diff(SPEC, 384, base, role="hype") == ["role"]
    assert fingerprint_diff(SPEC, 768, base, role="chunks") == ["dimension"]
    assert fingerprint_diff(EmbeddingSpec("hosted", "openai", SPEC.model), 384, base,
                            role="chunks") == ["kind", "normalize"]

    # A hosted endpoint with no `dimensions` cannot say: not compared, not a mismatch.
    assert fingerprint_diff(SPEC, None, base, role="chunks") == []

    # Several at once, in a fixed order (the order the guard's message lists them).
    assert fingerprint_diff(replace(SPEC, model="other", doc_prefix="p: ", query_prefix="q: "), 768, base,
                            role="chunks") == ["model", "dimension", "doc_prefix", "query_prefix"]

    # What does not change the vector space is not compared.
    assert fingerprint_diff(replace(SPEC, provider="renamed", base_url="http://elsewhere/v1",
                                    device="cpu", dimensions=99), 384, base, role="chunks") == []
