"""The optional second LLM pass (generation.verify_citations): its verdicts must
reach the citations whichever way the model spells the number, and reach the caller."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import serve_api
from src.generation.generator import Generator
from src.retrieval.retriever import RetrievedDoc

DOCS = [RetrievedDoc(id="a", text="alpha", metadata={"filename": "a.md"}),
        RetrievedDoc(id="b", text="beta", metadata={"filename": "b.md"})]
ANSWER = "Alpha [1] and beta [2].\n\nCONFIDENCE: HIGH"


class _LLM:
    provider, model, backend = "openai", "fake", "fake"

    def __init__(self, audit):
        self.replies = [ANSWER, audit]

    def complete(self, **kw):
        return SimpleNamespace(text=self.replies.pop(0), usage=None)


def _audit(n1, n2):
    return json.dumps({"overall": "FLAG", "verdicts": [
        {"citation": n1, "supported": True, "note": ""},
        {"citation": n2, "supported": False, "note": "not in the source"}]})


@pytest.mark.parametrize("n1, n2", [(1, 2), ("1", "2")])
def test_verdicts_reach_the_citations_as_int_or_string(n1, n2):
    ans = Generator(_LLM(_audit(n1, n2))).generate("q", DOCS)
    assert [c.supported for c in ans.citations] == [True, False]
    assert ans.citations[1].note == "not in the source"


def test_query_returns_the_verification(monkeypatch):
    gen = Generator(_LLM(_audit(1, 2)))
    rag = SimpleNamespace(generator=gen, search=lambda *a, **k: (DOCS, {"rerank_mode": "lexical"}))
    monkeypatch.setitem(serve_api._STATE, "rag", rag)
    monkeypatch.setattr(serve_api, "_generator_for", lambda provider, model: gen)
    out = serve_api.query(serve_api.QueryIn(q="q"))
    assert out.verification["overall"] == "FLAG"
    assert [c.supported for c in out.citations] == [True, False]


def test_with_verification_off_the_fields_are_empty(monkeypatch):
    gen = Generator(SimpleNamespace(provider="openai", model="fake", backend="fake",
                                    complete=lambda **kw: SimpleNamespace(text=ANSWER, usage=None)),
                    verify_citations=False)
    rag = SimpleNamespace(generator=gen, search=lambda *a, **k: (DOCS, {"rerank_mode": "lexical"}))
    monkeypatch.setitem(serve_api._STATE, "rag", rag)
    monkeypatch.setattr(serve_api, "_generator_for", lambda provider, model: gen)
    out = serve_api.query(serve_api.QueryIn(q="q"))
    assert out.verification is None
    assert [c.supported for c in out.citations] == [None, None]
