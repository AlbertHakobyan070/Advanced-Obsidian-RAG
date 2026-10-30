"""Generator provenance (slice D, E5) - src/generation/generator.py.

No LLM is constructed for the context lock: _format_context is a staticmethod.
The literal below was captured on the code before slice D, so with
provenance.label_generated off the numbered sources the model reads stay
byte-identical, even for a chunk that carries provenance: generated.
"""
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.generation.generator import Generator
from src.retrieval.retriever import RetrievedDoc
from src.utils.config_loader import Config


def _docs():
    return [
        RetrievedDoc(id="p1", text="  The soft margin adds slack variables.\n",
                     metadata={"file_type": "pdf", "filename": "Bishop PRML.pdf",
                               "source_file": "Books\\Bishop PRML.pdf",
                               "chapter": "Kernel Methods", "page_start": 332,
                               "page_end": 332, "provenance": "source"}),
        RetrievedDoc(id="n1", text="C trades margin width for violations.",
                     metadata={"file_type": "note", "filename": "SVM",
                               "source_file": "Notes\\SVM.md",
                               "course_name": "Machine Learning",
                               "heading": "Soft margin", "provenance": "authored"}),
        RetrievedDoc(id="c1", text="[Canvas: svm map (generated)]\nSoft margin SVM",
                     metadata={"file_type": "canvas", "filename": "svm map (generated)",
                               "source_file": "Maps/svm map (generated).canvas",
                               "canvas_node_id": "n1", "provenance": "generated"}),
    ]


# Captured from Generator._format_context(_docs()) on the unmodified code.
EXPECTED = ('[1] (from Bishop PRML.pdf \u203a Kernel Methods, p.332)\nThe soft margin adds slack variables.\n\n'
            '[2] (from Machine Learning \u203a Soft margin)\nC trades margin width for violations.\n\n'
            '[3] (from svm map (generated))\n[Canvas: svm map (generated)]\nSoft margin SVM')


def test_default_context_is_byte_identical():
    assert Generator._format_context(_docs()) == EXPECTED


class _FakeLLM:
    """Returns one canned reply and keeps every user prompt it was sent."""
    provider, model, backend = "openai", "fake", "fake"

    def __init__(self, reply):
        self.reply, self.users = reply, []

    def complete(self, **kw):
        self.users.append(kw["user"])
        return SimpleNamespace(text=self.reply, usage=None)


def _from_config(tmp_path, reply, **prov):
    cfg = Config({"generation": {"verify_citations": False}, "provenance": prov}, tmp_path)
    llm = _FakeLLM(reply)
    return Generator.from_config(cfg, llm), llm


def test_label_generated_marks_only_generated_sources(tmp_path):
    gen, llm = _from_config(tmp_path, "Slack [1].\n\nCONFIDENCE: HIGH", label_generated=True)
    gen.generate("q", _docs())
    user = llm.users[0]
    assert user.count("generated, not a primary source") == 1
    assert "[3] (from svm map (generated); generated, not a primary source)\n" in user
    assert "[1] (from Bishop PRML.pdf › Kernel Methods, p.332)\n" in user
    assert "[2] (from Machine Learning › Soft margin)\n" in user


def test_generated_only_citations_cap_confidence_and_say_so(tmp_path):
    canvas, pdf = _docs()[2], _docs()[0]
    gen, llm = _from_config(tmp_path, "The map says so [1].\n\nCONFIDENCE: HIGH",
                            generated_only_confidence="MEDIUM")
    ans = gen.generate("q", [canvas, pdf])
    assert ans.confidence == "MEDIUM"
    assert ans.confidence_note == "capped at MEDIUM: every cited source is generated"
    # The cap alone never relabels the context the model reads.
    assert "not a primary source" not in llm.users[0]
    # Already at or below the cap: untouched.
    low, _ = _from_config(tmp_path, "The map says so [1].\n\nCONFIDENCE: LOW",
                          generated_only_confidence="MEDIUM")
    ans = low.generate("q", [canvas, pdf])
    assert (ans.confidence, ans.confidence_note) == ("LOW", None)


def test_mixed_citations_are_not_capped(tmp_path):
    canvas, pdf = _docs()[2], _docs()[0]
    gen, _ = _from_config(tmp_path, "Map [1] and book [2].\n\nCONFIDENCE: HIGH",
                          generated_only_confidence="LOW")
    ans = gen.generate("q", [canvas, pdf])
    assert (ans.confidence, ans.confidence_note) == ("HIGH", None)


def test_unknown_confidence_untouched(tmp_path):
    canvas = _docs()[2]
    gen, _ = _from_config(tmp_path, "The map says so [1].", generated_only_confidence="LOW")
    ans = gen.generate("q", [canvas])
    assert (ans.confidence, ans.confidence_note) == ("UNKNOWN", None)


def test_bad_cap_fails_from_config(tmp_path):
    with pytest.raises(ValueError, match="provenance.generated_only_confidence"):
        _from_config(tmp_path, "", generated_only_confidence="HIGH")


@pytest.mark.parametrize("route", ["query", "answer", "compare"])
def test_routes_put_the_confidence_note_in_generation_only_when_capped(tmp_path, monkeypatch, route):
    import serve_api
    canvas = _docs()[2]
    rows = {"ids": [canvas.id], "documents": [canvas.text], "metadatas": [canvas.metadata]}
    monkeypatch.setattr(serve_api, "_HISTORY", deque(maxlen=50))

    def call(**prov):
        gen, _ = _from_config(tmp_path, "The map says so [1].\n\nCONFIDENCE: HIGH", **prov)
        monkeypatch.setattr(serve_api, "_generator_for", lambda provider, model: gen)
        # generated_mode off: this pins the confidence cap, not /answer's refusal
        # (tests/test_api_generated.py); /answer stamps provenance in every mode.
        monkeypatch.setitem(serve_api._STATE, "rag", SimpleNamespace(
            generator=gen, search=lambda q, **kw: ([canvas], {"timings": {"total": 1.0}}),
            retriever=SimpleNamespace(_get_collection=lambda: SimpleNamespace(get=lambda **kw: rows)),
            generated_mode="off", provenance_map={}))
        if route == "query":
            out = serve_api.query(serve_api.QueryIn(q="x"))
        elif route == "answer":
            out = serve_api.answer(serve_api.AnswerIn(q="x", docs=[canvas.id]))
        else:
            out = serve_api.compare(serve_api.CompareIn(
                q="x", mode="query", branches=[{"id": "a"}, {"id": "b"}]))["branches"][0]
            return out["confidence"], out["generation"]
        return out.confidence, out.generation

    confidence, generation = call(generated_only_confidence="LOW")
    assert confidence == "LOW"
    assert generation["confidence_note"] == "capped at LOW: every cited source is generated"
    confidence, generation = call()
    assert confidence == "HIGH" and "confidence_note" not in generation
