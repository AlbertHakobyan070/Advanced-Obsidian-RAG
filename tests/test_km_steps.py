"""Knowledge-map prompts and LLM steps (src/knowledge_map/steps.py). No network:
a real StepRunner over a fake client factory."""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from src.knowledge_map.config import RoleSpec
from src.knowledge_map.graph import Edge, Node, edge_id, node_id
from src.knowledge_map.llm_step import Budget, StepRunner, UnitFailed
from src.knowledge_map.locate import PageText
from src.knowledge_map.schema import Extraction
from src.knowledge_map.scout import Scout, Section, Unit
from src.knowledge_map.steps import (classify, extract_prompt, extract_unit, link, node_target,
                                     prompt_versions, trim_nodes, verify_edges, verify_nodes)
from src.knowledge_map.templates import load_templates
from src.prompts.loader import load_prompt


class Factory:
    """Replies are scripted in order; the last one repeats. Records every user prompt."""
    def __init__(self, *replies):
        self.replies, self.users, self.systems = list(replies), [], []

    def __call__(self, provider, model):
        def complete(system, user, temperature=None, max_tokens=None, json_mode=False):
            self.systems.append(system)
            self.users.append(user)
            item = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
            text = item if isinstance(item, str) else json.dumps(item)
            return NS(text=text, model="m", usage={"input_tokens": 1, "output_tokens": 1})
        return NS(complete=complete)


def _runner(factory, retries=1):
    attempts = []
    r = StepRunner(None, "x", RoleSpec("p", "m", 100, 0.0, ()), Budget(100, 100), attempts, retries,
                   0.0, client_factory=factory, sleep=lambda s: None)
    return r, attempts


def _node(title, quote, page, kind="concept", located=True):
    return Node(id=node_id(kind, title), kind=kind, title=title, summary=f"About {title}.",
                section="sec-1", unit="sec-1-u1", importance=3, quotes=[quote],
                located=[(quote, (page,))] if located else [])


QW = (5, 30)

PAGES = {
    12: "Opening words that set the scene. A ledger records every transaction in the order it "
        "arrives at the bank branch. Closing words on page twelve.",
    13: "Earlier remarks here. A journal is a book of original entry kept by the clerk on duty. "
        "Later remarks follow.",
}


def test_prompts_load_with_versions():
    v = prompt_versions()
    assert v == {"map_classify": 1, "map_extract": 4, "map_link": 2, "map_verify": 2, "map_trim": 1}
    for name in v:
        p = load_prompt(name)
        assert p["system"].strip() and p["user"].strip()


def test_example_quotes_are_verbatim():
    p = load_prompt("map_extract")
    ex = Extraction.model_validate_json(p["example_reply"])
    titles = {n.title for n in ex.nodes}
    for q in [q for n in ex.nodes for q in n.quotes] + [e.quote for e in ex.edges if e.quote]:
        assert q in p["example_passage"], q
        assert 5 <= len(q.split()) <= 30, q
    assert all(e.source in titles and e.target in titles for e in ex.edges)
    # every example edge is quoted, so the quote can state its relation and direction
    assert ex.edges and all(e.quote for e in ex.edges)
    system, _ = extract_prompt(Unit("u", "s", 1, 1, "x", 1), load_templates()["concept"], 5,
                               quote_words=QW)
    assert p["example_passage"].strip() in system
    assert "source <type> target" in system                   # edge direction is stated


def test_extract_prompt_carries_template_vocab():
    t = load_templates()["procedure"]
    text = PAGES[12] + "\n" + PAGES[13]
    unit = Unit("sec-1-u1", "sec-1", 12, 13, text, len(text))
    system, user = extract_prompt(unit, t, 7, quote_words=QW)
    both = system + "\n" + user
    for word in t.edge_types + t.node_kinds:
        assert word in both
    assert t.guidance in both
    assert "at most 7" in both
    assert text in user
    # the harness adds no page markers: the unit's page numbers appear nowhere
    assert not re.search(r"\b1[23]\b", both)
    assert not re.search(r"(?i)\b(page|pp?\.)\s*\d", both)


def test_extract_prompt_takes_quote_words_from_config():
    unit = Unit("sec-1-u1", "sec-1", 12, 12, PAGES[12], len(PAGES[12]))
    for t in load_templates().values():
        assert "5-30" not in t.guidance                     # the limit lives in config only
        system, user = extract_prompt(unit, t, 7, quote_words=(6, 25))
        both = system + user
        assert "6 to 25 words" in both
        assert "5 to 30" not in both and "5-30" not in both


def test_gapfill_prompt_names_headings_and_keeps_rules():
    t = load_templates()["concept"]
    unit = Unit("sec-1-u1", "sec-1", 12, 12, PAGES[12], len(PAGES[12]))
    system, user = extract_prompt(unit, t, 3, quote_words=QW, headings=["Ledgers (already in the map: Ledger)"])
    assert system == extract_prompt(unit, t, 3, quote_words=QW)[0]      # same rules, same schema
    assert "- Ledgers (already in the map: Ledger)" in user and PAGES[12] in user
    assert "never reuse a listed title" in user


def test_extract_rejects_unknown_kind():
    t = load_templates()["concept"]
    reply = {"nodes": [{"title": "Ledger", "kind": "widget", "summary": "A record.",
                        "quotes": ["A ledger records every transaction in the order"], "importance": 3}]}
    f = Factory(reply)
    r, attempts = _runner(f)
    unit = Unit("sec-1-u1", "sec-1", 12, 12, PAGES[12], len(PAGES[12]))
    with pytest.raises(UnitFailed) as ei:
        extract_unit(r, unit, t, 5, quote_words=QW)
    assert ei.value.unit == "sec-1-u1"
    assert [a["outcome"] for a in attempts] == ["check_error", "check_error"]
    assert "widget" in f.users[1]


def test_extract_ok():
    t = load_templates()["concept"]
    reply = {"nodes": [{"title": "Ledger", "kind": "concept", "summary": "A record.",
                        "quotes": ["A ledger records every transaction in the order"], "importance": 3}]}
    r, _ = _runner(Factory(reply))
    unit = Unit("sec-1-u1", "sec-1", 12, 12, PAGES[12], len(PAGES[12]))
    assert extract_unit(r, unit, t, 5, quote_words=QW).nodes[0].title == "Ledger"


def test_node_target_floor():
    assert node_target(100, 100_000, 40) == 3
    assert node_target(12_000, 24_000, 40) == 20
    assert node_target(24_000, 24_000, 40) == 40


@pytest.mark.parametrize("bad", [
    {"source": "concept-nope", "target": "concept-journal", "type": "uses"},      # unknown id
    {"source": "concept-ledger", "target": "concept-journal", "type": "befriends"},  # unknown type
    {"source": "concept-ledger", "target": "concept-ledger", "type": "uses"},     # self-edge
])
def test_link_drops_bad_edges_and_keeps_the_rest(bad):
    t = load_templates()["concept"]
    nodes = [_node("Ledger", "A ledger records every transaction in the order", 12),
             _node("Journal", "A journal is a book of original entry", 13)]
    nodes[0].quotes.insert(0, "a paraphrase that was never found")   # failed to locate
    good = {"source": "concept-journal", "target": "concept-ledger", "type": "contrasts-with", "quote": None}
    f = Factory({"edges": [bad, good]})
    r, attempts = _runner(f)
    out = link(r, nodes, t, quote_words=(6, 25))
    assert [e.type for e in out.edges] == ["contrasts-with"] and out.dropped == 1
    assert [a["outcome"] for a in attempts] == ["ok"]                  # no re-ask: one reply is enough
    first = f.users[0]
    assert "concept-ledger" in first and "A journal is a book of original entry" in first
    assert "at most 4" in f.systems[0] + first
    assert "a paraphrase that was never found" not in first   # the located quote is shown
    assert "A ledger records every transaction in the order" in first
    assert "6 to 25 words" in f.systems[0]


def test_link_falls_back_to_first_quote_when_unlocated():
    t = load_templates()["concept"]
    nodes = [_node("Ledger", "A ledger records every transaction in the order", 12, located=False),
             _node("Journal", "A journal is a book of original entry", 13)]
    f = Factory({"edges": []})
    r, _ = _runner(f)
    link(r, nodes, t, quote_words=QW)
    assert "A ledger records every transaction in the order" in f.users[0]


def test_verify_rejects_duplicate_ids_and_empty_batch():
    text = PageText(PAGES)
    a = _node("Ledger", "A ledger records every transaction in the order", 12)
    b = _node("Journal", "A journal is a book of original entry", 13)
    dup = {"verdicts": [{"id": a.id, "supported": True, "reason": "x"},
                        {"id": a.id, "supported": False, "reason": "y"},
                        {"id": b.id, "supported": True, "reason": "z"}]}
    f = Factory(dup)
    r, attempts = _runner(f)
    with pytest.raises(UnitFailed):
        verify_nodes(r, [a, b], text)
    assert [x["outcome"] for x in attempts] == ["check_error", "check_error"]
    assert "duplicate" in f.users[1]
    for fn, args in ((verify_nodes, ([], text)), (verify_edges, ([], {}, text))):
        with pytest.raises(ValueError, match="empty batch"):
            fn(r, *args)


def test_verify_missing_id_reasks():
    text = PageText(PAGES)
    a = _node("Ledger", "A ledger records every transaction in the order", 12)
    b = _node("Journal", "A journal is a book of original entry", 13)
    half = {"verdicts": [{"id": a.id, "supported": True, "reason": "stated"}]}
    full = {"verdicts": [{"id": a.id, "supported": True, "reason": "stated"},
                         {"id": b.id, "supported": False, "reason": "quote is about books"}]}
    f = Factory(half, full)
    r, attempts = _runner(f)
    out = verify_nodes(r, [a, b], text)
    assert set(out) == {a.id, b.id} and out[b.id].supported is False
    assert [x["outcome"] for x in attempts] == ["check_error", "ok"]
    assert b.id in f.users[1].split("rejected", 1)[1]
    # the claim and the quote's page context (normalised, so case-folded) are shown
    assert "About Ledger." in f.users[0] and "opening words that set the scene" in f.users[0]


def test_verify_edges_shows_claim_and_context():
    text = PageText(PAGES)
    a = _node("Ledger", "A ledger records every transaction in the order", 12)
    b = _node("Journal", "A journal is a book of original entry", 13)
    q = "kept by the clerk on duty. Later remarks"
    e = Edge(id=edge_id(b.id, a.id, "uses"), source=b.id, target=a.id, type="uses", quote=q, pages=(13,))
    f = Factory({"verdicts": [{"id": e.id, "supported": False, "reason": "no"}]})
    r, _ = _runner(f)
    out = verify_edges(r, [e], {a.id: a, b.id: b}, text)
    assert out[e.id].supported is False
    assert "Journal" in f.users[0] and "uses" in f.users[0] and "Ledger" in f.users[0]
    assert "earlier remarks here" in f.users[0]


def test_trim_shows_reason_and_context_and_reasks():
    text = PageText(PAGES)
    a = _node("Ledger", "A ledger records every transaction in the order", 12)
    b = _node("Journal", "A journal is a book of original entry", 13)
    half = {"revisions": [{"id": a.id, "summary": "A ledger records transactions in order."}]}
    empty = {"revisions": [{"id": a.id, "summary": "A ledger records transactions in order."},
                           {"id": b.id, "summary": "  "}]}
    full = {"revisions": [{"id": a.id, "summary": "A ledger records transactions in order."},
                          {"id": b.id, "summary": None}]}
    f = Factory(half, empty, full)
    r, attempts = _runner(f, retries=2)
    out = trim_nodes(r, [(a, "adds the branch"), (b, "quote is about books")], text)
    assert out == {a.id: "A ledger records transactions in order.", b.id: None}
    assert [x["outcome"] for x in attempts] == ["check_error", "check_error", "ok"]
    assert "missing" in f.users[1] and "empty" in f.users[2]
    assert "About Ledger." in f.users[0] and "adds the branch" in f.users[0]
    assert "opening words that set the scene" in f.users[0]
    with pytest.raises(ValueError, match="empty batch"):
        trim_nodes(r, [], text)


def _scout():
    body = "A" * 3000 + " TAILMARK " + "B" * 50
    return Scout(pdf_path=Path("x.pdf"), vault_rel="x.pdf", sha256="0", book_title="Book",
                 chapter_title="Ledgers and Journals", page_start=12, page_end=13,
                 pages={12: body, 13: "second page"},
                 sections=[Section("sec-1", "Double entry", 12, 12), Section("sec-2", "Closing books", 13, 13)],
                 units=[], captions={12: ["Figure 2-1. A ledger"], 13: ["Table 2-1. Accounts", "Figure 2-2. X"]},
                 image_counts={}, scanned_pages=[], toc_used=True)


def test_classify_unknown_template_reasks():
    templates = load_templates()
    f = Factory({"template": "novel", "reason": "it reads like one"},
                {"template": "concept", "reason": "it explains ideas"})
    r, attempts = _runner(f)
    out = classify(r, _scout(), templates)
    assert out.template == "concept"
    assert [a["outcome"] for a in attempts] == ["check_error", "ok"]
    u = f.users[0]
    assert "Ledgers and Journals" in u and "Double entry" in u and "Closing books" in u
    assert "Captions (figures and tables): 3" in u
    assert "TAILMARK" not in u                 # only the first 3,000 characters
    for name, t in templates.items():
        assert name in f.systems[0] and t.purpose in f.systems[0]


def test_graph_ids():
    assert node_id("concept", "Document Model (JSON)") == "concept-document-model-json"
    assert edge_id("a", "b", "uses") == edge_id("a", "b", "uses") != edge_id("b", "a", "uses")
    assert re.fullmatch(r"e-[0-9a-f]{10}", edge_id("a", "b", "uses"))
