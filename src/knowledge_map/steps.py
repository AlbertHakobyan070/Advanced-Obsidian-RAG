"""The four LLM steps of a map run (spec 5, 5.2): classify, extract, link, verify.

Each step renders its prompt from src/prompts/map_<step>.yaml and calls the
StepRunner with a `check` that rejects anything outside the run's vocabulary
(unknown template, node kind, edge type, node id or verdict id), so the runner
re-asks with the error and, when retries run out, fails the unit. The model
never sees or writes page numbers: unit text carries no page markers, and the
verifier gets each quote in its normalised page context, not its page.
"""
from src.prompts.loader import load_prompt

from .graph import Edge, Node
from .locate import PageText
from .schema import Classification, Extraction, Links, Revisions, Verdict, Verdicts
from .scout import Scout, Unit
from .templates import Template, check_extraction

PROMPTS = ("map_classify", "map_extract", "map_link", "map_verify", "map_trim")
_OPENING_CHARS = 3000          # spec 5.2: the classifier reads the chapter's first 3,000 characters


def prompt_versions() -> dict[str, int]:
    """Every map prompt's version; they enter the checkpoint key."""
    return {name: load_prompt(name)["version"] for name in PROMPTS}


def classify(runner, scout: Scout, templates: dict[str, Template]) -> Classification:
    p = load_prompt("map_classify")
    fill = {
        "templates": "\n".join(f"- {t.name}: {t.purpose}" for t in templates.values()),
        "names": ", ".join(templates),
        "chapter": scout.chapter_title,
        "sections": "\n".join(f"- {s.title}" for s in scout.sections) or "(none)",
        "captions": sum(len(c) for c in scout.captions.values()),
        "opening": "\n".join(scout.pages[n] for n in sorted(scout.pages))[:_OPENING_CHARS],
    }

    def check(c: Classification) -> list[str]:
        return [] if c.template in templates else [
            f"template '{c.template}' does not exist (choose one of: {', '.join(templates)})"]
    return runner.call(p["system"].format(**fill), p["user"].format(**fill), Classification,
                       "classify", check=check)


def node_target(unit_chars: int, chapter_chars: int, node_cap: int) -> int:
    """A unit's share of the chapter's node cap, by length; never fewer than 3."""
    return max(3, round(node_cap * unit_chars / chapter_chars))


def extract_prompt(unit: Unit, template: Template, node_target: int, *,
                   quote_words: tuple[int, int], headings: list[str] | None = None) -> tuple[str, str]:
    """(system, user) for one unit; also what --export-prompts writes.
    quote_words is knowledge_map.quote_words (min, max). With `headings` it is a gap-fill call:
    the same system prompt, and a user message asking only for the text under those headings."""
    p = load_prompt("map_extract")
    fill = {"node_kinds": ", ".join(template.node_kinds), "edge_types": ", ".join(template.edge_types),
            "guidance": template.guidance, "node_target": node_target,
            "quote_min": quote_words[0], "quote_max": quote_words[1],
            "example_passage": p["example_passage"], "example_reply": p["example_reply"],
            "text": unit.text}
    user = (p["gapfill_user"].format(headings="\n".join(f"- {h}" for h in headings), text=unit.text)
            if headings else p["user"].format(**fill))
    return p["system"].format(**fill), user


def extract_unit(runner, unit: Unit, template: Template, node_target: int, *,
                 quote_words: tuple[int, int], headings: list[str] | None = None) -> Extraction:
    system, user = extract_prompt(unit, template, node_target, quote_words=quote_words, headings=headings)
    return runner.call(system, user, Extraction, f"gapfill-{unit.id}" if headings else unit.id,
                       check=lambda e: check_extraction(e, template))


def link(runner, nodes: list[Node], template: Template, *, quote_words: tuple[int, int]) -> Links:
    """Cross-section edges between the given (verified) nodes, endpoints by node id.
    Each node shows its first located quote (its first quote when none is located)."""
    p = load_prompt("map_link")
    ids = {n.id for n in nodes}
    rows = [f"{n.id} | {n.title} | {n.summary} | \"{n.located[0][0] if n.located else n.quotes[0]}\""
            for n in nodes]
    fill = {"edge_types": ", ".join(template.edge_types), "max_edges": 2 * len(nodes),
            "quote_min": quote_words[0], "quote_max": quote_words[1], "nodes": "\n".join(rows)}

    # A bad edge (unknown id, disallowed type, self-loop) is dropped and counted, not re-asked:
    # link edges are optional cross-section extras, and a model that garbles one id at
    # temperature 0 garbles it on every retry, which failed the whole map at ~50 nodes.
    links = runner.call(p["system"].format(**fill), p["user"].format(**fill), Links, "link")
    good = [e for e in links.edges
            if e.source in ids and e.target in ids and e.type in template.edge_types and e.source != e.target]
    return Links(edges=good, dropped=len(links.edges) - len(good))


def _quote_block(text: PageText, quote: str, page: int) -> str:
    return f'  quote: "{quote}"\n  in context: ...{text.context(quote, page)}...'


def _verify(runner, items: list[tuple[str, str, list[str]]], unit: str) -> dict[str, Verdict]:
    """items: (id, claim, quote blocks). Exactly one verdict per id, or the runner re-asks."""
    p = load_prompt("map_verify")
    body = "\n\n".join(f"id: {i}\nclaim: {claim}\n" + "\n".join(blocks) for i, claim, blocks in items)
    want = [i for i, _, _ in items]

    def check(v: Verdicts) -> list[str]:
        return _id_errors(want, [x.id for x in v.verdicts], "verdict")
    out = runner.call(p["system"].format(items=body), p["user"].format(items=body), Verdicts, unit,
                      check=check)
    return {v.id: v for v in out.verdicts}


def _id_errors(want: list[str], ids: list[str], noun: str) -> list[str]:
    """Exactly one reply per wanted id, none for any other."""
    got = set(ids)
    return ([f"missing a {noun} for id '{i}'" for i in want if i not in got]
            + [f"unknown id '{i}' (not in the items)" for i in sorted(got - set(want))]
            + [f"duplicate {noun}s for id '{i}' (give exactly one)" for i in sorted(got) if ids.count(i) > 1])


def verify_nodes(runner, batch: list[Node], text: PageText) -> dict[str, Verdict]:
    if not batch:
        raise ValueError("empty batch")
    items = []
    for n in batch:
        if not n.located:
            raise ValueError(f"node '{n.id}' has no located quote to verify")
        items.append((n.id, f"{n.title} ({n.kind}): {n.summary}",
                      [_quote_block(text, q, pages[0]) for q, pages in n.located]))
    return _verify(runner, items, f"verify-nodes-{batch[0].id}")


def trim_nodes(runner, batch: list[tuple[Node, str]], text: PageText) -> dict[str, str | None]:
    """(unsupported node, the verifier's reason) -> a summary cut down to what its quotes
    say, or None when the quotes support no summary of it. The result is re-verified."""
    if not batch:
        raise ValueError("empty batch")
    p = load_prompt("map_trim")
    body = "\n\n".join(f"id: {n.id}\nclaim: {n.title} ({n.kind}): {n.summary}\nreason: {why}\n"
                       + "\n".join(_quote_block(text, q, pages[0]) for q, pages in n.located)
                       for n, why in batch)
    want = [n.id for n, _ in batch]

    def check(r: Revisions) -> list[str]:
        return (_id_errors(want, [x.id for x in r.revisions], "revision")
                + [f"empty summary for id '{x.id}' (write one, or null)"
                   for x in r.revisions if x.summary is not None and not x.summary.strip()])
    out = runner.call(p["system"].format(items=body), p["user"].format(items=body), Revisions,
                      f"trim-nodes-{batch[0][0].id}", check=check)
    return {x.id: x.summary.strip() if x.summary is not None else None for x in out.revisions}


def verify_edges(runner, batch: list[Edge], nodes: dict[str, Node], text: PageText) -> dict[str, Verdict]:
    if not batch:
        raise ValueError("empty batch")
    items = []
    for e in batch:
        if not e.quote or not e.pages:
            raise ValueError(f"edge '{e.id}' has no located quote to verify")
        src, tgt = nodes[e.source], nodes[e.target]
        items.append((e.id, f'"{src.title}" {e.type} "{tgt.title}"',
                      [_quote_block(text, e.quote, e.pages[0])]))
    return _verify(runner, items, f"verify-edges-{batch[0].id}")
