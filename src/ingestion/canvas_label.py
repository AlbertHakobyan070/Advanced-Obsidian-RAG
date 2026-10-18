"""
canvas_label.py — what the Graph tab's "suggest label" shows a model about one
canvas edge, how a long node is fitted, the prompt, and how the reply is read.

Pure on purpose: no FastAPI, no config reads, no client. The console endpoint
(manage_api.canvas_suggest_label) wires config, the LLM client and HTTP around
these functions, so all of this runs offline in tests.

    ctx = edge_context(data, "e1", "Maps/m.canvas", vault, read_chars=20000,
                       max_items=12, images=client.vision, max_image_bytes=4 << 20)
    fit_node(ctx["from"], node_chars=2500, long_nodes="summarize",
             input_chars=20000, complete=lambda s, u: client.complete(s, u))
    system, user = render_prompt(ctx, vocabulary, guidance)
    got = parse_reply(client.complete(system, user).text, vocabulary, 12)

THE CANVAS COMES FROM THE BROWSER
  The editor posts the canvas as it holds it, unsaved edits included, so its
  shape is checked here (ValueError). File nodes are read from the vault and
  only read: a path that resolves outside the vault, or is missing, becomes a
  typed placeholder. It is never read and never an error, because many file
  nodes in a real vault point at renamed folders.

WHAT A MODEL GETS FOR EACH ENDPOINT NODE
  text   its cleaned text
  file   .md/.markdown: the body without frontmatter; code and plain text as
         is (reads stop at read_chars + 1 characters, with a note when more
         remained); a sub-canvas: its text-node titles; an image: the image
         itself only when allowed and under the cap; anything else (pdf,
         binaries): a typed placeholder
  link   its URL, never fetched
  group  its label and the titles of the nodes inside its box
  and around them: the arrow, the canvas title, each endpoint's group, the
  other edges at each endpoint (the neighbours' titles), the canvas's groups
  and its labels. A label with no letter once its #tags are removed (a step
  number, a bare #p) names no relation, so it is never an example and its
  edge shows only the neighbour's title.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.ingestion.canvas_loader import (
    _collapse,
    _describe_target,
    _smallest_containing_group,
)
from src.ingestion.code_loader import DEFAULT_LANGUAGE_MAP
from src.ingestion.obsidian_parser import clean_text, parse_frontmatter
from src.llm.llm_client import IMAGE_MIMES
from src.prompts.loader import load_prompt

# Extension -> mime, for image file nodes. Every value is one the LLM client sends.
IMAGE_TYPES = {f".{mime.split('/')[1]}": mime for mime in IMAGE_MIMES}
IMAGE_TYPES[".jpg"] = IMAGE_TYPES[".jpeg"]

_MARKDOWN = {".md", ".markdown"}
_PLAIN = {".txt", ".py", ".r", ".rmd"} | set(DEFAULT_LANGUAGE_MAP)
LONG_NODES = ("summarize", "truncate")
CONFIDENCE = ("high", "medium", "low")

# A tag the loader's extract_tags would extract unchanged: an ASCII letter first.
_TAG_RE = re.compile(r"[a-z][\w/-]*")
# A #tag in a label, written the way obsidian_parser.extract_tags reads one.
_HASHTAG_RE = re.compile(r"(^|\s)#[a-zA-Z][\w/-]*")


def _says_something(label: str) -> bool:
    """Whether a label names a relation: a letter is left once its #tags are
    removed. A sequence marker (1, 1,2, 10,11) or a tag-only label (#p) says
    nothing about how two ideas relate, and shown to a model as an example it
    gets copied: a map numbered that way got "1" back as a suggestion."""
    return any(ch.isalpha() for ch in _HASHTAG_RE.sub(r"\1", label))


def arrow(edge: dict) -> str:
    """The arrow as drawn. JSON Canvas defaults: fromEnd none, toEnd arrow."""
    back = (edge.get("fromEnd") or "none") == "arrow"
    ahead = (edge.get("toEnd") or "arrow") == "arrow"
    return {(False, True): "→", (True, True): "↔",
            (False, False): "—", (True, False): "←"}[(back, ahead)]


def _note(node: dict, text: str) -> None:
    node["note"] = f"{node['note']}; {text}" if node.get("note") else text


def _read_text(p: Path, read_chars: int) -> tuple[str, bool]:
    """At most read_chars + 1 characters, and whether more remained."""
    with open(p, encoding="utf-8-sig", errors="replace") as f:
        text = f.read(read_chars + 1)
    return text, len(text) > read_chars


def _vault_file(vault: Path, rel: str) -> tuple[Path | None, str | None]:
    """(file, None) for an existing file inside the vault, else (None, why)."""
    root = vault.resolve()
    p = (root / rel).resolve()
    try:
        p.relative_to(root)
    except ValueError:
        return None, "outside the vault, so it was not read"
    if p == root or not p.is_file():
        return None, "not found in the vault"
    return p, None


def _file_node(info: dict, vault: Path, read_chars: int, max_items: int,
               images: bool, max_image_bytes: int, image_rows: list) -> None:
    """Fill info (kind, text, mode, note, image) for a file node. Never an
    error: a file that is missing, outside the vault or cannot be read becomes
    a typed placeholder that says why."""
    rel = str(info["path"] or "").replace("\\", "/").lstrip("/")
    ext = Path(rel).suffix.lower()
    info["kind"] = f'{ext.lstrip(".") or "a"} file "{rel}"'
    if not rel:
        info["kind"], info["text"], info["mode"] = "file", "(a file node with no path)", "placeholder"
        return
    try:
        _read_file_node(info, vault, rel, ext, read_chars, max_items, images,
                        max_image_bytes, image_rows)
    except OSError as e:                       # permissions, a locked or vanished file
        why = f"it could not be read: {type(e).__name__}: {e}"
        info["mode"], info["image"] = "placeholder", None
        info["text"] = f"(a file node: {rel}; {why})"
        _note(info, why)
        if ext in IMAGE_TYPES:                 # its row is written last, so not yet
            image_rows.append({"node": info["id"], "path": rel, "sent": False, "reason": why})


def _read_file_node(info: dict, vault: Path, rel: str, ext: str, read_chars: int,
                    max_items: int, images: bool, max_image_bytes: int,
                    image_rows: list) -> None:
    """_file_node's reads. Every OSError is left to _file_node."""
    p, why = _vault_file(vault, rel)
    if ext in IMAGE_TYPES:
        info["kind"] = f'image "{rel}"'
        reason = why
        if p and not images:
            reason = "the backend is not marked vision: true"
        elif p and p.stat().st_size > max_image_bytes:
            reason = (f"{p.stat().st_size} bytes is over the {max_image_bytes}-byte cap "
                      f"(canvas.augment.max_image_mb)")
        if reason is None:
            info["image"] = (p.read_bytes(), IMAGE_TYPES[ext])
            info["text"], info["mode"] = f"(the image {rel}, attached)", "image"
        else:
            info["text"], info["mode"] = f"(an image, {rel}; not attached: {reason})", "placeholder"
        image_rows.append({"node": info["id"], "path": rel, "sent": reason is None,
                           "reason": reason})
        return
    if p is None:
        info["text"], info["mode"] = f"(a file node: {rel}; {why})", "placeholder"
        _note(info, why)
        return
    if ext in _MARKDOWN or ext in _PLAIN:
        raw, more = _read_text(p, read_chars)
        info["kind"] = (f'markdown note "{rel}"' if ext in _MARKDOWN else info["kind"])
        info["text"] = clean_text(parse_frontmatter(raw)[1]) if ext in _MARKDOWN else raw
        if more:
            _note(info, f"the file is longer than {read_chars} characters; at most "
                        f"{read_chars} were used")
        return
    if ext == ".canvas":
        info["kind"] = f'canvas "{rel}"'
        try:
            sub = json.loads(p.read_text(encoding="utf-8-sig", errors="replace"))
            titles = [_describe_target(n) for n in sub.get("nodes") or []
                      if isinstance(n, dict) and n.get("type") == "text"]
        except (ValueError, AttributeError) as e:      # JSONDecodeError is a ValueError
            info["text"], info["mode"] = f"(a canvas, {rel}; unreadable)", "placeholder"
            _note(info, f"unreadable canvas: {type(e).__name__}: {e}")
            return
        titles = [t for t in titles if t]
        info["text"] = ("A canvas whose text nodes include: " + " · ".join(titles[:max_items])
                        if titles else "(a canvas with no text nodes)")
        if len(titles) > max_items:
            _note(info, f"{len(titles) - max_items} more node titles not listed "
                        f"(canvas.augment.max_items)")
        return
    info["mode"] = "placeholder"
    info["text"] = f"(a {ext.lstrip('.') or 'binary'} file, {rel}; its content is not read)"


def _endpoint(node: dict, nodes: list, groups: list, vault: Path, *, read_chars: int,
              max_items: int, images: bool, max_image_bytes: int, image_rows: list) -> dict:
    """One endpoint as the prompt will see it, before fit_node."""
    ntype = node.get("type")
    group = _smallest_containing_group(node, [g for g in groups if g is not node])
    info = {"id": node.get("id"), "type": ntype, "title": _describe_target(node),
            "path": node.get("file") if ntype == "file" else None,
            "group": _collapse(group.get("label")) or None if group else None,
            "kind": str(ntype or "node"), "text": "", "mode": "full", "note": None,
            "image": None}
    if ntype == "text":
        info["kind"], info["text"] = "text card", clean_text(node.get("text") or "")
    elif ntype == "file":
        _file_node(info, vault, read_chars, max_items, images, max_image_bytes, image_rows)
    elif ntype == "link":
        info["kind"], info["mode"] = "web link", "placeholder"
        info["text"] = f"(a link to {node.get('url') or '?'}; the page is not fetched)"
    elif ntype == "group":
        label = _collapse(node.get("label"))
        titles = [t for t in (_describe_target(n) for n in nodes
                              if isinstance(n, dict) and n is not node
                              and _smallest_containing_group(n, [node]) is node) if t]
        info["kind"] = f'group "{label}"' if label else "group"
        info["text"] = (f'A group box "{label}" holding: ' + " · ".join(titles[:max_items])
                        if titles else f'An empty group box "{label}"')
        if len(titles) > max_items:
            _note(info, f"{len(titles) - max_items} more members not listed "
                        f"(canvas.augment.max_items)")
    info["chars"] = len(info["text"])
    return info


def _neighbours(nid: str, edges: list, target: dict, nodes_by_id: dict,
                max_items: int) -> list[str]:
    """Every other edge at one endpoint, as "→ title" or "← title" with the
    label in brackets after the arrow when it names a relation
    (_says_something). The neighbours' titles are the context; a step number
    is not shown. Labelled lines first, each group in canvas order, at most
    max_items in all."""
    lines = []
    for e in edges:
        if e is target:
            continue
        if e.get("fromNode") == nid:
            way, other = "→", e.get("toNode")
        elif e.get("toNode") == nid:
            way, other = "←", e.get("fromNode")
        else:
            continue
        label, title = _collapse(e.get("label")), _describe_target(nodes_by_id.get(other))
        named = _says_something(label)
        lines.append((not named, f"{way} [{label}] {title}" if named else f"{way} {title}"))
    return [line for _, line in sorted(lines, key=lambda x: x[0])][:max_items]   # stable


def _distinct(values) -> list[str]:
    seen: list[str] = []
    for v in values:
        if v and v not in seen:
            seen.append(v)
    return seen


def edge_context(data: dict, edge_id: str, canvas_rel: str, vault: Path, *,
                 read_chars: int, max_items: int, images: bool,
                 max_image_bytes: int) -> dict:
    """Everything the prompt says about one edge of the posted canvas.

    Raises ValueError when the canvas is not {nodes: [...], edges: [...]}, the
    edge is not in it, or an endpoint is missing from its nodes."""
    if not (isinstance(data, dict) and isinstance(data.get("nodes"), list)
            and isinstance(data.get("edges"), list)):
        raise ValueError("the posted canvas must be {nodes: [...], edges: [...]}")
    nodes = [n for n in data["nodes"] if isinstance(n, dict)]
    edges = [e for e in data["edges"] if isinstance(e, dict)]
    nodes_by_id = {n["id"]: n for n in nodes if isinstance(n.get("id"), str) and n["id"]}
    edge = next((e for e in edges if e.get("id") == edge_id), None)
    if edge is None:
        raise ValueError(f"no edge {edge_id!r} in the posted canvas")
    for key in ("fromNode", "toNode"):
        if not isinstance(edge.get(key), str) or edge[key] not in nodes_by_id:
            raise ValueError(f"edge {edge_id!r} has {key} {edge.get(key)!r}, which is not "
                             f"a node of the posted canvas")
    groups = [n for n in nodes if n.get("type") == "group"]
    image_rows: list[dict] = []
    ends = {side: _endpoint(nodes_by_id[edge[key]], nodes, groups, vault,
                            read_chars=read_chars, max_items=max_items, images=images,
                            max_image_bytes=max_image_bytes, image_rows=image_rows)
            for side, key in (("from", "fromNode"), ("to", "toNode"))}
    return {
        "canvas": canvas_rel, "title": Path(canvas_rel).stem, "direction": arrow(edge),
        "current_label": _collapse(edge.get("label")),
        **ends,
        "neighbour_labels": {side: _neighbours(edge[key], edges, edge, nodes_by_id, max_items)
                             for side, key in (("from", "fromNode"), ("to", "toNode"))},
        "groups": _distinct(_collapse(g.get("label")) for g in groups)[:max_items],
        "examples": _distinct(label for label in (_collapse(e.get("label")) for e in edges
                                                  if e is not edge)
                              if _says_something(label))[:max_items],
        "images": image_rows,
    }


def fit_node(node: dict, *, node_chars: int, long_nodes: str, input_chars: int,
             complete) -> dict:
    """Set node["sent"] (what the prompt carries), node["mode"] and
    node["summary_model"]. A node longer than node_chars is cut there
    (truncate) or summarised by one complete(system, user) -> LLMResponse
    call (summarize), and the mode says which. An empty summary raises
    ValueError: a failed summary is never replaced by a silent cut."""
    if long_nodes not in LONG_NODES:
        raise ValueError(f"canvas.augment.long_nodes must be one of {LONG_NODES}, "
                         f"got {long_nodes!r}")
    text = node["text"]
    node["summary_model"] = None
    if len(text) <= node_chars:
        node["sent"] = text
        return node
    if long_nodes == "truncate":
        node["sent"], node["mode"] = text[:node_chars], "truncated"
        return node
    from src.generation.generator import _strip_reasoning
    if len(text) > input_chars:
        _note(node, f"the summary was given the first {input_chars} characters")
    prompt = load_prompt("canvas_node_summary")
    resp = complete(prompt["system"].format(max_chars=node_chars),
                    prompt["user"].format(kind=node["kind"], title=node["title"],
                                          text=text[:input_chars]))
    summary = _strip_reasoning(resp.text or "")
    if not summary:
        raise ValueError(f"the summary of node {node['id']} came back empty")
    node["sent"], node["mode"], node["summary_model"] = summary, "summarized", resp.model
    return node


def render_prompt(ctx: dict, vocabulary: list[str], guidance: str | None) -> tuple[str, str]:
    """(system, user) for the label call, from src/prompts/canvas_label.yaml."""
    def listed(items):
        return "\n".join(f"- {x}" for x in items) or "(none)"

    def inline(items):          # " | ": real labels and group names contain commas
        return " | ".join(items) or "(none)"
    fill = {"canvas": ctx["title"], "vocabulary": ", ".join(vocabulary),
            "groups": inline(ctx["groups"]), "examples": inline(ctx["examples"]),
            "direction": ctx["direction"], "current": ctx["current_label"] or "(none)",
            "guidance": (guidance or "").strip() or "(none)"}
    for side in ("from", "to"):
        node = ctx[side]
        fill[f"{side}_kind"] = node["kind"]
        fill[f"{side}_group"] = node["group"] or "(none)"
        fill[f"{side}_text"] = node["sent"] or "(empty)"
        fill[f"{side}_neighbours"] = listed(ctx["neighbour_labels"][side])
    prompt = load_prompt("canvas_label")
    return prompt["system"].format(**fill), prompt["user"].format(**fill)


def normalise_tag(raw) -> str | None:
    """'#Machine Learning' -> 'machine-learning'; None when the result is not a
    tag extract_tags would read back unchanged (an ASCII letter first)."""
    if not isinstance(raw, str):
        return None
    tag = re.sub(r"\s+", "-", raw.strip().lstrip("#").strip().lower())
    return tag if _TAG_RE.fullmatch(tag) else None


def _key(relation: str) -> str:
    return re.sub(r"[\s_-]+", "", relation.lower())


def parse_reply(text: str, vocabulary: list[str], max_items: int) -> dict:
    """The model's suggestion: the LAST JSON object in the reply with a
    non-empty string "label", after <think> blocks are dropped.

    Raises ValueError when there is none. A relation outside the vocabulary
    becomes None with a warning: the label is the product, the relation only a
    classification. Relations match case- and separator-insensitively and come
    back spelled as the vocabulary spells them."""
    from src.generation.generator import _strip_reasoning
    body = _strip_reasoning(text or "")
    decoder, got = json.JSONDecoder(), None
    for m in re.finditer(r"\{", body):
        try:
            obj, _ = decoder.raw_decode(body, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("label"), str) and obj["label"].strip():
            got = obj
    if got is None:
        raise ValueError('the model\'s reply has no JSON object with a non-empty "label"')
    warnings: list[str] = []
    label = " ".join(got["label"].split())

    raw_rel = got.get("relation")
    relation = {_key(v): v for v in vocabulary}.get(_key(raw_rel)) if isinstance(raw_rel, str) else None
    if relation is None and raw_rel not in (None, ""):
        warnings.append(f"relation {raw_rel!r} is not in canvas.augment.vocabulary, so it was left empty")

    raw_conf = got.get("confidence")
    confidence = raw_conf.strip().lower() if isinstance(raw_conf, str) else None
    if confidence not in CONFIDENCE:
        if raw_conf not in (None, ""):
            warnings.append(f"confidence {raw_conf!r} is not one of {', '.join(CONFIDENCE)}")
        confidence = None

    def as_list(field):
        value = got.get(field)
        if value in (None, ""):
            return []
        if isinstance(value, (str, list)):
            return [value] if isinstance(value, str) else value
        warnings.append(f"{field} was not a list, so it was ignored")
        return []

    tags: list[str] = []
    for raw in as_list("tags"):
        tag = normalise_tag(raw)
        if tag is None:
            warnings.append(f"tag {raw!r} was dropped: a tag starts with a letter a-z, "
                            f"as Obsidian #tags do")
        elif tag not in tags:
            tags.append(tag)
    if len(tags) > max_items:
        warnings.append(f"{len(tags) - max_items} tag(s) over canvas.augment.max_items dropped")
        tags = tags[:max_items]

    alternatives, seen = [], {label.lower()}
    for alt in as_list("alternatives"):
        if not isinstance(alt, str):
            warnings.append(f"alternative {alt!r} was dropped: not text")
            continue
        alt = " ".join(alt.split())
        if alt and alt.lower() not in seen:
            seen.add(alt.lower())
            alternatives.append(alt)

    why = got.get("why")
    return {"label": label, "relation": relation, "tags": tags,
            "alternatives": alternatives[:max_items], "confidence": confidence,
            "why": why.strip() or None if isinstance(why, str) else None,
            "warnings": warnings}


_NODE_OUT = ("id", "type", "title", "path", "group", "mode", "chars", "sent", "note",
             "summary_model")


def context_out(ctx: dict) -> dict:
    """The response's `context` block: what the model was shown, minus image bytes."""
    return {"canvas": ctx["canvas"], "title": ctx["title"], "direction": ctx["direction"],
            "current_label": ctx["current_label"],
            "from": {k: ctx["from"].get(k) for k in _NODE_OUT},
            "to": {k: ctx["to"].get(k) for k in _NODE_OUT},
            "neighbour_labels": ctx["neighbour_labels"], "groups": ctx["groups"],
            "examples": ctx["examples"], "images": ctx["images"]}
