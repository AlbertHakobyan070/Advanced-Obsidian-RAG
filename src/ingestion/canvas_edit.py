"""Edits to an Obsidian .canvas file, applied ONTO the file as it was loaded.

The console's Graph tab sends back the whole canvas, but the server never
writes what the browser sent. It starts from the original JSON and copies over
only the fields this editor edits:

    text nodes   text            (and x / y / width / height, for moving)
    other nodes  x / y / width / height
    edges        label           (an empty label removes the key, as Obsidian does)

plus new text nodes, new edges and deletions. Everything else — colours, file /
link / group nodes and their keys, edge sides and ends, unknown keys at any
level, the order of keys — comes from the original, so a client bug cannot drop
a field it never meant to touch. The JSON Canvas 1.0 spec
(https://jsoncanvas.org/spec/1.0/) is the validation reference.

The writer keeps the file's own bytes: it splices the changed values into the
text as it was loaded (see dump_canvas), so an unedited node, edge or member is
written exactly as Obsidian or a plugin wrote it, and a file with no edits is
never rewritten.
"""
from __future__ import annotations

import json
import re

NODE_TYPES = {"text", "file", "link", "group"}
SIDES = {"top", "right", "bottom", "left"}
ENDS = {"none", "arrow"}
GEOMETRY = ("x", "y", "width", "height")
_COLOR_RE = re.compile(r"^([1-6]|#[0-9A-Fa-f]{6})$")


def _int(v, what: str) -> int:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, got {v!r}")
    if f != f or abs(f) > 1e9:
        raise ValueError(f"{what} is out of range: {v!r}")
    return int(round(f))


def _id(v, what: str) -> str:
    if not isinstance(v, str) or not v.strip() or len(v) > 200:
        raise ValueError(f"{what} must be a non-empty string id")
    return v


def merge_canvas(orig: dict, new: dict) -> dict:
    """The original canvas with the editor's changes applied. Raises ValueError."""
    if not isinstance(orig, dict) or not isinstance(orig.get("nodes", []), list):
        raise ValueError("the file on disk is not a canvas (no nodes list)")
    if not isinstance(new, dict) or not isinstance(new.get("nodes"), list) \
            or not isinstance(new.get("edges", []), list):
        raise ValueError("data must be {nodes: [...], edges: [...]}")
    onodes = {n["id"]: n for n in orig.get("nodes") or [] if isinstance(n, dict) and "id" in n}
    oedges = {e["id"]: e for e in orig.get("edges") or [] if isinstance(e, dict) and "id" in e}

    nodes, ids = [], set()
    for n in new["nodes"]:
        if not isinstance(n, dict):
            raise ValueError("every node must be an object")
        nid = _id(n.get("id"), "node id")
        if nid in ids:
            raise ValueError(f"duplicate node id {nid!r}")
        ids.add(nid)
        if nid in onodes:
            m = dict(onodes[nid])                  # original keys, original order
            for k in GEOMETRY:
                if k in n:
                    v = _int(n[k], f"node {nid} {k}")
                    if float(n[k]) != m.get(k):    # a file's own 12.5 is not rounded unless moved
                        m[k] = v
            if "text" in n and n["text"] != m.get("text"):
                if m.get("type") != "text":
                    raise ValueError(f"node {nid} is a {m.get('type')} node; only text "
                                     f"nodes have editable text")
                if not isinstance(n["text"], str):
                    raise ValueError(f"node {nid} text must be a string")
                m["text"] = n["text"]
        else:
            if n.get("type", "text") != "text":
                raise ValueError(f"new node {nid}: this editor only adds text nodes")
            if not isinstance(n.get("text", ""), str):
                raise ValueError(f"new node {nid}: text must be a string")
            m = {"id": nid, "type": "text", "text": n.get("text", "")}
            for k in GEOMETRY:
                m[k] = _int(n.get(k), f"new node {nid} {k}")
            if m["width"] <= 0 or m["height"] <= 0:
                raise ValueError(f"new node {nid}: width and height must be positive")
            if n.get("color") is not None:
                if not _COLOR_RE.match(str(n["color"])):
                    raise ValueError(f"new node {nid}: color must be 1-6 or #RRGGBB")
                m["color"] = str(n["color"])
        nodes.append(m)

    edges, eids = [], set()
    for e in new.get("edges") or []:
        if not isinstance(e, dict):
            raise ValueError("every edge must be an object")
        eid = _id(e.get("id"), "edge id")
        if eid in eids:
            raise ValueError(f"duplicate edge id {eid!r}")
        eids.add(eid)
        if eid in oedges:
            m = dict(oedges[eid])
            for k in ("fromNode", "toNode"):
                if k in e and e[k] != m.get(k):
                    raise ValueError(f"edge {eid}: reconnecting an edge is not supported "
                                     f"(delete it and draw a new one)")
        else:
            m = {"id": eid, "fromNode": _id(e.get("fromNode"), f"edge {eid} fromNode"),
                 "toNode": _id(e.get("toNode"), f"edge {eid} toNode")}
            for k, allowed in (("fromSide", SIDES), ("toSide", SIDES),
                               ("fromEnd", ENDS), ("toEnd", ENDS)):
                if e.get(k) is not None:
                    if e[k] not in allowed:
                        raise ValueError(f"edge {eid}: {k} must be one of {sorted(allowed)}")
                    m[k] = e[k]
        if "label" in e:
            label = e["label"]
            if label is not None and not isinstance(label, str):
                raise ValueError(f"edge {eid}: label must be a string")
            if label:
                m["label"] = label
            else:
                m.pop("label", None)
        for k in ("fromNode", "toNode"):
            if m.get(k) not in ids:
                raise ValueError(f"edge {eid} points at node {m.get(k)!r}, which is not "
                                 f"in the canvas (delete the edge with the node)")
        edges.append(m)

    out = dict(orig)                               # unknown top-level keys survive
    out["nodes"] = nodes
    if edges or "edges" in orig:
        out["edges"] = edges
    return out


def _compact(v) -> str:
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


# ---- writer: splice the changes into the file's own text ----------------------
#
# Obsidian's serializer is not json.dumps with some indent: one file mixes
# one-line and expanded objects, and plugins (Advanced Canvas) write their own.
# So nothing is re-serialized that did not change. Every value whose JSON is
# unchanged is copied from the file byte for byte; a changed object is rebuilt
# member by member (unchanged members verbatim, changed ones as compact JSON in
# that object's own colon/separator style); a NEW node or edge is appended in
# the layout of the last one in its list.

_WS = re.compile(r"[ \t\r\n]*")
_DEC = json.JSONDecoder()


def _ws(s: str, i: int) -> int:
    return _WS.match(s, i).end()


def _members(s: str, i: int):
    """The object at s[i] == '{' -> ([(key, key_at, key_end, value_at, value_end)], end)."""
    out, i = [], _ws(s, i + 1)
    if s[i] == "}":
        return out, i + 1
    while True:
        key, kend = _DEC.raw_decode(s, i)
        j = _ws(s, kend)
        if s[j] != ":":
            raise ValueError(f"expected ':' at offset {j}")
        vat = _ws(s, j + 1)
        _, vend = _DEC.raw_decode(s, vat)
        out.append((key, i, kend, vat, vend))
        i = _ws(s, vend)
        if s[i] == "}":
            return out, i + 1
        if s[i] != ",":
            raise ValueError(f"expected ',' or '}}' at offset {i}")
        i = _ws(s, i + 1)


def _elements(s: str, i: int):
    """The array at s[i] == '[' -> ([(start, end)], end)."""
    out, i = [], _ws(s, i + 1)
    if s[i] == "]":
        return out, i + 1
    while True:
        _, end = _DEC.raw_decode(s, i)
        out.append((i, end))
        i = _ws(s, end)
        if s[i] == "]":
            return out, i + 1
        if s[i] != ",":
            raise ValueError(f"expected ',' or ']' at offset {i}")
        i = _ws(s, i + 1)


def _obj_style(s: str, at: int):
    """(open, separator, colon, close) of the object at s[at], or None if empty."""
    mem, end = _members(s, at)
    if not mem:
        return None
    open_ = s[at:mem[0][1]]
    sep = s[mem[0][4]:mem[1][1]] if len(mem) > 1 else "," + open_[1:]
    return open_, sep, s[mem[0][2]:mem[0][3]], s[mem[-1][4]:end]


def _write(v, s: str, span, el_style, arr_style) -> str:
    if span is not None:
        a, b = span
        if json.loads(s[a:b]) == v:
            return s[a:b]
        if isinstance(v, dict) and s[a] == "{":
            mem, _ = _members(s, a)
            open_, sep, colon, close = _obj_style(s, a) or el_style or ("{", ",", ":", "}")
            orig = {m[0]: m for m in mem}
            parts = []
            for k, x in v.items():
                m = orig.get(k)
                parts.append(_compact(k) + colon + _write(x, s, None, el_style, arr_style)
                             if m is None else
                             s[m[1]:m[3]] + _write(x, s, (m[3], m[4]), el_style, arr_style))
            return open_ + sep.join(parts) + close if parts else "{}"
        if isinstance(v, list) and s[a] == "[":
            els, end = _elements(s, a)
            if els:
                lead = s[a + 1:els[0][0]]
                sep = s[els[0][1]:els[1][0]] if len(els) > 1 else "," + lead
                trail = s[els[-1][1]:end - 1]
                if s[els[-1][0]] == "{":                # new ones are appended: match the last
                    el_style = _obj_style(s, els[-1][0]) or el_style
            else:
                lead, sep, trail = arr_style or ("", ",", "")
            by_id = {}
            for e in els:
                o = json.loads(s[e[0]:e[1]])
                if isinstance(o, dict) and "id" in o:
                    by_id.setdefault(o["id"], e)
            parts = [_write(x, s, by_id.get(x.get("id")) if isinstance(x, dict) else None,
                            el_style, arr_style) for x in v]
            return "[" + lead + sep.join(parts) + trail + "]" if parts else "[]"
    if isinstance(v, dict) and v and el_style:      # a new node / edge: its siblings' layout
        open_, sep, colon, close = el_style
        return open_ + sep.join(_compact(k) + colon + _compact(x) for k, x in v.items()) + close
    if isinstance(v, list) and v and arr_style:     # a new edges list: the nodes list's layout
        lead, sep, trail = arr_style
        return "[" + lead + sep.join(_write(x, s, None, el_style, arr_style) for x in v) + trail + "]"
    return _compact(v)


def dump_canvas(data: dict, like: str) -> str:
    """`data` written into the text of `like` (the file as it is on disk)."""
    start = _ws(like, 0)
    top, end = _members(like, start)
    el_style = arr_style = None
    for key, *_, vat, vend in top:     # layout for a list that is empty in the file
        if key in ("nodes", "edges") and like[vat] == "[":
            els, aend = _elements(like, vat)
            if els and like[els[-1][0]] == "{":
                el_style = _obj_style(like, els[-1][0])
                arr_style = (like[vat + 1:els[0][0]],
                             like[els[0][1]:els[1][0]] if len(els) > 1
                             else "," + like[vat + 1:els[0][0]],
                             like[els[-1][1]:aend - 1])
                break
    return like[:start] + _write(data, like, (start, end), el_style, arr_style) + like[end:]
