"""canvas_loader.py tests — Obsidian .canvas ingestion (JSON node/edge graphs).

Run:  python -m pytest tests/ -q          (project venv, not the the local agent one)

Covers the two properties the whole design hinges on:
  - doc_id is keyed on (canvas path, node id), NOT node text — two nodes with
    identical text must not collide, and editing a node's text must upsert
    the same id rather than orphan it.
  - the graph (edge labels, direction, group containment) is flattened into
    the chunk TEXT and METADATA at ingest time, with zero retrieval-time code.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.ingestion.canvas_loader import (
    CanvasLoader, CanvasChunk, decode_canvas_edges, encode_canvas_edge,
)


def _write_canvas(path: Path, nodes: list, edges: list | None = None) -> Path:
    path.write_text(json.dumps({"nodes": nodes, "edges": edges or []}), encoding="utf-8")
    return path


def _loader(tmp_path: Path, min_chunk: int = 10) -> CanvasLoader:
    return CanvasLoader(vault_path=tmp_path, output_file=tmp_path / "out.jsonl",
                        min_chunk=min_chunk)


# ---- doc_id: the whole reason for the node-id keying scheme ----

def test_identical_text_nodes_get_different_doc_ids(tmp_path):
    same_text = "Duplicate content that appears verbatim on two separate canvas nodes."
    canvas = _write_canvas(tmp_path / "dupes.canvas", nodes=[
        {"id": "n1", "type": "text", "text": same_text, "x": 0, "y": 0, "width": 200, "height": 100},
        {"id": "n2", "type": "text", "text": same_text, "x": 500, "y": 500, "width": 200, "height": 100},
    ])
    chunks = _loader(tmp_path).load_file(canvas)
    assert len(chunks) == 2
    ids = {c.doc_id for c in chunks}
    assert len(ids) == 2, "two nodes with identical text collided — doc_id is hashing text, not node id"


def test_doc_id_stable_across_runs_and_survives_a_text_edit(tmp_path):
    canvas_path = tmp_path / "stable.canvas"
    _write_canvas(canvas_path, nodes=[
        {"id": "node-abc", "type": "text", "text": "Version one of this node's text, long enough to keep.",
         "x": 0, "y": 0, "width": 200, "height": 100},
    ])
    loader = _loader(tmp_path)
    first = loader.load_file(canvas_path)[0]
    again = loader.load_file(canvas_path)[0]
    assert again.doc_id == first.doc_id

    # Edit the node's text in place, same node id -> same doc_id (upsert, not orphan+new)
    _write_canvas(canvas_path, nodes=[
        {"id": "node-abc", "type": "text", "text": "Version TWO — the text changed but the node id did not.",
         "x": 0, "y": 0, "width": 200, "height": 100},
    ])
    edited = loader.load_file(canvas_path)[0]
    assert edited.doc_id == first.doc_id
    assert edited.text != first.text


# ---- edge labels + direction marking ----

def test_edge_label_appears_and_both_directions_are_represented(tmp_path):
    canvas = _write_canvas(tmp_path / "edge.canvas", nodes=[
        {"id": "a1", "type": "text", "x": 0, "y": 0, "width": 200, "height": 100,
         "text": "Node A content, long enough to clear the minimum chunk floor."},
        {"id": "b1", "type": "text", "x": 400, "y": 0, "width": 200, "height": 100,
         "text": "Node B content, also long enough to clear the minimum chunk floor."},
    ], edges=[
        {"id": "e1", "fromNode": "a1", "toNode": "b1", "fromSide": "right", "toSide": "left",
         "label": "outputs"},
    ])
    chunks = {c.metadata["canvas_node_id"]: c for c in _loader(tmp_path).load_file(canvas)}
    assert "-> [outputs]" in chunks["a1"].text
    assert "Node B content" in chunks["a1"].text
    assert "<- [outputs]" in chunks["b1"].text
    assert "Node A content" in chunks["b1"].text


def test_edge_label_with_newline_is_collapsed(tmp_path):
    canvas = _write_canvas(tmp_path / "multiline_label.canvas", nodes=[
        {"id": "a1", "type": "text", "x": 0, "y": 0, "width": 200, "height": 100,
         "text": "Source node text, long enough to clear the minimum chunk floor."},
        {"id": "b1", "type": "text", "x": 400, "y": 0, "width": 200, "height": 100,
         "text": "Target node text, long enough to clear the minimum chunk floor."},
    ], edges=[
        {"id": "e1", "fromNode": "a1", "toNode": "b1", "label": "expanding\nwith LLMs"},
    ])
    chunks = {c.metadata["canvas_node_id"]: c for c in _loader(tmp_path).load_file(canvas)}
    assert "expanding with LLMs" in chunks["a1"].text
    assert "\n" not in chunks["a1"].metadata["canvas_edge_labels"][0]


# ---- neighbors + file refs ----

def test_neighbors_and_file_refs(tmp_path):
    canvas = _write_canvas(tmp_path / "filelink.canvas", nodes=[
        {"id": "t1", "type": "text", "x": 0, "y": 0, "width": 200, "height": 100,
         "text": "Text node linking out to a file node, long enough for the floor."},
        {"id": "f1", "type": "file", "file": "10 - CANVASes/Other Map.canvas",
         "x": 400, "y": 0, "width": 200, "height": 100},
    ], edges=[
        {"id": "e1", "fromNode": "t1", "toNode": "f1", "label": "code guide"},
    ])
    chunks = _loader(tmp_path).load_file(canvas)
    assert len(chunks) == 1  # the file node gets no chunk of its own
    meta = chunks[0].metadata
    assert meta["canvas_node_id"] == "t1"
    assert "f1" in meta["canvas_neighbors"]
    assert "10 - CANVASes/Other Map.canvas" in meta["canvas_file_refs"]
    assert meta["canvas_degree"] == 1


# ---- group containment: smallest wins ----

def test_group_containment_picks_the_smallest_enclosing_group(tmp_path):
    canvas = _write_canvas(tmp_path / "groups.canvas", nodes=[
        {"id": "g_outer", "type": "group", "label": "Outer Group",
         "x": 0, "y": 0, "width": 1000, "height": 1000},
        {"id": "g_inner", "type": "group", "label": "Inner Group",
         "x": 100, "y": 100, "width": 200, "height": 200},
        {"id": "t1", "type": "text", "x": 120, "y": 120, "width": 50, "height": 50,
         "text": "A node nested inside two groups, long enough to clear the floor."},
    ])
    chunks = _loader(tmp_path).load_file(canvas)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.metadata["canvas_group"] == "Inner Group"
    assert "Group: Inner Group" in c.text
    assert "Outer Group" not in c.text


# ---- malformed input: raise, name the file, never skip silently ----

def test_invalid_json_raises_naming_the_file(tmp_path):
    bad = tmp_path / "broken.canvas"
    bad.write_text("{not valid json at all", encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        _loader(tmp_path).load_file(bad)
    assert "broken.canvas" in str(exc_info.value)


def test_missing_nodes_key_raises_naming_the_file(tmp_path):
    bad = tmp_path / "no_nodes.canvas"
    bad.write_text(json.dumps({"edges": []}), encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        _loader(tmp_path).load_file(bad)
    assert "no_nodes.canvas" in str(exc_info.value)


# ---- non-text nodes never become chunks ----

def test_file_link_and_group_nodes_produce_no_chunks(tmp_path):
    canvas = _write_canvas(tmp_path / "no_text.canvas", nodes=[
        {"id": "f1", "type": "file", "file": "Some/Note.md", "x": 0, "y": 0, "width": 100, "height": 100},
        {"id": "l1", "type": "link", "url": "https://example.com", "x": 200, "y": 0, "width": 100, "height": 100},
        {"id": "g1", "type": "group", "label": "Empty group", "x": 0, "y": 0, "width": 500, "height": 500},
    ])
    chunks = _loader(tmp_path).load_file(canvas)
    assert chunks == []


# ---- min_chunk counts the COMPOSED text, not the raw node text ----

def test_short_node_below_min_chunk_is_skipped_and_counted(tmp_path):
    canvas = _write_canvas(tmp_path / "short.canvas", nodes=[
        {"id": "t1", "type": "text", "text": "hi", "x": 0, "y": 0, "width": 50, "height": 50},
    ])
    loader = _loader(tmp_path, min_chunk=1000)
    chunks = loader.load_file(canvas)
    assert chunks == []
    assert loader.stats["nodes_skipped"] == 1


# ---- CanvasChunk dataclass shape ----

def test_canvas_chunk_to_dict_shape():
    ch = CanvasChunk(text="hello", metadata={"a": 1}, doc_id="deadbeef")
    assert ch.to_dict() == {"doc_id": "deadbeef", "text": "hello", "metadata": {"a": 1}}


def test_canvas_chunk_post_init_hashes_when_doc_id_omitted():
    ch = CanvasChunk(text="hello", metadata={"source_file": "x.canvas"})
    assert ch.doc_id and ch.doc_id != ""


# ---- canvas_edges: the ALIGNED edge list a traversal can pair up ----

def _joined_for_chroma(values):
    """Exactly what Embedder._clean_meta does to a list before Chroma sees it."""
    return ", ".join(str(v) for v in values)


def test_canvas_edges_pairs_each_label_with_its_own_edge(tmp_path):
    """The bug this key exists to fix: canvas_neighbors and canvas_edge_labels
    have DIFFERENT lengths when some edges are unlabelled, so position n of one
    is not position n of the other."""
    canvas = _write_canvas(tmp_path / "mixed.canvas", nodes=[
        {"id": "hub", "type": "text", "x": 0, "y": 0, "width": 200, "height": 100,
         "text": "Hub node with one labelled and one unlabelled edge, long enough."},
        {"id": "plain", "type": "text", "x": 400, "y": 0, "width": 200, "height": 100,
         "text": "Target reached by an UNLABELLED edge, long enough for the floor."},
        {"id": "tagged", "type": "text", "x": 800, "y": 0, "width": 200, "height": 100,
         "text": "Target reached by a LABELLED edge, long enough for the floor."},
    ], edges=[
        {"id": "e1", "fromNode": "hub", "toNode": "plain"},
        {"id": "e2", "fromNode": "hub", "toNode": "tagged", "label": "outputs"},
    ])
    hub = {c.metadata["canvas_node_id"]: c
           for c in _loader(tmp_path).load_file(canvas)}["hub"]
    meta = hub.metadata

    # The two legacy lists cannot be zipped: 2 neighbours, 1 label.
    assert len(meta["canvas_neighbors"]) == 2
    assert len(meta["canvas_edge_labels"]) == 1

    edges = decode_canvas_edges(meta["canvas_edges"])
    assert edges == [
        {"neighbor": "plain", "direction": "->", "label": ""},
        {"neighbor": "tagged", "direction": "->", "label": "outputs"},
    ]


def test_canvas_edges_round_trips_a_label_containing_commas(tmp_path):
    """A REAL label from this vault. Chroma stores lists ", "-joined, so a
    comma inside a label would otherwise read back as several edges."""
    real_label = "այսինքն, հետևում է, որ"
    canvas = _write_canvas(tmp_path / "commas.canvas", nodes=[
        {"id": "a1", "type": "text", "x": 0, "y": 0, "width": 200, "height": 100,
         "text": "Source node text, long enough to clear the minimum chunk floor."},
        {"id": "b1", "type": "text", "x": 400, "y": 0, "width": 200, "height": 100,
         "text": "Target node text, long enough to clear the minimum chunk floor."},
    ], edges=[{"id": "e1", "fromNode": "a1", "toNode": "b1", "label": real_label}])
    a1 = {c.metadata["canvas_node_id"]: c
          for c in _loader(tmp_path).load_file(canvas)}["a1"]

    through_chroma = _joined_for_chroma(a1.metadata["canvas_edges"])
    edges = decode_canvas_edges(through_chroma)
    assert len(edges) == 1, "a comma inside the label split one edge into several"
    assert edges[0]["label"] == real_label
    assert edges[0]["neighbor"] == "b1"
    # The human-readable footer still shows the label verbatim.
    assert real_label in a1.text


def test_edge_codec_round_trips_its_own_separators():
    for label in ("plain", "a, b", "100% sure", "left|right", "%2C not a comma",
                  "", "%|,%"):
        entry = encode_canvas_edge("n1", "->", label)
        assert "," not in entry and entry.count("|") == 2
        assert decode_canvas_edges([entry])[0]["label"] == label


def test_decode_canvas_edges_accepts_both_stored_shapes():
    entries = [encode_canvas_edge("n1", "->", "a"), encode_canvas_edge("n2", "<-", "")]
    as_list = decode_canvas_edges(entries)
    as_string = decode_canvas_edges(_joined_for_chroma(entries))
    assert as_list == as_string
    assert [e["direction"] for e in as_list] == ["->", "<-"]


def test_decode_canvas_edges_is_empty_for_a_node_with_no_edges():
    assert decode_canvas_edges("") == []
    assert decode_canvas_edges([]) == []
    assert decode_canvas_edges(None) == []


def test_decode_canvas_edges_raises_on_a_malformed_entry():
    with pytest.raises(ValueError) as exc_info:
        decode_canvas_edges("n1|->")
    assert "neighbor_id|direction|label" in str(exc_info.value)


def test_canvas_edges_records_an_edge_to_a_file_node(tmp_path):
    """file nodes get no chunk of their own, but the edge to one is still a
    real edge and must appear — a traversal skips it as a dangling id rather
    than never seeing it."""
    canvas = _write_canvas(tmp_path / "toFile.canvas", nodes=[
        {"id": "t1", "type": "text", "x": 0, "y": 0, "width": 200, "height": 100,
         "text": "Text node linking out to a file node, long enough for the floor."},
        {"id": "f1", "type": "file", "file": "10 - CANVASes/Other Map.canvas",
         "x": 400, "y": 0, "width": 200, "height": 100},
    ], edges=[{"id": "e1", "fromNode": "t1", "toNode": "f1", "label": "code guide"}])
    chunk = _loader(tmp_path).load_file(canvas)[0]
    edges = decode_canvas_edges(chunk.metadata["canvas_edges"])
    assert edges == [{"neighbor": "f1", "direction": "->", "label": "code guide"}]


# ---- canvas.max_chunk_size: splitting oversized node bodies ----

def _long_node(node_id: str, chars: int, x: int = 0) -> dict:
    para = "Sentence about the topic that is long enough to pack. " * 4
    body = ""
    while len(body) < chars:
        body += para + "\n\n"
    return {"id": node_id, "type": "text", "text": body[:chars],
            "x": x, "y": 0, "width": 200, "height": 100}


def test_splitting_is_off_by_default_however_long_the_node(tmp_path):
    """Every canvas chunk currently in the index was produced with no
    max_chunk_size — the default must not silently re-chunk the corpus."""
    canvas = _write_canvas(tmp_path / "long.canvas", nodes=[_long_node("n1", 9000)])
    chunks = _loader(tmp_path).load_file(canvas)
    assert len(chunks) == 1
    assert len(chunks[0].text) > 9000


def test_max_chunk_size_splits_and_part_one_keeps_the_unsplit_doc_id(tmp_path):
    canvas = _write_canvas(tmp_path / "long.canvas", nodes=[_long_node("n1", 5000)])
    unsplit = _loader(tmp_path).load_file(canvas)
    loader = CanvasLoader(vault_path=tmp_path, output_file=tmp_path / "o.jsonl",
                          min_chunk=10, max_chunk=1500)
    parts = loader.load_file(canvas)

    assert len(parts) > 1
    # Turning splitting on must UPSERT the chunk that already exists and only
    # ADD the tail — the project's stable-doc_id rule.
    assert parts[0].doc_id == unsplit[0].doc_id
    assert len({c.doc_id for c in parts}) == len(parts)
    assert loader.stats["nodes_split"] == 1
    assert [c.metadata["canvas_part"] for c in parts] == list(range(1, len(parts) + 1))
    assert {c.metadata["canvas_part_count"] for c in parts} == {len(parts)}


def test_split_parts_share_one_canvas_node_id_so_a_graph_hop_reaches_all(tmp_path):
    canvas = _write_canvas(tmp_path / "long.canvas", nodes=[_long_node("n1", 5000)])
    parts = CanvasLoader(vault_path=tmp_path, output_file=tmp_path / "o.jsonl",
                         min_chunk=10, max_chunk=1200).load_file(canvas)
    assert {c.metadata["canvas_node_id"] for c in parts} == {"n1"}


def test_every_split_part_carries_the_full_connections_footer(tmp_path):
    """Splitting the footer across parts would leave some parts of a node with
    no graph context, which is the whole reason this lane exists."""
    canvas = _write_canvas(tmp_path / "long.canvas", nodes=[
        _long_node("n1", 5000),
        {"id": "n2", "type": "text", "x": 900, "y": 0, "width": 200, "height": 100,
         "text": "The neighbour node, long enough to clear the minimum floor."},
    ], edges=[{"id": "e1", "fromNode": "n1", "toNode": "n2", "label": "supports"}])
    parts = CanvasLoader(vault_path=tmp_path, output_file=tmp_path / "o.jsonl",
                         min_chunk=10, max_chunk=1200).load_file(canvas)
    parts = [c for c in parts if c.metadata["canvas_node_id"] == "n1"]
    assert len(parts) > 1
    for c in parts:
        assert "Connections:" in c.text
        assert "-> [supports]" in c.text
        assert c.text.startswith("[Canvas: long]")


def test_unknown_canvas_chunking_strategy_is_rejected(tmp_path):
    with pytest.raises(ValueError) as exc_info:
        CanvasLoader(vault_path=tmp_path, output_file=tmp_path / "o.jsonl",
                     chunking="sideways")
    assert "sideways" in str(exc_info.value)


# ---- canvas.context_depth: inlining neighbour text at ingest time ----

def _depth_canvas(tmp_path: Path) -> Path:
    return _write_canvas(tmp_path / "depth.canvas", nodes=[
        {"id": "a", "type": "text", "x": 0, "y": 0, "width": 200, "height": 100,
         "text": "Node A, the node being composed, long enough for the floor."},
        {"id": "b", "type": "text", "x": 400, "y": 0, "width": 200, "height": 100,
         "text": "Node B FIRST LINE\nNode B second line carries the detail that "
                 "only an inlined neighbour would ever surface."},
        {"id": "c", "type": "text", "x": 800, "y": 0, "width": 200, "height": 100,
         "text": "Node C SECOND HOP first line\nNode C body detail, reachable "
                 "only from B and never inlined at any depth."},
    ], edges=[
        {"id": "e1", "fromNode": "a", "toNode": "b", "label": "leads to"},
        {"id": "e2", "fromNode": "b", "toNode": "c", "label": "then"},
    ])


def _depth_chunk(tmp_path: Path, depth: int) -> CanvasChunk:
    loader = CanvasLoader(vault_path=tmp_path, output_file=tmp_path / "o.jsonl",
                          min_chunk=10, context_depth=depth)
    return {c.metadata["canvas_node_id"]: c
            for c in loader.load_file(_depth_canvas(tmp_path))}["a"]


def test_context_depth_zero_inlines_only_the_neighbour_title(tmp_path):
    text = _depth_chunk(tmp_path, 0).text
    assert "Node B FIRST LINE" in text          # the title line
    assert "Node B second line" not in text     # the body is NOT inlined
    assert "Node C SECOND HOP" not in text


def test_context_depth_one_inlines_the_neighbour_body(tmp_path):
    text = _depth_chunk(tmp_path, 1).text
    assert "Node B second line" in text
    assert "Node C SECOND HOP" not in text, "depth 1 must not reach the second hop"


def test_context_depth_two_adds_second_hop_titles_only(tmp_path):
    text = _depth_chunk(tmp_path, 2).text
    assert "Node B second line" in text
    assert "also links:" in text
    assert "Node C SECOND HOP" in text
    # The second hop contributes a TITLE, never its full body.
    assert "Node C body detail" not in text
    # And it never points back at the node being composed.
    assert "Node A, the node being composed" not in text.split("also links:")[1]


def test_context_depth_inflation_is_measured_not_assumed(tmp_path):
    """The cost of duplicating neighbour text has to be reportable, since
    that is the whole basis for deciding whether depth >= 1 is worth it."""
    sizes = {}
    for depth in (0, 1, 2):
        loader = CanvasLoader(vault_path=tmp_path, output_file=tmp_path / f"o{depth}.jsonl",
                              min_chunk=10, context_depth=depth)
        _depth_canvas(tmp_path)
        loader.ingest_vault(verbose=False)
        sizes[depth] = loader.stats["chars_total"]
        assert loader.stats["chars_context"] > 0
        assert loader.stats["chars_context"] <= loader.stats["chars_total"]
    assert sizes[0] < sizes[1] < sizes[2]


def test_context_depth_outside_zero_one_two_is_rejected(tmp_path):
    with pytest.raises(ValueError) as exc_info:
        CanvasLoader(vault_path=tmp_path, output_file=tmp_path / "o.jsonl",
                     context_depth=3)
    assert "0, 1 or 2" in str(exc_info.value)


# ---- discovery walks, and prunes, rather than pattern-globbing ----

def test_discovery_finds_canvases_at_every_depth(tmp_path):
    from src.ingestion.canvas_loader import iter_canvas_files
    (tmp_path / "a" / "b" / "c").mkdir(parents=True)
    for rel in ("top.canvas", "a/mid.canvas", "a/b/c/deep.canvas"):
        _write_canvas(tmp_path / rel, nodes=[])
    (tmp_path / "a" / "not_a_canvas.md").write_text("x", encoding="utf-8")
    found = {f.relative_to(tmp_path).as_posix() for f in iter_canvas_files(tmp_path)}
    assert found == {"top.canvas", "a/mid.canvas", "a/b/c/deep.canvas"}


def test_discovery_prunes_skipped_directories(tmp_path):
    from src.ingestion.canvas_loader import iter_canvas_files
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / "keep").mkdir()
    _write_canvas(tmp_path / ".obsidian" / "hidden.canvas", nodes=[])
    _write_canvas(tmp_path / "keep" / "kept.canvas", nodes=[])
    found = {f.name for f in iter_canvas_files(tmp_path)}
    assert found == {"kept.canvas"}


def test_a_vault_under_a_skip_named_parent_is_still_walked(tmp_path):
    """The old membership test compared EVERY path component, including the
    ones above the vault root — so a vault living under a folder called
    `.git` or `node_modules` found nothing at all."""
    from src.ingestion.canvas_loader import iter_canvas_files
    vault = tmp_path / "node_modules" / "my vault"
    vault.mkdir(parents=True)
    _write_canvas(vault / "real.canvas", nodes=[])
    assert [f.name for f in iter_canvas_files(vault)] == ["real.canvas"]


def test_discovery_matches_the_extension_case_insensitively(tmp_path):
    from src.ingestion.canvas_loader import iter_canvas_files
    _write_canvas(tmp_path / "Upper.CANVAS", nodes=[])
    assert [f.name for f in iter_canvas_files(tmp_path)] == ["Upper.CANVAS"]
