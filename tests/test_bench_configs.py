import os
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.bench.configs import FACTORS, factor_config, factorial, ladder, leave_one_out, resolve
from eval.bench.fingerprint import chunk_files_digest, config_digest, git_state, index_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def test_full_set_is_the_shipped_default_shape():
    c = factor_config(frozenset(FACTORS))
    assert c["lanes"] == sorted(["dense", "sparse", "dense_scope", "sparse_scope",
                                 "dense_code", "sparse_code"])
    assert c["metadata_boost"] and c["auto_preset"] and c["hyde"]
    assert c["rerank"] == "cross_encoder" and c["top_k"] == 10


def test_routing_without_a_family_adds_nothing():
    c = factor_config(frozenset({"sparse", "routing"}))
    assert c["lanes"] == ["sparse", "sparse_scope"] and c["metadata_boost"] is True


def test_routing_forces_the_scope_lanes_open_and_their_absence_closes_them():
    # config.yaml ships retrieval.scope_lanes off, so a routing rung that only
    # LISTED the scope lanes would run without them.
    assert factor_config(frozenset({"dense", "sparse", "routing"}))["scope_lanes"] is True
    assert factor_config(frozenset({"dense", "sparse"}))["scope_lanes"] is False


def test_empty_family_yields_no_lanes():
    assert factor_config(frozenset({"rerank", "hyde"}))["lanes"] == []


def test_ladder_loo_factorial_shapes():
    names = [n for n, _ in ladder()]
    assert names == ["R0-bm25", "R1-dense", "R2-hybrid", "R3-routing", "R4-code",
                     "R5-rerank", "R6-hyde"]
    loo = dict(leave_one_out())
    assert set(loo) == {"full"} | {f"-{f}" for f in FACTORS}
    assert "sparse" not in loo["-sparse"]["lanes"]
    combos = factorial()
    assert len(combos) == 64 and len({c[2] for c in combos}) == 64


def test_resolve_named(tmp_path):
    p = tmp_path / "configs.yaml"
    p.write_text("configs:\n  wide:\n    top_k: 10\n    dense_top_k: 60\n", encoding="utf-8")
    [(name, ov)] = resolve("wide", p)
    assert name == "wide" and ov["dense_top_k"] == 60 and ov["omnisearch"] is False


def test_shipped_configs_file_parses_and_resolves():
    got = resolve("default-k10,wide-pools", ROOT / "eval" / "configs.yaml")
    assert [n for n, _ in got] == ["default-k10", "wide-pools"]
    assert got[1][1]["dense_top_k"] == 60 and got[0][1]["omnisearch"] is False


def test_factorial_names_are_unique_and_the_empty_set_is_f_none():
    # The runner keys its rows by config name, so two subsets must never share one.
    names = [n for n, _, _ in factorial()]
    assert len(set(names)) == 64 and "F:none" in names and "F:dense+hyde" in names


def test_resolve_generated_sets_need_no_file_and_unknown_names_are_reported(tmp_path):
    absent = tmp_path / "absent.yaml"                       # ladder/loo/factorial never read it
    assert resolve("ladder", absent)[0][0] == "R0-bm25"
    assert len(resolve("loo", absent)) == 7 and len(resolve("factorial", absent)) == 64
    p = tmp_path / "configs.yaml"
    p.write_text("configs:\n  wide:\n    top_k: 10\n", encoding="utf-8")
    with pytest.raises(KeyError) as e:
        resolve("wide,nope", p)
    assert "nope" in str(e.value) and "wide" in str(e.value)    # names the culprit, lists the known


def test_chunk_files_digest_tracks_the_chunk_files_only(tmp_path):
    chunks, pdf = tmp_path / "chunks.jsonl", tmp_path / "pdf_chunks.jsonl"
    chunks.write_text("a\n", encoding="utf-8")
    pdf.write_text("b\n", encoding="utf-8")
    base = chunk_files_digest(tmp_path)
    assert chunk_files_digest(tmp_path) == base                      # stable while nothing changes
    (tmp_path / "notes.txt").write_text("unrelated", encoding="utf-8")
    assert chunk_files_digest(tmp_path) == base                      # only *chunks.jsonl count
    st = pdf.stat()
    os.utime(pdf, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))    # touched, same bytes
    touched = chunk_files_digest(tmp_path)
    assert touched != base
    st = chunks.stat()
    chunks.write_text("a longer line\n", encoding="utf-8")
    os.utime(chunks, ns=(st.st_atime_ns, st.st_mtime_ns))            # mtime restored: size alone
    assert chunk_files_digest(tmp_path) != touched


def test_config_digest_ignores_key_order_but_sees_values():
    cfg = lambda d: SimpleNamespace(as_dict=lambda: d)
    a = config_digest(cfg({"a": 1, "b": {"x": 2}}))
    assert len(a) == 16 and a == config_digest(cfg({"b": {"x": 2}, "a": 1}))
    assert a != config_digest(cfg({"a": 1, "b": {"x": 3}}))


def test_git_state_reads_this_checkout():
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout (e.g. a source tarball)")
    g = git_state(ROOT)
    assert re.fullmatch(r"[0-9a-f]{40,64}", g["sha"]) and isinstance(g["dirty"], bool)


def test_git_state_raises_outside_a_repo_instead_of_inventing_a_sha(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))   # never find a parent repo
    with pytest.raises(RuntimeError, match="rev-parse"):
        git_state(tmp_path)


def test_index_fingerprint_shape_and_bm25_sidecar(tmp_path):
    (tmp_path / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    cfg = SimpleNamespace(
        get=lambda k, d=None: {"embedding.provider": "local", "embedding.local_model": "emb-m",
                               "retrieval.cross_encoder_model": "rr-m"}.get(k, d),
        path=lambda k, d=None: {"paths.chunks_file": tmp_path / "chunks.jsonl",
                                "paths.bm25_index": tmp_path / "bm25.pkl"}[k])
    rag = SimpleNamespace(route_coverage=None, retriever=SimpleNamespace(
        _get_collection=lambda: SimpleNamespace(count=lambda: 7)))
    assert index_fingerprint(cfg, rag)["bm25"] is None                # no sidecar written yet
    (tmp_path / "bm25.pkl.meta.json").write_text('{"count": 5, "built_at": "t"}', encoding="utf-8")
    assert index_fingerprint(cfg, rag) == {
        "dense_count": 7, "collection": "obsidian_vault", "bm25": {"count": 5, "built_at": "t"},
        "embedding": {"provider": "local", "model": "emb-m"}, "reranker": "rr-m",
        "chunk_files": chunk_files_digest(tmp_path)}


def test_index_fingerprint_names_the_loaded_route_sidecar(tmp_path):
    """A GM-route arm's numbers depend on the pointer sidecar: the run record names the
    loaded one (coverage, built_at, sha256). With none loaded, the record is as before."""
    import hashlib
    (tmp_path / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    cfg = SimpleNamespace(
        get=lambda k, d=None: {"embedding.provider": "local", "embedding.local_model": "emb-m"}.get(k, d),
        path=lambda k, d=None: {"paths.chunks_file": tmp_path / "chunks.jsonl",
                                "paths.bm25_index": tmp_path / "bm25.pkl"}[k])
    rag = lambda coverage: SimpleNamespace(route_coverage=coverage, retriever=SimpleNamespace(
        _get_collection=lambda: SimpleNamespace(count=lambda: 7)))
    before = index_fingerprint(cfg, rag(None))
    unloaded = {"generated_rows": 3, "with_pointers": 0, "sidecar": None}    # a store with no sidecar
    assert index_fingerprint(cfg, rag(unloaded)) == before and "route" not in before
    sidecar = tmp_path / "route_pointers.json"
    raw = b'{"version": 1, "built_at": "2026-10-29T00:00:00Z", "pointers": {}, "unrouted": {}}'
    sidecar.write_bytes(raw)
    loaded = {"generated_rows": 3, "with_pointers": 2, "sidecar": str(sidecar)}
    got = index_fingerprint(cfg, rag(loaded))
    assert got == {**before, "route": {**loaded, "built_at": "2026-10-29T00:00:00Z",
                                       "sha256": hashlib.sha256(raw).hexdigest()}}


def test_graph_ablation_shapes_and_baselines(tmp_path):
    from eval.bench.configs import BASELINES
    absent = tmp_path / "absent.yaml"                       # generated: the file is never read
    got = resolve("graph", absent)
    assert [n for n, _ in got] == ["G-off", "G-legacy", "G-e1e4"]
    c = dict(got)
    # the gate and decompose pinned off: no arm may shorten or re-sort the search's list
    assert c["G-off"] == {**dict(resolve("diversity", absent))["D-off"], "gate": False, "decompose": False,
                          "graph": None}
    # direction pinned: the arms are defined as both-ways walks, whatever graph.direction says
    assert c["G-legacy"]["graph"] == {"bridges": False, "cross_canvas": False, "wikilinks": False,
                                      "direction": "both"}
    assert c["G-e1e4"]["graph"] == {"bridges": True, "cross_canvas": True, "wikilinks": True,
                                    "direction": "both"}
    assert all({**o, "graph": None} == c["G-off"] for o in c.values())   # only the walk differs
    assert BASELINES["G-legacy"] == "G-off" and BASELINES["G-e1e4"] == "G-legacy"
    p = tmp_path / "configs.yaml"
    p.write_text("configs:\n  wide:\n    top_k: 10\n", encoding="utf-8")
    with pytest.raises(KeyError, match="graph"):            # the error lists the generated sets
        resolve("nope", p)


def test_generated_ablation_gains_route_and_its_two_pairings(tmp_path):
    from eval.bench.configs import BASELINES, GATE_PAIRS, graph_ablation
    arms = dict(resolve("generated", tmp_path / "x.yaml"))   # generated: the file is never read
    assert list(arms) == ["GM-off", "GM-drop", "GM-route"]
    assert [o["generated_mode"] for o in arms.values()] == ["off", "drop", "route"]
    g_off = dict(graph_ablation())["G-off"]
    for name in ("GM-off", "GM-drop"):                      # G-off plus the knob, nothing else
        assert {k: v for k, v in arms[name].items() if k != "generated_mode"} == g_off
    r = arms["GM-route"]                                    # ...plus the routed lane, or route == drop
    assert "routed" in r["lanes"] and set(r["lanes"]) - {"routed"} == set(g_off["lanes"])
    assert {k: v for k, v in r.items() if k not in ("generated_mode", "lanes")} == \
        {k: v for k, v in g_off.items() if k != "lanes"}
    assert BASELINES["GM-route"] == "GM-drop" and GATE_PAIRS["GM-route"] == "GM-off"
    assert BASELINES["GM-drop"] == "GM-off"
    p = tmp_path / "configs.yaml"
    p.write_text("configs:\n  wide:\n    top_k: 10\n", encoding="utf-8")
    with pytest.raises(KeyError, match="generated"):        # the error lists the new keyword
        resolve("nope", p)
