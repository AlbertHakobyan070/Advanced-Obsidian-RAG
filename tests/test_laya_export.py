"""tools/laya/export_pairs.py: the training-pair export for the Laya fine-tune.

Every test runs against a temp world: a real on-disk Chroma collection whose vectors are
hand-made points on a circle (so "whose nearest neighbours are these" is arithmetic, not luck),
two chunk JSONLs, and a sets/ and a smoke/ directory with one question each. The tool finds it
all the way the real one does, through a config.yaml in the working directory. Nothing here
touches data/ or the live store, and no embedding model is ever loaded.

The chunk table below is the whole story; read it first. `pos` is where a chunk sits on the
circle: the cosine distance between two chunks grows with the gap between their positions, so
the table also says who is whose neighbour.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from math import cos, sin
from pathlib import Path
from typing import NamedTuple

import pytest

chromadb = pytest.importorskip("chromadb")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.bench.questions import SchemaError, question_from_dict, save_suite
from src.utils.config_loader import load_config
from tools.laya import export_pairs as EP

COLLECTION = "vault_test"
SENT = "Maximum likelihood picks the parameter θ that makes the observed sample most probable. "
PROSE = SENT * 5                                    # ~430 chars: over the sampler's 300-char floor
CODE = "def step(x):\n    return x * 2\n" * 12      # newlines matter in code: they must survive
CODE_TYPES = ("py", "ipynb")


class C(NamedTuple):
    id: str
    file: str                  # as the loaders write source_file: "\" in most lanes, "/" in canvas rows
    type: str                  # file_type
    pos: "int | None"          # place on the circle; None = no vector in the store
    jsonl: bool = True         # has a row in the chunk files
    text: "str | None" = None  # default: PROSE or CODE, tagged with the id so every text is unique
    course: str = "Stats"


CHUNKS = [
    # prose side
    C("a1", r"notes\alpha.md", "note", 0),
    C("a2", r"notes\alpha.md", "note", 5),                        # a1's own file: its nearest neighbour
    C("a3", "NOTES/Alpha.md", "note", 11),                        # the same file, spelt another way
    C("g1", r"gold\Gold File.md", "note", 18),                    # the sets name this file as gold
    C("g2", r"gold\Gold File.md", "note", 26),
    C("g3", r"gold\Gold File.md", "note", 35, jsonl=False),       # gold, but only the store knows the chunk
    C("b1", r"courses\beta – notes.pdf", "pdf", 45),
    C("b2", r"courses\beta – notes.pdf", "pdf", 56),
    C("c1", r"courses\gamma.md", "note", 68),
    C("c2", r"courses\gamma.md", "note", 82),
    C("z1", r"store\only.md", "note", 95, jsonl=False),           # a vector with no JSONL row: only a negative
    C("k1", r"smoke\Smoke Gold.md", "note", 109),                 # the smoke set names this file as gold
    C("k2", r"smoke\Smoke Gold.md", "note", 127),
    # chunks the sampler's skip rules refuse as passages (they stay in the store)
    C("t1", r"misc\junk.md", "note", 140, text="too short to teach anything"),
    C("toc1", r"misc\toc.md", "note", 157, text=PROSE + " . . ." * 6),
    C("sk1", r"misc\skip.md", "note", 178, course="__SKIP__"),
    # code side
    C("d1", r"code\delta.py", "py", 195),
    C("d2", r"code\delta.py", "py", 214),
    C("d3", r"code\eps.ipynb", "ipynb", 234),
    C("d4", r"code\eps.ipynb", "ipynb", 255),
    # a JSONL row whose vector the store never got
    C("m1", r"misc\missing.md", "note", None),
]
PROSE_IDS = {"a1", "a2", "a3", "b1", "b2", "c1", "c2"}
CODE_IDS = {"d1", "d2", "d3", "d4"}
GOLD_IDS = {"g1", "g2", "g3", "k1", "k2"}
GOLD_FILES = {"gold/gold file.md", "smoke/smoke gold.md"}     # the sets' spelling, normalised


def _norm(p):
    """The file-identity rule, restated dumbly: '/' separators, case-insensitive."""
    return p.replace("\\", "/").casefold()


def _text_of(c):
    if c.text is not None:
        return c.text
    return f"# {c.id}\n{CODE}" if c.type in CODE_TYPES else f"[{c.id}] {PROSE}"


def _key(seed, did):
    return hashlib.sha256(f"{seed}:{did}".encode("utf-8")).hexdigest()


def _q(qid, suite, gold_file, status="draft"):
    return question_from_dict({
        "id": qid, "question": f"What does {qid} ask about?", "suite": suite, "tier": "T1",
        "split": None, "answerable": True, "gold": [{"file": gold_file}], "nuggets": ["a fact"],
        "expect_course": None,
        "provenance": {"author": "draft", "status": status, "seed_chunks": []}}, "t")


class World:
    def __init__(self, root, capsys):
        self.root = root.resolve()                 # the spelling load_config() resolves to
        self.capsys = capsys
        self.data, self.chroma = self.root / "data", self.root / "chroma_db"
        self.sets, self.smoke = self.root / "sets", self.root / "smoke"
        self.out = self.root / "out" / "pairs.jsonl"
        self.by_id = {c.id: c for c in CHUNKS}
        self.text = {c.id: _text_of(c) for c in CHUNKS}

        # Equal distances would leave the order of two neighbours to the index, and the
        # oracle below could not say who is right.
        pos = [c.pos for c in CHUNKS if c.pos is not None]
        for p in pos:
            gaps = [abs(q - p) for q in pos if q != p]
            assert len(set(gaps)) == len(gaps), f"tied distances from position {p}"

        self.data.mkdir()
        lanes: dict[str, list] = {"chunks.jsonl": [], "code_chunks.jsonl": []}     # two lanes, as on disk
        for c in CHUNKS:
            if c.jsonl:
                lanes["code_chunks.jsonl" if c.type in CODE_TYPES else "chunks.jsonl"].append(
                    {"doc_id": c.id, "text": self.text[c.id],
                     "metadata": {"source_file": c.file, "file_type": c.type, "course_name": c.course}})
        for name, recs in lanes.items():
            (self.data / name).write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")

        stored = [c for c in CHUNKS if c.pos is not None]
        col = chromadb.PersistentClient(path=str(self.chroma)).create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"})
        col.add(ids=[c.id for c in stored],
                embeddings=[[cos(c.pos / 100), sin(c.pos / 100)] for c in stored],
                documents=[self.text[c.id] for c in stored],
                metadatas=[{"source_file": c.file, "file_type": c.type, "course_name": c.course}
                           for c in stored])

        self.sets.mkdir()
        self.smoke.mkdir()
        # Neither question spells its gold file the way the corpus does.
        save_suite(self.sets / "lexical.yaml", [_q("lex-0001", "lexical", "gold/gold file.md")])
        save_suite(self.smoke / "smoke.yaml", [_q("smk-0001", "books", "SMOKE\\SMOKE GOLD.MD")])
        self.write_config()

    def write_config(self, chunks_file="data/chunks.jsonl", chroma_dir="chroma_db",
                     collection_name=COLLECTION, extra=""):
        (self.root / "config.yaml").write_text(
            "paths:\n"
            f"  chunks_file: {chunks_file}\n"
            f"  chroma_dir: {chroma_dir}\n"
            f"  collection_name: {collection_name}\n" + extra, encoding="utf-8")

    def run(self, *flags, default_out=True):
        """main() as typed on a command line, aimed at this world. Returns (rows, stdout)."""
        argv = list(flags)
        for flag, path in (("--sets", self.sets), ("--smoke", self.smoke)):
            if flag not in argv:
                argv += [flag, str(path)]
        if default_out and "--out" not in argv:
            argv += ["--out", str(self.out)]
        EP.main(argv)
        stdout = self.capsys.readouterr().out
        rows = ([json.loads(line) for line in self.out.read_text(encoding="utf-8").splitlines()]
                if self.out.exists() else [])
        return rows, stdout

    def snapshot(self):
        """Everything the store holds: count, ids, documents, metadata, vectors."""
        col = chromadb.PersistentClient(path=str(self.chroma)).get_collection(COLLECTION)
        got = col.get(include=["documents", "metadatas", "embeddings"])
        order = sorted(range(len(got["ids"])), key=lambda i: got["ids"][i])
        return (col.count(), [got["ids"][i] for i in order], [got["documents"][i] for i in order],
                [got["metadatas"][i] for i in order], [got["embeddings"][i].tolist() for i in order])

    def collection_names(self):
        return {c.name for c in chromadb.PersistentClient(path=str(self.chroma)).list_collections()}

    def expected_negatives(self, pid, k):
        """The first k stored chunks by distance that are of another file and not gold: the rule
        computed by brute force, without asking Chroma anything."""
        me = self.by_id[pid]
        near = sorted((c for c in CHUNKS if c.pos is not None and c.id != pid),
                      key=lambda c: abs(c.pos - me.pos))
        return [c.id for c in near
                if _norm(c.file) != _norm(me.file) and _norm(c.file) not in GOLD_FILES][:k]


@pytest.fixture
def world(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path.resolve())          # load_config() looks for ./config.yaml first
    return World(tmp_path, capsys)


def _pids(rows):
    return [r["pid"] for r in rows]


def _negative_ids(rows):
    return {n["id"] for r in rows for n in r["negatives"]}


# --- the leakage guard ------------------------------------------------------------------

def test_gold_file_chunks_are_neither_positives_nor_negatives_and_the_count_is_printed(world):
    # --negatives 50 reaches every chunk in the store, so a gold chunk that the guard let
    # through would show up as somebody's negative.
    rows, out = world.run("--n-passages", "100", "--negatives", "50")
    assert rows
    assert not GOLD_IDS & set(_pids(rows))
    # g3 has no JSONL row: only the store's own metadata can say it is a gold file's chunk.
    assert not GOLD_IDS & _negative_ids(rows)
    # g1, g2 (the sets) and k1, k2 (the smoke set): four JSONL rows in two files.
    assert re.search(r"excluded\s+4 chunk\(s\) in 2 file\(s\)", out), out


def test_a_rejected_questions_gold_file_is_still_kept_out(world):
    # A rejected draft can be un-rejected, or have its gold re-pointed, in review; over-excluding
    # costs some training chunks, under-excluding contaminates the eval.
    save_suite(world.sets / "rejected.yaml",
               [_q("lex-0002", "lexical", "COURSES/GAMMA.MD", status="rejected")])
    rows, _ = world.run("--n-passages", "100", "--negatives", "50")
    assert not {"c1", "c2"} & set(_pids(rows))
    assert not {"c1", "c2"} & _negative_ids(rows)


def test_a_gold_file_that_matches_no_chunk_is_reported(world):
    # A typo'd gold name guards nothing, and nothing else would say so.
    save_suite(world.sets / "typo.yaml", [_q("lex-0002", "lexical", "gold/a file nobody has.md")])
    _, out = world.run("--n-passages", "100")
    assert "1 of the 3 gold file(s) named match no chunk row" in out, out


def test_a_missing_sets_dir_means_no_gold_files_and_the_output_says_so(world):
    shutil.rmtree(world.sets)
    rows, out = world.run("--n-passages", "100")
    assert {"g1", "g2"} <= set(_pids(rows)) and not {"k1", "k2"} & set(_pids(rows))
    assert f"{world.sets}" in out and "does not exist" in out
    assert re.search(r"excluded\s+2 chunk\(s\) in 1 file\(s\)", out), out

    shutil.rmtree(world.smoke)
    rows, out = world.run("--n-passages", "100")
    assert {"g1", "g2", "k1", "k2"} <= set(_pids(rows))
    assert re.search(r"excluded\s+0 chunk\(s\) in 0 file\(s\)", out), out


def test_a_malformed_set_file_raises_and_nothing_is_written(world):
    (world.sets / "broken.yaml").write_text("- id: nope\n", encoding="utf-8")
    with pytest.raises(SchemaError, match=r"broken\.yaml"):
        world.run()
    assert not world.out.exists()


# --- passages ---------------------------------------------------------------------------

def test_only_chunks_that_pass_the_samplers_skip_rules_and_have_a_vector_become_passages(world):
    rows, _ = world.run("--n-passages", "100", "--negatives", "50")
    # not t1 (too short), toc1 (a contents page), sk1 (__SKIP__), z1 (no JSONL row) or m1 (no vector)
    assert set(_pids(rows)) == PROSE_IDS | CODE_IDS


def test_an_id_with_no_stored_vector_is_skipped_and_counted(world):
    rows, out = world.run("--n-passages", "100")
    assert "m1" not in _pids(rows)
    assert re.search(r"no stored vector\s+1 chunk\(s\)", out) and "m1" in out, out


def test_passages_come_out_in_the_documented_hash_order(world):
    rows, _ = world.run("--n-passages", "100", "--seed", "7")
    assert _pids(rows) == sorted(PROSE_IDS | CODE_IDS, key=lambda d: _key(7, d))


@pytest.mark.parametrize("n, want", [
    (6, {"prose": 3, "code": 3}),        # half and half...
    (10, {"prose": 6, "code": 4}),       # ...but the code side only has four: prose takes up the slack
    (100, {"prose": 7, "code": 4}),      # and asking for more than there is gives everything
])
def test_prose_and_code_are_both_drawn(world, n, want):
    rows, out = world.run("--n-passages", str(n), "--seed", "0")
    kinds = Counter("code" if r["meta"]["file_type"] in CODE_TYPES else "prose" for r in rows)
    assert kinds == want
    if n == 6:      # and within each side the picks are the first by hash
        assert set(_pids(rows)) == (set(sorted(PROSE_IDS, key=lambda d: _key(0, d))[:3])
                                    | set(sorted(CODE_IDS, key=lambda d: _key(0, d))[:3]))
    if n == 100:
        assert "asked for 100" in out


@pytest.mark.parametrize("n, prose, code, want", [
    (10, 100, 100, (5, 5)), (11, 100, 100, (6, 5)),      # an odd one goes to prose
    (10, 100, 3, (7, 3)), (10, 2, 100, (2, 8)),          # the short side gives its slack away
    (10, 2, 3, (2, 3)),                                  # nothing left to give
    (1, 5, 5, (1, 0)), (1, 0, 5, (0, 1)),
])
def test_split_budget(n, prose, code, want):
    assert EP._split_budget(n, prose, code) == want


def test_same_seed_is_byte_identical_and_another_seed_differs(world):
    world.run("--n-passages", "5", "--seed", "0")
    first = world.out.read_bytes()
    world.run("--n-passages", "5", "--seed", "0")
    assert world.out.read_bytes() == first
    others = set()
    for seed in (1, 2, 3):                 # five of eleven: another seed is free to pick other chunks
        world.run("--n-passages", "5", "--seed", str(seed))
        others.add(world.out.read_bytes())
    assert others != {first}


# --- negatives --------------------------------------------------------------------------

def test_negatives_are_the_nearest_chunks_of_other_files_in_order(world):
    rows, _ = world.run("--n-passages", "100", "--negatives", "5")
    for r in rows:
        assert [n["id"] for n in r["negatives"]] == world.expected_negatives(r["pid"], 5), r["pid"]
    a1 = next(r for r in rows if r["pid"] == "a1")
    # a2 and a3 (a1's own file), then g1, g2, g3 (gold) are passed over; z1 has no JSONL row
    # at all and is still a fair negative.
    assert [n["id"] for n in a1["negatives"]] == ["b1", "b2", "c1", "c2", "z1"]


def test_negatives_never_share_the_positives_file_even_when_only_separator_or_case_differs(world):
    rows, _ = world.run("--n-passages", "100", "--negatives", "50")
    for r in rows:
        mine = _norm(world.by_id[r["pid"]].file)
        for n in r["negatives"]:
            assert _norm(world.by_id[n["id"]].file) != mine, (r["pid"], n["id"])
    by_pid = {r["pid"]: {n["id"] for n in r["negatives"]} for r in rows}
    assert not {"a2", "a3"} & by_pid["a1"]          # a3 is "NOTES/Alpha.md": a1's file, spelt otherwise
    assert not {"a1", "a2"} & by_pid["a3"]


@pytest.mark.parametrize("k", [1, 3, 12])
def test_negatives_flag_is_respected(world, k):
    rows, _ = world.run("--n-passages", "100", "--negatives", str(k))
    assert {len(r["negatives"]) for r in rows} == {k}


def test_a_passage_with_too_few_neighbours_keeps_what_it_got_and_is_counted(world):
    # The alpha file's three passages can find only 12 chunks of other, non-gold files.
    rows, out = world.run("--n-passages", "100", "--negatives", "13")
    short = {r["pid"]: len(r["negatives"]) for r in rows if len(r["negatives"]) < 13}
    assert short == {"a1": 12, "a2": 12, "a3": 12}
    assert re.search(r"3 of 11 passage\(s\) got fewer than 13", out), out


# --- the store and the output -----------------------------------------------------------

def test_the_store_is_unchanged_afterwards(world):
    before = world.snapshot()
    world.run("--n-passages", "100")
    assert world.snapshot() == before


def test_batching_and_paging_do_not_change_the_output(world, monkeypatch):
    world.run("--n-passages", "100", "--negatives", "5")
    whole = world.out.read_bytes()
    monkeypatch.setattr(EP, "BATCH", 3)          # eleven passages in four batches
    monkeypatch.setattr(EP, "ID_PAGE", 4)        # twenty stored ids in five pages
    world.run("--n-passages", "100", "--negatives", "5")
    assert world.out.read_bytes() == whole
    col = chromadb.PersistentClient(path=str(world.chroma)).get_collection(COLLECTION)
    assert EP._stored_ids(col) == {c.id for c in CHUNKS if c.pos is not None}


def test_a_passage_whose_vector_has_gone_missing_is_an_error_and_writes_nothing(world, monkeypatch):
    real = EP._stored_ids
    monkeypatch.setattr(EP, "_stored_ids", lambda col: real(col) | {"m1"})     # claims m1 has a vector
    with pytest.raises(EP.ExportError, match="m1"):
        world.run("--n-passages", "100")
    assert not world.out.exists() and not any(world.out.parent.glob("pairs*"))


def test_a_missing_collection_is_a_readable_error_and_is_not_created(world):
    world.write_config(collection_name="no_such_collection")
    with pytest.raises(EP.ExportError, match=r"no_such_collection.*never creates"):
        world.run()
    assert world.collection_names() == {COLLECTION}
    assert not world.out.exists()


def test_a_missing_chroma_directory_is_an_error_and_is_not_created(world):
    world.write_config(chroma_dir="nowhere/chroma_db")
    with pytest.raises(EP.ExportError, match="nowhere"):
        world.run()
    assert not (world.root / "nowhere").exists()       # PersistentClient would have made it


@pytest.mark.parametrize("chunks_file, match", [
    ("empty/chunks.jsonl", "no passage"),               # the folder exists and holds no chunk files
    ("nowhere/chunks.jsonl", "does not exist"),
])
def test_nothing_to_export_is_an_error_not_an_empty_file(world, chunks_file, match):
    (world.root / "empty").mkdir()
    world.write_config(chunks_file=chunks_file)
    with pytest.raises(EP.ExportError, match=match):
        world.run()
    assert not world.out.exists()


def test_rows_have_the_documented_shape_and_the_text_is_kept_verbatim(world):
    rows, _ = world.run("--n-passages", "100", "--negatives", "2")
    r = next(r for r in rows if r["pid"] == "b1")
    assert list(r) == ["pid", "passage", "negatives", "meta"]
    assert r["passage"] == world.text["b1"]
    assert r["meta"] == {"source_file": r"courses\beta – notes.pdf", "file_type": "pdf"}
    assert all(list(n) == ["id", "text"] and n["text"] == world.text[n["id"]] for n in r["negatives"])
    d1 = next(r for r in rows if r["pid"] == "d1")
    assert d1["passage"] == world.text["d1"] and "\n" in d1["passage"]     # code keeps its line breaks


def test_output_is_utf8_with_lf_endings_and_not_ascii_escaped(world):
    world.run("--n-passages", "100")
    raw = world.out.read_bytes()
    assert "–".encode("utf-8") in raw and "θ".encode("utf-8") in raw
    assert b"\\u2013" not in raw and b"\r" not in raw and raw.endswith(b"\n")


def test_the_output_size_is_printed(world):
    _, out = world.run("--n-passages", "100")
    assert f"{world.out.stat().st_size} bytes" in out and str(world.out) in out


def test_defaults_sets_from_config_smoke_from_eval_smoke_out_under_data_laya(world):
    # Move the question sets to where the defaults look, then give no path flags at all.
    shutil.move(str(world.sets), str(world.root / "configured_sets"))
    (world.root / "eval").mkdir()
    shutil.move(str(world.smoke), str(world.root / "eval" / "smoke"))
    world.write_config(extra="eval:\n  sets_dir: configured_sets\n")
    EP.main(["--n-passages", "100"])
    out = world.capsys.readouterr().out
    default_out = world.root / "data" / "laya" / "pairs.jsonl"            # data/laya did not exist
    rows = [json.loads(line) for line in default_out.read_text(encoding="utf-8").splitlines()]
    assert not GOLD_IDS & set(_pids(rows))
    assert re.search(r"excluded\s+4 chunk\(s\) in 2 file\(s\)", out), out


def test_default_paths_resolve_against_the_projects_root_not_the_working_directory(world, monkeypatch):
    # Run from a subfolder, load_config() falls back to the project's own config.yaml (modelled
    # here by pointing it at this world's); nothing may then be read or written relative to the
    # folder the command happens to be typed in.
    (world.root / "eval").mkdir()
    shutil.move(str(world.smoke), str(world.root / "eval" / "smoke"))
    elsewhere = world.root / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = world.root / "config.yaml"
    monkeypatch.setattr(EP, "load_config", lambda: load_config(config))
    EP.main(["--n-passages", "100", "--sets", str(world.sets)])             # no --out, no --smoke
    rows = [json.loads(line) for line in
            (world.root / "data" / "laya" / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert not {"k1", "k2"} & set(_pids(rows))                             # eval/smoke found under the root
    assert not (elsewhere / "data").exists()


def test_a_failed_write_leaves_the_old_file_and_no_partial_one(tmp_path):
    out = tmp_path / "laya" / "pairs.jsonl"
    EP._write_atomic(out, iter([{"k": "old"}]))                  # also makes the parent directory
    assert json.loads(out.read_text(encoding="utf-8")) == {"k": "old"}

    def dies_midway():
        yield {"k": "new"}
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        EP._write_atomic(out, dies_midway())
    assert json.loads(out.read_text(encoding="utf-8")) == {"k": "old"}
    assert [p.name for p in out.parent.iterdir()] == ["pairs.jsonl"]


# --- the command line -------------------------------------------------------------------

def test_it_runs_as_a_script_by_path_with_no_help_from_pythonpath(world):
    # Everything above imports the module as a package, with the project already on sys.path.
    # Run by path, sys.path[0] is tools/laya, and only the module's own bootstrap puts the
    # project there; a PYTHONPATH that happens to hold the project would hide a broken one.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, str(Path(EP.__file__)), "--n-passages", "100", "--sets", str(world.sets),
         "--smoke", str(world.smoke), "--out", str(world.out)],
        cwd=world.root, env=env, capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    assert "Wrote" in proc.stdout
    assert len(world.out.read_text(encoding="utf-8").splitlines()) == 11


def test_flag_defaults_are_the_plans():
    a = EP.build_parser().parse_args([])
    assert (a.n_passages, a.negatives, a.seed) == (4000, 15, 0)
    assert (a.out, a.sets, a.smoke) == (None, None, None)            # None: resolved from the project
    assert EP.DEFAULT_OUT == "data/laya/pairs.jsonl" and EP.DEFAULT_SMOKE == "eval/smoke"


@pytest.mark.parametrize("flag, value", [("--n-passages", "0"), ("--n-passages", "-3"),
                                         ("--negatives", "0"), ("--negatives", "x")])
def test_counts_must_be_positive_integers(flag, value, capsys):
    with pytest.raises(SystemExit) as e:
        EP.main([flag, value])
    assert e.value.code == 2 and flag in capsys.readouterr().err
