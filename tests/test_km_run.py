"""Knowledge-map run orchestration (src/knowledge_map/run.py), end to end. No network:
a synthetic PDF in a fake vault, staging in a separate dir, a fake client scripted per role
(the role is the requested model's prefix, e.g. "extract-m")."""
import dataclasses
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS

import fitz
import pytest

from src.ingestion.canvas_loader import SOURCE_LINE_RE
from src.knowledge_map import run as run_module
from src.knowledge_map.config import KMConfig, RoleSpec
from src.knowledge_map.graph import edge_id
from src.knowledge_map.run import run_map
from src.knowledge_map.schema import Extraction
from src.knowledge_map.templates import check_extraction, load_templates

PAGES = [
    "A ledger records every transaction in order.\nIt is kept by the branch clerk each day.",
    "A journal is the book of original entry.\nEntries move from the journal to the ledger.",
    "Double entry means every debit has a credit.\nThe trial balance checks that totals agree.",
    "An audit reviews the books at year end.\nAuditors sample entries and trace them back.",
]


def node(title, quote, importance=3):
    return {"title": title, "kind": "concept", "summary": f"About {title}.", "quotes": [quote],
            "importance": importance}


SEC1 = {"nodes": [node("Ledger", "A ledger records every transaction in order"),
                  node("Journal", "A journal is the book of original entry")],
        "edges": [{"source": "Journal", "target": "Ledger", "type": "leads-to",
                   "quote": "Entries move from the journal to the ledger"},
                  {"source": "Journal", "target": "Nowhere", "type": "uses", "quote": None}]}
SEC2 = {"nodes": [node("Double Entry", "Double entry means every debit has a credit"),
                  node("Audit", "An audit reviews the books at year end")], "edges": []}
# keyed by a phrase in the unit's text
EXTRACT = {"ledger records every": SEC1, "Double entry means": SEC2}
LINK = {"edges": [{"source": "concept-double-entry", "target": "concept-ledger", "type": "uses",
                   "quote": None},
                  {"source": "concept-journal", "target": "concept-ledger", "type": "leads-to",
                   "quote": None}]}      # a duplicate of the quoted extract edge


class Fake:
    """Replies per role; counts calls per role. `fail` roles raise on every call;
    verify answers unsupported for the ids in `unsupported` while their claim still holds
    no summary from `trim`, or always for ids in `still_unsupported`;
    the trim prompt (sent to the verify role) answers from `trim` ({id: summary or None})."""
    def __init__(self, extract=EXTRACT, link=LINK, fail=(), unsupported=(), template="concept",
                 trim=None, still_unsupported=(), gapfill=None):
        self.extract, self.link, self.fail, self.unsupported = extract, link, set(fail), set(unsupported)
        self.template, self.trim, self.still = template, trim or {}, set(still_unsupported)
        self.gapfill, self.gapfill_users = gapfill or {}, []
        self.calls = Counter()

    def __call__(self, provider, model):
        role = model.split("-")[0]

        def complete(system, user, temperature=None, max_tokens=None, json_mode=False):
            self.calls[role] += 1
            if role in self.fail:
                raise RuntimeError(f"{role} is down")
            if role == "classify":
                reply = {"template": self.template, "reason": "explains bookkeeping terms"}
            elif role == "extract" and "too few nodes" in user:
                self.calls["gapfill"] += 1
                self.gapfill_users.append(user)
                reply = next(r for k, r in self.gapfill.items() if k in user)
            elif role == "extract":
                reply = next(r for k, r in self.extract.items() if k in user)
            elif role == "link":
                reply = self.link
            elif '"revisions"' in system:
                self.calls["trim"] += 1
                ids = re.findall(r"^id: (\S+)$", user, re.M)
                reply = {"revisions": [{"id": i, "summary": self.trim.get(i)} for i in ids]}
            else:
                items = re.findall(r"^id: (\S+)\nclaim: (.*)$", user, re.M)
                trimmed = [s for s in self.trim.values() if s]
                reply = {"verdicts": [{"id": i, "reason": "checked",
                                       "supported": i not in self.still and
                                       not (i in self.unsupported and not any(s in claim for s in trimmed))}
                                      for i, claim in items]}
            return NS(text=json.dumps(reply), model=model, usage={"input_tokens": 10, "output_tokens": 5})
        return NS(complete=complete)


def roles():
    return {r: RoleSpec("fake", f"{r}-m", 100, 0.0, ())
            for r in ("classify", "extract", "link", "verify", "reverify")}


@pytest.fixture
def env(tmp_path):
    vault = tmp_path / "vault"
    (vault / "Books").mkdir(parents=True)
    pdf = vault / "Books" / "book.pdf"
    doc = fitz.open()
    for text in PAGES:
        page = doc.new_page()
        for i, line in enumerate(text.split("\n")):
            page.insert_text((72, 72 + 14 * i), line)
    doc.set_metadata({"title": "Test Book"})
    doc.save(pdf)
    doc.close()
    k = KMConfig(staging_dir=tmp_path / "staging", vault_root=vault, min_grounded_ratio=0.9,
                 max_span_pages=3, node_cap=40, fanout_max=8, quote_words=(5, 30),
                 max_wall_minutes=10, max_llm_calls=50, retries=0, verify_batch=10,
                 max_unit_chars=10_000, window_pages=2, max_scanned_ratio=0.2, scanned_threshold=50,
                 heading_min_ratio=1.15, min_section_chars=400, chars_per_node=2000,
                 rate_limit_sleep_s=0.0, request_timeout_s=60.0, generated_markers=("(generated)",),
                 roles=roles())
    return NS(vault=vault, pdf=pdf, k=k, tmp=tmp_path, folder=tmp_path / "staging" / "test-book" / "pages-1-4")


def run(env, fake, **kw):
    kw.setdefault("pages", (1, 4))
    kw.setdefault("template", "concept")
    return run_map(kw.pop("k", env.k), env.pdf, client_factory=fake, progress=lambda s: None, **kw)


def files(folder: Path, work=False) -> list[str]:
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*")
                  if p.is_file() and (work or not p.relative_to(folder).as_posix().startswith(".work/")))


def snapshot(d: Path) -> dict:
    return {p: p.read_bytes() for p in d.rglob("*") if p.is_file()}


def test_end_to_end_writes_staging_only(env):
    before = snapshot(env.vault)
    fake = Fake()
    rec = run(env, fake, template=None, argv=["map", "book.pdf"])
    assert rec["status"] == "ok", rec["reason"]
    assert fake.calls["classify"] == 1 and rec["template"] == "concept"
    assert snapshot(env.vault) == before
    out = files(env.folder)
    canvas = [f for f in out if f.endswith(".canvas")]
    assert len(canvas) == 1 and canvas[0].endswith(" (generated).canvas")
    assert {"map.json", "run.json"} <= set(out) and len([f for f in out if f.startswith("runs/")]) == 1
    graph = json.loads((env.folder / "map.json").read_text(encoding="utf-8"))
    assert {n["id"] for n in graph["nodes"] if n["kind"] != "hub"} == {
        "concept-ledger", "concept-journal", "concept-double-entry", "concept-audit"}
    sem = {(e["source"], e["type"], e["target"]): e["status"] for e in graph["edges"] if e["status"] != "structural"}
    assert sem == {("concept-journal", "leads-to", "concept-ledger"): "stated",
                   ("concept-double-entry", "uses", "concept-ledger"): "inferred"}
    cv = json.loads((env.folder / canvas[0]).read_text(encoding="utf-8"))
    texts = [n for n in cv["nodes"] if n["type"] == "text" and n["id"] != "chapter"]
    assert len(texts) == 4
    for n in texts:
        assert SOURCE_LINE_RE.fullmatch(n["text"].splitlines()[-1]), n["text"]
    saved = json.loads((env.folder / "run.json").read_text(encoding="utf-8"))
    assert saved["argv"] == ["map", "book.pdf"] and saved["status"] == "ok"
    assert saved["counts"]["candidates"] == 4 and saved["counts"]["kept"] == 4
    assert saved["counts"]["edges"]["discarded"] == 1          # "Nowhere" is not a node of its reply
    assert saved["counts"]["edges"]["duplicates"] == 1         # link repeated the quoted leads-to edge
    assert saved["totals"]["calls"] == sum(fake.calls.values()) and saved["totals"]["retries"] == 0
    assert saved["totals"]["peak_rss_mb"] > 0
    assert saved["config"]["roles"]["verify"]["model"] == "verify-m"


def test_a_bad_link_edge_is_dropped_not_fatal(env):
    bad = {"edges": LINK["edges"] + [{"source": "concept-ledger", "target": "concept-audit", "type": "supports",
                                      "quote": None},                  # a type the template lacks
                                     {"source": "concept-ledgr", "target": "concept-audit", "type": "uses",
                                      "quote": None}]}                 # a garbled id
    fake = Fake(link=bad)
    rec = run(env, fake)
    assert rec["status"] == "ok", rec["reason"]
    assert fake.calls["link"] == 1                                    # no re-ask
    assert rec["counts"]["edges"]["discarded"] == 1 + 2               # "Nowhere" from extract + the two


def test_pages_are_derived_not_model_written(env):
    one = {"nodes": [node("Double Entry", "Double entry means every debit has a credit")], "edges": []}
    rec = run(env, Fake(extract={"journal is the book": one}), pages=(2, 3))
    assert rec["status"] == "ok", rec["reason"]
    graph = json.loads((env.tmp / "staging" / "test-book" / "pages-2-3" / "map.json").read_text(encoding="utf-8"))
    n = next(n for n in graph["nodes"] if n["id"] == "concept-double-entry")
    assert n["source_pages"] == [[3, 3]] and n["quotes"][0]["pages"] == [3]


def test_gate_fails_below_ratio(env):
    para = {"ledger records every": {"nodes": [node("Ledger", "A ledger records every transaction in order"),
                                               node("Journal", "The journal is where entries are first written")]},
            "Double entry means": {"nodes": [node("Double Entry", "Double entry means every debit has a credit"),
                                             node("Audit", "Audits look over the books once a year")]}}
    fake = Fake(extract=para)
    rec = run(env, fake)
    assert rec["status"] == "failed" and "grounded" in rec["reason"]
    assert rec["gate"]["grounded"] == 2 and rec["gate"]["candidates"] == 4 and not rec["gate"]["passed"]
    assert {r["reason"] for r in rec["rejects"]} == {"not_found"}
    assert fake.calls["link"] == 0 and fake.calls["verify"] == 1    # R6: the gate stops the run before link
    out = files(env.folder)
    assert len(out) == 1 and out[0].startswith("runs/")
    resumed = run(env, Fake(extract=para, fail={"extract", "verify"}))     # from checkpoints
    assert resumed["status"] == "failed"
    assert {r["served"] for r in resumed["rejects"]} == {"extract-m"}      # kept beside the checkpoint


def test_unit_failure_fails_run(env):
    rec = run(env, Fake(fail={"extract"}))
    assert rec["status"] == "failed" and "sec-1-u1" in rec["reason"]
    assert [f for f in files(env.folder) if not f.startswith("runs/")] == []


def test_rerun_resumes_from_checkpoints(env):
    assert run(env, Fake())["status"] == "ok"
    again = Fake(fail={"extract", "link", "verify"})
    rec = run(env, again)
    assert rec["status"] == "ok", rec["reason"]
    assert sum(again.calls.values()) == 0
    assert len([f for f in files(env.folder) if f.startswith("runs/")]) == 2


def test_budget_failure_keeps_checkpoints(env):
    rec = run(env, Fake(), k=dataclasses.replace(env.k, max_llm_calls=2))
    assert rec["status"] == "failed" and rec["reason"] == "max_llm_calls"
    work = [f for f in files(env.folder, work=True) if f.startswith(".work/")]
    assert sorted(Path(f).name for f in work) == ["extract-sec-1-u1.json", "extract-sec-2-u1.json"]
    assert "map.json" not in files(env.folder)
    resumed = Fake()
    assert run(env, resumed)["status"] == "ok" and resumed.calls["extract"] == 0


def test_dry_run_writes_nothing(env):
    lines = []
    fake = Fake()
    rec = run_map(env.k, env.pdf, pages=(1, 4), dry_run=True, client_factory=fake, progress=lines.append)
    assert rec["status"] == "dry_run" and rec["template"] == "concept"
    assert [u["unit"] for u in rec["plan"]] == ["sec-1-u1", "sec-2-u1"]
    assert fake.calls["extract"] == 0 and not (env.tmp / "staging").exists()
    assert any("sec-2-u1" in s for s in lines)


def test_export_then_import_round_trip(env):
    prompts = env.tmp / "prompts"
    rec = run(env, Fake(), export_prompts=prompts)
    assert rec["status"] == "exported" and not (env.tmp / "staging").exists()
    names = sorted(p.name for p in prompts.iterdir() if p.name != "export.json")
    assert names == ["extract-sec-1-u1.prompt.json", "extract-sec-2-u1.prompt.json"]
    p = json.loads((prompts / names[0]).read_text(encoding="utf-8"))
    assert set(p) == {"system", "user", "unit", "section"} and p["unit"] == "sec-1-u1"
    assert "ledger records every" in p["user"]

    replies = env.tmp / "replies"
    replies.mkdir()
    (replies / "extract-sec-1-u1.json").write_text(json.dumps(SEC1), encoding="utf-8")
    (replies / "extract-sec-2-u1.json").write_text(json.dumps(SEC2), encoding="utf-8")
    fake = Fake(fail={"extract"})
    rec = run(env, fake, import_extract=replies)
    assert rec["status"] == "ok", rec["reason"]
    imported = [a for a in rec["attempts"] if a["provider"] == "import"]
    assert [a["unit"] for a in imported] == ["sec-1-u1", "sec-2-u1"]
    assert all(a["requested"] is None and a["served"] is None and len(a["sha256"]) == 64 for a in imported)
    assert fake.calls["extract"] == 0 and rec["totals"]["calls"] == sum(fake.calls.values())

    (replies / "extract-sec-2-u1.json").unlink()
    rec = run(env, Fake(), import_extract=replies)
    assert rec["status"] == "failed" and "extract-sec-2-u1.json" in rec["reason"]


def _export_with_replies(env, fake):
    """Export into DIR, then write the replies into the same DIR (the documented flow)."""
    d = env.tmp / "exchange"
    rec = run(env, fake, template=None, export_prompts=d)
    assert rec["status"] == "exported", rec["reason"]
    (d / "extract-sec-1-u1.json").write_text(json.dumps(SEC1), encoding="utf-8")
    (d / "extract-sec-2-u1.json").write_text(json.dumps(SEC2), encoding="utf-8")
    return d


def test_import_uses_the_exported_template(env):
    """I1: import without --template takes export.json's template and never classifies again,
    even when the classifier would now answer differently."""
    d = _export_with_replies(env, Fake())
    m = json.loads((d / "export.json").read_text(encoding="utf-8"))
    assert m["template"] == "concept" and m["classifier_reason"] == "explains bookkeeping terms"
    assert m["units"] == ["sec-1-u1", "sec-2-u1"] and m["pages"] == [1, 4] and len(m["pdf_sha256"]) == 64
    assert set(m["prompt_versions"]) >= {"map_classify"}
    second = Fake(template="procedure", fail={"extract"})
    rec = run(env, second, template=None, import_extract=d)
    assert rec["status"] == "ok", rec["reason"]
    assert rec["template"] == "concept" and rec["classifier_reason"] == "explains bookkeeping terms"
    assert second.calls["classify"] == 0


def test_import_refuses_a_mismatched_export(env):
    d = _export_with_replies(env, Fake())
    rec = run(env, Fake(), template="procedure", import_extract=d)
    assert rec["status"] == "failed" and "template 'concept'" in rec["reason"]
    m = json.loads((d / "export.json").read_text(encoding="utf-8"))
    for field, bad in (("pdf_sha256", "0" * 64), ("pages", [1, 3]), ("units", ["sec-1-u1"])):
        (d / "export.json").write_text(json.dumps({**m, field: bad}), encoding="utf-8")
        rec = run(env, Fake(), template=None, import_extract=d)
        assert rec["status"] == "failed" and rec["reason"].startswith(f"{d / 'export.json'}: {field} "), rec["reason"]
    (d / "export.json").unlink()
    fake = Fake()
    rec = run(env, fake, template=None, import_extract=d)
    assert rec["status"] == "failed" and "pass --template or re-export" in rec["reason"]
    assert fake.calls["classify"] == 0


def test_no_ingest_imports():
    bad = re.compile(r"^\s*(from|import)\s+(src\.(index_ops|embeddings|retrieval|pipeline)\b|main\b|manage_api\b"
                     r"|serve_api\b|src\.ingestion\b(?!\.(canvas_edit|canvas_loader)\b)"
                     r"|src\s+import\s+\(?[\s\w,]*\b(index_ops|embeddings|retrieval|pipeline)\b)", re.M)
    for line in ("from src.ingestion import pdf_loader", "from src import retrieval", "import src.index_ops",
                 "from src import (\n    pipeline,\n)", "from src.ingestion.pdf_loader import load",
                 "from src.retrieval.route import x", "import main"):
        assert bad.search(line), line
    for line in ("from src.ingestion.canvas_edit import merge_canvas",
                 "from src.ingestion.canvas_loader import SOURCE_LINE_RE", "from src.ocr_studio.pages import atomic_write"):
        assert not bad.search(line), line
    for f in (Path(__file__).resolve().parents[1] / "src" / "knowledge_map").glob("*.py"):
        assert not bad.search(f.read_text(encoding="utf-8")), f.name


def test_staging_inside_vault_refused(env):
    k = dataclasses.replace(env.k, staging_dir=env.vault / "maps")
    before = snapshot(env.vault)
    with pytest.raises(ValueError, match="inside the vault"):
        run(env, Fake(), k=k)
    assert snapshot(env.vault) == before and not (env.vault / "maps").exists()


def test_rerun_skips_classify(env):
    assert run(env, Fake(), template=None)["status"] == "ok"
    again = Fake(fail={"classify", "extract", "link", "verify"})
    rec = run(env, again, template=None)
    assert rec["status"] == "ok" and rec["template"] == "concept", rec["reason"]
    assert sum(again.calls.values()) == 0


def test_verify_batch_change_reverifies(env):
    assert run(env, Fake())["status"] == "ok"
    fake = Fake()
    rec = run(env, fake, k=dataclasses.replace(env.k, verify_batch=1))
    assert rec["status"] == "ok", rec["reason"]
    assert fake.calls["verify"] == 5                    # 4 single-node batches + 1 edge batch
    names = [Path(f).name for f in files(env.folder, work=True) if "/verify-" in f]
    assert names and all(re.fullmatch(r"verify-(nodes|edges)-[0-9a-f]{12}\.json", n) for n in names)


def test_symbol_only_title_is_rejected(env):
    t = load_templates()["concept"]
    for title in ("???", "___", "_ _"):                  # slug splits on [\W_]+, so "_" is no letter either
        ext = Extraction.model_validate({"nodes": [node(title, "A ledger records every transaction in order")]})
        assert any("no letters or digits" in e for e in check_extraction(ext, t)), title
    ok = Extraction.model_validate({"nodes": [node("B_2", "A ledger records every transaction in order")]})
    assert check_extraction(ok, t) == []
    bad = {"ledger records every": {"nodes": [node("???", "A ledger records every transaction in order")]},
           "Double entry means": SEC2}
    rec = run(env, Fake(extract=bad))                   # the model reply is re-asked, then the unit fails
    assert rec["status"] == "failed" and "sec-1-u1" in rec["reason"]
    assert not any(f.endswith("extract-sec-1-u1.json") for f in files(env.folder, work=True))
    replies = env.tmp / "replies"
    replies.mkdir()
    (replies / "extract-sec-1-u1.json").write_text(json.dumps(bad["ledger records every"]), encoding="utf-8")
    (replies / "extract-sec-2-u1.json").write_text(json.dumps(SEC2), encoding="utf-8")
    rec = run(env, Fake(), import_extract=replies)
    assert rec["status"] == "failed" and "extract-sec-1-u1.json" in rec["reason"]


def test_crash_leaves_run_record(env, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("layout bug")
    monkeypatch.setattr(run_module, "render_canvas", boom)
    with pytest.raises(RuntimeError, match="layout bug"):
        run(env, Fake())
    out = files(env.folder)
    assert len(out) == 1 and out[0].startswith("runs/")
    saved = json.loads((env.folder / out[0]).read_text(encoding="utf-8"))
    assert saved["status"] == "failed" and saved["reason"].startswith("crash: RuntimeError")


def test_export_dir_inside_vault_refused(env):
    before = snapshot(env.vault)
    with pytest.raises(ValueError, match="inside the vault"):
        run(env, Fake(), export_prompts=env.vault / "prompts")
    assert snapshot(env.vault) == before and not (env.vault / "prompts").exists()


def test_edge_downgrades_to_inferred(env):
    sec1 = {"nodes": SEC1["nodes"],
            "edges": [{"source": "Journal", "target": "Ledger", "type": "leads-to",        # located, unsupported
                       "quote": "Entries move from the journal to the ledger"},
                      {"source": "Ledger", "target": "Journal", "type": "contrasts-with",  # not in the text
                       "quote": "The ledger and the journal differ in many ways"}]}
    unsupported = {edge_id("concept-journal", "concept-ledger", "leads-to")}
    rec = run(env, Fake(extract={"ledger records every": sec1, "Double entry means": SEC2}, link={"edges": []},
                        unsupported=unsupported))
    assert rec["status"] == "ok", rec["reason"]
    assert rec["counts"]["edges"]["downgraded"] == 2 and rec["counts"]["edges"]["stated"] == 0
    graph = json.loads((env.folder / "map.json").read_text(encoding="utf-8"))
    sem = [e for e in graph["edges"] if e["status"] != "structural"]
    assert len(sem) == 2 and all(e["status"] == "inferred" and e["quote"] is None for e in sem)


def test_too_wide_node_rejected(env):
    wide = {"nodes": [{"title": "Ledger", "kind": "concept", "summary": "About ledgers.", "importance": 3,
                       "quotes": ["A ledger records every transaction in order",
                                  "Entries move from the journal to the ledger"]}]}
    rec = run(env, Fake(extract={"ledger records every": wide, "Double entry means": SEC2}),
              k=dataclasses.replace(env.k, max_span_pages=1))
    r = next(r for r in rec["rejects"] if r["id"] == "concept-ledger")
    assert r["reason"] == "too_wide" and sorted(q["pages"][0] for q in r["quotes"]) == [1, 2]


def test_overclaiming_node_is_trimmed_and_kept(env):
    fake = Fake(unsupported={"concept-audit"}, trim={"concept-audit": "An audit reviews the books at year end."})
    rec = run(env, fake)
    assert rec["status"] == "ok", rec["reason"]
    assert rec["gate"]["grounded"] == 4 and rec["counts"]["revised"] == 1 and rec["rejects"] == []
    assert fake.calls["trim"] == 1 and fake.calls["verify"] == 3     # verify, trim, edges
    assert fake.calls["reverify"] == 1                                 # the trim is checked by another model
    graph = json.loads((env.folder / "map.json").read_text(encoding="utf-8"))
    audit = next(n for n in graph["nodes"] if n["id"] == "concept-audit")
    assert audit["summary"] == "An audit reviews the books at year end."
    assert audit["summary_revised_from"] == "About Audit."
    assert "summary_revised_from" not in next(n for n in graph["nodes"] if n["id"] == "concept-ledger")
    again = Fake(fail={"extract", "link", "verify", "reverify"})      # trim and re-verify are checkpointed
    assert run(env, again)["counts"]["revised"] == 1 and sum(again.calls.values()) == 0


def test_trim_null_drops_without_reverify(env):
    fake = Fake(unsupported={"concept-audit"}, trim={"concept-audit": None})
    rec = run(env, fake, k=dataclasses.replace(env.k, min_grounded_ratio=0.5))
    assert rec["status"] == "ok", rec["reason"]
    assert fake.calls["trim"] == 1 and fake.calls["verify"] == 3     # verify, trim, edges
    assert fake.calls["reverify"] == 0                                 # nothing to re-verify
    r = next(r for r in rec["rejects"] if r["id"] == "concept-audit")
    assert r["reason"] == "unsupported" and r["summary"] == "About Audit." and "no supported summary" in r["detail"]
    assert rec["counts"]["revised"] == 0 and rec["gate"]["grounded"] == 3


def test_trimmed_summary_still_unsupported_drops(env):
    fake = Fake(unsupported={"concept-audit"}, still_unsupported={"concept-audit"},
                trim={"concept-audit": "An audit reviews the books at year end."})
    rec = run(env, fake, k=dataclasses.replace(env.k, min_grounded_ratio=0.5))
    r = next(r for r in rec["rejects"] if r["id"] == "concept-audit")
    assert r["reason"] == "unsupported" and r["summary"] == "About Audit."
    assert r["detail"].count("checked") == 2 and "after trim" in r["detail"]
    assert rec["counts"]["revised"] == 0


def with_headings(env, heads=None):
    """Rewrite the book with a large heading on pages 1 and 3; returns a KMConfig whose density
    leaves both headings short of nodes after the first extraction. The default headings are
    named by the first extraction's nodes (Ledger, Double Entry)."""
    heads = heads or {1: "Ledgers", 3: "Double Entry"}
    doc = fitz.open()
    for i, text in enumerate(PAGES, start=1):
        page = doc.new_page()
        y = 72
        if i in (1, 3):
            page.insert_text((72, y), heads[i], fontsize=20)
            y += 28
        for line in text.split("\n"):
            page.insert_text((72, y), line)
            y += 14
    doc.set_metadata({"title": "Test Book"})
    doc.save(env.pdf)
    doc.close()
    return dataclasses.replace(env.k, min_section_chars=20, chars_per_node=60)


GAPFILL = {"ledger records every": {"nodes": [node("Branch Clerk", "It is kept by the branch clerk each day"),
                                              node("Ledger", "A ledger records every transaction in order")],
                                    "edges": []},
           "Double entry means": {"nodes": [node("Trial Balance", "The trial balance checks that totals agree")],
                                  "edges": []}}


def test_heading_short_of_nodes_is_gap_filled(env):
    k = with_headings(env)
    fake = Fake(gapfill=GAPFILL)
    rec = run(env, fake, k=k)
    assert rec["status"] == "ok", rec["reason"]
    assert fake.calls["gapfill"] == 2                       # one call per short heading
    assert "- Ledgers (already in the map: Ledger; Journal)" in fake.gapfill_users[0]
    graph = json.loads((env.folder / "map.json").read_text(encoding="utf-8"))
    ids = {n["id"] for n in graph["nodes"]}
    assert {"concept-branch-clerk", "concept-trial-balance"} <= ids
    assert rec["counts"]["candidates"] == 6                 # the repeated "Ledger" is not a new candidate
    cov = rec["coverage"]
    assert cov["headings"] == 2 and cov["short_before_gapfill"] == 2 and cov["unplaced"] == []
    assert cov["short"] == ["Double Entry (p. 3): 3 of 4 nodes"]    # it gained one; Ledgers (needs 3) is full
    assert {a["unit"] for a in rec["attempts"] if a["unit"].startswith("gapfill-")} == {
        "gapfill-ledgers-p1", "gapfill-double-entry-p3"}
    again = Fake(fail={"extract", "link", "verify", "reverify"})
    assert run(env, again, k=k)["status"] == "ok"           # gap-fill resumes from its checkpoints


def test_cap_rises_to_what_the_headings_need(env):
    k = dataclasses.replace(with_headings(env), node_cap=4)        # the headings need 3 + 4 nodes
    rec = run(env, Fake(gapfill=GAPFILL), k=k)
    assert rec["status"] == "ok", rec["reason"]
    assert rec["coverage"]["node_cap"] == 7
    assert rec["counts"]["kept"] == 6 and "cap" not in rec["counts"]["dropped"]   # gap-fill nodes stay


def test_heading_whose_idea_is_not_a_node_is_gap_filled(env):
    """Enough nodes for the length, but none named after "Bookkeeping": gap-fill asks for its
    own idea, and the cap makes room for that one node."""
    k = dataclasses.replace(with_headings(env, {1: "Bookkeeping", 3: "Double Entry"}), chars_per_node=1000, node_cap=2)
    fake = Fake(gapfill={"ledger records every": {"nodes": [
        node("Bookkeeping", "It is kept by the branch clerk each day")], "edges": []}})
    rec = run(env, fake, k=k)
    assert rec["status"] == "ok", rec["reason"]
    assert fake.calls["gapfill"] == 1                       # Double Entry is named by its node
    assert "Bookkeeping (already in the map: Ledger; Journal; still missing: a node for this heading's own idea)" \
        in fake.gapfill_users[0]
    assert rec["coverage"]["node_cap"] == 3                 # 1 + 1 for the length, + 1 for the missing idea
    # 5 candidates: the cap keeps section 1 (Ledger, Journal, Bookkeeping) and cuts Double Entry's
    assert rec["coverage"]["short"] == ["Double Entry (p. 3): 0 of 1 nodes, none named after the heading"]
    graph = json.loads((env.folder / "map.json").read_text(encoding="utf-8"))
    assert "concept-bookkeeping" in {n["id"] for n in graph["nodes"]}


def test_coverage_is_reported_on_the_nodes_the_cap_keeps(env):
    k = dataclasses.replace(with_headings(env), node_cap=2, chars_per_node=1000)   # needs 1 + 1: the floor holds
    rec = run(env, Fake(gapfill=GAPFILL), k=k)
    assert rec["status"] == "ok", rec["reason"]
    assert rec["coverage"]["node_cap"] == 2
    assert rec["counts"]["kept"] == 2 and rec["counts"]["dropped"] == {"cap": 2}
    assert rec["coverage"]["short"] == ["Double Entry (p. 3): 0 of 1 nodes, none named after the heading"]   # both covered before the cap


def test_imported_run_reports_gaps_without_gap_fill(env):
    k = with_headings(env)
    replies = env.tmp / "replies"
    replies.mkdir()
    (replies / "extract-sec-1-u1.json").write_text(json.dumps(SEC1), encoding="utf-8")
    (replies / "extract-sec-2-u1.json").write_text(json.dumps(SEC2), encoding="utf-8")
    fake = Fake(gapfill=GAPFILL)
    rec = run(env, fake, k=k, import_extract=replies)
    assert rec["status"] == "ok", rec["reason"]
    assert fake.calls["gapfill"] == 0 and fake.calls["extract"] == 0
    assert rec["coverage"]["short_before_gapfill"] == 2 and len(rec["coverage"]["short"]) == 2


def test_chars_per_node_zero_turns_gap_fill_off(env):
    k = dataclasses.replace(with_headings(env), chars_per_node=0)
    fake = Fake(gapfill=GAPFILL)
    rec = run(env, fake, k=k)
    assert rec["status"] == "ok" and fake.calls["gapfill"] == 0 and rec["coverage"]["short"] == []
