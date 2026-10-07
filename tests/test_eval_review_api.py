"""Eval review API (manage_api): progress, the filtered list, detail + review cache,
and the verify / reject / edit writes.

Every test runs against a temp sets dir (manage_api.EVAL_SETS_DIR is monkeypatched);
the real eval/sets is never read or written.
"""
import datetime as dt
import json
import threading
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from eval.bench.questions import (
    SUITES, load_sets, question_from_dict, question_to_dict, save_suite,
)

ROOT = Path(__file__).resolve().parents[1]


def _d(qid="lex-0001", suite="lexical", status="draft", **over):
    d = {"id": qid, "question": f"Question text for {qid}?", "suite": suite, "tier": "T1",
         "split": "dev", "answerable": True, "gold": [{"file": "a.md", "heading": "BM25"}],
         "nuggets": ["term-frequency saturation"], "expect_course": None,
         "provenance": {"author": "draft", "drafted_by": "claude-sonnet-5.5",
                        "seed_chunks": ["x"], "status": status, "reviewed_at": None},
         "notes": ""}
    d.update(over)
    return d


def _write(sets, name, *records):
    save_suite(sets / f"{name}.yaml", [question_from_dict(r, "test") for r in records])


@pytest.fixture
def sets(tmp_path, monkeypatch, management_module):
    d = tmp_path / "sets"
    d.mkdir()
    monkeypatch.setattr(management_module, "EVAL_SETS_DIR", d)
    return d


@pytest.fixture
def client(sets, management_module):
    return TestClient(management_module.app)      # no `with`: no startup hooks to run


# ---- reading ------------------------------------------------------------------

def test_progress_counts_per_suite_and_totals(sets, client):
    _write(sets, "lexical", _d("lex-0001"), _d("lex-0002"),
           _d("lex-0003", status="verified"), _d("lex-0004", status="rejected"))
    _write(sets, "code", _d("code-0001", suite="code", status="edited"))
    p = client.get("/api/eval/progress").json()
    assert set(p) == {*SUITES, "totals"}
    assert p["lexical"] == {"quota": 40, "total": 4, "draft": 2, "verified": 1,
                            "edited": 0, "rejected": 1}
    assert p["code"] == {"quota": 40, "total": 1, "draft": 0, "verified": 0,
                         "edited": 1, "rejected": 0}
    assert p["unanswerable"] == {"quota": 45, "total": 0, "draft": 0, "verified": 0,
                                 "edited": 0, "rejected": 0}       # empty suites still listed
    assert p["totals"] == {"quota": 400, "total": 5, "draft": 2, "verified": 1,
                           "edited": 1, "rejected": 1}


def test_a_sets_dir_that_does_not_exist_yet_is_an_empty_queue(
        tmp_path, monkeypatch, management_module):
    monkeypatch.setattr(management_module, "EVAL_SETS_DIR", tmp_path / "never_made")
    c = TestClient(management_module.app)
    assert c.get("/api/eval/questions").json() == []
    assert c.get("/api/eval/progress").json()["totals"]["total"] == 0


def test_list_rows_filters_and_validator_counts(sets, client):
    _write(sets, "lexical", _d("lex-0001", split="dev"),
           _d("lex-0002", split="test", status="verified"),
           _d("lex-0003", split="dev", status="rejected"))
    _write(sets, "paraphrase", _d("para-0001", suite="paraphrase", split="test", tier="T2"))
    (sets / ".review_cache.json").write_text(json.dumps({"lex-0001": {"gold_texts": [], "findings": [
        {"level": "error", "code": "locator-empty", "message": "m"},
        {"level": "warn", "code": "copied-phrasing", "message": "m"},
        {"level": "warn", "code": "near-duplicate", "message": "m"}]}}), encoding="utf-8")

    rows = client.get("/api/eval/questions").json()
    assert [r["id"] for r in rows] == ["lex-0001", "lex-0002", "lex-0003", "para-0001"]  # rejected stay
    assert rows[0] == {"id": "lex-0001", "suite": "lexical", "tier": "T1", "split": "dev",
                       "status": "draft", "author": "draft",
                       "question": "Question text for lex-0001?", "n_errors": 1, "n_warnings": 2}
    assert (rows[1]["n_errors"], rows[1]["n_warnings"]) == (0, 0)      # no cache entry

    def ids(**params):
        return [r["id"] for r in client.get("/api/eval/questions", params=params).json()]

    assert ids(suite="paraphrase") == ["para-0001"]
    assert ids(status="verified") == ["lex-0002"]
    assert ids(split="test") == ["lex-0002", "para-0001"]
    assert ids(suite="lexical", split="dev", status="rejected") == ["lex-0003"]
    # the console sends an unset dropdown as an empty value: that means "no filter"
    assert ids(suite="", status="", split="") == ["lex-0001", "lex-0002", "lex-0003", "para-0001"]


def test_detail_joins_the_cache_and_is_empty_without_one(sets, client):
    _write(sets, "lexical", _d("lex-0001"), _d("lex-0002"))
    body = client.get("/api/eval/questions/lex-0001").json()
    assert body["record"] == question_to_dict(question_from_dict(_d("lex-0001"), "t"))
    assert body["gold_texts"] == [] and body["findings"] == []          # no cache file at all

    cache = {"lex-0001": {
        "gold_texts": [{"file": "a.md", "pages": None, "heading": "BM25", "text": "k1 sets tf saturation"}],
        "findings": [{"level": "warn", "code": "nugget-unsupported", "message": "m"}]}}
    (sets / ".review_cache.json").write_text(json.dumps(cache), encoding="utf-8")
    body = client.get("/api/eval/questions/lex-0001").json()
    assert body["gold_texts"] == cache["lex-0001"]["gold_texts"]
    assert body["findings"] == cache["lex-0001"]["findings"]
    other = client.get("/api/eval/questions/lex-0002").json()           # cache exists, misses this id
    assert other["gold_texts"] == [] and other["findings"] == []


def test_a_corrupt_review_cache_raises_instead_of_reading_as_no_findings(sets, client):
    _write(sets, "lexical", _d("lex-0001"))
    (sets / ".review_cache.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        client.get("/api/eval/questions/lex-0001")


def test_a_sets_file_the_loader_rejects_is_a_readable_500_not_an_empty_queue(sets, client):
    _write(sets, "lexical", _d("lex-0001"))
    path = sets / "lexical.yaml"
    bad = yaml.safe_load(path.read_text(encoding="utf-8"))
    bad[0]["tier"] = "T9"                                               # a hand edit with a typo
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    before = path.read_bytes()
    for r in (client.get("/api/eval/progress"), client.get("/api/eval/questions"),
              client.post("/api/eval/questions/lex-0001", json={"action": "verify"})):
        assert r.status_code == 500
        assert "lexical.yaml[0]: lex-0001: tier 'T9'" in r.json()["error"]
    assert path.read_bytes() == before


def test_unknown_question_is_404_on_read_and_write(sets, client):
    _write(sets, "lexical", _d("lex-0001"))
    path = sets / "lexical.yaml"
    before = path.read_bytes()
    assert client.get("/api/eval/questions/lex-9999").status_code == 404
    for body in ({"action": "verify"}, {"action": "edit", "record": _d("lex-9999")}):
        r = client.post("/api/eval/questions/lex-9999", json=body)
        assert r.status_code == 404 and r.json()["ok"] is False
    assert path.read_bytes() == before


# ---- writing ------------------------------------------------------------------

def test_verify_reject_edit_round_trip_on_disk(sets, client):
    _write(sets, "lexical", _d("lex-0001"), _d("lex-0002"), _d("lex-0003"))
    first = dt.date.today()

    r = client.post("/api/eval/questions/lex-0001", json={"action": "verify"})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert r.json()["record"]["provenance"]["status"] == "verified"
    assert client.post("/api/eval/questions/lex-0002", json={"action": "reject"}).status_code == 200
    edited = _d("lex-0003", question="Reworded question?", tier="T3",
                nuggets=["one", "two"], notes="gold fixed")
    r = client.post("/api/eval/questions/lex-0003", json={"action": "edit", "record": edited})
    assert r.status_code == 200 and r.json()["record"]["question"] == "Reworded question?"
    last = dt.date.today()

    on_disk = {q.id: q for q in load_sets(sets, include_rejected=True)}
    assert list(on_disk) == ["lex-0001", "lex-0002", "lex-0003"]        # rejected kept, order kept
    for qid, status in (("lex-0001", "verified"), ("lex-0002", "rejected"), ("lex-0003", "edited")):
        prov = on_disk[qid].provenance
        assert prov["status"] == status
        assert first <= dt.date.fromisoformat(prov["reviewed_at"]) <= last
        assert prov["seed_chunks"] == ["x"] and prov["author"] == "draft"   # the rest survives
    assert on_disk["lex-0001"].question == "Question text for lex-0001?"    # verify changed nothing else
    c = on_disk["lex-0003"]
    assert (c.question, c.tier, c.nuggets, c.notes) == ("Reworded question?", "T3",
                                                        ["one", "two"], "gold fixed")
    assert not list(sets.glob("*.tmp"))                                  # no temp file left behind
    # what the detail endpoint serves is what is on disk now
    served = client.get("/api/eval/questions/lex-0003").json()["record"]
    assert served == question_to_dict(c)


@pytest.mark.parametrize("over, fragment", [
    ({"tier": "T9"}, "tier 'T9'"),
    ({"nuggets": []}, "at least one nugget"),
    ({"question": "  "}, "question must be a non-empty string"),
])
def test_an_invalid_edit_is_a_400_with_the_schema_text_and_leaves_the_file_byte_identical(
        sets, client, over, fragment):
    _write(sets, "lexical", _d("lex-0001"), _d("lex-0002"))
    path = sets / "lexical.yaml"
    before = path.read_bytes()
    r = client.post("/api/eval/questions/lex-0001",
                    json={"action": "edit", "record": _d("lex-0001", **over)})
    assert r.status_code == 400
    assert r.json()["ok"] is False and fragment in r.json()["error"]
    assert path.read_bytes() == before
    assert not list(sets.glob("*.tmp"))


@pytest.mark.parametrize("over", [{"id": "lex-0002"}, {"id": "lex-0099"},
                                  {"suite": "paraphrase"}, {"split": "test"}])
def test_an_edit_cannot_change_id_suite_or_split(sets, client, over):
    # The split is locked too: moving a question between dev and test leaks the
    # sealed test set.
    _write(sets, "lexical", _d("lex-0001"), _d("lex-0002"))
    path = sets / "lexical.yaml"
    before = path.read_bytes()
    r = client.post("/api/eval/questions/lex-0001",
                    json={"action": "edit", "record": _d("lex-0001", **over)})
    assert r.status_code == 400 and "cannot change id, suite or split" in r.json()["error"]
    assert path.read_bytes() == before


def test_an_edit_that_omits_the_split_keeps_the_stored_one(sets, client):
    _write(sets, "lexical", _d("lex-0001"))
    r = client.post("/api/eval/questions/lex-0001", json={
        "action": "edit", "record": _d("lex-0001", split=None, question="Reworded?")})
    assert r.status_code == 200 and r.json()["record"]["split"] == "dev"


def test_the_sets_are_parsed_once_until_a_file_changes(sets, client, management_module,
                                                       monkeypatch):
    _write(sets, "lexical", _d("lex-0001"))
    calls = []
    real = management_module.load_sets
    monkeypatch.setattr(management_module, "load_sets",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    client.get("/api/eval/progress")
    client.get("/api/eval/questions")
    assert len(calls) == 1
    client.post("/api/eval/questions/lex-0001", json={"action": "verify"})
    client.get("/api/eval/progress")
    assert len(calls) == 2          # the write changed the file; the next read re-parses


@pytest.mark.parametrize("body", [
    {"action": "approve"},
    {"action": "verify", "record": _d()},       # a record the server would silently drop
    {"action": "reject", "record": _d()},
    {"action": "edit"},
])
def test_a_malformed_review_request_is_a_400_and_writes_nothing(sets, client, body):
    _write(sets, "lexical", _d("lex-0001"))
    path = sets / "lexical.yaml"
    before = path.read_bytes()
    r = client.post("/api/eval/questions/lex-0001", json=body)
    assert r.status_code == 400 and r.json()["ok"] is False
    assert path.read_bytes() == before


def test_a_write_that_dies_midway_leaves_the_file_whole_and_no_temp_behind(
        sets, client, management_module, monkeypatch):
    _write(sets, "lexical", _d("lex-0001"), _d("lex-0002"))
    path = sets / "lexical.yaml"
    before = path.read_bytes()
    real_write = Path.write_text

    def dies_midway(self, text, *args, **kwargs):    # the real save_suite, its temp write cut short
        if self.suffix == ".tmp":
            real_write(self, text[:20], *args, **kwargs)
            raise OSError("disk full")
        return real_write(self, text, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", dies_midway)
    with pytest.raises(OSError, match="disk full"):        # raised, not swallowed
        client.post("/api/eval/questions/lex-0001", json={"action": "verify"})
    assert path.read_bytes() == before                      # the real file never saw the partial text
    assert not list(sets.glob("*.tmp"))


def test_two_threads_posting_to_one_suite_both_land(sets, management_module, monkeypatch):
    _write(sets, "lexical", _d("lex-0001"), _d("lex-0002"))
    real_load = management_module.load_sets

    def slow_load(*args, **kwargs):
        out = real_load(*args, **kwargs)
        time.sleep(0.25)       # widen load -> write: with no lock both threads read the same
        return out             # old file and the second write would erase the first
    monkeypatch.setattr(management_module, "load_sets", slow_load)

    barrier = threading.Barrier(2)
    responses = {}

    def post(qid, action):
        c = TestClient(management_module.app)                           # one client per thread
        barrier.wait()
        responses[qid] = c.post(f"/api/eval/questions/{qid}", json={"action": action})

    threads = [threading.Thread(target=post, args=a)
               for a in (("lex-0001", "verify"), ("lex-0002", "reject"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads)
    assert {qid: r.status_code for qid, r in responses.items()} == {"lex-0001": 200, "lex-0002": 200}
    status = {q.id: q.provenance["status"] for q in load_sets(sets, include_rejected=True)}
    assert status == {"lex-0001": "verified", "lex-0002": "rejected"}


def test_a_question_is_written_back_to_the_file_that_holds_it(sets, client):
    # A suite spread over two files and a file that mixes suites: neither may be
    # rewritten as <suite>.yaml (that would duplicate ids and break every load).
    _write(sets, "a_notes", _d("para-0001", suite="paraphrase", notes="same idea as lex-0003"))
    _write(sets, "lexical", _d("lex-0001"), _d("lex-0002"))
    _write(sets, "lexical_more", _d("lex-0003"))
    _write(sets, "misc", _d("code-0001", suite="code"), _d("lex-0004"))
    bytes_of = lambda name: (sets / f"{name}.yaml").read_bytes()
    a_notes, lexical = bytes_of("a_notes"), bytes_of("lexical")

    assert client.post("/api/eval/questions/lex-0003", json={"action": "verify"}).status_code == 200
    assert (bytes_of("a_notes"), bytes_of("lexical")) == (a_notes, lexical)  # a mention is not a hit
    misc = bytes_of("misc")
    assert client.post("/api/eval/questions/lex-0004", json={"action": "reject"}).status_code == 200
    assert bytes_of("lexical") == lexical and bytes_of("misc") != misc

    ids_in = {p.stem: [d["id"] for d in yaml.safe_load(p.read_text(encoding="utf-8"))]
              for p in sets.glob("*.yaml")}
    assert ids_in == {"a_notes": ["para-0001"], "lexical": ["lex-0001", "lex-0002"],
                      "lexical_more": ["lex-0003"], "misc": ["code-0001", "lex-0004"]}
    status = {q.id: q.provenance["status"] for q in load_sets(sets, include_rejected=True)}
    assert status["lex-0003"] == "verified" and status["lex-0004"] == "rejected"


# ---- contract -----------------------------------------------------------------

def test_review_endpoints_are_in_the_schema_with_their_tiers(management_module):
    endpoints = management_module.api_schema()["endpoints"]
    for key in ("GET /api/eval/progress", "GET /api/eval/questions",
                "GET /api/eval/questions/{qid}"):
        assert endpoints[key]["permission"] == "read"
    assert endpoints["POST /api/eval/questions/{qid}"]["permission"] == "mutating"


def test_the_schema_says_an_edit_cannot_change_id_suite_or_split(management_module):
    # The endpoint refuses all three (test_an_edit_cannot_change_id_suite_or_split);
    # the contract an agent reads has to say the same.
    purpose = management_module.api_schema()["endpoints"]["POST /api/eval/questions/{qid}"]["purpose"]
    assert "id, suite or split" in purpose


@pytest.mark.parametrize("name", ["config.yaml", "config.example.yaml"])
def test_sets_dir_is_a_config_key_in_both_configs(name):
    from src.utils.config_loader import load_config
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"no {name} in this checkout")
    assert load_config(path).get("eval.sets_dir") == "eval/sets"
