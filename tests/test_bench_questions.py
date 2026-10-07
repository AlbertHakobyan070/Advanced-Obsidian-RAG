import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.bench.questions import (
    SUITES, GoldSource, Question, SchemaError, assign_splits, load_sets,
    question_from_dict, question_to_dict, save_suite, sets_lock,
)


def _d(**over):
    d = {"id": "lex-0001", "question": "What does BM25's k1 control?", "suite": "lexical",
         "tier": "T1", "split": None, "answerable": True,
         "gold": [{"file": "a.md", "heading": "BM25"}], "nuggets": ["term-frequency saturation"],
         "expect_course": None,
         "provenance": {"author": "draft", "drafted_by": "claude-sonnet-5.5",
                        "seed_chunks": ["x"], "status": "draft", "reviewed_at": None},
         "notes": ""}
    d.update(over)
    return d


def test_roundtrip_through_dict():
    q = question_from_dict(_d(gold=[{"file": "b.pdf", "pages": [3, 4]}]), "t")
    assert q.gold == [GoldSource(file="b.pdf", pages=(3, 4))]
    assert question_from_dict(question_to_dict(q), "t") == q


@pytest.mark.parametrize("over", [
    {"id": "LEX-1"}, {"suite": "nope"}, {"tier": "T9"}, {"split": "train"},
    {"answerable": True, "gold": []}, {"answerable": True, "nuggets": []},
    {"answerable": False},                                  # unanswerable must have no gold
    {"gold": [{"file": "b.pdf", "pages": [9, 3]}]},
    {"provenance": {"author": "someone", "status": "draft"}},
])
def test_schema_rejects(over):
    with pytest.raises(SchemaError):
        question_from_dict(_d(**over), "t")


def test_single_page_shorthand():
    q = question_from_dict(_d(gold=[{"file": "b.pdf", "pages": 7}]), "t")
    assert q.gold[0].pages == (7, 7)


def test_load_and_save(tmp_path):
    q1 = question_from_dict(_d(), "t")
    q2 = question_from_dict(_d(id="lex-0002"), "t")
    save_suite(tmp_path / "lexical.yaml", [q1, q2])
    assert [q.id for q in load_sets(tmp_path)] == ["lex-0001", "lex-0002"]


def test_duplicate_ids_across_files_are_an_error(tmp_path):
    q = question_from_dict(_d(), "t")
    save_suite(tmp_path / "lexical.yaml", [q])
    save_suite(tmp_path / "paraphrase.yaml", [question_from_dict(_d(suite="paraphrase"), "t")])
    with pytest.raises(SchemaError):
        load_sets(tmp_path)


def test_rejected_are_skipped_unless_asked(tmp_path):
    rej = question_from_dict(_d(provenance={"author": "draft", "status": "rejected"}), "t")
    save_suite(tmp_path / "lexical.yaml", [rej])
    assert load_sets(tmp_path) == []
    assert len(load_sets(tmp_path, include_rejected=True)) == 1


def test_suite_quotas_are_nine_suites_totalling_400():
    # Pinned: the quota table is the plan for the whole question set. The
    # multilingual suite was dropped 2026-10-05, so a record naming it must be
    # rejected rather than quietly accepted as a tenth suite.
    assert SUITES == {"lexical": 40, "paraphrase": 50, "code": 40, "scoped": 40, "books": 50,
                      "multihop": 55, "canvas": 40, "personal": 40, "unanswerable": 45}
    assert sum(SUITES.values()) == 400
    with pytest.raises(SchemaError):
        question_from_dict(_d(suite="multilingual"), "t")


def test_split_balancer_is_balanced_stable_and_deterministic():
    qs = [question_from_dict(_d(id=f"lex-{i:04d}"), "t") for i in range(1, 11)]
    assign_splits(qs)
    assert sum(q.split == "dev" for q in qs) == 5
    before = {q.id: q.split for q in qs}
    qs.append(question_from_dict(_d(id="lex-0011"), "t"))
    assign_splits(qs)
    assert all(q.split == before[q.id] for q in qs if q.id in before)   # never reshuffles
    again = [question_from_dict(_d(id=f"lex-{i:04d}"), "t") for i in range(1, 11)]
    assert [q.split for q in assign_splits(again)] == [before[q.id] for q in again]


# ---- gold alternatives (twins: the same passage in another file) ----

def test_alternatives_round_trip_and_are_omitted_when_empty():
    rec = _d(gold=[{"file": "b.pdf", "pages": [3, 3],
                    "alternatives": [{"file": "b.md", "heading": "Slide 3"}]}])
    q = question_from_dict(rec, "t")
    assert q.gold[0].alternatives == (GoldSource(file="b.md", heading="Slide 3"),)
    assert question_from_dict(question_to_dict(q), "t") == q
    assert "alternatives" not in question_to_dict(question_from_dict(_d(), "t"))["gold"][0]


def test_an_alternative_cannot_have_alternatives():
    with pytest.raises(SchemaError):
        question_from_dict(_d(gold=[{"file": "a.md", "alternatives": [
            {"file": "b.md", "alternatives": [{"file": "c.md"}]}]}]), "t")


# ---- an unknown key is an error, never silently ignored ----
# A misspelt `heding:` on a gold entry used to load as a file-only locator: the
# question got easier to match and every retrieval metric quietly inflated.

@pytest.mark.parametrize("record,bad,an_allowed_key", [
    (_d(expect_cource="Stats"), "expect_cource", "expect_course"),
    (_d(gold=[{"file": "a.md", "heding": "BM25"}]), "heding", "heading"),
    (_d(gold=[{"file": "a.md", "alternatives": [{"file": "b.md", "required": True}]}]),
     "required", "pages"),                  # an alternative carries file / pages / heading only
    (_d(provenance={"author": "draft", "status": "draft", "draftd_by": "x"}),
     "draftd_by", "drafted_by"),
], ids=["top-level", "gold", "alternative", "provenance"])
def test_an_unknown_key_names_itself_and_lists_the_allowed_keys(record, bad, an_allowed_key):
    with pytest.raises(SchemaError) as e:
        question_from_dict(record, "t")
    message = str(e.value)
    assert repr(bad) in message
    assert an_allowed_key in message.split("allowed:", 1)[1]


def test_every_documented_key_is_still_accepted():
    q = question_from_dict(_d(gold=[{"file": "a.md", "pages": [1, 2], "heading": "H",
                                     "required": False,
                                     "alternatives": [{"file": "b.md", "pages": 3,
                                                       "heading": "H2"}]}]), "t")
    assert q.gold[0].alternatives == (GoldSource(file="b.md", pages=(3, 3), heading="H2"),)
    assert q.provenance["seed_chunks"] == ["x"] and q.provenance["reviewed_at"] is None


@pytest.mark.parametrize("over", [{"expect_course": 5}, {"expect_course": ["Stats"]},
                                  {"notes": 5}, {"notes": ["a"]}, {"notes": 0}])
def test_expect_course_and_notes_must_be_a_string_or_null(over):
    with pytest.raises(SchemaError):
        question_from_dict(_d(**over), "t")


def test_null_notes_load_as_an_empty_string_and_null_expect_course_stays_null():
    q = question_from_dict(_d(notes=None, expect_course=None), "t")
    assert q.notes == "" and q.expect_course is None
    kept = question_from_dict(_d(expect_course="Stats", notes="why"), "t")
    assert kept.expect_course == "Stats" and kept.notes == "why"


# ---- save_suite is atomic ----

def test_a_crash_while_writing_never_truncates_the_existing_file(tmp_path, monkeypatch):
    target = tmp_path / "lexical.yaml"
    save_suite(target, [question_from_dict(_d(), "t")])
    before = target.read_bytes()
    real_write = Path.write_text

    def dies_midway(self, text, *args, **kwargs):
        real_write(self, text[:20], *args, **kwargs)        # half a file reaches the disk...
        raise OSError("power cut")                          # ...and then the process dies

    monkeypatch.setattr(Path, "write_text", dies_midway)
    with pytest.raises(OSError, match="power cut"):
        save_suite(target, [question_from_dict(_d(id="lex-0002"), "t")])
    monkeypatch.undo()
    assert target.read_bytes() == before                    # the old file is whole
    assert [p.name for p in tmp_path.iterdir()] == ["lexical.yaml"]     # and no temp is left


def test_a_failed_swap_keeps_the_old_file_and_removes_the_temp(tmp_path, monkeypatch):
    target = tmp_path / "lexical.yaml"
    save_suite(target, [question_from_dict(_d(), "t")])
    before = target.read_bytes()

    def refuse(src, dst):
        raise OSError("target is locked")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError, match="locked"):
        save_suite(target, [question_from_dict(_d(id="lex-0002"), "t")])
    monkeypatch.undo()
    assert target.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["lexical.yaml"]


def test_a_briefly_locked_target_is_retried_until_the_swap_succeeds(tmp_path, monkeypatch):
    # Windows: a scanner holding the file for a moment makes os.replace raise
    # PermissionError (WinError 5); that, and only that, is retried.
    import eval.bench.questions as qmod
    target, real, tries = tmp_path / "lexical.yaml", os.replace, []

    def locked_twice(src, dst):
        tries.append(1)
        if len(tries) <= 2:
            raise PermissionError(5, "Access is denied")
        real(src, dst)

    monkeypatch.setattr(os, "replace", locked_twice)
    monkeypatch.setattr(qmod.time, "sleep", lambda s: None)
    save_suite(target, [question_from_dict(_d(), "t")])
    assert len(tries) == 3
    assert [q.id for q in load_sets(tmp_path)] == ["lex-0001"]
    assert [p.name for p in tmp_path.iterdir()] == ["lexical.yaml"]


def test_a_lock_that_outlasts_the_retries_raises_and_keeps_the_old_file(tmp_path, monkeypatch):
    import eval.bench.questions as qmod
    target = tmp_path / "lexical.yaml"
    save_suite(target, [question_from_dict(_d(), "t")])
    before, tries = target.read_bytes(), []

    def always_locked(src, dst):
        tries.append(1)
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(os, "replace", always_locked)
    monkeypatch.setattr(qmod.time, "sleep", lambda s: None)
    with pytest.raises(PermissionError):
        save_suite(target, [question_from_dict(_d(id="lex-0002"), "t")])
    monkeypatch.undo()
    assert len(tries) == qmod._REPLACE_TRIES
    assert target.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["lexical.yaml"]


# ---- one writer of the sets at a time, across processes ----

def test_the_sets_lock_is_released_after_use(tmp_path):
    with sets_lock(tmp_path):
        assert (tmp_path / ".sets.lock").exists()
    assert not (tmp_path / ".sets.lock").exists()


def test_a_live_holder_is_waited_for_then_reported(tmp_path, monkeypatch):
    import eval.bench.questions as qmod
    (tmp_path / ".sets.lock").write_text("123")              # another writer, still working
    monkeypatch.setattr(qmod, "_LOCK_TIMEOUT", 0.2)
    with pytest.raises(TimeoutError, match="another writer"):
        with sets_lock(tmp_path):
            pass
    assert (tmp_path / ".sets.lock").read_text() == "123"    # someone else's lock is left alone


def test_a_dead_writers_lock_is_broken(tmp_path):
    import time
    lock = tmp_path / ".sets.lock"
    lock.write_text("123")
    old = time.time() - 120                                  # far older than any read-modify-write
    os.utime(lock, (old, old))
    with sets_lock(tmp_path):
        assert lock.read_text() == str(os.getpid())          # taken over
    assert not lock.exists()
