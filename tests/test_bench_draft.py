"""
Tests for `bench draft` (eval/bench/draft.py).

No network and no real model: the LLM is a FakeLLM that replays canned replies and
the query API is a FakeSearch callable. Everything else is real: the seed pack, a
small on-disk chunk corpus, load_gold_chunks, validate() and the YAML writer, so a
record is accepted here only if the real validator accepts it.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.bench import cli, draft
from eval.bench.draft import DraftError, SearchUnreachable, draft_suite, parse_reply
from eval.bench.questions import SUITES, load_sets, question_from_dict, save_suite
from eval.bench.sample import write_pack
from src.llm.llm_client import LLMResponse

PDF = "B\\book.pdf"
NOTE = "N\\notes.md"
BISECT = ("Bisection halves the bracketing interval at each step, so after k steps the interval "
          "length is (b - a) / 2^k and the root stays trapped until the tolerance is reached.")
SECANT = ("Secant iteration replaces the derivative in Newton's method with a finite difference "
          "quotient built from the two most recent iterates, which gives superlinear convergence.")
PIVOT = ("Gaussian elimination with partial pivoting swaps rows so that the largest available "
         "pivot is used, which controls the growth of rounding errors in the multipliers.")
MULLER = ("Muller's method fits a parabola through three previous iterates and takes the root "
          "closest to the latest one as the next approximation of the zero.")

# What a well-behaved model answers for each text: a question that copies no 5-word run,
# and nuggets built from the passage's own content words.
GOOD = {
    BISECT: dict(question="How quickly does the enclosing bracket shrink as bisection proceeds, and what ends it?",
                 nuggets=["The interval is halved at each step", "The root stays trapped until the tolerance is reached"]),
    SECANT: dict(question="What does the secant iteration use in place of Newton's derivative, and how fast does it converge?",
                 nuggets=["It replaces the derivative with a finite difference quotient from two recent iterates",
                          "Convergence is superlinear"]),
    PIVOT: dict(question="Which row is chosen as the pivot in partial pivoting, and why does it matter for rounding?",
                nuggets=["The largest available pivot is used", "Partial pivoting controls the growth of rounding errors"]),
    MULLER: dict(question="What does Muller's method fit through three previous iterates?",
                 nuggets=["It fits a parabola through three previous iterates"]),
}


def reply(text=BISECT, **over):
    d = {"skip": False, "reason": "", "tier": "T2", "expect_course": None, "notes": "tests the fake drafter",
         **GOOD[text], **over}
    return json.dumps(d)


def answer_for_seed(system, user):
    """A reply for whichever passage the prompt carries."""
    return next(reply(t) for t in GOOD if t in user)


def seed(doc_id, text=BISECT, **over):
    rec = {"doc_id": doc_id, "lane": "x_chunks.jsonl", "source_file": NOTE, "file_type": "note",
           "page_start": None, "page_end": None, "heading_path": "", "course_name": "Numerical Methods",
           "domain": "math", "text": text}
    rec.update(over)
    return rec


def pdf_seed(doc_id, text=BISECT, start=5, end=6, **over):
    return seed(doc_id, text, source_file=PDF, file_type="pdf", page_start=start, page_end=end, **over)


class FakeLLM:
    """Replays `replies` in order: a str is returned, an Exception raised, a callable
    called with (system, user). A call past the end fails the test (pytest.fail is
    not an Exception, so the drafter's own error handling cannot swallow it)."""
    model = "configured-model"

    def __init__(self, *replies, reported="reported-model"):
        self.replies, self.reported, self.calls = list(replies), reported, []

    def complete(self, system, user, temperature=None, max_tokens=None):
        self.calls.append((system, user))
        if not self.replies:
            pytest.fail("unexpected extra LLM call")
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        if callable(r):
            r = r(system, user)
        return LLMResponse(text=r, model=self.reported, provider="openai")


def hit(cid, text="", score=0.0, label="a label", **over):
    return {"id": cid, "origin_id": cid, "lookup_available": True, "n": 1, "label": label,
            "cited": False, "live": False, "score": score, "text": text, **over}


class FakeSearch:
    """The warm query API: /search -> `hits` (a list, or a callable of the payload),
    /chunks/<id> -> `meta[id]`. A lookup the test did not expect is a KeyError."""

    def __init__(self, hits=(), meta=None, unreachable=False):
        self.hits, self.meta, self.unreachable, self.calls = hits, meta or {}, unreachable, []

    def __call__(self, path, payload=None):
        self.calls.append((path, payload))
        if self.unreachable:
            raise SearchUnreachable("search service is unreachable (refused)")
        if path == "/search":
            return {"results": self.hits(payload) if callable(self.hits) else self.hits, "retrieval": {}}
        assert path.startswith("/chunks/"), path
        cid = unquote(path[len("/chunks/"):].split("?")[0])
        return {"id": cid, "kind": "chunk", "text": None, "metadata": self.meta[cid]}

    def searches(self):
        return [p for path, p in self.calls if path == "/search"]

    def lookups(self):
        return [path for path, _ in self.calls if path.startswith("/chunks/")]


def corpus(tmp_path, *chunks):
    """data/x_chunks.jsonl holding each seed chunk, as the corpus the validator reads."""
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    keys = ("source_file", "file_type", "page_start", "page_end", "heading_path", "course_name", "domain")
    rows = [{"doc_id": c["doc_id"], "text": c["text"],
             "metadata": {k: c[k] for k in keys if c.get(k) is not None}} for c in chunks]
    (d / "x_chunks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return d


def drafted(tmp_path, suite, records, llm, fetch=None, **kw):
    """draft_suite over `records` written as a pack, with a corpus built from their chunks."""
    logs, sleeps = [], []
    chunks = [c for r in records for c in draft._chunks_of(suite, r)]
    pack = tmp_path / "pack.jsonl"
    write_pack(pack, records)
    sets = tmp_path / "sets"
    kw = {"n": 10, "max_score": -2.0, **kw}
    d = draft_suite(suite, llm, fetch or FakeSearch(), seeds_path=pack, sets_dir=sets,
                    data_dir=corpus(tmp_path, *chunks), sleep=sleeps.append, log=logs.append, **kw)
    return SimpleNamespace(d=d, logs=logs, sleeps=sleeps, sets=sets,
                           out=lambda: load_sets(sets, include_rejected=True))


def existing(sets, suite, *specs):
    """Write <suite>.yaml: specs are (id, status, seed_chunks) triples."""
    sets.mkdir(exist_ok=True)
    save_suite(sets / f"{suite}.yaml", [question_from_dict({
        "id": qid, "question": f"already drafted {qid}", "suite": suite, "tier": "T1", "split": None,
        "answerable": True, "gold": [{"file": "a.md"}], "nuggets": ["n"], "expect_course": None,
        "provenance": {"author": "draft", "status": status, "seed_chunks": seeds}}, "t")
        for qid, status, seeds in specs])


# --- the schema's prefixes and the reply parser -----------------------------------

def test_every_suite_has_a_prefix_that_makes_a_valid_id():
    assert set(draft.PREFIX) == set(SUITES)
    for suite, prefix in draft.PREFIX.items():
        q = question_from_dict({"id": f"{prefix}-0001", "question": "q?", "suite": suite, "tier": "T1",
                                "answerable": False, "provenance": {"author": "draft", "status": "draft"}}, "t")
        assert q.id == f"{prefix}-0001"


def test_parse_reply_strips_markdown_fences():
    plain = reply()
    assert parse_reply("```json\n" + plain + "\n```")["question"] == GOOD[BISECT]["question"]
    assert parse_reply("Here you go:\n```\n" + plain + "\n```\nDone.")["tier"] == "T2"
    assert parse_reply(plain)["nuggets"] == GOOD[BISECT]["nuggets"]


@pytest.mark.parametrize("text", [
    "not json at all",
    "[1, 2]",                                                    # JSON, but not an object
    json.dumps({"question": "q", "nuggets": ["n"], "tier": "T1"}),            # no "skip"
    json.dumps({"skip": "no"}),                                  # skip is not a bool
    reply(question="  "),
    reply(nuggets=[]),
    reply(nuggets=["n"] * 6),                                    # the schema allows 1-5
    reply(nuggets=["fine", 3]),
    reply(tier="T9"),
    reply(expect_course=7),
])
def test_parse_reply_rejects_a_reply_that_is_not_what_was_asked_for(text):
    with pytest.raises(ValueError):
        parse_reply(text)


def test_parse_reply_for_a_skip_needs_only_the_reason():
    assert parse_reply(json.dumps({"skip": True, "reason": "a contents page"})) == {
        "skip": True, "reason": "a contents page"}


def test_a_pack_with_a_line_separator_inside_a_text_is_read_whole(tmp_path):
    # splitlines() would cut a record at U+2028; write_pack leaves it raw (ensure_ascii=False)
    write_pack(tmp_path / "p.jsonl", [seed("a", "left right"), seed("b")])
    assert [r["doc_id"] for r in draft.load_pack(tmp_path / "p.jsonl")] == ["a", "b"]
    (tmp_path / "bad.jsonl").write_text('{"doc_id": "a"}\n{oops\n', encoding="utf-8")
    with pytest.raises(DraftError, match="line 2"):
        draft.load_pack(tmp_path / "bad.jsonl")


# --- gold, provenance, ids -----------------------------------------------------------

def test_gold_comes_from_the_seed_never_from_the_model(tmp_path):
    seeds = [pdf_seed("p1", BISECT, 5, 6), seed("n1", SECANT, heading_path="Ch 4 > Secant"),
             seed("n2", PIVOT), pdf_seed("p2", MULLER, 7, None)]
    run = drafted(tmp_path, "paraphrase", seeds, FakeLLM(*[answer_for_seed] * 4))
    q_pdf, q_head, q_file, q_one = run.out()
    assert [(g.file, g.pages, g.heading, g.required) for g in q_pdf.gold] == [(PDF, (5, 6), None, True)]
    assert [(g.file, g.pages, g.heading) for g in q_head.gold] == [(NOTE, None, "Ch 4 > Secant")]
    assert [(g.file, g.pages, g.heading) for g in q_file.gold] == [(NOTE, None, None)]    # file-level
    assert [(g.pages, g.heading) for g in q_one.gold] == [((7, 7), None)]                 # no page_end
    assert [q.id for q in run.out()] == ["para-0001", "para-0002", "para-0003", "para-0004"]
    assert q_pdf.provenance == {"author": "draft", "drafted_by": "reported-model", "seed_chunks": ["p1"],
                                "status": "draft", "reviewed_at": None}
    assert q_pdf.split is None and q_pdf.suite == "paraphrase" and q_pdf.answerable
    assert q_pdf.expect_course is None and q_pdf.tier == "T2" and "tests the fake drafter" in q_pdf.notes


def test_drafted_by_falls_back_to_the_configured_model(tmp_path):
    run = drafted(tmp_path, "paraphrase", [seed("n1")], FakeLLM(reply(), reported=""))
    assert run.out()[0].provenance["drafted_by"] == "configured-model"
    assert "configured-model" in run.d.summary()


# --- resume ---------------------------------------------------------------------------

def test_resume_skips_used_seeds_and_continues_the_ids(tmp_path):
    sets = tmp_path / "sets"
    existing(sets, "lexical", ("lex-0003", "draft", ["s1"]), ("lex-0004", "rejected", ["s2"]))
    existing(sets, "paraphrase", ("para-0001", "draft", ["s3"]))      # another suite's seed is used up too
    llm = FakeLLM(answer_for_seed)
    run = drafted(tmp_path, "lexical", [seed("s1", BISECT), seed("s2", SECANT), seed("s3", PIVOT),
                                        seed("s4", MULLER)], llm, n=3)
    assert len(llm.calls) == 1 and MULLER in llm.calls[0][1]
    new = [q for q in run.out() if q.suite == "lexical"]
    assert [q.id for q in new] == ["lex-0003", "lex-0004", "lex-0005"]    # numbering counts the rejected one
    assert new[-1].provenance["seed_chunks"] == ["s4"]
    assert new[1].provenance["status"] == "rejected"                    # a review decision is kept
    assert "lexical now holds 2 of 3" in run.d.summary()                # a rejected draft is not live


def test_a_full_suite_asks_nothing_and_leaves_the_file_alone(tmp_path):
    sets = tmp_path / "sets"
    existing(sets, "lexical", ("lex-0001", "verified", ["a"]), ("lex-0002", "draft", ["b"]))
    before = (sets / "lexical.yaml").read_bytes()
    llm = FakeLLM()
    run = drafted(tmp_path, "lexical", [seed("s1")], llm, n=2)
    assert llm.calls == [] and (sets / "lexical.yaml").read_bytes() == before
    assert run.d.summary().startswith("accepted 0; lexical now holds 2 of 2")


def test_the_run_stops_at_n_in_the_middle_of_the_pack(tmp_path):
    existing(tmp_path / "sets", "lexical", ("lex-0001", "verified", ["a"]), ("lex-0002", "draft", ["b"]))
    run = drafted(tmp_path, "lexical", [seed("s1"), seed("s2", SECANT)], FakeLLM(answer_for_seed), n=3)
    assert [q.id for q in run.out()] == ["lex-0001", "lex-0002", "lex-0003"]     # one more, then it stops
    assert run.out()[-1].provenance["seed_chunks"] == ["s1"]


def test_rejected_questions_do_not_count_toward_the_quota(tmp_path):
    existing(tmp_path / "sets", "lexical", ("lex-0001", "rejected", ["a"]))
    run = drafted(tmp_path, "lexical", [seed("s1")], FakeLLM(answer_for_seed), n=1)
    assert [q.id for q in run.out()] == ["lex-0001", "lex-0002"]        # topped up past the rejected one


# --- validation, retry, skip -----------------------------------------------------------

COPIED = reply(question="Why does the bracketing interval at each step halve under bisection?")


def test_a_failing_validation_gets_exactly_one_retry_with_the_finding_then_the_seed_is_skipped(tmp_path):
    llm = FakeLLM(COPIED, COPIED)                                       # a third call would fail the test
    run = drafted(tmp_path, "paraphrase", [seed("n1")], llm)
    assert len(llm.calls) == 2
    feedback = llm.calls[1][1]
    assert "copied-phrasing" in feedback and "paraphrase it" in feedback and "REJECTED" in feedback
    assert "REJECTED" not in llm.calls[0][1]
    assert run.d.skipped == {"invalid:copied-phrasing": 1} and run.d.count["retries"] == 1
    assert not (run.sets / "paraphrase.yaml").exists()                  # an invalid record is never written
    assert any("skipped (invalid:copied-phrasing)" in line for line in run.logs)


def test_the_retry_can_succeed(tmp_path):
    run = drafted(tmp_path, "paraphrase", [seed("n1")], FakeLLM(COPIED, reply()))
    assert [q.id for q in run.out()] == ["para-0001"] and run.d.count["retries"] == 1
    assert not run.d.skipped


def test_lexical_questions_may_repeat_the_term(tmp_path):
    # the validator exempts `lexical` from the copied-phrasing rule; the drafter inherits that
    run = drafted(tmp_path, "lexical", [seed("n1")], FakeLLM(COPIED))
    assert len(run.out()) == 1 and run.d.count["retries"] == 0


def test_an_unsupported_nugget_is_retried_like_a_copied_phrase(tmp_path):
    bad = reply(nuggets=["Kolmogorov complexity diverges under Wasserstein regularisation"])
    llm = FakeLLM(bad, reply())
    run = drafted(tmp_path, "paraphrase", [seed("n1")], llm)
    assert "nugget-unsupported" in llm.calls[1][1] and len(run.out()) == 1


def test_a_seed_whose_file_is_not_in_the_corpus_is_never_written(tmp_path):
    llm = FakeLLM(reply(), reply())
    pack = tmp_path / "pack.jsonl"
    write_pack(pack, [seed("n1")])
    d = draft_suite("paraphrase", llm, FakeSearch(), seeds_path=pack, sets_dir=tmp_path / "sets",
                    data_dir=corpus(tmp_path), n=5, max_score=-2.0, sleep=lambda s: None, log=lambda s: None)
    assert d.skipped == {"invalid:gold-file-missing": 1} and not (tmp_path / "sets").exists()


def test_a_malformed_reply_is_a_retry(tmp_path):
    llm = FakeLLM("Sure! Here is your question.", reply())
    run = drafted(tmp_path, "paraphrase", [seed("n1")], llm)
    assert "not valid JSON" in llm.calls[1][1] and len(run.out()) == 1
    assert run.d.count["retries"] == 1


def test_a_fenced_reply_needs_no_retry(tmp_path):
    llm = FakeLLM("```json\n" + reply() + "\n```")
    assert len(drafted(tmp_path, "paraphrase", [seed("n1")], llm).out()) == 1 and len(llm.calls) == 1


def test_two_malformed_replies_skip_the_seed(tmp_path):
    run = drafted(tmp_path, "paraphrase", [seed("n1")], FakeLLM("nope", "still nope"))
    assert run.d.skipped == {"malformed-reply": 1} and not run.sets.exists()


def test_a_model_that_skips_is_not_asked_again(tmp_path):
    llm = FakeLLM(json.dumps({"skip": True, "reason": "a table of contents"}))
    run = drafted(tmp_path, "paraphrase", [seed("n1")], llm)
    assert len(llm.calls) == 1 and run.d.skipped == {"llm-skip": 1}
    assert any("a table of contents" in line for line in run.logs) and not run.sets.exists()


def test_the_model_never_decides_expect_course_outside_scoped(tmp_path):
    run = drafted(tmp_path, "paraphrase", [seed("n1")], FakeLLM(reply(expect_course="Anything")))
    assert run.out()[0].expect_course is None


def test_scoped_expect_course_is_the_seeds_course(tmp_path):
    ok = drafted(tmp_path, "scoped", [seed("n1")], FakeLLM(reply(expect_course="numerical methods")))
    assert ok.out()[0].expect_course == "Numerical Methods"               # the corpus' own spelling
    llm = FakeLLM(reply(expect_course=None), reply(expect_course="Numerical Methods"))
    fixed = drafted(tmp_path / "again", "scoped", [seed("n1")], llm)
    assert "'Numerical Methods'" in llm.calls[1][1] and len(fixed.out()) == 1
    assert "Numerical Methods" in llm.calls[0][1]                         # the course is in the prompt


# --- LLM and transport errors, --max-calls ------------------------------------------------

def test_an_llm_error_is_retried_with_backoff(tmp_path):
    llm = FakeLLM(ConnectionError("proxy down"), reply())
    run = drafted(tmp_path, "paraphrase", [seed("n1")], llm)
    assert len(run.out()) == 1 and run.sleeps == [draft.BACKOFF_SECONDS]
    assert run.d.count["llm_errors"] == 1 and run.d.calls == 2          # a failed try is still a call


def test_failing_every_try_skips_the_seed_and_the_next_seed_is_still_drafted(tmp_path):
    boom = ConnectionError("proxy down")
    llm = FakeLLM(*[boom] * draft.LLM_TRIES, reply(SECANT))
    run = drafted(tmp_path, "paraphrase", [seed("n1"), seed("n2", SECANT)], llm)
    assert run.sleeps == [draft.BACKOFF_SECONDS * 2 ** i for i in range(draft.LLM_TRIES - 1)]
    assert run.d.skipped == {"llm-error": 1} and [q.provenance["seed_chunks"] for q in run.out()] == [["n2"]]
    assert any("ConnectionError: proxy down" in line for line in run.logs)


def test_max_calls_caps_the_llm_calls_and_stops_the_run(tmp_path):
    llm = FakeLLM(*[answer_for_seed] * 5)
    run = drafted(tmp_path, "paraphrase", [seed(f"s{i}") for i in range(5)], llm, max_calls=2)
    assert len(llm.calls) == 2 and len(run.out()) == 2
    assert run.d.stopped == "--max-calls 2 reached" and "stopped early" in run.d.summary()


def test_a_retry_counts_against_max_calls(tmp_path):
    llm = FakeLLM(COPIED, reply())
    run = drafted(tmp_path, "paraphrase", [seed("n1")], llm, max_calls=1)
    assert len(llm.calls) == 1 and not run.sets.exists() and run.d.stopped


def test_the_file_is_a_valid_suite_after_every_write_and_no_temp_is_left(tmp_path):
    sets, seen = tmp_path / "sets", []

    def snapshot(system, user):
        # runs at the start of each seed's call, i.e. right after the previous write
        if sets.exists():
            assert not list(sets.glob("*.tmp"))
            seen.append([q.id for q in load_sets(sets)])
        else:
            seen.append([])
        return answer_for_seed(system, user)

    run = drafted(tmp_path, "paraphrase", [seed(f"s{i}") for i in range(3)], FakeLLM(snapshot, snapshot, snapshot))
    assert seen == [[], ["para-0001"], ["para-0001", "para-0002"]]
    assert len(run.out()) == 3 and not list(sets.glob("*.tmp"))


def test_a_review_made_while_the_run_is_going_survives_the_next_write(tmp_path):
    sets = tmp_path / "sets"

    def console_verifies_first_draft_then_answer(system, user):
        if sets.exists():                                     # the second seed: the author has reviewed draft 1
            qs = load_sets(sets, include_rejected=True)
            qs[0].provenance.update(status="verified", reviewed_at="2026-10-07")
            save_suite(sets / "paraphrase.yaml", qs)
        return answer_for_seed(system, user)

    run = drafted(tmp_path, "paraphrase", [seed("s1"), seed("s2")],
                  FakeLLM(console_verifies_first_draft_then_answer, console_verifies_first_draft_then_answer))
    assert [q.provenance["status"] for q in run.out()] == ["verified", "draft"]


# --- twins --------------------------------------------------------------------------------------

def test_twins_from_other_files_become_alternatives_with_their_own_locators(tmp_path):
    text_twin = BISECT + " As a note."                                   # the same passage, a few words more
    hits = [hit("p1", BISECT),                                           # the seed itself
            hit("same", BISECT),                                         # same file: not a twin
            hit("t_note", text_twin), hit("t_pdf", BISECT), hit("t_pdf2", BISECT),
            hit("far", PIVOT),                                           # another passage: no lookup spent
            hit("live1", BISECT, live=True, lookup_available=False)]     # a live-vault excerpt
    meta = {"same": {"source_file": PDF, "page_start": 5, "page_end": 6},
            "t_note": {"source_file": "T\\copy.md", "heading_path": "Ch 4 > Bisection"},
            "t_pdf": {"source_file": "T\\slides.pdf", "page_start": 12, "page_end": 12,
                      "heading_path": "ignored: pages win"},
            "t_pdf2": {"source_file": "t/slides.pdf", "page_start": 12, "page_end": 12}}   # same twin, spelled otherwise
    search = FakeSearch(hits, meta)
    run = drafted(tmp_path, "paraphrase", [pdf_seed("p1")], FakeLLM(reply()), search)
    (q,) = run.out()
    assert [(a.file, a.pages, a.heading) for a in q.gold[0].alternatives] == [
        ("T\\copy.md", None, "Ch 4 > Bisection"), ("T\\slides.pdf", (12, 12), None)]
    assert q.gold[0].file == PDF and q.gold[0].pages == (5, 6)
    assert "twin auto-detected" in q.notes and "T\\copy.md" in q.notes and run.d.count["twins"] == 2
    assert search.lookups() == ["/chunks/same?include_text=0", "/chunks/t_note?include_text=0",
                                "/chunks/t_pdf?include_text=0", "/chunks/t_pdf2?include_text=0"]
    assert search.searches() == [{
        "q": BISECT[:300], "hyde": False, "rerank": "none", "top_k": 10, "include_text": 1500,
        "parent_context": False, "neighbor_context": False, "gate": False}]


def test_a_seed_the_model_skips_costs_no_twin_search(tmp_path):
    search = FakeSearch()
    drafted(tmp_path, "paraphrase", [seed("n1")], FakeLLM(json.dumps({"skip": True})), search)
    assert search.calls == []


def test_an_unreachable_search_service_is_the_one_fail_soft_path(tmp_path):
    search = FakeSearch(unreachable=True)
    run = drafted(tmp_path, "paraphrase", [seed("n1"), seed("n2", SECANT)],
                  FakeLLM(answer_for_seed, answer_for_seed), search)
    assert len(run.out()) == 2 and not any(g.alternatives for q in run.out() for g in q.gold)
    assert len([m for m in run.logs if m.startswith("WARNING")]) == 1        # warned once...
    assert len(search.calls) == 1                                           # ...and not asked again
    assert run.d.twins_off and "unreachable" in run.d.summary()


def test_any_other_search_failure_raises(tmp_path):
    def broken(path, payload=None):
        return {"error": "Reranking failed: out of memory", "results": [], "retrieval": {}}

    with pytest.raises(DraftError, match="out of memory"):
        drafted(tmp_path, "paraphrase", [seed("n1")], FakeLLM(reply()), broken)


# --- multihop -------------------------------------------------------------------------------------

HOP = {"term": "root finding", "chunks": [pdf_seed("h1", BISECT, 5, 5), seed("h2", SECANT, heading_path="Ch 4 > Secant"),
                                          seed("h3", PIVOT, source_file="N\\other.md")]}
HOP_REPLY = reply(BISECT, question="Compare how bisection and the secant iteration use earlier steps when locating a root.",
                  nuggets=["Bisection halves the bracketing interval", "Secant iteration uses a finite difference quotient",
                           "Partial pivoting uses the largest available pivot"])


def test_multihop_has_one_required_gold_per_chunk_and_all_ids_as_seeds(tmp_path):
    search = FakeSearch()
    run = drafted(tmp_path, "multihop", [HOP], FakeLLM(HOP_REPLY), search)
    (q,) = run.out()
    assert [(g.file, g.pages, g.heading, g.required) for g in q.gold] == [
        (PDF, (5, 5), None, True), (NOTE, None, "Ch 4 > Secant", True), ("N\\other.md", None, None, True)]
    assert q.provenance["seed_chunks"] == ["h1", "h2", "h3"] and q.id == "hop-0001"
    assert q.tier == "T3"                                   # the model's T2 does not make it single-hop
    assert len(search.searches()) == 3                      # twins are looked up per chunk


def test_a_paraphrase_question_is_never_tiered_t1(tmp_path):
    # T1 means "near the passage's wording"; a paraphrase question is reworded by construction
    t1 = drafted(tmp_path, "paraphrase", [seed("p")], FakeLLM(reply(tier="T1")))
    assert t1.out()[0].tier == "T2"
    t3 = drafted(tmp_path / "t3", "paraphrase", [seed("p")], FakeLLM(reply(tier="T3")))
    assert t3.out()[0].tier == "T3"                         # only T1 is overridden
    lex = drafted(tmp_path / "lex", "lexical", [seed("p")], FakeLLM(reply(tier="T1")))
    assert lex.out()[0].tier == "T1"                        # other suites keep the model's tier


def test_multihop_tier_is_t3_unless_the_model_says_t4(tmp_path):
    t4 = drafted(tmp_path, "multihop", [HOP], FakeLLM(json.dumps({**json.loads(HOP_REPLY), "tier": "T4"})))
    assert t4.out()[0].tier == "T4"
    t1 = drafted(tmp_path / "t1", "multihop", [HOP], FakeLLM(json.dumps({**json.loads(HOP_REPLY), "tier": "T1"})))
    assert t1.out()[0].tier == "T3"


def test_multihop_prompt_shows_every_passage(tmp_path):
    llm = FakeLLM(HOP_REPLY)
    drafted(tmp_path, "multihop", [HOP], llm)
    prompt = llm.calls[0][1]
    assert "root finding" in prompt and all(c["text"] in prompt for c in HOP["chunks"])


def test_a_multihop_group_with_a_used_chunk_is_not_asked_again(tmp_path):
    existing(tmp_path / "sets", "lexical", ("lex-0001", "draft", ["h2"]))            # h2 seeded another question
    llm = FakeLLM()
    run = drafted(tmp_path, "multihop", [HOP], llm)
    assert llm.calls == [] and [q.suite for q in run.out()] == ["lexical"]


# --- unanswerable ---------------------------------------------------------------------------------

UNANS = {"course": "Numerical Methods", "n_chunks": 120, "headings": ["Bisection", "Secant method", "Pivoting"]}
ASKED = ["How does Muller's method fit a parabola through three iterates?",
         "What is the Illinois variant of regula falsi?",
         "How are Chebyshev nodes chosen for interpolation?"]
SCORES = {ASKED[0]: (-6.0, "Intro to numerics"), ASKED[1]: (1.5, "Regula falsi notes"),
          ASKED[2]: (-2.5, "Interpolation basics")}


def scored(payload):
    score, label = SCORES[payload["q"]]
    return [hit("c1", score=score, label=label), hit("c2", score=score - 3, label="a worse hit")]


def test_unanswerable_keeps_only_candidates_below_the_score_cutoff(tmp_path):
    search = FakeSearch(scored)
    run = drafted(tmp_path, "unanswerable", [UNANS], FakeLLM(json.dumps({"questions": ASKED})), search)
    kept = run.out()
    assert [q.question for q in kept] == [ASKED[0], ASKED[2]]
    assert run.d.skipped == {"abstain-check": 1}
    first = kept[0]
    assert (first.id, first.suite, first.tier, first.answerable, first.gold, first.nuggets) == (
        "none-0001", "unanswerable", "T2", False, [], [])
    assert first.expect_course == "Numerical Methods" and first.provenance["seed_chunks"] == []
    assert first.provenance["drafted_by"] == "reported-model" and first.split is None
    assert "Intro to numerics" in first.notes and "-6.00" in first.notes
    assert "Interpolation basics" in kept[1].notes and "-2.50" in kept[1].notes
    assert search.searches()[0] == {"q": ASKED[0], "hyde": False, "top_k": 5, "include_text": 300, "gate": False}
    # a pass leaves the suite short: the way forward is another pass, not `bench sample`
    assert "run again for another" in run.d.summary() and "bench sample" not in run.d.summary()


def test_max_score_moves_the_cutoff(tmp_path):
    run = drafted(tmp_path, "unanswerable", [UNANS], FakeLLM(json.dumps({"questions": ASKED})),
                  FakeSearch(scored), max_score=2.0)
    assert len(run.out()) == 3 and not run.d.skipped


def test_unanswerable_stops_at_n_without_checking_the_rest(tmp_path):
    search = FakeSearch(scored)
    run = drafted(tmp_path, "unanswerable", [UNANS], FakeLLM(json.dumps({"questions": ASKED})), search, n=1)
    assert [q.question for q in run.out()] == [ASKED[0]] and len(search.searches()) == 1


def test_unanswerable_does_not_repeat_what_was_already_asked(tmp_path):
    sets = tmp_path / "sets"
    existing(sets, "unanswerable", ("none-0001", "rejected", []))
    qs = load_sets(sets, include_rejected=True)
    qs[0].question, qs[0].expect_course = ASKED[0], UNANS["course"]
    save_suite(sets / "unanswerable.yaml", qs)
    llm, search = FakeLLM(json.dumps({"questions": ASKED[:2]})), FakeSearch(scored)
    run = drafted(tmp_path, "unanswerable", [UNANS], llm, search, max_score=2.0)
    assert ASKED[0] in llm.calls[0][1]                                   # listed as "already asked"
    assert run.d.skipped == {"near-duplicate": 1}                        # the validator's own rule
    assert [q.question for q in run.out()] == [ASKED[0], ASKED[1]] and len(search.searches()) == 1


def test_unanswerable_needs_the_search_service(tmp_path):
    with pytest.raises(SearchUnreachable):
        drafted(tmp_path, "unanswerable", [UNANS], FakeLLM(json.dumps({"questions": ASKED})),
                FakeSearch(unreachable=True))
    assert not (tmp_path / "sets").exists()                              # nothing is kept unchecked


def test_unanswerable_hits_without_scores_raise_and_no_hits_means_absent(tmp_path):
    with pytest.raises(DraftError, match="without scores"):
        drafted(tmp_path, "unanswerable", [UNANS], FakeLLM(json.dumps({"questions": ASKED})),
                FakeSearch(lambda p: [hit("c1", score=None)]))
    run = drafted(tmp_path / "empty", "unanswerable", [UNANS], FakeLLM(json.dumps({"questions": ASKED[:1]})),
                  FakeSearch([]))
    assert "returned no hits" in run.out()[0].notes


# --- the search client and the command line ------------------------------------------------------

def test_http_fetch_maps_a_refused_connection_and_an_http_error(monkeypatch):
    import requests
    fetch, calls = draft.http_fetch("http://host:1/"), []

    def refuse(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", refuse)
    with pytest.raises(SearchUnreachable, match="unreachable"):
        fetch("/search", {"q": "x"})
    monkeypatch.setattr(requests, "get", lambda url, timeout=None: SimpleNamespace(ok=False, status_code=503, text="not ready"))
    with pytest.raises(DraftError, match="503.*not ready"):
        fetch("/chunks/a?include_text=0")
    monkeypatch.setattr(requests, "post", lambda url, json=None, timeout=None: (
        calls.append((url, json)) or SimpleNamespace(ok=True, json=lambda: {"results": []})))
    assert fetch("/search", {"q": "x"}) == {"results": []} and calls == [("http://host:1/search", {"q": "x"})]


def _parse(*argv):
    import main
    return main.build_parser().parse_args(list(argv))


def _run(*argv):
    args = _parse(*argv)
    args.func(args)


def test_draft_is_wired_into_the_parser():
    a = _parse("bench", "draft", "--suite", "lexical")
    assert a.func is cli.bench_cmd and a.bench_command == "draft"
    assert (a.suite, a.n, a.provider, a.model, a.seeds, a.sets, a.search_url, a.max_calls, a.max_score) == (
        "lexical", None, "freellmapi", None, None, None, None, None, -2.0)    # seeds: None = the checkout's eval/seeds
    with pytest.raises(SystemExit):
        _parse("bench", "draft")                                    # --suite is required
    with pytest.raises(SystemExit):
        _parse("bench", "draft", "--suite", "nope")


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """`bench draft` with the config, the LLM client and the HTTP client replaced."""
    st = SimpleNamespace(llm=FakeLLM(answer_for_seed), search=FakeSearch(), built=[], urls=[],
                         config={}, tmp=tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda path=None: SimpleNamespace(get=lambda k, d=None: st.config.get(k, d)))

    def build(cfg, provider, model=None, role="generation"):
        st.built.append((provider, model))
        if isinstance(st.llm, Exception):
            raise st.llm
        return st.llm

    monkeypatch.setattr("src.llm.llm_client.LLMClient.from_provider_override", build)
    monkeypatch.setattr(draft, "http_fetch", lambda url: st.urls.append(url) or st.search)
    # the CLI passes no `sleep`, and the real backoff would wait over a minute
    real_draft_suite = draft.draft_suite
    monkeypatch.setattr(draft, "draft_suite", lambda *a, **kw: real_draft_suite(*a, sleep=lambda s: None, **kw))
    seeds = tmp_path / "seeds"
    write_pack(seeds / "paraphrase.jsonl", [seed("n1")])
    st.argv = ["bench", "draft", "--suite", "paraphrase", "--seeds", str(seeds), "--sets", str(tmp_path / "sets"),
               "--data-dir", str(corpus(tmp_path, seed("n1")))]
    return st


def test_bench_draft_end_to_end_through_the_cli(wired, capsys):
    _run(*wired.argv, "--provider", "minimax", "--model", "m-1", "--search-url", "http://x:9")
    out = capsys.readouterr().out
    assert "accepted 1" in out and "model(s): reported-model x1" in out
    assert [q.id for q in load_sets(wired.tmp / "sets")] == ["para-0001"]
    assert wired.built == [("minimax", "m-1")] and wired.urls == ["http://x:9"]


def test_the_cli_defaults_to_freellmapi_and_the_consoles_query_api(wired):
    wired.config["webui.rag_api"] = "http://cfg:8051"
    _run(*wired.argv)
    assert wired.built == [("freellmapi", None)] and wired.urls == ["http://cfg:8051"]


def test_a_draft_run_finds_its_seed_pack_in_the_checkout_not_the_working_directory(
        wired, monkeypatch):
    """--seeds defaults to where `bench sample` writes by default: the checkout's
    eval/seeds, not an eval/seeds under whatever directory this was started from."""
    monkeypatch.setattr(cli, "DEFAULT_SEEDS", wired.tmp / "seeds")      # the pack the fixture wrote
    elsewhere = wired.tmp / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    without_seeds_flag = wired.argv[:4] + wired.argv[6:]
    assert "--seeds" not in without_seeds_flag
    _run(*without_seeds_flag)
    assert [q.id for q in load_sets(wired.tmp / "sets")] == ["para-0001"]


def test_the_cli_exits_1_when_a_seed_was_lost_to_llm_errors(wired, capsys):
    boom = ConnectionError("proxy down")
    wired.llm = FakeLLM(*[boom] * draft.LLM_TRIES)
    with pytest.raises(SystemExit) as e:
        _run(*wired.argv)
    assert e.value.code == 1 and "llm-error 1" in capsys.readouterr().out


def test_cli_mistakes_are_errors_with_exit_2(wired, capsys, tmp_path):
    wired.llm = ValueError("generation.provider = 'nope' is not in the `providers:` registry")
    with pytest.raises(SystemExit) as e:
        _run(*wired.argv, "--provider", "nope")
    assert e.value.code == 2 and "providers" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        _run(*wired.argv[:2], "--suite", "lexical", *wired.argv[4:])        # no lexical pack in --seeds
    assert e.value.code == 2 and "bench sample --suites lexical" in capsys.readouterr().err
    for flag in ("--n", "--max-calls"):
        with pytest.raises(SystemExit) as e:
            _run(*wired.argv, flag, "0")
        assert e.value.code == 2 and flag in capsys.readouterr().err


# --- other writers, LaTeX escapes, cold questions, a dead provider --------------------------

def test_another_writers_question_is_numbered_around_never_overwritten(tmp_path):
    # Two drafters on one suite numbered the same id live: the id comes from the files as
    # they are at write time, and the other writer's question survives.
    sets = tmp_path / "sets"

    def meanwhile(system, user):
        existing(sets, "paraphrase", ("para-0001", "draft", ["other"]))
        return reply()
    run = drafted(tmp_path, "paraphrase", [seed("n1")], FakeLLM(meanwhile))
    assert sorted(q.id for q in run.out()) == ["para-0001", "para-0002"]


def test_a_seed_another_writer_used_meanwhile_is_not_drafted_twice(tmp_path):
    sets = tmp_path / "sets"

    def meanwhile(system, user):
        existing(sets, "lexical", ("lex-0001", "draft", ["n1"]))
        return reply()
    run = drafted(tmp_path, "paraphrase", [seed("n1")], FakeLLM(meanwhile))
    assert [q.id for q in run.out()] == ["lex-0001"] and run.d.skipped == {"seed-taken": 1}


def test_a_suite_another_writer_filled_meanwhile_takes_no_more(tmp_path):
    sets = tmp_path / "sets"

    def meanwhile(system, user):
        existing(sets, "paraphrase", ("para-0001", "draft", ["x1"]))
        return reply()
    run = drafted(tmp_path, "paraphrase", [seed("n1"), seed("n2", SECANT)], FakeLLM(meanwhile), n=1)
    assert [q.id for q in run.out()] == ["para-0001"] and run.d.skipped == {"suite-full": 1}


def test_drafted_ids_stay_below_the_hand_written_range(tmp_path):
    existing(tmp_path / "sets", "personal", ("pers-0501", "verified", []))
    run = drafted(tmp_path, "personal", [seed("n1")], FakeLLM(reply()))
    assert sorted(q.id for q in run.out()) == ["pers-0001", "pers-0501"]


def test_a_latex_backslash_read_as_a_json_escape_is_sent_back(tmp_path):
    # json.loads turns "\beta" into a backspace and "eta" (seen live): such a reply is
    # rejected and the model asked again; a control character is never stored.
    bad = reply().replace('"question": "', '"question": "With \\beta, ', 1)   # JSON text: \beta
    with pytest.raises(ValueError, match="Escape every backslash"):
        parse_reply(bad)
    llm = FakeLLM(bad, reply())
    run = drafted(tmp_path, "paraphrase", [seed("n1")], llm)
    assert len(run.out()) == 1 and "\x08" not in run.out()[0].question
    assert "Escape every backslash" in llm.calls[1][1]


def test_a_question_pointing_at_an_unseen_passage_is_sent_back(tmp_path):
    cold = reply(question="In the provided example, how is the interval halved at each step?")
    llm = FakeLLM(cold, reply())
    run = drafted(tmp_path, "paraphrase", [seed("n1")], llm)
    assert len(run.out()) == 1 and "provided example" not in run.out()[0].question
    assert "asked cold" in llm.calls[1][1]


def test_a_run_stops_after_a_streak_of_seeds_lost_to_llm_errors(tmp_path):
    boom = ConnectionError("proxy down")
    seeds = [seed(f"s{i}") for i in range(draft.LLM_ERROR_STREAK + 2)]
    llm = FakeLLM(*[boom] * (draft.LLM_TRIES * draft.LLM_ERROR_STREAK))   # one more call fails the test
    run = drafted(tmp_path, "paraphrase", seeds, llm)
    assert run.d.skipped == {"llm-error": draft.LLM_ERROR_STREAK} and "in a row" in run.d.stopped


def test_an_overridden_tier_says_so_in_the_notes(tmp_path):
    run = drafted(tmp_path, "paraphrase", [seed("p")], FakeLLM(reply(tier="T1")))
    assert "tier T1 set to T2" in run.out()[0].notes
