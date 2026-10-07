"""
draft.py — `bench draft`: a resumable, validated question drafter that runs on the
project's own LLM provider registry (FreeLLMAPI by default, which is free), so the
questions still missing from the suites cost no Claude tokens. the author verifies every
draft in the console afterwards, so the drafter's one job is never to hand him a
record the validator rejects.

Who decides what:

  The LLM writes the question, its nuggets, a tier and a note. It never writes
  gold. A locator is read off the seed chunk (a PDF's page range, a note's heading
  path) and twins come from the search service, so a model cannot invent a file
  or a page.

  Nothing invalid is written. Each assembled record goes through question_from_dict
  and validate() against the real chunk files; an error, a copied phrasing or an
  unsupported nugget earns ONE retry with the finding as feedback, then the seed
  is skipped (and logged with the reason).

  The run is resumable. The suite's YAML is rewritten whole, atomically, after
  every accepted question, so a crash loses at most one; a seed whose id is in any
  question's provenance.seed_chunks is not asked again; ids continue the suite's
  prefix. A skipped seed leaves no trace, so a later run asks it again.

Twins (the same passage in another file; 24% of the books suite). The seed's first
~300 chars go to the warm query API's /search; a hit from ANOTHER file whose
word-set Jaccard with the seed text reaches TWIN_JACCARD becomes an `alternatives`
entry. /search hits carry no file or page (only id, label, score, text), so a
qualifying hit's metadata is fetched from GET /chunks/{id}. The twin's own text
joins the gold chunks validate() reads, because its file is not among the seed
files the chunk pass loaded.

Unanswerable. A seed is a course neighbourhood, not a chunk. The LLM proposes
questions next to the course's headings; each is kept only if /search's best
score for it is below --max-score (cross-encoder logits: the vault has nothing
that answers it), and the score and top label go into the notes.

Failure policy. An LLM or transport error is retried LLM_TRIES times with backoff,
then the seed is skipped. The search service being UNREACHABLE is the one fail-soft
path (twins are an enrichment): warn once, draft without them, say so in the
summary. Every other search problem, and an unreachable service for `unanswerable`
(where the search IS the check), raises DraftError.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import yaml

from eval.bench.questions import (
    TIERS, Question, SchemaError, load_sets, question_from_dict, save_suite, sets_lock)
from eval.bench.relevance import norm_file
from eval.bench.sample import used_seed_ids
from eval.bench.validate import _words, load_gold_chunks, validate

# Rule constants. They define what the drafter MEANS, so they live beside the rules
# (like the suite quotas and the validator's thresholds); the two a user is expected
# to turn are CLI flags.
PREFIX = {"lexical": "lex", "paraphrase": "para", "code": "code", "scoped": "scope",
          "books": "book", "multihop": "hop", "canvas": "canv", "personal": "pers",
          "unanswerable": "none"}                 # spec 3.2: suite prefix + counter
TWIN_JACCARD = 0.6          # word-set overlap at which another file's chunk is "the same passage"
TWIN_QUERY_CHARS = 300      # the part of the seed text that is the twin search's query
TWIN_TOP_K = 10             # hits the twin search looks at...
TWIN_TEXT_CHARS = 1500      # ...and how much of each it reads
ABSTAIN_TOP_K = 5           # hits the unanswerable check looks at (only the best score counts)
ABSTAIN_TEXT_CHARS = 300
UNANSWERABLE_ASKED = 3      # candidate questions asked of the LLM per course
UNANSWERABLE_TIER = "T2"    # as in the smoke set: the right answer is "nothing among similar sources"
# Proxy failures come in bursts: while its good providers are rate-limited, FreeLLMAPI
# falls back to dead ones and every try in a few seconds fails alike (live: 3 tries
# 2 s apart lost ~40% of seeds). Waits that double to cover a minute ride a burst out.
LLM_TRIES = 5               # calls per prompt before a transport error skips the seed
BACKOFF_SECONDS = 5.0       # the first wait between tries; it doubles (5+10+20+40 s)
LLM_ERROR_STREAK = 5        # seeds lost to LLM errors in a row before the run stops: the provider is down
RETRIES = 1                 # re-asks per seed after a rejected reply
MAX_NUGGETS = 5             # spec 3.2: "1-5 atomic facts"
DRAFT_ID_LIMIT = 500        # drafted ids stay below this; hand-written ones start at 0501 (WRITING_QUESTIONS.md)
HTTP_TIMEOUT = (5, 120)     # connect, read: the first search loads the index and the cross-encoder
# Findings that send a record back to the LLM: every error, plus these warnings.
_BLOCKING_WARNINGS = frozenset({"copied-phrasing", "nugget-unsupported", "nugget-number-unsupported"})
# JSON reads a lone backslash in LaTeX as an escape: \beta arrives as a backspace and
# "eta", \theta as a tab, \nabla as a newline (seen live). A reply string carrying a
# control character goes back to the model, which then escapes its backslashes.
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
# A question is asked cold. These phrasings point at a passage the reader never sees
# ("in the provided example", "this passage"); seen live in drafts.
_NOT_STANDALONE = re.compile(
    r"\b(?:the|this|that)\s+(?:passage|excerpt|snippet|seed|cheat\s?sheet)s?\b"
    r"|\b(?:provided|above|following|given)\s+(?:passage|excerpt|snippet|text|example|code|script|cell)s?\b",
    re.IGNORECASE)


class DraftError(RuntimeError):
    """The run cannot go on (as opposed to one seed that cannot be drafted)."""


class SearchUnreachable(DraftError):
    """The query API did not answer at all: refused, DNS, connect timeout."""


class _CallCap(Exception):
    """--max-calls is reached: the run stops here (a stop, not a skip)."""


class _GaveUp(Exception):
    """An LLM call failed LLM_TRIES times in a row."""


Fetch = Callable[[str, "dict | None"], dict]     # (path, JSON body or None for GET) -> parsed JSON


def _say(msg: str) -> None:
    print(msg, flush=True)


# --- prompts --------------------------------------------------------------------

_SYSTEM = """\
You draft evaluation questions for Noetrix, a retrieval system over the author's personal Obsidian vault (course notes, textbooks, notebooks, canvases, daily notes). the author reviews every question you write, so honesty and quality matter more than quantity.

You are shown the seed passage(s) for ONE question. Write at most one question they answer, or skip the seed.

RULES
1. Skip a seed that cannot carry a good question: a table of contents, boilerplate, a fragment, a bare formula with no context, garbled text, or text that is not mostly English. Skipping is cheap; a forced question is not.
2. Prefer facts specific to the author's material (a worked example, a stated assumption, a course's own notation, a notebook's numbers, a canvas's own relations) over textbook generalities any model already knows. A question answerable from general knowledge alone is weak: make the vault-specific detail the point.
3. Paraphrase: the question must NOT copy a run of 5 or more consecutive words from the passage{exception}.
4. Nuggets are 1 to 3 (never more than 5) atomic facts a complete answer must contain. Write each with the CONTENT WORDS of the passage: at least 60% of a nugget's content words must occur in the passage (this is checked mechanically). No nugget the passage alone cannot verify.
5. Tier honestly. T1 = near-verbatim, one passage. T2 = paraphrased, or the right source must be picked among similar ones. T3 = needs two or more sources or sections. T4 = needs a follow-up search to resolve a reference. {tier_rule}

{suite_rules}

{reply}"""

_REPLY = """\
REPLY with one JSON object and nothing else (no prose, no markdown fence):
{"skip": false, "reason": "", "question": "...", "nuggets": ["..."], "tier": "T1", "expect_course": null, "notes": "..."}
- To skip the seed set "skip": true and give the "reason"; the other fields may then be empty.
- "question" must stand alone: it is asked cold, with no passage in front of the reader.
- "tier" is T1, T2, T3 or T4. "notes" is ONE line: what the question tests and why that tier."""

_SUITE_RULES = {
    "lexical": "SUITE lexical (exact-term search). Build the question around the passage's most distinctive exact token: an identifier, function or error name, formula or theorem name, acronym. Keep that term verbatim in the question and paraphrase everything around it.",
    "paraphrase": "SUITE paraphrase (meaning, not vocabulary). Word the question unlike the passage: synonyms, a different sentence shape, a description instead of the passage's own rare terms, so that only the meaning can match it.",
    "code": "SUITE code. The passage is code from a notebook or script. Ask what it does, which function, parameter or library it uses, or what number or output it shows. Nuggets name the actual calls, arguments or values.",
    "scoped": 'SUITE scoped (course-aware retrieval). The seed names its course. The question must name that course in natural words ("In my <course> notes ...") and ask about something specific to its material. Set "expect_course" to the course name exactly as given.',
    "books": "SUITE books (textbook pages, some of them OCR'd scans: ignore garbled characters). Ask for the page's specific content: the numbers of a worked example, the parts of an exercise, a definition or theorem with its conditions. Name the book or its subject so the right book can be picked.",
    "canvas": "SUITE canvas. The passage is an Obsidian canvas node with its connections. Ask about the relation the canvas draws: how two ideas connect, what a labelled edge says, what a group holds. Nuggets state those relations.",
    "personal": "SUITE personal. The passage is one of the author's own notes (daily note, strategy, idea, workflow). Ask about the plans, decisions, projects or routines it records. Never invent a detail; skip a note that records nothing worth asking about.",
    "multihop": "SUITE multihop. You get 2 or 3 passages from different files that share a heading term. Write ONE question that needs ALL of them (compare, combine, relate). Skip when the passages repeat the same content, are unrelated, or one alone answers the question. Nuggets cover facts from each passage.",
}

_UNANSWERABLE_SYSTEM = f"""\
You draft ABSTENTION test questions for Noetrix, a retrieval system over the author's personal Obsidian vault. A correct system answers them by saying the vault does not contain the answer.

You are given one course and the headings its notes cover. The vault is large: besides this course's notes it holds dozens of standard textbooks (statistics, machine learning, calculus, databases, AI engineering, marketing and more), so a question about any STANDARD topic of those subjects is answered somewhere in it and is useless here. Write {UNANSWERABLE_ASKED} questions a student of the course could plausibly ask, near the course's topics, that no standard textbook or lecture note would answer:
- something from 2025 or later (a new model, library release, paper or regulation);
- a precise figure or detail of one real-world system or organisation that no textbook states (a company's internal number, a product's current price or limit);
- a niche, named method or result outside the standard curriculum;
- a detail of the author's own projects that notes would not record (the size of his dataset, a hyperparameter he chose).
Each must be a real, self-contained question with a real answer (not nonsense, not a trick) and must not repeat a question already asked.

REPLY with one JSON object and nothing else (no prose, no markdown fence):
{{"questions": ["...", "...", "..."]}}"""


def _system(suite: str) -> str:
    if suite == "unanswerable":
        return _UNANSWERABLE_SYSTEM
    multi = suite == "multihop"
    return _SYSTEM.format(
        exception=(" (the one exception is the exact term the question is built around: keep that verbatim)"
                   if suite == "lexical" else ""),
        tier_rule=("Several passages: T3, or T4 when answering needs a follow-up search to resolve a reference."
                   if multi else
                   "One passage is T1 or T2; use T4 only when it points at something to look up elsewhere."),
        suite_rules=_SUITE_RULES[suite], reply=_REPLY)


def _feedback(problem: str, previous: str) -> str:
    return (f"\n\nYOUR PREVIOUS REPLY WAS REJECTED.\nProblem(s): {problem}\nPrevious reply:\n{previous[:1500]}\n"
            f"Fix every problem and reply again, with the JSON object only.")


# --- reading a reply ------------------------------------------------------------

_FENCE = re.compile(r"```[\w-]*[ \t]*\n?(.*?)```", re.DOTALL)


def _json_object(text: str) -> dict:
    """The reply as a dict, markdown fences stripped. ValueError (its text is the
    feedback sent back to the model) when it is not one JSON object."""
    body = text.strip()
    fence = _FENCE.search(body)
    if fence and not body.startswith("{"):
        body = fence.group(1).strip()
    try:
        d = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"the reply is not valid JSON ({e}); reply with ONE JSON object and nothing else") from None
    if not isinstance(d, dict):
        raise ValueError(f"the reply must be one JSON object, not a {type(d).__name__}")
    return d


def parse_reply(text: str) -> dict:
    """A grounded (single-seed or multihop) reply, shape-checked and normalised:
    {"skip": True, "reason"} or {"skip": False, question, nuggets, tier,
    expect_course, notes}. ValueError when the shape is wrong."""
    d = _json_object(text)
    if not isinstance(d.get("skip"), bool):
        raise ValueError('"skip" must be true or false')
    if d["skip"]:
        return {"skip": True, "reason": str(d.get("reason") or "no reason given").strip()}
    question, nuggets, course = d.get("question"), d.get("nuggets"), d.get("expect_course")
    if not isinstance(question, str) or not question.strip():
        raise ValueError('"question" must be a non-empty string')
    if not (isinstance(nuggets, list) and 1 <= len(nuggets) <= MAX_NUGGETS
            and all(isinstance(n, str) and n.strip() for n in nuggets)):
        raise ValueError(f'"nuggets" must be a list of 1 to {MAX_NUGGETS} non-empty strings')
    if d.get("tier") not in TIERS:
        raise ValueError(f'"tier" must be one of {", ".join(TIERS)}')
    if course is not None and not isinstance(course, str):
        raise ValueError('"expect_course" must be a string or null')
    question, nuggets = question.strip(), [n.strip() for n in nuggets]
    notes = str(d.get("notes") or "").strip()
    _no_control([question, *nuggets, notes])
    return {"skip": False, "question": question, "nuggets": nuggets,
            "tier": d["tier"], "expect_course": course, "notes": notes}


def parse_questions(text: str) -> list[str]:
    """An unanswerable reply: {"questions": [...]}."""
    qs = _json_object(text).get("questions")
    if not (isinstance(qs, list) and qs and all(isinstance(q, str) and q.strip() for q in qs)):
        raise ValueError('"questions" must be a non-empty list of question strings')
    qs = [q.strip() for q in qs]
    _no_control(qs)
    return qs


def _no_control(texts: list[str]) -> None:
    for t in texts:
        m = _CONTROL.search(t)
        if m:
            raise ValueError(
                f"a string holds the control character U+{ord(m.group()):04X}: JSON read a "
                f"backslash in your LaTeX as an escape (\\beta became a backspace). Escape every "
                f"backslash: write \\\\beta, \\\\theta, \\\\frac")


# --- seeds, gold, the suite file ------------------------------------------------

def load_pack(path) -> list[dict]:
    """The records of a seed pack, one JSON object per line as write_pack wrote it.
    Split on "\\n" only: splitlines() would also cut at U+2028 and friends, which
    a chunk's text can hold and ensure_ascii=False leaves raw."""
    path = Path(path)
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise DraftError(f"{path.name} line {i}: not valid JSON ({e})") from e
    return out


def _chunks_of(suite: str, rec: dict) -> list[dict]:
    """The seed chunks behind a pack record: itself, a multihop group's chunks,
    or none (a course neighbourhood has no chunk)."""
    if suite == "multihop":
        return rec["chunks"]
    return [] if suite == "unanswerable" else [rec]


def _label(rec: dict) -> str:
    return str(rec.get("doc_id") or rec.get("term") or rec.get("course"))


def _locator(meta: dict) -> dict:
    """pages when the chunk has them (a PDF), else its heading path, else nothing
    (file-level). `meta` is a seed record or chunk metadata: same key names."""
    start = meta.get("page_start")
    if start is not None:
        end = meta.get("page_end")
        return {"pages": [int(start), int(start if end is None else end)]}
    heading = meta.get("heading_path") or ""
    return {"heading": heading} if heading.strip() else {}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def _read_suite(path: Path) -> list[Question]:
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SchemaError(f"{path.name}: expected a YAML list of question records")
    return [question_from_dict(d, f"{path.name}[{i}]") for i, d in enumerate(raw)]


def _write_suite(path: Path, questions: list[Question]) -> None:
    """save_suite is atomic itself (a temp file, then a retried os.replace): a reader
    sees the old file or the new one, never half. A second temp layer here would
    only double the renames a Windows scanner can trip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_suite(path, questions)


class _SuiteFile:
    """<sets>/<suite>.yaml and the counters that follow from it. Ids and seeds are
    global (load_sets rejects a reused id in ANY file), so the numbering and the
    peers come from every file; rejected questions keep their ids but do not count
    toward the quota, as in quota_table."""

    def __init__(self, sets_dir: Path, suite: str, n: int):
        self.sets_dir, self.suite = Path(sets_dir), suite
        self.path = self.sets_dir / f"{suite}.yaml"
        self.prefix, self.n = PREFIX[suite], n
        self._count(load_sets(sets_dir, include_rejected=True))

    def _count(self, every: list[Question]) -> None:
        """The counters, from the sets as they are on disk. Draft ids stay below
        DRAFT_ID_LIMIT: a hand-written 0501 must not pull the drafts' numbering up."""
        stem = self.prefix + "-"
        nums = [int(q.id[len(stem):]) for q in every if q.id.startswith(stem)]
        self.next_num = 1 + max((x for x in nums if x < DRAFT_ID_LIMIT), default=0)
        self.peers = [q for q in every if q.suite == self.suite]       # rejected ones too
        self.live = sum(1 for q in self.peers if q.provenance.get("status") != "rejected")

    @property
    def full(self) -> bool:
        return self.live >= self.n

    def new_id(self) -> str:
        return f"{self.prefix}-{self.next_num:04d}"

    def append(self, q: Question) -> str | None:
        """Write q into the suite file, or say why not ("seed-taken", "suite-full").
        Under the sets lock, with everything re-read: another drafter or a console
        review may have written since this run began, so the id is numbered from the
        files as they are now (q.id is reassigned), a seed another question already
        uses is not drafted twice, the quota is counted afresh, and a review made in
        the meantime is not undone by rewriting a stale copy."""
        with sets_lock(self.sets_dir):
            every = load_sets(self.sets_dir, include_rejected=True)
            self._count(every)
            taken = {s for p in every for s in p.provenance.get("seed_chunks") or []}
            if taken & set(q.provenance.get("seed_chunks") or []):
                return "seed-taken"
            if self.full:
                return "suite-full"
            q.id = self.new_id()
            questions = _read_suite(self.path)
            questions.append(q)
            _write_suite(self.path, questions)
        self.next_num += 1
        self.live += 1
        self.peers.append(q)
        return None


# --- the search service ---------------------------------------------------------

def http_fetch(base_url: str) -> Fetch:
    """A Fetch over the warm query API (`rag serve`). A connection that cannot be
    made is SearchUnreachable; an HTTP error is a DraftError carrying the body."""
    import requests

    base = base_url.rstrip("/")

    def fetch(path: str, payload: dict | None = None) -> dict:
        try:
            r = (requests.get(base + path, timeout=HTTP_TIMEOUT) if payload is None
                 else requests.post(base + path, json=payload, timeout=HTTP_TIMEOUT))
        except requests.ConnectionError as e:        # refused, DNS, connect timeout: not a slow read
            raise SearchUnreachable(f"search service {base} is unreachable ({type(e).__name__}: {e})") from e
        if not r.ok:
            raise DraftError(f"{base}{path}: HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    return fetch


# --- the drafter ----------------------------------------------------------------

class Drafter:
    """One suite's run. `llm` needs `.complete(system, user)` returning an object
    with `.text` and `.model`, and `.model`; `fetch` is a Fetch; `gold_chunks` is
    load_gold_chunks' result for the pack's files, loaded once by the caller."""

    def __init__(self, suite: str, store: _SuiteFile, llm, fetch: Fetch, gold_chunks: dict, *,
                 max_score: float, max_calls: int | None = None,
                 sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = _say):
        self.suite, self.store, self.llm, self.fetch, self.gold_chunks = suite, store, llm, fetch, gold_chunks
        self.max_score, self.max_calls, self.sleep, self.log = max_score, max_calls, sleep, log
        self.system = _system(suite)
        self.calls = 0
        self.count: Counter = Counter()      # accepted, retries, twins, llm_errors
        self.skipped: Counter = Counter()    # reason -> seeds (candidates, for unanswerable)
        self.models: Counter = Counter()
        self.twins_off = False               # the search service was unreachable
        self.stopped: str | None = None      # why the run ended before the pack did

    # -- the loop --

    def run(self, records: list[dict]) -> None:
        handle = self._unanswerable if self.suite == "unanswerable" else self._grounded
        streak = 0
        for i, rec in enumerate(records, 1):
            if self.store.full:
                break
            self.log(f"[{i}/{len(records)}] {_label(rec)}")
            lost = self.skipped["llm-error"]
            try:
                handle(rec)
            except _CallCap:
                self.stopped = f"--max-calls {self.max_calls} reached"
                break
            streak = streak + 1 if self.skipped["llm-error"] > lost else 0
            if streak >= LLM_ERROR_STREAK:      # every seed now waits out the whole backoff for nothing
                self.stopped = (f"{streak} seeds in a row were lost to LLM errors: the provider looks "
                                f"down; run again once it answers (those seeds are asked again)")
                break

    def summary(self) -> str:
        skipped = ", ".join(f"{k} {v}" for k, v in sorted(self.skipped.items()))
        lines = [
            f"accepted {self.count['accepted']}; {self.suite} now holds {self.store.live} of {self.store.n}",
            f"skipped {sum(self.skipped.values())}" + (f": {skipped}" if skipped else ""),
            f"retries {self.count['retries']}, twins added {self.count['twins']}, "
            f"LLM calls {self.calls} ({self.count['llm_errors']} failed)",
            "model(s): " + (", ".join(f"{m} x{c}" for m, c in self.models.most_common()) or "none called"),
        ]
        if self.twins_off:
            lines.append("NOTE: the search service was unreachable, so twin detection did not run for some "
                         "or all seeds (the one fail-soft path); drafts may lack `alternatives`")
        if self.stopped:
            lines.append(f"stopped early: {self.stopped}")
        elif not self.store.full:
            lines.append("one pass over the courses is done; run again for another (kept questions are "
                         "not asked twice)" if self.suite == "unanswerable" else
                         "the pack is used up before the suite is full; draw more with `bench sample` "
                         "(it skips used seeds) and run again")
        return "\n".join(lines)

    # -- talking to the LLM and the search service --

    def _ask(self, system: str, user: str) -> tuple[str, str]:
        """(reply text, model id). Each try is one call against --max-calls."""
        for attempt in range(1, LLM_TRIES + 1):
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise _CallCap
            self.calls += 1
            try:
                resp = self.llm.complete(system, user)
            except Exception as e:      # noqa: BLE001 - counted, logged, then reported as the skip reason
                self.count["llm_errors"] += 1
                self.log(f"  LLM call failed ({attempt}/{LLM_TRIES}): {type(e).__name__}: {e}")
                if attempt == LLM_TRIES:
                    raise _GaveUp(f"{type(e).__name__}: {e}") from e
                self.sleep(BACKOFF_SECONDS * 2 ** (attempt - 1))
            else:
                model = resp.model or self.llm.model
                self.models[model] += 1
                return resp.text, model

    def _call(self, path: str, payload: dict | None = None) -> dict:
        res = self.fetch(path, payload)
        if res.get("error"):                  # /search reports a failure as 200 + {"error": ...}
            raise DraftError(f"{path}: {res['error']}")
        return res

    def _skip(self, kind: str, why: str) -> None:
        self.skipped[kind] += 1
        self.log(f"  skipped ({kind}): {why}")

    # -- single seeds and multihop groups --

    def _prompt(self, rec: dict, chunks: list[dict]) -> str:
        def passage(c: dict, title: str) -> str:
            loc = _locator(c)
            where = (f"pages {loc['pages'][0]}-{loc['pages'][1]}" if "pages" in loc
                     else f"heading {loc['heading']}" if "heading" in loc else "whole file")
            return (f"{title}\nSource file: {c['source_file']}\nKind: {c.get('file_type')}   "
                    f"Course: {c.get('course_name')}   Domain: {c.get('domain')}\n"
                    f"Location: {where}\nText:\n{c['text']}")

        if self.suite == "multihop":
            return (f"Shared heading term: {rec['term']}\n\n"
                    + "\n\n".join(passage(c, f"PASSAGE {i}") for i, c in enumerate(chunks, 1)))
        extra = f"\nCourse to name in the question: {chunks[0].get('course_name')}" if self.suite == "scoped" else ""
        return passage(chunks[0], "SEED PASSAGE") + extra

    def _grounded(self, rec: dict) -> None:
        chunks = _chunks_of(self.suite, rec)
        prompt = self._prompt(rec, chunks)
        twins: list[list[dict]] | None = None        # looked up once, and only for a seed the model takes
        feedback, kind, problem = "", "", ""
        for attempt in range(RETRIES + 1):
            try:
                text, model = self._ask(self.system, prompt + feedback)
            except _GaveUp as e:
                return self._skip("llm-error", str(e))
            self.count["retries"] += attempt           # counted once the re-ask was really sent
            try:
                reply = parse_reply(text)
            except ValueError as e:
                kind, problem = "malformed-reply", str(e)
            else:
                if reply["skip"]:
                    return self._skip("llm-skip", reply["reason"])
                if twins is None:
                    twins = [self._twins(c) for c in chunks]
                q, kind, problem = self._assemble(chunks, twins, reply, model)
                if q is not None:
                    return self._accept(q)
            if attempt < RETRIES:
                self.log(f"  rejected, asking again: {problem}")
                feedback = _feedback(problem, text)
        self._skip(kind, problem)

    def _assemble(self, chunks: list[dict], twins: list[list[dict]], reply: dict,
                  model: str) -> tuple[Question | None, str, str]:
        """(question, "", "") for a record that validates, else (None, kind, problem)."""
        cold = _NOT_STANDALONE.search(reply["question"])
        if cold:
            return None, "not-standalone", (
                f'the question says "{cold.group(0)}", but it is asked cold, with no passage in front '
                f"of the reader: name the book, course or notebook instead")
        course = chunks[0].get("course_name") or ""
        if self.suite == "scoped" and (reply["expect_course"] or "").strip().casefold() != course.strip().casefold():
            return None, "invalid:expect_course", (
                f'"expect_course" must be exactly {course!r}: the question has to name that course')
        gold, twin_chunks, notes = [], {}, [reply["notes"]]
        for chunk, found in zip(chunks, twins):
            entry = {"file": chunk["source_file"], **_locator(chunk), "required": True}
            if found:
                entry["alternatives"] = [t["gold"] for t in found]
                for t in found:
                    twin_chunks.setdefault(norm_file(t["gold"]["file"]), []).append(
                        {"meta": t["meta"], "text": t["text"]})
                notes.append("twin auto-detected: "
                             + "; ".join(f"{t['gold']['file']} ({t['jaccard']:.2f})" for t in found))
            gold.append(entry)
        # The rubric overrides the model where a suite fixes the tier by construction:
        # multihop needs several sources (T3) or a follow-up search (T4), and a paraphrase
        # question is never near its passage's wording (T1), so it is at least T2.
        tier = reply["tier"]
        if self.suite == "multihop":
            tier = "T4" if tier == "T4" else "T3"
        elif self.suite == "paraphrase" and tier == "T1":
            tier = "T2"
        if tier != reply["tier"]:     # the model's note may still argue for its own tier
            notes.append(f"(tier {reply['tier']} set to {tier} by the {self.suite} suite's rule)")
        record = {
            "id": self.store.new_id(), "question": reply["question"], "suite": self.suite,
            "tier": tier,
            "split": None, "answerable": True, "gold": gold, "nuggets": reply["nuggets"],
            "expect_course": course if self.suite == "scoped" else None,
            "provenance": {"author": "draft", "drafted_by": model,
                           "seed_chunks": [c["doc_id"] for c in chunks],
                           "status": "draft", "reviewed_at": None},
            "notes": " ".join(n for n in notes if n),
        }
        try:
            q = question_from_dict(record, "draft")
        except SchemaError as e:
            return None, "invalid:schema", str(e)
        # The real chunks win over a twin hit's text; validate() needs the twin's file
        # present, and the chunk pass only loaded the seed files.
        findings = [f for f in validate([q], {**twin_chunks, **self.gold_chunks})
                    if f.level == "error" or f.code in _BLOCKING_WARNINGS]
        if findings:
            return None, f"invalid:{findings[0].code}", "; ".join(f"{f.code}: {f.message}" for f in findings)
        return q, "", ""

    def _accept(self, q: Question) -> None:
        why = self.store.append(q)
        if why:                       # another writer got there first since this seed was read
            return self._skip(why, f"{q.question[:80]!r} not written: {why}")
        twins = sum(len(g.alternatives) for g in q.gold)
        self.count["accepted"] += 1
        self.count["twins"] += twins
        self.log(f"  accepted {q.id} ({q.tier}{f', {twins} twin(s)' if twins else ''})")

    def _twins(self, seed: dict) -> list[dict]:
        """Other files' chunks that hold the seed's passage: [{gold, meta, text,
        jaccard}], `gold` being the alternatives entry. [] when the search service
        is unreachable (warned once, then not asked again)."""
        if self.twins_off:
            return []
        try:
            res = self._call("/search", {
                "q": seed["text"][:TWIN_QUERY_CHARS], "hyde": False, "rerank": "none",
                "top_k": TWIN_TOP_K, "include_text": TWIN_TEXT_CHARS,
                # a hit must be one chunk's own text, so no small-to-big swap; the gate
                # would refuse rerank=none and has nothing to say about twins
                "parent_context": False, "neighbor_context": False, "gate": False})
        except SearchUnreachable as e:
            self.twins_off = True
            self.log(f"WARNING: {e}; drafting WITHOUT twin detection from here on")
            return []
        mine, seen, out = set(_words(seed["text"])), set(), []
        for hit in res["results"]:
            if hit.get("live") or not hit.get("lookup_available", True) or hit["origin_id"] == seed["doc_id"]:
                continue
            text = hit.get("text") or ""
            jaccard = _jaccard(mine, set(_words(text)))
            if jaccard < TWIN_JACCARD:
                continue                                  # not the same passage: no lookup spent on it
            meta = self._call(f"/chunks/{quote(hit['origin_id'], safe='')}?include_text=0")["metadata"]
            file = meta.get("source_file") or ""
            if norm_file(file) == norm_file(seed["source_file"]):
                continue                                  # the same file is not a twin
            locator = _locator(meta)
            key = (norm_file(file), json.dumps(locator, sort_keys=True))
            if key not in seen:                               # two hits can be one twin locator
                seen.add(key)
                out.append({"gold": {"file": file, **locator}, "meta": meta, "text": text, "jaccard": jaccard})
        return out

    # -- unanswerable --

    def _unanswerable(self, rec: dict) -> None:
        course = rec["course"]
        asked = [q.question for q in self.store.peers if q.expect_course == course]
        prompt = (f"Course: {course} ({rec['n_chunks']} chunks)\nHeadings the notes cover:\n"
                  + "\n".join(f"- {h}" for h in rec["headings"])
                  + ("\n\nAlready asked, do not repeat:\n" + "\n".join(f"- {q}" for q in asked) if asked else ""))
        feedback = ""
        for attempt in range(RETRIES + 1):
            try:
                text, model = self._ask(self.system, prompt + feedback)
            except _GaveUp as e:
                return self._skip("llm-error", str(e))
            self.count["retries"] += attempt
            try:
                candidates = parse_questions(text)
                break
            except ValueError as e:
                if attempt == RETRIES:
                    return self._skip("malformed-reply", str(e))
                self.log(f"  rejected, asking again: {e}")
                feedback = _feedback(str(e), text)
        for question in candidates[:UNANSWERABLE_ASKED]:
            if self.store.full:
                break
            self._abstain(course, question, model)

    def _abstain(self, course: str, question: str, model: str) -> None:
        try:
            q = question_from_dict({
                "id": self.store.new_id(), "question": question, "suite": self.suite,
                "tier": UNANSWERABLE_TIER, "split": None, "answerable": False, "gold": [],
                "nuggets": [], "expect_course": course,
                "provenance": {"author": "draft", "drafted_by": model, "seed_chunks": [],
                               "status": "draft", "reviewed_at": None},
                "notes": ""}, "draft")
        except SchemaError as e:
            return self._skip("invalid:schema", str(e))
        # The validator's own near-duplicate rule, against everything already asked
        # (rejected questions included: the author threw those away on purpose).
        dup = [f for f in validate([*self.store.peers, q], {}) if f.qid == q.id and f.code == "near-duplicate"]
        if dup:
            return self._skip("near-duplicate", f"{dup[0].message}: {question}")
        hits = self._call("/search", {"q": question, "hyde": False, "top_k": ABSTAIN_TOP_K,
                                      "include_text": ABSTAIN_TEXT_CHARS, "gate": False})["results"]
        scored = [h for h in hits if h.get("score") is not None]
        if hits and not scored:
            raise DraftError("/search returned hits without scores; the unanswerable check needs a "
                             "scoring reranker (retrieval.rerank_mode)")
        if scored:
            top = max(scored, key=lambda h: h["score"])
            if top["score"] >= self.max_score:
                return self._skip("abstain-check", f"top hit {top['label']!r} scored {top['score']:.2f}, "
                                                   f"not below {self.max_score}: {question}")
            q.notes = (f"abstention check: top hit {top['label']!r} scored {top['score']:.2f} "
                       f"(kept below {self.max_score})")
        else:
            q.notes = "abstention check: /search returned no hits"
        self._accept(q)


def draft_suite(suite: str, llm, fetch: Fetch, *, seeds_path, sets_dir, data_dir, n: int,
                max_score: float, max_calls: int | None = None,
                sleep: Callable[[float], None] = time.sleep,
                log: Callable[[str], None] = _say) -> Drafter:
    """Draft `suite` until its file holds `n` live questions, the pack is used up
    or --max-calls is reached. Returns the Drafter; print its summary()."""
    records = load_pack(seeds_path)
    store = _SuiteFile(sets_dir, suite, n)
    used = used_seed_ids(sets_dir)
    todo = [r for r in records if not used.intersection(c["doc_id"] for c in _chunks_of(suite, r))]
    log(f"{suite}: the file holds {store.live} of {n}; {len(todo)} of {len(records)} seeds are unused")
    gold_chunks: dict = {}
    if todo and not store.full:
        files = {c["source_file"] for r in todo for c in _chunks_of(suite, r) if c.get("source_file")}
        if files:
            log(f"reading the chunk files for {len(files)} seed file(s)...")
        gold_chunks = load_gold_chunks(Path(data_dir), files)
    drafter = Drafter(suite, store, llm, fetch, gold_chunks, max_score=max_score,
                      max_calls=max_calls, sleep=sleep, log=log)
    drafter.run(todo)
    return drafter
