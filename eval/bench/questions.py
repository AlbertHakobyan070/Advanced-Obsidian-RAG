"""
questions.py — the v2 question-set schema, its YAML files, and the dev/test split.

Three decisions live here, each for a reason:

  Gold is a LOCATOR, never a chunk id. A GoldSource is a vault-relative file
  plus an optional page range and/or heading prefix. Chunk ids change whenever
  the corpus is re-chunked or the embedding model is swapped — exactly the
  components this eval exists to ablate — while a locator survives both. What
  counts as a match is decided in relevance.py.

  The split is assigned ONCE and stored in the YAML. Recomputing it per run
  would reshuffle existing questions whenever one is added, and a "sealed"
  test split that moves is not sealed. assign_splits only ever fills in
  questions that have no split yet.

  Validation is strict and loud. These files are hand-edited (the review
  queue, the author's own questions), so a bad record raises SchemaError naming the
  file, the question id and the problem instead of quietly scoring as zero. A
  key the schema does not define is such a problem too: a misspelt `heding:` on
  a gold entry would otherwise load as a file-only locator and inflate every
  retrieval metric. Only the schema is checked here; whether a gold file
  actually exists in the corpus is a corpus question, answered by `bench validate`.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import yaml

# Target number of questions per suite; they sum to 400 (spec §3.1, minus
# `multilingual`, dropped 2026-10-05: no quality demand). A quota is a target
# the validator reports against, not something load_sets enforces; a record
# naming a suite that is not listed here is a SchemaError.
SUITES: dict[str, int] = {
    "lexical": 40, "paraphrase": 50, "code": 40, "scoped": 40, "books": 50,
    "multihop": 55, "canvas": 40, "personal": 40, "unanswerable": 45,
}
TIERS = ("T1", "T2", "T3", "T4")
SPLITS = ("dev", "test")
STATUSES = ("draft", "verified", "edited", "rejected")
AUTHORS = ("draft", "owner")

# fullmatch, not `^...$`: `$` also matches before a trailing newline, and \d
# also matches non-ASCII digits.
_ID_RE = re.compile(r"[a-z]+-[0-9]{4}")

# The keys a record may carry, level by level (spec §3.2). Anything else is a
# SchemaError. An alternative carries no `required` (it is the same passage as
# its gold entry, so it is required exactly when that entry is) and no
# alternatives of its own.
_TOP_KEYS = ("id", "question", "suite", "tier", "split", "answerable", "gold", "nuggets",
             "expect_course", "provenance", "notes")
_GOLD_KEYS = ("file", "pages", "heading", "required", "alternatives")
_ALT_KEYS = ("file", "pages", "heading")
_PROV_KEYS = ("author", "drafted_by", "seed_chunks", "status", "reviewed_at")


class SchemaError(ValueError):
    """A question record that breaks the schema."""


@dataclass(frozen=True)
class GoldSource:
    file: str                              # vault-relative, as in chunk metadata
    pages: tuple[int, int] | None = None   # inclusive PDF page range
    heading: str | None = None             # heading_path prefix
    required: bool = True                  # multihop: every required source must be found
    # Other files holding the SAME passage — a slide PDF and its .md study-note copy,
    # a book stored twice. A chunk from any of them satisfies this entry; without
    # them, retrieving the twin would score as a miss. Each carries its own locator;
    # an alternative has no alternatives of its own.
    alternatives: tuple["GoldSource", ...] = ()

    def sources(self) -> tuple["GoldSource", ...]:
        return (self, *self.alternatives)


@dataclass
class Question:
    id: str
    question: str
    suite: str
    tier: str
    split: str | None
    answerable: bool
    gold: list[GoldSource]
    nuggets: list[str]
    expect_course: str | None
    provenance: dict
    notes: str = ""


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)    # True is not page 1


def _pages(raw, err) -> tuple[int, int]:
    if _is_int(raw):
        start = end = raw
    elif isinstance(raw, (list, tuple)) and len(raw) == 2 and all(_is_int(x) for x in raw):
        start, end = raw
    else:
        raise err(f"pages must be an int or [start, end], got {raw!r}")
    if not 1 <= start <= end:
        raise err(f"pages {raw!r}: need 1 <= start <= end")
    return (start, end)


def _only_keys(d: dict, allowed: tuple[str, ...], what: str, err) -> None:
    unknown = [k for k in d if k not in allowed]
    if unknown:
        raise err(f"unknown key(s) in {what}: {', '.join(repr(k) for k in unknown)} "
                  f"(allowed: {', '.join(allowed)})")


def _gold_from_dict(g, err, nested: bool = False) -> GoldSource:
    if not isinstance(g, dict) or not isinstance(g.get("file"), str) or not g["file"].strip():
        raise err(f"every gold entry needs a 'file', got {g!r}")
    if nested and "alternatives" in g:      # said plainly, before the generic unknown-key error
        raise err("an alternative cannot have alternatives of its own")
    _only_keys(g, _ALT_KEYS if nested else _GOLD_KEYS,
               "an alternative" if nested else "a gold entry", err)
    pages = g.get("pages")
    heading = g.get("heading")
    required = g.get("required", True)
    alts = g.get("alternatives") or []
    if heading is not None and not isinstance(heading, str):
        raise err(f"gold heading must be a string, got {heading!r}")
    if not isinstance(required, bool):
        raise err(f"gold 'required' must be true or false, got {required!r}")
    if not isinstance(alts, list):
        raise err(f"gold alternatives must be a list, got {alts!r}")
    return GoldSource(file=g["file"], pages=None if pages is None else _pages(pages, err),
                      heading=heading, required=required,
                      alternatives=tuple(_gold_from_dict(a, err, nested=True) for a in alts))


def question_from_dict(d: dict, where: str) -> Question:
    """Validate one record and build the Question; `where` (a file name, a
    test label) prefixes every error so a bad record can be found."""
    if not isinstance(d, dict):
        raise SchemaError(f"{where}: expected a mapping, got {type(d).__name__}")
    qid = d.get("id")

    def err(problem: str) -> SchemaError:
        return SchemaError(f"{where}: {qid}: {problem}")

    if not isinstance(qid, str) or not _ID_RE.fullmatch(qid):
        raise err("id must be lowercase letters, '-' and four digits, like 'lex-0001'")
    _only_keys(d, _TOP_KEYS, "the record", err)       # before `missing`: a typo'd name reads as one
    missing = [k for k in ("question", "suite", "tier", "answerable", "provenance") if k not in d]
    if missing:
        raise err(f"missing key(s): {', '.join(missing)}")
    if not isinstance(d["question"], str) or not d["question"].strip():
        raise err("question must be a non-empty string")
    # `in tuple(...)`, not `in dict`: an unhashable value (a YAML list) must be
    # a SchemaError, not a TypeError.
    if d["suite"] not in tuple(SUITES):
        raise err(f"suite {d['suite']!r} is not one of {', '.join(SUITES)}")
    if d["tier"] not in TIERS:
        raise err(f"tier {d['tier']!r} is not one of {', '.join(TIERS)}")
    split = d.get("split")
    if split is not None and split not in SPLITS:
        raise err(f"split {split!r} is not one of {', '.join(SPLITS)} (or null)")
    answerable = d["answerable"]
    if not isinstance(answerable, bool):
        raise err(f"answerable must be true or false, got {answerable!r}")

    raw_gold = d.get("gold") or []
    raw_nuggets = d.get("nuggets") or []
    if not isinstance(raw_gold, list):
        raise err("gold must be a list")
    if not isinstance(raw_nuggets, list) or not all(isinstance(n, str) and n.strip() for n in raw_nuggets):
        raise err("nuggets must be a list of non-empty strings")
    gold = [_gold_from_dict(g, err) for g in raw_gold]
    if answerable and not gold:
        raise err("an answerable question needs at least one gold source")
    if answerable and not raw_nuggets:
        raise err("an answerable question needs at least one nugget")
    if not answerable and gold:
        raise err("an unanswerable question must have no gold sources (abstaining is the right answer)")

    prov = d["provenance"]
    if not isinstance(prov, dict):
        raise err("provenance must be a mapping")
    _only_keys(prov, _PROV_KEYS, "provenance", err)
    if prov.get("author") not in AUTHORS:
        raise err(f"provenance.author {prov.get('author')!r} is not one of {', '.join(AUTHORS)}")
    if prov.get("status") not in STATUSES:
        raise err(f"provenance.status {prov.get('status')!r} is not one of {', '.join(STATUSES)}")

    expect_course, notes = d.get("expect_course"), d.get("notes")
    if expect_course is not None and not isinstance(expect_course, str):
        raise err(f"expect_course must be a string or null, got {expect_course!r}")
    if notes is not None and not isinstance(notes, str):
        raise err(f"notes must be a string or null, got {notes!r}")

    return Question(
        id=qid, question=d["question"], suite=d["suite"], tier=d["tier"], split=split,
        answerable=answerable, gold=gold, nuggets=list(raw_nuggets),
        expect_course=expect_course, provenance=dict(prov), notes=notes or "")


def _gold_to_dict(g: GoldSource) -> dict:
    out: dict = {"file": g.file}
    if g.pages is not None:
        out["pages"] = list(g.pages)           # safe_dump cannot write a tuple
    if g.heading is not None:
        out["heading"] = g.heading
    out["required"] = g.required
    if g.alternatives:                         # written only when present: most entries have none
        out["alternatives"] = [{k: v for k, v in _gold_to_dict(a).items() if k != "required"}
                               for a in g.alternatives]
    return out


def question_to_dict(q: Question) -> dict:
    """Plain dict in schema order (spec §3.2), ready for yaml/json."""
    return {
        "id": q.id, "question": q.question, "suite": q.suite, "tier": q.tier,
        "split": q.split, "answerable": q.answerable,
        "gold": [_gold_to_dict(g) for g in q.gold], "nuggets": list(q.nuggets),
        "expect_course": q.expect_course, "provenance": dict(q.provenance), "notes": q.notes,
    }


def load_sets(sets_dir, suites=None, include_rejected: bool = False) -> list[Question]:
    """Every question in sets_dir/*.yaml, files in sorted order, records in file order.

    All files are read and every id checked for duplicates before anything is
    filtered, so a reused id is an error even when its suite was not asked for
    (ids are never reused). `suites` then keeps the questions whose own `suite`
    is listed; rejected questions are dropped unless include_rejected.
    """
    questions: list[Question] = []
    first_seen: dict[str, str] = {}
    for path in sorted(Path(sets_dir).glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise SchemaError(f"{path.name}: expected a YAML list of question records")
        for i, d in enumerate(raw):
            where = f"{path.name}[{i}]"
            q = question_from_dict(d, where)
            if q.id in first_seen:
                raise SchemaError(f"{where}: {q.id}: duplicate id (first seen at {first_seen[q.id]})")
            first_seen[q.id] = where
            questions.append(q)
    wanted = None if suites is None else set(suites)
    return [q for q in questions
            if (wanted is None or q.suite in wanted)
            and (include_rejected or q.provenance.get("status") != "rejected")]


def save_suite(path, questions: list[Question]) -> None:
    """Write the suite whole or not at all. The text goes to a temp file beside
    the target, then os.replace swaps it in: a crash or a full disk mid-write
    leaves the old file as it was instead of a truncated one. The temp name does
    not end in .yaml, so load_sets never reads it."""
    path = Path(path)
    text = yaml.safe_dump([question_to_dict(q) for q in questions],
                          sort_keys=False, allow_unicode=True, width=100)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        _retry_locked(os.replace, tmp, path)
    finally:
        tmp.unlink(missing_ok=True)           # a no-op once the replace has happened


# Windows lets a virus scanner, indexer or sync client hold a just-written file open
# for a moment, and os.replace (or an unlink) then fails with PermissionError although
# nothing is wrong (seen live: `bench draft` lost its lexical run to WinError 5). The
# operation is retried briefly; a file still locked after that raises as before.
_REPLACE_TRIES = 6
_REPLACE_WAIT = 0.1              # seconds before the 2nd try; doubles each time (~3 s in all)


def _retry_locked(op, *args):
    for attempt in range(_REPLACE_TRIES):
        try:
            return op(*args)
        except PermissionError:
            if attempt == _REPLACE_TRIES - 1:
                raise
            time.sleep(_REPLACE_WAIT * 2 ** attempt)


# One writer of the question sets at a time, across processes. The console's review
# save, a `bench draft` append and `bench split` each re-read, change and rewrite a
# sets file: two of them interleaving lose one's write, and two drafters on one suite
# number the same id (seen live). The lock is a file created with O_EXCL and held for
# one read-modify-write (well under a second), so one older than _LOCK_STALE was left
# by a writer that died holding it.
_LOCK_NAME = ".sets.lock"
_LOCK_STALE = 30.0               # seconds
_LOCK_TIMEOUT = 60.0             # seconds to wait for a live holder


@contextmanager
def sets_lock(sets_dir):
    path = Path(sets_dir) / _LOCK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + _LOCK_TIMEOUT
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):     # Windows: a lock mid-delete is EACCES
            try:
                age = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue                                 # released between the two calls
            if age > _LOCK_STALE:
                # ponytail: two writers finding the same dead lock can both take it;
                # fine for a few local writers, an OS file lock if that ever matters
                _retry_locked(path.unlink, True)
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"could not take {path} within {_LOCK_TIMEOUT:.0f} s: another writer of "
                    f"the question sets holds it. Delete the file if no bench command or "
                    f"console save is running.")
            time.sleep(0.05)
            continue
        try:
            os.write(fd, str(os.getpid()).encode())
        finally:
            os.close(fd)
        break
    try:
        yield
    finally:
        _retry_locked(path.unlink, True)


def assign_splits(questions: list[Question]) -> list[Question]:
    """Fill in a split for every question that has none; mutates and returns the list.

    Greedy balance within each suite x tier stratum: questions are taken in id
    order and each goes to whichever split has fewer members in its stratum
    (existing, already-stored splits count). A tie is broken by sha256(id), so
    the outcome depends only on the ids, never on file order or the run date.
    Questions that already have a split are never touched: adding questions
    cannot reshuffle the old ones.
    """
    counts = Counter((q.suite, q.tier, q.split) for q in questions if q.split)
    for q in sorted((q for q in questions if not q.split), key=lambda q: q.id):
        dev, test = counts[(q.suite, q.tier, "dev")], counts[(q.suite, q.tier, "test")]
        if dev != test:
            q.split = "dev" if dev < test else "test"
        else:
            q.split = SPLITS[int(hashlib.sha256(q.id.encode("utf-8")).hexdigest(), 16) % 2]
        counts[(q.suite, q.tier, q.split)] += 1
    return questions
