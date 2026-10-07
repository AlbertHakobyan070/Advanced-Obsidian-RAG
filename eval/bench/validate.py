"""
validate.py — does a question set agree with the corpus it points into?

`bench validate` runs these checks over hand-edited YAML, so the two levels are
the whole design:

  error  the question cannot be scored as written. A gold file that is not in
         the corpus, or a locator (pages / heading) that no chunk of that file
         satisfies, scores 0 for EVERY configuration and so masquerades as a
         retrieval failure. The CLI exits 1 on any error.
  warn   the question can be scored but may not measure retrieval well: a
         nugget the gold text does not back up, a question lifted from the gold
         text, a near-duplicate. These are lexical heuristics — they flag, a
         human decides.

Gold paths are compared through relevance.norm_file, so '\\' vs '/', letter
case and Unicode form all collapse into one file; an en dash vs a hyphen does
NOT collapse — that is a different path, and it is reported as missing.

A question none of whose gold entries matched a chunk has no gold text to judge
against, so the nugget and copied-phrasing checks are skipped for it: the errors
already say what to fix, and "every nugget unsupported" would bury them.

Chunk files are read with the same streaming reader the index builders use
(chunk text contains U+2028 and friends, and the biggest file is hundreds of MB).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from eval.bench.questions import SUITES, Question
from eval.bench.relevance import chunk_matches, norm_file
from src.embeddings.embedder import iter_jsonl_records

# Rule constants. They define what the checks MEAN (spec §3), so they live next
# to the rules rather than in config.yaml, like the suite quotas in questions.py.
NUGGET_SUPPORT = 0.6        # a nugget needs >= 60% of its content words in the gold text
COPY_RUN = 5                # this many consecutive shared words is copying, not overlap
DUPLICATE_JACCARD = 0.8     # word-set Jaccard at or above this is a near-duplicate
# What the review cache keeps per gold entry: enough text to judge a question by,
# not a book.
CACHE_CHUNKS_PER_GOLD = 3
CACHE_TEXT_CHARS = 1500

# \w+ is Unicode-aware, so a Greek or Armenian nugget is judged like an English
# one instead of reading as "no words". The stoplist is short on purpose: it only
# has to stop "the/and/with" from counting as evidence that a nugget is supported.
_WORD = re.compile(r"\w+")
# Numbers, compared by value so 0.2 matches 0.20 and 1,000 matches 1000; a comma
# counts only between groups of three digits ("1,2,3" is three numbers). A nugget's
# side is strict: digits inside a word are part of a name ("k1", "bge-v2", "T4"). The
# gold side is lenient, so a number the text does state is found even where OCR glued
# it to a word ("2.1Visualize") or a table dropped its zero (".20"), both seen live.
_NUMBER = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w)")
_ANY_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d*\.\d+|\d+")
_STOPWORDS = frozenset(
    "the and for are was were with that this from into which what when where how why who "
    "its not but can will would should could has have had does did than then there their "
    "they them these those been being also such each any all over under about between "
    "using used use via per our your you".split())


@dataclass
class Finding:
    qid: str
    level: str          # "error" | "warn"
    code: str
    message: str


def _words(text: str) -> list[str]:
    # casefold, not lower: final sigma and sharp s compare equal across spellings
    return _WORD.findall(text.casefold())


def _content_words(text: str) -> set[str]:
    return {w for w in _words(text) if len(w) >= 3 and w not in _STOPWORDS}


def load_gold_chunks(data_dir: Path, files: set[str]) -> dict[str, list[dict]]:
    """Normalised file -> [{"meta": ..., "text": ...}] for the chunks of `files`
    only, streamed out of every *chunks.jsonl in `data_dir`. A file with no chunk
    in the corpus gets no key, which is how `validate` knows it is missing."""
    wanted = {norm_file(f) for f in files}
    out: dict[str, list[dict]] = {}
    if not wanted:                       # nothing to look for: do not stream hundreds of MB
        return out
    for path in sorted(Path(data_dir).glob("*chunks.jsonl")):
        for rec in iter_jsonl_records(path):
            meta = rec.get("metadata") or {}
            key = norm_file(meta.get("source_file") or "")
            if key in wanted:
                out.setdefault(key, []).append({"meta": meta, "text": rec.get("text") or ""})
    return out


def _locator(g) -> str:
    parts = []
    if g.pages is not None:
        parts.append(f"pages {g.pages[0]}-{g.pages[1]}")
    if g.heading:
        parts.append(f"heading {g.heading!r}")
    return " and ".join(parts)


def _shared_run(question_words: list[str], chunk_words: list[list[str]]) -> str | None:
    """The first COPY_RUN-word run the question shares with any chunk, or None."""
    n = COPY_RUN
    runs = {tuple(question_words[i:i + n]) for i in range(len(question_words) - n + 1)}
    if not runs:
        return None
    for words in chunk_words:
        for i in range(len(words) - n + 1):
            if tuple(words[i:i + n]) in runs:
                return " ".join(words[i:i + n])
    return None


def _numbers(text: str, pattern=_NUMBER) -> list[tuple[str, float]]:
    return [(tok, float(tok.replace(",", ""))) for tok in pattern.findall(text)]


def _check_text(q: Question, matched: list[dict]) -> list[Finding]:
    out: list[Finding] = []
    chunk_words = [_words(c["text"]) for c in matched]
    vocabulary = set().union(*chunk_words)
    gold_numbers = {v for c in matched for _, v in _numbers(c["text"], _ANY_NUMBER)}
    for nugget in q.nuggets:
        want = _content_words(nugget)
        # A nugget with no content words (all tokens under three characters,
        # e.g. "k1 = 1.2") has nothing lexical to check: not judged.
        if want and len(want & vocabulary) / len(want) < NUGGET_SUPPORT:
            out.append(Finding(q.id, "warn", "nugget-unsupported",
                               f"only {len(want & vocabulary)} of {len(want)} content words of "
                               f"nugget {nugget!r} appear in the gold text"))
        # Words can be paraphrased, numbers cannot: a number the gold text never
        # states was invented (seen live: a drafted sum of 92 where the page says 273).
        invented = [tok for tok, v in _numbers(nugget) if v not in gold_numbers]
        if invented:
            out.append(Finding(q.id, "warn", "nugget-number-unsupported",
                               f"nugget {nugget!r} states {', '.join(invented)}, which no gold "
                               f"chunk does"))
    # `lexical` is exempt: its point is that the question carries the exact term.
    if q.suite != "lexical":
        phrase = _shared_run(_words(q.question), chunk_words)
        if phrase:
            out.append(Finding(q.id, "warn", "copied-phrasing",
                               f"the question repeats {phrase!r} from the gold text "
                               f"(a run of {COPY_RUN}+ words); paraphrase it"))
    return out


def _near_duplicates(questions: list[Question]) -> list[Finding]:
    sets = [set(_words(q.question)) for q in questions]
    out: list[Finding] = []
    for j in range(len(questions)):
        for i in range(j):
            a, b = sets[i], sets[j]
            if a and b and len(a & b) / len(a | b) >= DUPLICATE_JACCARD:
                # attach to the later question: one finding per pair
                out.append(Finding(questions[j].id, "warn", "near-duplicate",
                                   f"near-duplicate of {questions[i].id} "
                                   f"(word overlap {len(a & b) / len(a | b):.0%})"))
    return out


def validate(questions: list[Question], gold_chunks: dict[str, list[dict]]) -> list[Finding]:
    """Findings for `questions`, in question order (then gold order, then rule
    order), with the near-duplicate pairs last."""
    findings: list[Finding] = []
    for q in questions:
        matched: list[dict] = []
        for g in q.gold:
            for s in g.sources():            # the entry itself, then each twin
                kind = "gold" if s is g else "alternative"
                chunks = gold_chunks.get(norm_file(s.file))
                if not chunks:
                    findings.append(Finding(q.id, "error", "gold-file-missing",
                                            f"{kind} file {s.file!r} is not in the corpus"))
                    continue
                hits = [c for c in chunks if chunk_matches(c["meta"], s)]
                if not hits:
                    findings.append(Finding(q.id, "error", "locator-empty",
                                            f"no chunk of {s.file!r} satisfies {_locator(s)}"))
                    continue
                matched += hits
        if matched:
            findings += _check_text(q, matched)
    findings += _near_duplicates(questions)
    return findings


def quota_table(questions: list[Question]) -> list[tuple[str, int, int]]:
    """(suite, have, target) for every suite, in SUITES order. Counts the
    questions it is given; load_sets has already dropped the rejected ones."""
    have = Counter(q.suite for q in questions)
    return [(suite, have[suite], target) for suite, target in SUITES.items()]


def write_review_cache(path, questions: list[Question], gold_chunks: dict[str, list[dict]],
                       findings: list[Finding]) -> None:
    """Write {qid: {"gold_texts": [...], "findings": [...]}} for the review queue.

    One gold_texts entry per matching chunk (at most CACHE_CHUNKS_PER_GOLD per
    gold entry, text cut to CACHE_TEXT_CHARS). Its file / pages / heading are the
    gold entry's own locator as written in the record, not the chunk's: that is
    the key the review UI joins a gold source to its texts on.

    The write is atomic — a temp file in the SAME directory, then os.replace — so
    a console reading the cache while `bench validate` rewrites it sees the old
    file or the new one, never half of either; a failure leaves the old file in
    place and removes the temp.
    """
    by_qid: dict[str, list[dict]] = {}
    for f in findings:
        by_qid.setdefault(f.qid, []).append({"level": f.level, "code": f.code, "message": f.message})
    cache = {}
    for q in questions:
        texts = []
        for s in (s for g in q.gold for s in g.sources()):
            hits = [c for c in gold_chunks.get(norm_file(s.file), []) if chunk_matches(c["meta"], s)]
            for c in hits[:CACHE_CHUNKS_PER_GOLD]:
                texts.append({"file": s.file, "pages": None if s.pages is None else list(s.pages),
                              "heading": s.heading, "text": c["text"][:CACHE_TEXT_CHARS]})
        cache[q.id] = {"gold_texts": texts, "findings": by_qid.get(q.id, [])}

    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
