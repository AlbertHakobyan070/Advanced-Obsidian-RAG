"""
citations.py — the ONE definition of how an answer's citation markers and its
confidence line are written, and therefore how they are read back.

WHY THIS MODULE EXISTS
  These patterns lived in two places: `src/generation/generator.py`, which
  parses a fresh answer, and `eval/metrics.py`, which scores a stored one. The
  same regex, copy-pasted. On 2026-09-17 a model started emitting fullwidth
  CJK citation brackets and a bolded confidence label, and both copies broke —
  but a fix applied to one of them would have left the eval suite quietly
  scoring correctly-cited answers as uncited, which is worse than both being
  broken. One definition, two importers.

  It lives in `src/utils/` and imports nothing, because `eval/metrics.py` is
  deliberately dependency-light: it is the LLM-free half of the eval suite and
  must not drag in the LLM client, the prompt loader or the retriever just to
  recognise a bracket.

WHAT MODELS ACTUALLY EMIT
  The generation prompt asks for `[1]` and a bare `CONFIDENCE: HIGH` line, and
  now says so more forcefully. Models still improvise. Observed live from the
  freellmapi proxy:

      **CONFIDENCE:** HIGH      -> confidence read as UNKNOWN
      【1】【2】【3】            -> zero citations; every source `cited: false`

  Both failures were silent: a correct, fully grounded answer scored UNKNOWN
  with no citations. So the parsers tolerate the decoration — but only around
  an otherwise exact label or number. Prose that merely contains the word
  "confidence" must never be mistaken for a confidence line, and the tests
  pin that limit as hard as they pin the tolerance.
"""
from __future__ import annotations

import re

# Markdown decoration a model may wrap a label in: **bold**, _italic_,
# `code`, ### heading.
_MARK = r"[*_`#]{0,4}"

# The CONFIDENCE line the generation prompt asks for, tolerant of that
# decoration and of a fullwidth colon (a model that switches scripts for its
# brackets tends to switch for its punctuation too).
#
# The trailing \b is load-bearing: without it "My confidence: lower bounds are
# covered in lecture 4" parses as LOW.
CONFIDENCE_RE = re.compile(
    rf"{_MARK}\s*CONFIDENCE\s*{_MARK}\s*[:：]\s*{_MARK}\s*"
    rf"(HIGH|MEDIUM|LOW)\b{_MARK}",
    re.IGNORECASE,
)

# Citation bracket pairs: ASCII, fullwidth, CJK lenticular, tortoise-shell.
# Openers and closers are separate character classes on purpose — a model that
# switches scripts mid-answer sometimes closes an ASCII bracket with a
# fullwidth one.
#
# Deliberately NOT matched: grouped markers like `[1, 2]`. Supporting them
# would also match numpy-style indexing `a[1, 2]`, which is ordinary content
# in this corpus, and the prompt asks for `[1], [2]` anyway.
CITATION_RE = re.compile(r"[\[［【〔](\d{1,3})[\]］】〕]")

# A fragment that is nothing but citation markers belongs to the sentence
# BEFORE it ("...gradient clipping. [1][2]"). Same bracket families.
ONLY_CITATIONS_RE = re.compile(
    r"^(?:\s*[\[［【〔]\d{1,3}[\]］】〕]\s*)+[.,;]?$")
