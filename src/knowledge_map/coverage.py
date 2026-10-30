"""Coverage: which of the chapter's headings are short of kept nodes.

A heading's span is the normalised chapter text from the heading to the next one (the
last runs to the end). A node belongs to a heading when one of its located quotes starts
inside that span. A heading is short when it holds fewer than ceil(span chars /
chars_per_node) nodes, or when none of its nodes is named after it (its own idea is
missing). Pure Python.
"""
import math
import re
from dataclasses import dataclass

from .graph import Node
from .locate import PageText, norm
from .scout import Heading

# Words a heading's idea is not made of: function words, and headings that name no idea.
_SKIP = {"a", "an", "and", "the", "of", "to", "for", "from", "in", "on", "with", "using", "vs", "other",
         "summary", "introduction", "conclusion", "overview", "exercises", "references"}


@dataclass(frozen=True)
class HeadingSpan:
    title: str
    page: int
    start: int          # offsets into PageText.text
    end: int


@dataclass(frozen=True)
class Gap:
    span: HeadingSpan
    have: list[str]     # titles of the nodes already under the heading
    need: int
    named: bool         # one of `have` is named after the heading

    def line(self) -> str:
        """How a gap-fill prompt names it."""
        have = f"already in the map: {'; '.join(self.have)}" if self.have else "nothing in the map yet"
        return f"{self.span.title} ({have}" + ("" if self.named else "; still missing: a node for this heading's own idea") + ")"

    @property
    def target(self) -> int:
        """Nodes to ask for: what the length still needs, and at least one when the heading's own idea is missing."""
        return max(self.need - len(self.have), 0 if self.named else 1)


def _stems(s: str) -> set[str]:
    words = [w for w in re.findall(r"[^\W_]+", s.casefold()) if len(w) > 2 and w not in _SKIP]
    return {(w[:-1] if w.endswith("s") and len(w) > 3 else w)[:5] for w in words}


def names(heading: str, title: str) -> bool:
    """A node title names a heading's idea when it holds every significant word of the heading
    (compared by a 5-letter stem, so "Extract" matches "extraction"). A heading with no
    significant word ("Summary") names no idea and is always named."""
    return _stems(heading) <= _stems(title)


def heading_spans(headings: list[Heading], text: PageText, min_chars: int) -> tuple[list[HeadingSpan], list[str]]:
    """(spans with at least `min_chars` of text under the heading, titles not found on their page)."""
    placed, unplaced = [], []
    for h in headings:
        hits = text._spans(h.title, h.page, h.page)
        if hits:
            placed.append((hits[0][0], h))
        else:
            unplaced.append(h.title)
    placed.sort(key=lambda x: x[0])
    ends = [s for s, _ in placed[1:]] + [len(text.text)]
    return [HeadingSpan(h.title, h.page, s, e) for (s, h), e in zip(placed, ends)
            if e - s - len(norm(h.title)) >= min_chars], unplaced


def need(span: HeadingSpan, chars_per_node: int) -> int:
    """How many nodes a heading's text needs."""
    return math.ceil((span.end - span.start) / chars_per_node)


def gaps(spans: list[HeadingSpan], nodes: list[Node], text: PageText, chars_per_node: int) -> list[Gap]:
    """The spans holding fewer nodes than their length needs, or none named after the heading."""
    starts = [(s, n.title) for n in nodes for q, pages in n.located for s, _ in text._spans(q, pages[0], pages[0])]
    out = []
    for h in spans:
        have = list(dict.fromkeys(t for s, t in starts if h.start <= s < h.end))
        g = Gap(h, have, need(h, chars_per_node), any(names(h.title, t) for t in have))
        if g.target > 0:
            out.append(g)
    return out
