"""
generator.py — Grounded answer generation with inline citations.

Takes the reranked top-k docs, formats them as numbered sources, asks the LLM to
answer using ONLY those sources with [n] citations, parses out a confidence
signal, and optionally runs a second LLM pass to verify each citation is actually
supported by its source.

Usage:
    from src.generation.generator import Generator
    gen = Generator.from_config(cfg, llm)
    answer = gen.generate("What is ARIMA?", reranked_docs)
    print(answer.text, answer.confidence, answer.citations)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.llm.llm_client import LLMClient
from src.prompts.loader import load_prompt
from src.retrieval.retriever import RetrievedDoc
from src.utils.citations import CITATION_RE, CONFIDENCE_RE
from src.utils.config_loader import Config
from src.utils.logger import get_logger
from src.utils.timing import StageTrace

log = get_logger(__name__)

# Both patterns are shared with eval/metrics.py, which scores stored answers
# with the same rules. See src/utils/citations.py for what models actually
# emit and why the parsers tolerate it.
_CONFIDENCE_RE = CONFIDENCE_RE
_CITATION_RE = CITATION_RE
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _strip_reasoning(text: str) -> str:
    # remove <think>…</think> reasoning blocks some fallback models emit
    text = _THINK_RE.sub("", text)
    # drop an unclosed leading <think> (model got cut off)
    if "<think>" in text.lower() and "</think>" not in text.lower():
        text = text[text.lower().rfind("<think>") + 7:]
    return text.strip()


@dataclass
class Citation:
    """One [n] marker the answer used, resolved to the source it points at.
    `supported` stays None unless the verification pass ran and judged it."""
    number: int
    source_label: str
    chunk_id: str
    supported: bool | None = None
    note: str = ""


@dataclass
class Answer:
    """What generate() returns: the answer text with the CONFIDENCE line
    removed, that line's level (HIGH | MEDIUM | LOW, or UNKNOWN when the
    model omitted or mangled it), the citations it used and the sources it
    was given."""
    text: str
    confidence: str
    citations: list[Citation]
    sources: list[RetrievedDoc]
    verification: dict[str, Any] | None = None
    usage: dict[str, Any] | None = field(default=None)
    # Effective retrieval settings for this query (preset, top_k, hyde...),
    # filled in by RAGPipeline.query so callers can see what actually ran.
    retrieval: dict[str, Any] | None = field(default=None)
    # Wall-clock ms of this answer's own stages — generate (the LLM call),
    # parse, and verify when it ran. Retrieval's stages are in retrieval["timings"].
    timings: dict[str, Any] | None = field(default=None)


class Generator:
    def __init__(self, llm: LLMClient, verify_citations: bool = True):
        self.llm = llm
        self.verify_citations = verify_citations
        self._gen_prompt = load_prompt("generation")
        self._verify_prompt = load_prompt("verify")

    @classmethod
    def from_config(cls, cfg: Config, llm: LLMClient) -> "Generator":
        return cls(
            llm=llm,
            verify_citations=cfg.get("generation.verify_citations", True),
        )

    # ---- context formatting ----

    @staticmethod
    def _source_label(meta: dict, fallback: str = "") -> str:
        """
        Build a human-useful citation label from chunk metadata.
          PDF books -> "Goodfellow Deep Learning › Chapter 6: Deep Feedforward, p.206"
          Notes     -> "Natural Language Processing › Vanishing Gradients"
        Falls back to the retriever's label, then filename, then a generic tag.
        Works for unknown-course book chunks too (uses the book filename + page,
        not the bare domain), which is the whole point of the fix.
        """
        def clip(s: Any, n: int) -> str:
            s = str(s)
            return s if len(s) <= n else s[: n - 1].rstrip() + "…"

        meta = meta or {}
        ft = str(meta.get("file_type", "")).lower()
        fname = meta.get("filename") or meta.get("source_file") or ""

        # Live-vault hits (Omnisearch lane): not in the index yet — say so.
        if meta.get("live"):
            return f"{clip(Path(str(fname)).stem if fname else 'note', 50)} (live vault)"

        if ft == "pdf":
            label = clip(fname, 55) if fname else (fallback or "source")
            chapter = str(meta.get("chapter") or "").strip()
            if chapter and chapter.lower() not in str(fname).lower():
                label += f" › {clip(chapter, 45)}"
            page = meta.get("page_start") or meta.get("page")
            if page:
                label += f", p.{page}"
            return label

        # notes / markdown: course + heading (handle the key drift across loaders)
        course = str(meta.get("course_name") or meta.get("course")
                     or meta.get("course_code") or "").strip()
        if course.lower() == "unknown":
            course = ""
        heading = str(meta.get("heading") or "").strip()
        parts = [p for p in (course, heading) if p]
        if len(parts) == 2 and parts[0].lower() == parts[1].lower():
            parts = parts[:1]
        if parts:
            return " › ".join(clip(p, 50) for p in parts)
        return clip(fname, 55) if fname else (fallback or "note")

    @staticmethod
    def _format_context(docs: list[RetrievedDoc]) -> str:
        blocks = []
        for i, d in enumerate(docs, start=1):
            label = Generator._source_label(d.metadata, getattr(d, "source_label", ""))
            blocks.append(f"[{i}] (from {label})\n{d.text.strip()}")
        return "\n\n".join(blocks)

    # ---- main ----

    def generate(
        self, query: str, docs: list[RetrievedDoc],
        max_tokens: int | None = None,
    ) -> Answer:
        """Answer `query` from `docs` only, numbered [1..n] in the order given.

        No docs -> a fixed LOW-confidence "nothing relevant" answer, and the
        LLM is never called. Otherwise: one completion, <think> blocks
        stripped, the CONFIDENCE line parsed out of the text, citations
        resolved against `docs`, then the optional verification pass. A
        failing generation call raises; only verification is fail-soft."""
        if not docs:
            return Answer(
                text="Your notes don't contain anything relevant to this question.",
                confidence="LOW",
                citations=[],
                sources=[],
                timings={},
            )

        tr = StageTrace()
        context = self._format_context(docs)
        with tr.span("generate"):
            resp = self.llm.complete(
                system=self._gen_prompt["system"],
                user=self._gen_prompt["user"].format(query=query, context=context),
                # None = the client's configured default (generation.max_tokens).
                max_tokens=max_tokens,
            )
        with tr.span("parse"):
            raw = resp.text
            raw = _strip_reasoning(raw)

            confidence = self._extract_confidence(raw)
            clean_text = _CONFIDENCE_RE.sub("", raw).strip()
            citations = self._extract_citations(clean_text, docs)

        answer = Answer(
            text=clean_text,
            confidence=confidence,
            citations=citations,
            sources=docs,
            usage=resp.usage,
        )

        if self.verify_citations and citations:
            with tr.span("verify"):
                answer.verification = self._verify(clean_text, docs, citations)
        answer.timings = tr.timings()

        log.info(
            "generated answer: %d chars, confidence=%s, %d citations",
            len(clean_text), confidence, len(citations),
        )
        return answer

    # ---- parsing helpers ----

    @staticmethod
    def _extract_confidence(text: str) -> str:
        m = _CONFIDENCE_RE.search(text)
        return m.group(1).upper() if m else "UNKNOWN"

    @staticmethod
    def _extract_citations(text: str, docs: list[RetrievedDoc]) -> list[Citation]:
        used = sorted({int(n) for n in _CITATION_RE.findall(text)})
        citations = []
        for n in used:
            # An out-of-range marker (a [9] against 7 sources) is dropped
            # here without a trace; eval/metrics.citation_validity is what
            # counts those as dangling.
            if 1 <= n <= len(docs):
                d = docs[n - 1]
                citations.append(
                    Citation(
                        number=n,
                        source_label=Generator._source_label(
                            d.metadata, getattr(d, "source_label", "")
                        ),
                        chunk_id=d.id,
                    )
                )
        return citations

    # ---- citation verification (second LLM pass) ----

    def _verify(
        self, answer_text: str, docs: list[RetrievedDoc], citations: list[Citation]
    ) -> dict[str, Any]:
        context = self._format_context(docs)
        try:
            resp = self.llm.complete(
                system=self._verify_prompt["system"],
                user=self._verify_prompt["user"].format(answer=answer_text, context=context),
                temperature=0.0,
                max_tokens=500,
            )
            data = self._parse_json(resp.text)
            if not data:
                return {"overall": "UNKNOWN", "verdicts": []}

            # Fold verdicts back into the Citation objects
            verdict_map = {v.get("citation"): v for v in data.get("verdicts", [])}
            for c in citations:
                v = verdict_map.get(c.number)
                if v:
                    c.supported = bool(v.get("supported", False))
                    c.note = str(v.get("note", ""))
            return data
        except Exception as e:
            log.warning("citation verification failed: %s", e)
            return {"overall": "UNKNOWN", "verdicts": []}

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        text = text.strip()
        # Strip accidental markdown fences
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        # Grab the outermost {...}
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
