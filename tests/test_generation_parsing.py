"""Answer-parsing tolerance — citation markers and the confidence line.

Run:  python -m pytest tests/ -q

Models do not reliably obey the generation prompt's formatting. Two variants
seen live from the freellmapi provider on 2026-09-17 broke the parser silently,
on /query and /compare as much as on the newer /answer:

  * "**CONFIDENCE:** HIGH"     -> confidence reported as UNKNOWN
  * fullwidth citations 【1】   -> zero citations, every source `cited: false`

Both produced a correct, fully grounded answer that scored UNKNOWN with no
citations — the worst kind of failure, because nothing errored.

These tests pin the tolerance AND its limits: the parser must still refuse
ordinary prose that merely contains the word "confidence", and must still
refuse a grouped `[1, 2]` marker, which in this corpus is far more likely to
be numpy indexing than a citation.

No LLM is called — only the static parsing helpers, which is the whole surface
that broke.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.generation.generator import _CONFIDENCE_RE, Generator
from src.retrieval.retriever import RetrievedDoc
from src.utils.citations import CITATION_RE, CONFIDENCE_RE, ONLY_CITATIONS_RE


def docs(n):
    return [RetrievedDoc(id=f"c{i}", text=f"source {i}",
                         metadata={"filename": f"f{i}.md"})
            for i in range(1, n + 1)]


# ---- confidence: tolerate the decoration ----

@pytest.mark.parametrize("line, expected", [
    ("CONFIDENCE: HIGH", "HIGH"),                 # what the prompt asks for
    ("**CONFIDENCE:** HIGH", "HIGH"),             # observed live
    ("**CONFIDENCE: MEDIUM**", "MEDIUM"),         # emphasis around the whole line
    ("__CONFIDENCE:__ low", "LOW"),               # underscores + lowercase level
    ("### CONFIDENCE: LOW", "LOW"),               # emitted as a heading
    ("`CONFIDENCE:` HIGH", "HIGH"),               # code span
    ("CONFIDENCE:HIGH", "HIGH"),                  # no space at all
    ("CONFIDENCE： HIGH", "HIGH"),                # fullwidth colon
])
def test_confidence_survives_markdown_decoration(line, expected):
    body = f"ARIMA is a time-series model [1].\n\n{line}"
    assert Generator._extract_confidence(body) == expected


# ---- confidence: and refuse prose ----

@pytest.mark.parametrize("prose", [
    "The sources support this, so I have high confidence in the answer.",
    "Confidence intervals are narrower at higher n.",
    "confidence lower than the usual threshold",
    "My confidence: lower bounds are covered in lecture 4.",
    "Report the confidence: highest posterior density comes first.",
])
def test_prose_mentioning_confidence_is_not_a_confidence_line(prose):
    assert Generator._extract_confidence(prose) == "UNKNOWN"


def test_an_answer_with_no_confidence_line_is_unknown():
    assert Generator._extract_confidence("Just an answer [1].") == "UNKNOWN"


# ---- confidence: the same regex strips the line from the answer text ----

def test_decorated_confidence_line_is_stripped_from_the_answer_text():
    """Leftover '**' would leak into the console and every markdown export."""
    raw = "ARIMA is a time-series model [1].\n\n**CONFIDENCE:** HIGH"
    assert _CONFIDENCE_RE.sub("", raw).strip() == "ARIMA is a time-series model [1]."


def test_confidence_line_wrapped_in_bold_leaves_no_stray_markers():
    raw = "Answer [1].\n\n**CONFIDENCE: MEDIUM**"
    assert _CONFIDENCE_RE.sub("", raw).strip() == "Answer [1]."


def test_heading_style_confidence_line_leaves_no_stray_hashes():
    raw = "Answer [1].\n\n### CONFIDENCE: LOW"
    assert _CONFIDENCE_RE.sub("", raw).strip() == "Answer [1]."


# ---- citations: tolerate the bracket family ----

def test_ascii_citations_are_extracted():
    text = "Gradient clipping helps [1], and batch norm also helps [2]."
    assert [c.number for c in Generator._extract_citations(text, docs(6))] == [1, 2]


def test_fullwidth_cjk_citations_are_extracted():
    """Observed live: the model cited all six sources with 【n】 and the
    response reported none of them."""
    text = ("Gradient clipping helps【1】【2】, batch norm also helps"
            "【3】【4】【5】【6】.")
    got = [c.number for c in Generator._extract_citations(text, docs(6))]
    assert got == [1, 2, 3, 4, 5, 6]


def test_mixed_ascii_and_fullwidth_citations_in_one_answer():
    text = "First claim [1]. Second claim【2】. Third claim［3］."
    assert [c.number for c in Generator._extract_citations(text, docs(3))] == [1, 2, 3]


def test_citation_numbers_outside_the_source_list_are_dropped():
    text = "Real [1] but hallucinated [9], and an array index [0]."
    assert [c.number for c in Generator._extract_citations(text, docs(2))] == [1]


def test_citations_carry_the_source_label_and_chunk_id():
    cites = Generator._extract_citations("Claim【1】", docs(1))
    assert len(cites) == 1
    assert cites[0].chunk_id == "c1"
    assert cites[0].source_label


def test_a_grouped_marker_is_deliberately_not_a_citation():
    """`a[1, 2]` is ordinary numpy indexing in this corpus, and the prompt asks
    for one number per bracket. Supporting `[1, 2]` would silently turn every
    2-D index into two citations."""
    assert CITATION_RE.findall("values = a[1, 2]") == []


# ---- the two files must not drift apart again ----

def test_the_generator_and_the_eval_suite_share_one_citation_pattern():
    """This pattern lived in both files as a copy. A fix applied to one of them
    would have left the eval suite scoring correctly-cited answers as uncited,
    which is worse than both being broken."""
    from eval import metrics
    assert metrics._CITATION_RE is CITATION_RE
    assert metrics._ONLY_CITATIONS_RE is ONLY_CITATIONS_RE

    import src.generation.generator as gen
    assert gen._CITATION_RE is CITATION_RE
    assert gen._CONFIDENCE_RE is CONFIDENCE_RE


def test_the_eval_suite_scores_fullwidth_citations_as_cited():
    from eval.metrics import citation_validity
    rate, dangling = citation_validity("Claim【1】 and claim【2】.", n_docs=3)
    assert rate == 1.0 and dangling == []


def test_the_eval_suite_still_flags_a_dangling_fullwidth_citation():
    from eval.metrics import citation_validity
    rate, dangling = citation_validity("Claim【1】 and claim【9】.", n_docs=3)
    assert dangling == [9] and rate == 0.5


def test_a_citation_only_fragment_is_recognised_in_every_bracket_family():
    for frag in ("[1][2]", "【1】【2】", "［3］.", " [1] "):
        assert ONLY_CITATIONS_RE.match(frag), frag
    assert not ONLY_CITATIONS_RE.match("gradient clipping [1]")


# ---- the prompt asks for what the parser prefers ----

def test_the_generation_prompt_still_asks_for_the_plain_forms():
    """The parser tolerates decoration; the prompt should still discourage it,
    so tolerance stays a safety net rather than the normal path."""
    from src.prompts.loader import load_prompt
    system = load_prompt("generation")["system"]
    assert "ASCII square brackets" in system
    assert "CONFIDENCE:" in system
    assert "no bold" in system
