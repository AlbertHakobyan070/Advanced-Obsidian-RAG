"""SKILL_NAVIGATOR.md names only what exists: its tools are MCP tools, its JSON
fields are request-model fields, its routes are serve_api routes. A rename on
the engine side fails here instead of leaving the skill calling a ghost."""
import re
from pathlib import Path

import pytest

import serve_api
from src.mcp_bridge import tools
from src.utils import release

SKILL = Path(__file__).resolve().parents[1] / "SKILL_NAVIGATOR.md"
if not SKILL.is_file():
    pytest.skip(f"{SKILL.name} is not in the public release; the skills ship "
                "from their own repo", allow_module_level=True)
TEXT = SKILL.read_text(encoding="utf-8")


def _norm(path: str) -> str:
    return re.sub(r"\{[^}]*\}", "{}", path)


def test_frontmatter_names_the_skill_and_carries_the_version_line():
    front = re.match(r"---\n(.*?)\n---\n", TEXT, re.S)
    assert front and re.search(r"^name: noetrix-navigator$", front.group(1), re.M)
    assert re.search(r"^description: \S", front.group(1), re.M)
    assert release.skill_version(SKILL) == release.PRODUCT_VERSION


def test_every_tool_it_names_exists():
    named = set(re.findall(r"\bnoetrix_\w+", TEXT))
    real = {t.name for t in tools.NAMED} | set(tools.DISPATCHERS)
    assert {"noetrix_evidence_pack", "noetrix_graph_expand", "noetrix_get_chunk"} <= named
    assert named <= real, f"not MCP tools: {sorted(named - real)}"


def test_every_request_field_it_names_exists():
    # A field is a quoted name used as a JSON key ("graph": ...) or alone in
    # backticks (`"top_k"`); quoted values ("up") are neither.
    fields = set(re.findall(r'"(\w+)"\s*:', TEXT)) | set(re.findall(r'`"(\w+)"`', TEXT))
    real = (set(serve_api.EvidencePackIn.model_fields) | set(serve_api.GraphExpandIn.model_fields)
            | set(serve_api.QueryIn.model_fields))
    assert {"graph", "edge_types", "direction", "bridges", "min_score", "seeds"} <= fields
    assert fields <= real, f"not request fields: {sorted(fields - real)}"


def test_every_route_it_names_exists():
    named = {_norm(p) for p in re.findall(r"\b(?:GET|POST) (/[\w/{}]+)", TEXT)}
    real = {_norm(r.path) for r in serve_api.app.routes}
    assert {"/evidence_pack", "/graph/expand", "/schema", "/chunks/{}"} <= named
    assert named <= real, f"not serve_api routes: {sorted(named - real)}"
