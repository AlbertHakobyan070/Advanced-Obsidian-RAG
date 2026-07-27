"""Config-writer contract: the Settings tab must be able to write every key it
offers, and must never rewrite the wrong line.

The regression that prompted these: switching the generation backend from the
console raised ``generation.model: found 2 matches in config.yaml; refusing to
rewrite ambiguously``, because the writer matched on the LAST path segment
inside a top-level section and ``generation:`` carries both its own ``model``
and the legacy ``local:`` block's. Every provider switch was therefore dead in
any config that kept the legacy block — which is every shipped config.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SAMPLE = """\
generation:
  provider: freellmapi
  base_url: "http://localhost:3001/v1"
  model: auto                       # keep this comment
  temperature: 0.1
  local:
    base_url: "http://localhost:5001/v1"
    model: "some-local-model"

retrieval:
  cross_encoder_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  rerank_http:
    model: bge-reranker-v2-m3
  presets:
    code: {rerank_top_k: 10, use_hyde: false}

webui:
  themes:
    - name: ledger
      model: not-a-real-key
  port: 8052
"""


@pytest.fixture
def sample_cfg(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(SAMPLE, encoding="utf-8")
    return p


def test_nested_leaf_no_longer_shadows_the_section_key(management_module, sample_cfg):
    """The exact failure from the console: generation.model + generation.local.model."""
    written = management_module._persist_section_keys(
        sample_cfg, {"generation.model": "MiniMax-M3"})

    assert written == ["generation.model"]
    text = sample_cfg.read_text(encoding="utf-8")
    assert "  model: MiniMax-M3                       # keep this comment" in text
    assert '    model: "some-local-model"' in text          # untouched


def test_deeper_path_targets_the_deeper_line(management_module, sample_cfg):
    management_module._persist_section_keys(
        sample_cfg, {"generation.local.model": "gemma-local"})

    text = sample_cfg.read_text(encoding="utf-8")
    assert "  model: auto" in text                          # untouched
    assert "    model: gemma-local" in text


def test_repeated_leaf_in_different_subtrees_is_addressable(management_module, sample_cfg):
    """`model` appears four times; each full path still resolves to one line."""
    management_module._persist_section_keys(
        sample_cfg, {"retrieval.rerank_http.model": "bge-reranker-base"})

    text = sample_cfg.read_text(encoding="utf-8")
    assert "    model: bge-reranker-base" in text
    assert '  cross_encoder_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"' in text


def test_sequence_items_do_not_shift_the_path_index(management_module, sample_cfg):
    """A `- name:` list item must not become a parent for the keys after it."""
    management_module._persist_section_keys(sample_cfg, {"webui.port": "8062"})

    assert "  port: 8062" in sample_cfg.read_text(encoding="utf-8")


def test_unknown_path_raises_instead_of_writing(management_module, sample_cfg):
    before = sample_cfg.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="no such key"):
        management_module._persist_section_keys(
            sample_cfg, {"generation.nonexistent": "x"})
    assert sample_cfg.read_text(encoding="utf-8") == before


def test_ambiguity_still_refuses_and_names_the_lines(management_module, tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("a:\n  b: 1\n  b: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"found 2 matches.*line 2, line 3"):
        management_module._persist_section_keys(p, {"a.b": "3"})


def test_a_failed_key_writes_nothing_at_all(management_module, sample_cfg):
    """Multi-key saves are all-or-nothing; a bad key can't half-apply a batch."""
    before = sample_cfg.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        management_module._persist_section_keys(
            sample_cfg, {"generation.provider": "minimax",
                         "generation.bogus": "x"})
    assert sample_cfg.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("config_name", ["config.example.yaml"])
def test_every_editable_setting_is_writable_in_the_shipped_config(
        management_module, tmp_path, config_name):
    """The console must not offer a key the writer cannot resolve.

    Without this, a setting can be added to the dropdown, look saveable, and
    fail only when a user clicks Save — which is exactly how the provider
    switch shipped broken.
    """
    src = ROOT / config_name
    if not src.exists():
        pytest.skip(f"{config_name} is not present")
    target = tmp_path / "probe.yaml"
    target.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    unresolved = []
    for key in management_module.EDITABLE_SETTINGS:
        try:
            management_module._persist_section_keys(target, {key: "probe-value"})
        except ValueError as e:
            unresolved.append(f"{key}: {e}")
    assert not unresolved, "\n".join(unresolved)
