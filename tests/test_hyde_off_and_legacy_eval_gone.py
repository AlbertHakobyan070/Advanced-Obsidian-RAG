"""HyDE ships off, and the first-generation golden-suite harness is gone.

Two owner decisions, pinned so neither drifts back:

  1. HyDE is OFF by default and in every preset. It costs an LLM round-trip per
     prose query and measured no gain, so only a call turns it on
     ({"hyde": true}). The default lives in three places that must agree: the
     class's own defaults, config.example.yaml and config.yaml.
  2. `main.py eval`, the console's `eval` job kind and the eval.judge /
     eval.ngram_n config keys went with the old harness. The bench
     (`main.py bench`) and the review queue's eval.sets_dir stay.

Offline: the LLM is a fake and nothing is written under the checkout.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import yaml

from src.retrieval.hyde import HyDE
from src.utils.config_loader import Config

ROOT = Path(__file__).resolve().parents[1]


class _FakeLLM:
    model = "fake-model"

    def __init__(self):
        self.calls = 0

    def complete(self, **kw):
        self.calls += 1
        return type("R", (), {"text": "a hypothetical answer"})()


def _shipped(name: str) -> dict:
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} is not present in this checkout")
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


# ---- 1. HyDE is off unless something turns it on ----

def test_hyde_is_off_unless_the_config_or_the_call_turns_it_on(tmp_path):
    llm = _FakeLLM()
    assert HyDE(llm).enabled is False                                      # the class default
    assert HyDE.from_config(Config({}, tmp_path), llm).enabled is False    # an absent key
    on = Config({"retrieval": {"use_hyde": True}}, tmp_path)
    assert HyDE.from_config(on, llm).enabled is True                       # the config

    text, status = HyDE(llm).expand_with_info("explain arima")
    assert (text, status, llm.calls) == ("explain arima", "off", 0)        # off costs no LLM call
    assert HyDE(llm).expand_with_info("explain arima", enabled=True)[1] == "nocache"   # a call turns it on
    assert llm.calls == 1


@pytest.mark.parametrize("name", ["config.example.yaml", "config.yaml"])
def test_the_shipped_configs_ship_hyde_off_in_every_preset(name):
    retrieval = _shipped(name)["retrieval"]
    assert retrieval["use_hyde"] is False
    presets = retrieval["presets"]
    for preset in ("code", "concept", "synthesis"):
        assert presets[preset]["use_hyde"] is False, preset
    assert not [p for p, v in presets.items() if v.get("use_hyde")], "a preset turns HyDE on"


# ---- 2. the old harness is gone, the bench and the review queue are not ----

def test_the_golden_suite_command_is_gone_and_the_bench_command_remains():
    import main            # imported here: it configures logging at import, as test_bench_cli does
    parser = main.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "--retrieval-only"])
    assert parser.parse_args(["bench", "run"]).bench_command == "run"


def test_the_console_no_longer_queues_an_eval_job(management_module):
    assert "eval" not in management_module.api_schema()["job_kinds"]
    with pytest.raises(ValueError, match="unknown job kind"):
        management_module._build_argv("eval", {})


@pytest.mark.parametrize("name", ["config.example.yaml", "config.yaml"])
def test_the_eval_config_section_keeps_the_review_queues_key_and_drops_the_judge(name):
    section = _shipped(name)["eval"]
    assert section["sets_dir"] == "eval/sets"
    assert "judge" not in section and "ngram_n" not in section
