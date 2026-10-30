"""Knowledge-map step runner (src/knowledge_map/llm_step.py). No network: every client is a fake."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
from pydantic import BaseModel

from src.knowledge_map.config import RoleSpec
from src.knowledge_map.llm_step import (Budget, BudgetExceeded, StepRunner, UnitFailed,
                                        parse_json, served_matches)


class Out(BaseModel):
    answer: str


class RateLimitError(Exception):
    status_code = 429


def _resp(text, model="gpt-oss-120b", usage=(7, 3)):
    return NS(text=text, model=model,
              usage={"input_tokens": usage[0], "output_tokens": usage[1]} if usage else None)


class Factory:
    """script: {(provider, model): [reply | Exception, ...]}; the last item repeats."""
    def __init__(self, script):
        self.script, self.calls, self.built = script, [], []

    def __call__(self, provider, model):
        self.built.append((provider, model))
        key = (provider, model)

        def complete(system, user, temperature=None, max_tokens=None, json_mode=False):
            self.calls.append({"key": key, "user": user, "json_mode": json_mode,
                               "temperature": temperature, "max_tokens": max_tokens})
            queue = self.script[key]
            item = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(item, Exception):
                raise item
            return item
        return NS(complete=complete)


def _runner(factory, *, model="gpt-oss-120b", fallbacks=(("p", "llama-3.2-90b"),), retries=2,
            budget=None, sleep=None, attempts=None):
    spec = RoleSpec("p", model, 100, 0.0, tuple(fallbacks))
    attempts = [] if attempts is None else attempts
    return StepRunner(None, "extract", spec, budget or Budget(100, 100), attempts, retries, 20.0,
                      client_factory=factory, sleep=sleep or (lambda s: None)), attempts


def test_ok_first_try():
    f = Factory({("p", "gpt-oss-120b"): [_resp('{"answer": "hi"}', "openai/gpt-oss-120b")]})
    r, attempts = _runner(f)
    out = r.call("sys", "usr", Out, "unit-1")
    assert out.answer == "hi"
    assert len(attempts) == 1
    a = attempts[0]
    assert a["outcome"] == "ok" and a["input_tokens"] == 7 and a["output_tokens"] == 3
    assert (a["role"], a["unit"], a["provider"], a["requested"]) == ("extract", "unit-1", "p", "gpt-oss-120b")
    assert a["served"] == "openai/gpt-oss-120b" and "seconds" in a
    assert f.calls[0]["json_mode"] is True and f.calls[0]["max_tokens"] == 100


def test_invalid_json_reasks_with_error():
    f = Factory({("p", "gpt-oss-120b"): [_resp("not json"), _resp('{"answer": "x"}')]})
    r, attempts = _runner(f)
    assert r.call("sys", "usr", Out, "u").answer == "x"
    assert "rejected" not in f.calls[0]["user"]
    assert "Your previous reply was rejected" in f.calls[1]["user"]
    assert f.calls[1]["user"].startswith("usr")
    assert [a["outcome"] for a in attempts] == ["invalid_json", "ok"]


def test_schema_error_reasks():
    f = Factory({("p", "gpt-oss-120b"): [_resp('{"wrong": 1}'), _resp('{"answer": "x"}')]})
    r, attempts = _runner(f)
    r.call("sys", "usr", Out, "u")
    assert [a["outcome"] for a in attempts] == ["schema_error", "ok"]
    assert "rejected" in f.calls[1]["user"]


def test_served_mismatch_counts_as_failure():
    f = Factory({("p", "gpt-oss-120b"): [_resp('{"answer": "bad"}', "dots-studio/dots-3-note-preview:free")],
                 ("p", "llama-3.2-90b"): [_resp('{"answer": "good"}', "meta/llama-3.2-90b-instruct")]})
    r, attempts = _runner(f)
    assert r.call("sys", "usr", Out, "u").answer == "good"
    assert [a["outcome"] for a in attempts] == ["served_mismatch"] * 3 + ["ok"]
    assert "rejected" not in f.calls[1]["user"]
    assert [c["key"][1] for c in f.calls] == ["gpt-oss-120b"] * 3 + ["llama-3.2-90b"]


@pytest.mark.parametrize("req,served,expected", [
    ("gpt-oss-120b", "openai/gpt-oss-120b", True),
    ("llama-3.2-90b-vision", "meta/llama-3.2-90b-vision-instruct", True),
    ("auto", "x", True),
    ("gpt-oss-120b", "dots-3:free", False),
    ("claude-sonnet-4-5", "dots-studio/dots-3-note-preview:free", False),
    ("gpt-oss-120b", None, False),
    ("auto", None, True),
    ("claude-sonnet-4-5", "anthropic/claude-sonnet-4-5:free", True),
])
def test_served_matches_forms(req, served, expected):
    assert served_matches(req, served) is expected


def test_all_fail_raises_unitfailed():
    f = Factory({("p", "gpt-oss-120b"): [RuntimeError("boom")], ("p", "llama-3.2-90b"): [_resp("nope", "llama-3.2-90b")]})
    r, attempts = _runner(f)
    with pytest.raises(UnitFailed) as ei:
        r.call("sys", "usr", Out, "u9")
    assert len(attempts) == (1 + 2) * 2 and ei.value.unit == "u9"
    assert ei.value.attempts == attempts
    assert [a["outcome"] for a in attempts] == ["transport_error"] * 3 + ["invalid_json"] * 3


def test_rate_limit_sleeps():
    slept = []
    f = Factory({("p", "gpt-oss-120b"): [RateLimitError("429"), _resp('{"answer": "x"}')]})
    r, attempts = _runner(f, sleep=slept.append)
    r.call("sys", "usr", Out, "u")
    assert slept == [20.0]
    assert [a["outcome"] for a in attempts] == ["rate_limited", "ok"]


def test_rate_limit_by_class_name():
    class OpenAIRateLimitError(Exception):
        pass
    slept = []
    f = Factory({("p", "gpt-oss-120b"): [OpenAIRateLimitError("x"), _resp('{"answer": "x"}')]})
    r, _ = _runner(f, sleep=slept.append)
    r.call("sys", "usr", Out, "u")
    assert slept == [20.0]


def test_budget_calls_and_wall():
    b = Budget(2, 10)
    b.charge(); b.charge()
    with pytest.raises(BudgetExceeded, match="max_llm_calls"):
        b.charge()
    t = [0.0]
    b = Budget(100, 1, clock=lambda: t[0])
    b.charge()
    t[0] = 61.0
    with pytest.raises(BudgetExceeded, match="max_wall_minutes"):
        b.charge()


def test_budget_exceeded_propagates_without_retry():
    f = Factory({("p", "gpt-oss-120b"): [RuntimeError("x")]})
    r, attempts = _runner(f, budget=Budget(1, 10))
    with pytest.raises(BudgetExceeded):
        r.call("sys", "usr", Out, "u")
    assert len(attempts) == 1


def test_check_errors_reask():
    seen = []

    def check(obj):
        seen.append(obj.answer)
        return ["unknown kind 'x'"] if len(seen) == 1 else []
    f = Factory({("p", "gpt-oss-120b"): [_resp('{"answer": "a"}'), _resp('{"answer": "b"}')]})
    r, attempts = _runner(f)
    assert r.call("sys", "usr", Out, "u", check=check).answer == "b"
    assert [a["outcome"] for a in attempts] == ["check_error", "ok"]
    assert "unknown kind 'x'" in f.calls[1]["user"]


def test_parse_json_forms():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('Sure:\n```json\n{"a": 2}\n```\nbye') == {"a": 2}
    assert parse_json('here you go {"a": {"b": 3}} thanks') == {"a": {"b": 3}}
    for bad in ("nothing", "[1, 2]", "{broken"):
        with pytest.raises(ValueError):
            parse_json(bad)


def test_unitfailed_carries_only_this_units_attempts():
    seeded = [{"unit": "other", "outcome": "ok"}]
    f = Factory({("p", "gpt-oss-120b"): [RuntimeError("boom")], ("p", "llama-3.2-90b"): [RuntimeError("last")]})
    r, attempts = _runner(f, attempts=seeded)
    with pytest.raises(UnitFailed) as ei:
        r.call("sys", "usr", Out, "mine")
    assert len(attempts) == 1 + 6 and attempts[0] is seeded[0]
    assert len(ei.value.attempts) == 6 and all(a["unit"] == "mine" for a in ei.value.attempts)
    assert "after 6 attempts" in str(ei.value) and "transport_error" in str(ei.value) and "last" in str(ei.value)


def test_fallback_does_not_hear_the_primarys_rejection():
    f = Factory({("p", "gpt-oss-120b"): [_resp("bad")],
                 ("p", "llama-3.2-90b"): [_resp('{"answer": "x"}', "llama-3.2-90b")]})
    r, _ = _runner(f, retries=1)
    r.call("sys", "usr", Out, "u")
    assert [("rejected" in c["user"]) for c in f.calls] == [False, True, False]


def test_rate_limited_seconds_exclude_the_sleep():
    import time
    f = Factory({("p", "gpt-oss-120b"): [RateLimitError("429"), _resp('{"answer": "x"}')]})
    r, attempts = _runner(f, sleep=lambda s: time.sleep(0.2))
    r.call("sys", "usr", Out, "u")
    assert attempts[0]["outcome"] == "rate_limited" and attempts[0]["seconds"] < 0.1


def test_request_timeout_is_a_logged_attempt_and_sdk_retries_are_off(tmp_path):
    """Real LLMClient via the default factory, against a socket that accepts and never answers:
    each attempt times out once, and the server sees exactly one connection per logged attempt."""
    import socket
    import threading

    from src.utils.config_loader import Config

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    conns = []

    def accept():
        while True:
            try:
                conns.append(srv.accept()[0])
            except OSError:
                return
    threading.Thread(target=accept, daemon=True).start()
    try:
        cfg = Config({"providers": {"stall": {"kind": "openai", "model": "m",
                                              "base_url": f"http://127.0.0.1:{srv.getsockname()[1]}/v1"}}},
                     tmp_path)
        attempts = []
        r = StepRunner(cfg, "extract", RoleSpec("stall", "m", 100, 0.0, ()), Budget(100, 100), attempts, 1,
                       0.0, sleep=lambda s: None, request_timeout_s=0.3)
        with pytest.raises(UnitFailed):
            r.call("sys", "usr", Out, "u")
        assert [a["outcome"] for a in attempts] == ["transport_error"] * 2
        assert all("Timeout" in a["error"] for a in attempts)
        assert len(conns) == 2                          # max_retries=0: no hidden SDK retries
    finally:
        srv.close()
        for c in conns:
            c.close()


def test_default_factory_needs_a_timeout():
    with pytest.raises(ValueError, match="request_timeout_s"):
        StepRunner(None, "extract", RoleSpec("p", "m", 100, 0.0, ()), Budget(1, 1), [], 0, 0.0)
