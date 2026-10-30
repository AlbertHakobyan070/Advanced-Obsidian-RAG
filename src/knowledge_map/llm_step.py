"""One LLM step: JSON in, validated model out, with retries, fallbacks, a served-model check
and run budgets (spec 3.6). Every attempt is logged into `attempts`; nothing is swallowed."""
import json
import re
import time
from typing import Callable

from pydantic import BaseModel, ValidationError

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_REJECTED = ("invalid_json", "schema_error", "check_error")


class BudgetExceeded(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class UnitFailed(RuntimeError):
    def __init__(self, unit: str, attempts: list[dict]):
        last = attempts[-1] if attempts else {}
        super().__init__(f"unit {unit!r} failed after {len(attempts)} attempts; last: "
                         f"{last.get('outcome')} {last.get('error', '')[:200]}".rstrip())
        self.unit, self.attempts = unit, attempts


class Budget:
    def __init__(self, max_calls: int, max_wall_minutes: float, clock=time.monotonic):
        self.max_calls, self.max_wall_minutes, self._clock = max_calls, max_wall_minutes, clock
        self._start = clock()
        self.calls = 0

    def charge(self) -> None:
        if self.calls >= self.max_calls:
            raise BudgetExceeded("max_llm_calls")
        if self._clock() - self._start > self.max_wall_minutes * 60:
            raise BudgetExceeded("max_wall_minutes")
        self.calls += 1


def _norm(name: str) -> str:
    return name.rsplit("/", 1)[-1].split(":", 1)[0]


def served_matches(requested: str, served: str | None) -> bool:
    """`auto` accepts anything; a pinned model must be what answered (a proxy may substitute)."""
    if requested == "auto":
        return True
    if not served:
        return False
    r, s = _norm(requested), _norm(served)
    return s == r or s.startswith(r)


def parse_json(text: str) -> dict:
    cands = [text]
    m = _FENCE.search(text)
    if m:
        cands.append(m.group(1))
    if "{" in text and "}" in text:
        cands.append(text[text.index("{"):text.rindex("}") + 1])
    for c in cands:
        try:
            val = json.loads(c)
        except ValueError:
            continue
        if isinstance(val, dict):
            return val
    raise ValueError("reply is not a JSON object")


def _is_rate_limit(e: Exception) -> bool:
    return getattr(e, "status_code", None) == 429 or "RateLimit" in type(e).__name__


class StepRunner:
    def __init__(self, cfg, role: str, spec, budget: Budget, attempts: list[dict], retries: int,
                 rate_limit_sleep_s: float, client_factory=None, sleep=time.sleep,
                 request_timeout_s: float | None = None):
        self.role, self.spec, self.budget, self.attempts = role, spec, budget, attempts
        self.retries, self.rate_limit_sleep_s, self.sleep = retries, rate_limit_sleep_s, sleep
        if client_factory is None:
            if request_timeout_s is None:
                raise ValueError("StepRunner needs request_timeout_s (knowledge_map.request_timeout_s) "
                                 "to build its own clients")
            from src.llm.llm_client import LLMClient

            def client_factory(provider, model):
                c = LLMClient.from_provider_override(cfg, provider, model=model,
                                                     role=f"knowledge_map.{role}")
                # No SDK-internal retries: they would be invisible to the attempt log and the
                # budget (spec 3.6); this loop does the retrying. The timeout bounds one request.
                c._client = c._client.with_options(max_retries=0, timeout=request_timeout_s)
                return c
        self._factory = client_factory
        self._clients: dict[tuple[str, str], object] = {}

    def _client(self, provider: str, model: str):
        if (provider, model) not in self._clients:
            self._clients[(provider, model)] = self._factory(provider, model)
        return self._clients[(provider, model)]

    def call(self, system: str, user: str, model_cls: type[BaseModel], unit: str,
             check: Callable[[BaseModel], list[str]] | None = None) -> BaseModel:
        targets = [(self.spec.provider, self.spec.model), *self.spec.fallbacks]
        plan = [t for t in targets for _ in range(1 + self.retries)]
        start, feedback = len(self.attempts), None
        for i, (provider, model) in enumerate(plan):
            if i % (1 + self.retries) == 0:   # a new model never hears about another model's reply
                feedback = None
            self.budget.charge()
            prompt = user if feedback is None else (
                f"{user}\n\nYour previous reply was rejected: {feedback[:500]}. "
                "Reply again with only the JSON object.")
            feedback = None
            rec = {"role": self.role, "unit": unit, "provider": provider, "requested": model,
                   "served": None, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0,
                   "outcome": None}
            self.attempts.append(rec)
            t0 = time.monotonic()
            result, error = None, None
            try:
                resp = self._client(provider, model).complete(
                    system, prompt, temperature=self.spec.temperature,
                    max_tokens=self.spec.max_tokens, json_mode=True)
            except Exception as e:  # recorded in the attempt log, surfaced via UnitFailed
                rec["outcome"] = "rate_limited" if _is_rate_limit(e) else "transport_error"
                rec["error"] = f"{type(e).__name__}: {e}"
            else:
                rec["served"] = resp.model
                usage = resp.usage or {}
                rec["input_tokens"] = usage.get("input_tokens", 0)
                rec["output_tokens"] = usage.get("output_tokens", 0)
                if not served_matches(model, resp.model):
                    rec["outcome"] = "served_mismatch"
                else:
                    result, error, rec["outcome"] = self._validate(resp.text, model_cls, check)
                    if error:
                        rec["error"] = error
                        feedback = error
            rec["seconds"] = round(time.monotonic() - t0, 3)
            if rec["outcome"] == "ok":
                return result
            if rec["outcome"] == "rate_limited" and i < len(plan) - 1:
                self.sleep(self.rate_limit_sleep_s)
        raise UnitFailed(unit, self.attempts[start:])

    @staticmethod
    def _validate(text, model_cls, check):
        try:
            data = parse_json(text)
        except ValueError as e:
            return None, str(e), "invalid_json"
        try:
            obj = model_cls.model_validate(data)
        except ValidationError as e:
            return None, str(e), "schema_error"
        if check is not None:
            errs = check(obj)
            if errs:
                return None, "; ".join(errs), "check_error"
        return obj, None, "ok"
