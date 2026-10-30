"""
timing.py — per-stage wall-clock spans for one request.

time.perf_counter, never time.time: the wall clock can step (NTP, DST) and
has coarse resolution on Windows, so differences of time.time() are not a
duration. One StageTrace per request; spans with the same name accumulate.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

# What generated_mode route's routed lane did (route-then-cite §5.4); routed + unrouted
# == router_hits.
ROUTE_KEYS = ("router_hits", "routed", "unrouted", "targets", "targets_missing", "new")


@dataclass
class StageTrace:
    ms: dict[str, float] = field(default_factory=dict)
    # lane name -> number of candidates that lane returned
    lanes: dict[str, int] = field(default_factory=dict)
    # ROUTE_KEYS -> count, summed across retrieve() calls; all 0 when nothing routed
    route: dict[str, int] = field(default_factory=lambda: dict.fromkeys(ROUTE_KEYS, 0))

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = self.ms.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0

    def timings(self) -> dict[str, float]:
        return {k: round(v, 2) for k, v in self.ms.items()}
