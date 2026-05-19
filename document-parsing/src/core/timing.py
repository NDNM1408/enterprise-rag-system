"""Lightweight timing helpers for parse pipelines.

Two patterns:

    with timed("layout") as t:
        ...
    log.info("layout took %.3fs", t.seconds)

    timer = StageTimer()
    with timer.stage("render"):
        ...
    with timer.stage("layout"):
        ...
    log.info("breakdown: %s", timer.summary())   # "render=0.12s layout=1.83s ..."
    log.info("totals: %s", timer.totals())       # ditto, summed across all stages
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class _Span:
    seconds: float = 0.0


@contextmanager
def timed(_label: str = ""):
    span = _Span()
    t0 = time.perf_counter()
    try:
        yield span
    finally:
        span.seconds = time.perf_counter() - t0


@dataclass
class StageTimer:
    """Accumulate elapsed time per named stage across many invocations."""
    totals_s: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self.totals_s[name] = self.totals_s.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1

    def add(self, name: str, seconds: float) -> None:
        self.totals_s[name] = self.totals_s.get(name, 0.0) + seconds
        self.counts[name] = self.counts.get(name, 0) + 1

    def summary(self) -> str:
        return " ".join(
            f"{k}={v:.2f}s({self.counts.get(k, 0)})"
            for k, v in sorted(self.totals_s.items(), key=lambda kv: -kv[1])
        )

    def as_dict(self) -> dict[str, float]:
        return dict(self.totals_s)
