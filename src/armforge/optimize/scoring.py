"""Transparent scoring of measured candidates.

The scoring rule is deliberately simple and fully inspectable: each metric is
normalised to [0, 1] across the candidate set, multiplied by a configurable
weight, and summed. Every score exposes its components, so "why did this win"
is answered with arithmetic rather than assertion.

A single opaque number would be worse than useless here -- it would let a
recommendation look justified without being checkable.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass

from ..bench.types import BenchmarkResult


class Objective(str, enum.Enum):
    """What the user is optimising for."""

    FASTEST_PREFILL = "fastest-prefill"
    """Prompt processing throughput. Matters for long-context and RAG."""
    FASTEST_DECODE = "fastest-decode"
    """Token generation throughput. Matters for interactive chat."""
    FASTEST = "fastest"
    """Both phases weighted equally."""
    LOWEST_MEMORY = "lowest-memory"
    """Peak resident memory and model size."""
    BEST_BALANCE = "best-balance"
    """Speed and footprint together."""


#: Weight per metric for each objective. Metrics absent from a mapping are
#: ignored entirely rather than contributing zero.
DEFAULT_WEIGHTS: dict[Objective, dict[str, float]] = {
    Objective.FASTEST_PREFILL: {"prefill_tps": 1.0},
    Objective.FASTEST_DECODE: {"decode_tps": 1.0},
    Objective.FASTEST: {"prefill_tps": 0.5, "decode_tps": 0.5},
    Objective.LOWEST_MEMORY: {"peak_memory": 0.6, "model_size": 0.4},
    Objective.BEST_BALANCE: {
        "prefill_tps": 0.3,
        "decode_tps": 0.3,
        "peak_memory": 0.25,
        "model_size": 0.15,
    },
}

#: True when a larger raw value is better for that metric.
HIGHER_IS_BETTER: dict[str, bool] = {
    "prefill_tps": True,
    "decode_tps": True,
    "peak_memory": False,
    "model_size": False,
}


@dataclass(frozen=True)
class Component:
    """One metric's contribution to a total score."""

    metric: str
    raw: float
    normalized: float
    weight: float

    @property
    def contribution(self) -> float:
        return self.normalized * self.weight

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "raw": self.raw,
            "normalized": round(self.normalized, 4),
            "weight": self.weight,
            "contribution": round(self.contribution, 4),
        }


@dataclass(frozen=True)
class Score:
    """A candidate's score under one objective."""

    result: BenchmarkResult
    total: float
    components: tuple[Component, ...]

    @property
    def label(self) -> str:
        return self.result.config.label

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "total": round(self.total, 4),
            "components": [c.to_dict() for c in self.components],
        }


def extract(result: BenchmarkResult, metric: str) -> float | None:
    """Pull a raw metric value out of a result, or ``None`` if unmeasured."""
    if metric == "prefill_tps":
        return result.prefill_tps.mean if result.prefill_tps else None
    if metric == "decode_tps":
        return result.decode_tps.mean if result.decode_tps else None
    if metric == "peak_memory":
        return float(result.peak_memory_bytes) if result.peak_memory_bytes else None
    if metric == "model_size":
        return float(result.config.model.size_bytes) or None
    raise KeyError(f"unknown metric {metric!r}")


def _normalize(value: float, low: float, high: float, higher_is_better: bool) -> float:
    """Map a raw value onto [0, 1], where 1 is always the better end.

    When every candidate shares the same value the metric carries no
    information, so all candidates get 1.0 and the weighting effectively drops
    it -- rather than dividing by a zero range.
    """
    if high == low:
        return 1.0
    fraction = (value - low) / (high - low)
    return fraction if higher_is_better else 1.0 - fraction


def score_all(
    results: Sequence[BenchmarkResult],
    objective: Objective,
    *,
    weights: dict[str, float] | None = None,
) -> list[Score]:
    """Score every successful result, best first.

    Failed, skipped and unsupported results are excluded: they have no metrics
    to compare. They are still reported elsewhere, because "this did not work"
    is part of the answer.
    """
    usable = [r for r in results if r.ok]
    if not usable:
        return []

    active = weights if weights is not None else DEFAULT_WEIGHTS[objective]

    # A metric only participates if every usable candidate has it. Scoring a
    # partially-measured metric would rank candidates on data some of them
    # never produced.
    ranges: dict[str, tuple[float, float]] = {}
    for metric in active:
        values = [extract(r, metric) for r in usable]
        if any(v is None for v in values):
            continue
        ranges[metric] = (min(values), max(values))  # type: ignore[arg-type]

    scores: list[Score] = []
    for result in usable:
        components: list[Component] = []
        for metric, weight in active.items():
            if metric not in ranges:
                continue
            raw = extract(result, metric)
            if raw is None:
                continue
            low, high = ranges[metric]
            components.append(
                Component(
                    metric=metric,
                    raw=raw,
                    normalized=_normalize(raw, low, high, HIGHER_IS_BETTER[metric]),
                    weight=weight,
                )
            )

        total_weight = sum(c.weight for c in components)
        total = sum(c.contribution for c in components) / total_weight if total_weight else 0.0
        scores.append(Score(result=result, total=total, components=tuple(components)))

    return sorted(scores, key=lambda s: s.total, reverse=True)
