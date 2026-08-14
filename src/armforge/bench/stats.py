"""Statistics over repeated measurements."""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from .types import MetricStats


def summarize(samples: Sequence[float], unit: str) -> MetricStats | None:
    """Summarise repeated measurements of one metric.

    Returns ``None`` for an empty sample set rather than a zero-filled record,
    so that "not measured" never renders as "measured zero".
    """
    values = [float(v) for v in samples]
    if not values:
        return None

    return MetricStats(
        mean=statistics.fmean(values),
        median=statistics.median(values),
        minimum=min(values),
        maximum=max(values),
        # A single sample has no spread; stdev() would raise.
        stddev=statistics.stdev(values) if len(values) > 1 else 0.0,
        samples=len(values),
        unit=unit,
    )


def percent_change(baseline: float, candidate: float) -> float | None:
    """Signed percentage change from ``baseline`` to ``candidate``.

    ``None`` when the baseline is zero, since the change is undefined rather
    than infinite.
    """
    if baseline == 0:
        return None
    return ((candidate - baseline) / baseline) * 100.0


def is_meaningfully_different(
    baseline: MetricStats, candidate: MetricStats, *, sigma: float = 2.0
) -> bool:
    """Whether two measurements differ by more than their combined noise.

    A deliberately conservative check: the gap between the means must exceed
    ``sigma`` times the pooled standard deviation. Without it, run-to-run
    jitter on a thermally throttled laptop reads as a real optimisation.
    """
    pooled = (baseline.stddev**2 + candidate.stddev**2) ** 0.5
    if pooled == 0:
        return baseline.mean != candidate.mean
    return abs(candidate.mean - baseline.mean) > sigma * pooled
