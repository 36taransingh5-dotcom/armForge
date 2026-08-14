"""Running a candidate plan and collecting results.

Deliberately thin. Orchestration owns progress reporting and failure
isolation; it owns no measurement logic and no judgement about what the
numbers mean.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..bench.llamacpp import LlamaCppRunner
from ..bench.runner import BenchmarkRunner
from ..bench.types import BenchmarkResult
from ..hardware.types import HostProfile
from .candidates import CandidatePlan

#: Called with (index, total, candidate label) before each measurement.
ProgressCallback = Callable[[int, int, str], None]


@dataclass
class SweepReport:
    """Everything a sweep produced, measured and unmeasured alike."""

    host: HostProfile
    plan: CandidatePlan
    results: list[BenchmarkResult] = field(default_factory=list)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    finished_at: str | None = None

    @property
    def succeeded(self) -> list[BenchmarkResult]:
        return [r for r in self.results if r.ok]

    @property
    def failed(self) -> list[BenchmarkResult]:
        return [r for r in self.results if not r.ok]

    def to_dict(self) -> dict:
        return {
            "host": self.host.to_dict(),
            "plan": self.plan.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total": len(self.results),
                "succeeded": len(self.succeeded),
                "failed": len(self.failed),
                "pruned": len(self.plan.pruned),
            },
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def _runner_for(candidate) -> BenchmarkRunner:
    """Pick a runner for a candidate. Only llama.cpp exists so far."""
    return LlamaCppRunner(candidate.config.runtime)


def run_sweep(
    plan: CandidatePlan,
    host: HostProfile,
    *,
    on_progress: ProgressCallback | None = None,
    runner_factory: Callable[[object], BenchmarkRunner] = _runner_for,
) -> SweepReport:
    """Measure every candidate in ``plan``.

    Runners convert their own failures into results, so one broken
    configuration cannot end the sweep -- the report keeps the failure and
    carries on.
    """
    report = SweepReport(host=host, plan=plan)
    total = len(plan.candidates)

    # One runner per runtime, so the ggml feature probe is paid for once each
    # rather than once per thread count.
    runners: dict[str, BenchmarkRunner] = {}

    for index, candidate in enumerate(plan.candidates, start=1):
        if on_progress:
            on_progress(index, total, candidate.label)

        key = candidate.config.runtime.binary_path
        if key not in runners:
            runners[key] = runner_factory(candidate)

        report.results.append(runners[key].run(candidate.config, host))

    report.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return report


def estimate_duration(plan: CandidatePlan, *, seconds_per_candidate: float = 25.0) -> float:
    """Rough wall-clock estimate, so a long sweep can warn before it starts."""
    return len(plan.candidates) * seconds_per_candidate


def group_by_thread_count(
    results: Sequence[BenchmarkResult],
) -> dict[int, list[BenchmarkResult]]:
    """Results keyed by thread count, for plotting the two phase curves."""
    grouped: dict[int, list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.config.threads, []).append(result)
    return dict(sorted(grouped.items()))
