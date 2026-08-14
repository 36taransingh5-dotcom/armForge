"""Tests for the sweep orchestration loop."""

from __future__ import annotations

from armforge.bench.mock import MockRunner
from armforge.bench.types import Status
from armforge.optimize.candidates import Candidate, CandidatePlan
from armforge.optimize.sweep import run_sweep


def _plan(bench_config, threads: list[int]) -> CandidatePlan:
    from dataclasses import replace

    candidates = tuple(
        Candidate(config=replace(bench_config, threads=t), rationale="test") for t in threads
    )
    return CandidatePlan(candidates=candidates, pruned=(), notes=())


def test_run_sweep_calls_on_progress_before_each_measurement(bench_config, apple_host):
    plan = _plan(bench_config, [1, 2, 4])
    seen = []

    run_sweep(
        plan,
        apple_host,
        on_progress=lambda i, t, label: seen.append((i, t, label)),
        runner_factory=lambda c: MockRunner(),
    )

    assert [s[0] for s in seen] == [1, 2, 3]
    assert all(s[1] == 3 for s in seen)
    assert [s[2] for s in seen] == [c.label for c in plan.candidates]


def test_run_sweep_calls_on_result_after_each_measurement(bench_config, apple_host):
    plan = _plan(bench_config, [1, 2, 4])
    results = []

    report = run_sweep(
        plan,
        apple_host,
        on_result=results.append,
        runner_factory=lambda c: MockRunner(),
    )

    assert len(results) == 3
    assert all(r.status is Status.OK for r in results)
    # on_result must have fired with the exact objects that ended up in the report.
    assert results == report.results


def test_on_result_fires_even_for_a_failed_candidate(bench_config, apple_host):
    plan = _plan(bench_config, [1, 4])
    seen_statuses = []

    run_sweep(
        plan,
        apple_host,
        on_result=lambda r: seen_statuses.append(r.status),
        runner_factory=lambda c: MockRunner(fail_on_threads=frozenset({4})),
    )

    assert seen_statuses == [Status.OK, Status.FAILED]


def test_run_sweep_works_with_neither_callback(bench_config, apple_host):
    """Both callbacks are optional; omitting them must not raise."""
    plan = _plan(bench_config, [1, 2])
    report = run_sweep(plan, apple_host, runner_factory=lambda c: MockRunner())
    assert len(report.results) == 2


def test_run_sweep_reuses_one_runner_per_runtime(bench_config, apple_host):
    """The ggml feature probe should be paid for once per runtime, not per candidate."""
    plan = _plan(bench_config, [1, 2, 4, 6])
    built = []

    def factory(candidate):
        runner = MockRunner()
        built.append(runner)
        return runner

    run_sweep(plan, apple_host, runner_factory=factory)
    assert len(built) == 1, "all candidates share one runtime, so one runner should suffice"
