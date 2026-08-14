"""Turning measurements into a recommendation that can be acted on.

Two things distinguish this from picking the top row of a table.

First, the recommendation is **per phase**. Prefill and decode are bound by
different resources and, on a heterogeneous CPU, measurably prefer different
thread counts. llama.cpp can express that split -- ``-t`` sets generation
threads and ``-tb`` sets prompt-processing threads -- so ArmForge emits both
rather than averaging them into one number that is wrong for each.

Second, every claim is checked against measurement noise before it is made.
A 3% gap between two candidates whose runs varied by 15% is not a finding, and
saying so is more useful than a confident wrong answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..bench.stats import is_meaningfully_different, percent_change
from ..bench.types import BenchmarkResult
from ..hardware.types import HostProfile
from .scoring import Objective, Score, score_all

#: Relative standard deviation above which a measurement is called noisy.
NOISE_THRESHOLD = 0.05


@dataclass(frozen=True)
class PhaseChoice:
    """The best thread count for one phase, within a fixed model and runtime."""

    phase: str
    threads: int
    throughput: float
    stddev: float
    result: BenchmarkResult
    runner_up: BenchmarkResult | None = None
    decisive: bool = True
    """False when the runner-up is within measurement noise."""

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "threads": self.threads,
            "throughput_tok_s": round(self.throughput, 2),
            "stddev": round(self.stddev, 2),
            "decisive": self.decisive,
            "runner_up_threads": (self.runner_up.config.threads if self.runner_up else None),
        }


@dataclass(frozen=True)
class Recommendation:
    objective: Objective
    winner: Score
    prefill: PhaseChoice | None
    decode: PhaseChoice | None
    baseline: BenchmarkResult | None
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    pareto: tuple[BenchmarkResult, ...] = ()
    failures: tuple[BenchmarkResult, ...] = ()
    improvements: dict[str, float | None] = field(default_factory=dict)

    @property
    def deployment_command(self) -> str:
        """A llama.cpp command line expressing the per-phase thread split."""
        config = self.winner.result.config
        parts = ["llama-completion", "-m", config.model.path]
        if self.decode:
            parts += ["-t", str(self.decode.threads)]
        if self.prefill:
            parts += ["-tb", str(self.prefill.threads)]
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "objective": self.objective.value,
            "winner": self.winner.to_dict(),
            "winning_config": self.winner.result.config.to_dict(),
            "prefill": self.prefill.to_dict() if self.prefill else None,
            "decode": self.decode.to_dict() if self.decode else None,
            "baseline": self.baseline.config.label if self.baseline else None,
            "improvements_vs_baseline_pct": self.improvements,
            "deployment_command": self.deployment_command,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "pareto": [r.config.label for r in self.pareto],
            "failures": [
                {"label": r.config.label, "status": r.status.value, "error": r.error}
                for r in self.failures
            ],
        }


def pick_baseline(
    results: Sequence[BenchmarkResult], host: HostProfile
) -> BenchmarkResult | None:
    """The configuration a developer would land on without ArmForge.

    Defined as: every core (what ``nproc`` reports), on the plainest available
    build. This is the honest comparison point -- improvements are measured
    against what someone would actually have done, not against an artificially
    crippled single-threaded run.
    """
    usable = [r for r in results if r.ok]
    if not usable:
        return None

    nproc = host.cpu.physical_cores

    def rank(result: BenchmarkResult) -> tuple[int, int, int]:
        flags = result.config.runtime.build_flags
        special = int(bool(flags.get("kleidiai"))) + int(bool(flags.get("accelerate")))
        return (
            abs(result.config.threads - nproc),  # closest to nproc
            special,  # plainest build
            -result.config.model.size_bytes,  # largest (least quantised) model
        )

    return min(usable, key=rank)


def _best_for_phase(results: Sequence[BenchmarkResult], phase: str) -> PhaseChoice | None:
    """Best thread count for one phase among results sharing model and runtime."""
    getter = (lambda r: r.prefill_tps) if phase == "prefill" else (lambda r: r.decode_tps)
    scored = [(r, getter(r)) for r in results if r.ok and getter(r) is not None]
    if not scored:
        return None

    scored.sort(key=lambda pair: pair[1].mean, reverse=True)
    best, best_stats = scored[0]
    runner_up, runner_stats = scored[1] if len(scored) > 1 else (None, None)

    decisive = True
    if runner_stats is not None:
        decisive = is_meaningfully_different(runner_stats, best_stats)

    return PhaseChoice(
        phase=phase,
        threads=best.config.threads,
        throughput=best_stats.mean,
        stddev=best_stats.stddev,
        result=best,
        runner_up=runner_up,
        decisive=decisive,
    )


#: Relative gap below which two values on a Pareto axis count as equal.
#: Deliberately the same as NOISE_THRESHOLD: a difference smaller than
#: run-to-run variation is not a trade-off worth preserving a candidate for.
PARETO_TOLERANCE = NOISE_THRESHOLD


def pareto_frontier(
    results: Sequence[BenchmarkResult], *, tolerance: float = PARETO_TOLERANCE
) -> list[BenchmarkResult]:
    """Candidates not beaten on every axis at once.

    Axes are prefill throughput (higher better), decode throughput (higher
    better) and peak memory (lower better). A candidate is on the frontier
    unless some other candidate is at least as good on all three and
    meaningfully better on one -- meaning there is no reason to ever choose it.

    ``tolerance`` matters more than it looks. Peak memory barely moves across
    thread counts of the same model: on an M4 sweep it varied by under 4%,
    which is measurement noise, not a real trade-off. Comparing those values
    exactly leaves almost every candidate technically non-dominated and turns
    the frontier into a list of everything. Differences below the tolerance are
    therefore treated as ties.
    """
    usable = [
        r for r in results if r.ok and r.prefill_tps and r.decode_tps and r.peak_memory_bytes
    ]

    def axes(result: BenchmarkResult) -> tuple[float, float, float]:
        return (
            result.prefill_tps.mean,
            result.decode_tps.mean,
            -float(result.peak_memory_bytes),
        )

    def at_least_as_good(a: float, b: float) -> bool:
        return a >= b - abs(b) * tolerance

    def clearly_better(a: float, b: float) -> bool:
        return a > b + abs(b) * tolerance

    frontier: list[BenchmarkResult] = []
    for candidate in usable:
        c_axes = axes(candidate)
        dominated = any(
            all(at_least_as_good(o, c) for o, c in zip(axes(other), c_axes, strict=True))
            and any(clearly_better(o, c) for o, c in zip(axes(other), c_axes, strict=True))
            for other in usable
            if other is not candidate
        )
        if not dominated:
            frontier.append(candidate)
    return frontier


def recommend(
    results: Sequence[BenchmarkResult],
    host: HostProfile,
    objective: Objective = Objective.BEST_BALANCE,
) -> Recommendation | None:
    """Pick and justify a configuration. ``None`` when nothing was measured."""
    scores = score_all(results, objective)
    failures = tuple(r for r in results if not r.ok)
    if not scores:
        return None

    winner = scores[0]
    config = winner.result.config

    # Fix model and runtime from the overall winner, then choose thread counts
    # per phase within that group -- so the two numbers describe one
    # deployable configuration rather than two incompatible ones.
    siblings = [
        r
        for r in results
        if r.ok
        and r.config.model.path == config.model.path
        and r.config.runtime.binary_path == config.runtime.binary_path
    ]
    prefill = _best_for_phase(siblings, "prefill")
    decode = _best_for_phase(siblings, "decode")

    baseline = pick_baseline(results, host)
    improvements: dict[str, float | None] = {}
    reasons: list[str] = []
    warnings: list[str] = []

    if baseline is not None and baseline is not winner.result:
        if prefill and baseline.prefill_tps:
            improvements["prefill"] = percent_change(
                baseline.prefill_tps.mean, prefill.throughput
            )
        if decode and baseline.decode_tps:
            improvements["decode"] = percent_change(baseline.decode_tps.mean, decode.throughput)
        if winner.result.peak_memory_bytes and baseline.peak_memory_bytes:
            improvements["peak_memory"] = percent_change(
                baseline.peak_memory_bytes, winner.result.peak_memory_bytes
            )
        improvements["model_size"] = percent_change(
            baseline.config.model.size_bytes, config.model.size_bytes
        )

        for metric, label in (
            ("prefill", "prompt processing"),
            ("decode", "token generation"),
        ):
            change = improvements.get(metric)
            if change is not None and abs(change) >= 1:
                direction = "faster" if change > 0 else "slower"
                reasons.append(
                    f"{abs(change):.0f}% {direction} {label} than the "
                    f"{baseline.config.threads}-thread baseline"
                )
        size_change = improvements.get("model_size")
        if size_change is not None and abs(size_change) >= 1:
            reasons.append(
                f"{abs(size_change):.0f}% {'smaller' if size_change < 0 else 'larger'} on disk"
            )

    if prefill and decode and prefill.threads != decode.threads:
        reasons.append(
            f"prompt processing peaks at {prefill.threads} threads while token "
            f"generation peaks at {decode.threads}; the two phases are bound by "
            "different resources, so they are configured separately"
        )

    # Honesty checks: say when a claim is not supported by the spread.
    for choice in (prefill, decode):
        if choice is None:
            continue
        if not choice.decisive and choice.runner_up is not None:
            warnings.append(
                f"{choice.phase}: {choice.threads} threads and "
                f"{choice.runner_up.config.threads} threads are within "
                "measurement noise; either is defensible"
            )
        if choice.throughput and (choice.stddev / choice.throughput) > NOISE_THRESHOLD:
            warnings.append(
                f"{choice.phase} measurement varied by "
                f"{choice.stddev / choice.throughput * 100:.0f}% between "
                "repetitions; treat the exact figure with caution"
            )

    if len(scores) > 1:
        runner_up = scores[1]
        if abs(winner.total - runner_up.total) < 0.02:
            warnings.append(
                f"{winner.label} and {runner_up.label} scored within 0.02 of each "
                "other; the choice between them is close"
            )

    return Recommendation(
        objective=objective,
        winner=winner,
        prefill=prefill,
        decode=decode,
        baseline=baseline,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        pareto=tuple(pareto_frontier(results)),
        failures=failures,
        improvements=improvements,
    )
