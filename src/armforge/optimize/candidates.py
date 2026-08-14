"""Capability-driven candidate generation.

This is where ArmForge differs from a configuration sweeper. A sweeper
enumerates the cross product of every option and measures all of it. ArmForge
starts from what the CPU reports it can do and uses Arm-specific knowledge to
decide what is *worth* measuring -- and, just as importantly, records why it
included or excluded each option.

Every candidate carries a rationale, and every exclusion carries a reason.
A plan that cannot explain itself is a black box, and a black box is not
evidence.

Nothing here predicts a winner. It decides what to put on the bench. The
measurement decides the rest, and where measurement contradicts the model,
the model is what was wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..analyzer.gguf import GGUFError, GGUFModel, read_gguf
from ..bench.types import BenchConfig, ModelRef, RuntimeSpec, Workload
from ..hardware.types import CpuProfile, HostProfile


@dataclass(frozen=True)
class Candidate:
    """One configuration worth measuring, with the reason it made the cut."""

    config: BenchConfig
    rationale: str

    @property
    def label(self) -> str:
        return self.config.label


@dataclass(frozen=True)
class Pruned:
    """A configuration deliberately not measured, and why."""

    label: str
    reason: str


@dataclass(frozen=True)
class CandidatePlan:
    """The full set of decisions made before any measurement happened."""

    candidates: tuple[Candidate, ...]
    pruned: tuple[Pruned, ...]
    notes: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.candidates)

    def to_dict(self) -> dict:
        return {
            "candidates": [
                {"label": c.label, "rationale": c.rationale, "config": c.config.to_dict()}
                for c in self.candidates
            ],
            "pruned": [{"label": p.label, "reason": p.reason} for p in self.pruned],
            "notes": list(self.notes),
        }


def thread_candidates(cpu: CpuProfile) -> list[tuple[int, str]]:
    """Thread counts worth testing on this CPU, with reasons.

    The set is chosen from the core topology rather than from a fixed ladder.
    On a heterogeneous part the interesting region is around the performance
    cluster boundary; on a uniform part it is the usual doubling.

    ``nproc`` is always included -- not because it is expected to win, but
    because it is what everyone actually uses, and demonstrating that it loses
    requires measuring it.
    """
    performance = max(cpu.performance_cores, 1)
    total = max(cpu.physical_cores, 1)
    chosen: dict[int, str] = {}

    def offer(count: int, reason: str) -> None:
        count = max(1, min(count, total))
        chosen.setdefault(count, reason)

    offer(1, "single-thread reference point")

    if cpu.is_heterogeneous:
        offer(
            max(performance // 2, 1),
            "half the performance cluster, in case memory bandwidth saturates early",
        )
        offer(
            performance,
            f"the {performance} performance cores only -- the topology model's "
            "expectation for memory-bound decode",
        )

        # The optimum for compute-bound prefill sits somewhere between "the
        # performance cluster" and "everything", and a single midpoint guess
        # misses it: on a 4P+6E M4 the measured prefill peak is 6 threads,
        # while performance + efficiency/2 would have proposed 7. Sample the
        # range instead of guessing a point in it.
        span = total - performance
        if span > 0:
            step = max(1, -(-span // 3))  # ceil(span / 3)
            for count in range(performance + step, total, step):
                offer(
                    count,
                    "performance cluster plus part of the efficiency cluster, "
                    "sampling for the point where slower cores stop paying for "
                    "themselves",
                )
        offer(
            total,
            f"every core ({total}) -- what nproc would pick, included to test "
            "whether it is actually worse",
        )
    else:
        offer(max(total // 4, 1), "quarter of the cores")
        offer(max(total // 2, 1), "half the cores")
        offer(total, f"all {total} cores -- the nproc default on a uniform CPU")

    return sorted(chosen.items())


def _quantization_note(model: GGUFModel, cpu: CpuProfile) -> str:
    """How this model's format interacts with this CPU's instruction set."""
    quant = model.quantization or "unknown"

    if not cpu.is_arm64:
        return f"{quant} on a non-Arm host; no Arm-specific expectation"
    if model.repackable_for_i8mm and cpu.has("i8mm"):
        return (
            f"{quant} can be repacked into a blocked layout that feeds SMMLA, "
            "and this CPU reports FEAT_I8MM"
        )
    if model.repackable_for_i8mm:
        return (
            f"{quant} is repackable, but this CPU lacks FEAT_I8MM so the layout "
            "buys nothing here"
        )
    return (
        f"{quant} has no int8 matrix-multiply fast path, so FEAT_I8MM goes "
        "unused regardless of the hardware"
    )


def _runtime_note(runtime: RuntimeSpec, cpu: CpuProfile) -> str:
    flags = runtime.build_flags
    if flags.get("accelerate"):
        return (
            "Apple Accelerate build: routes prompt processing through Apple's "
            "BLAS rather than ggml's Arm kernels, so it measures a different "
            "thing than the other variants"
        )
    if flags.get("kleidiai"):
        return (
            "KleidiAI build: Arm's micro-kernels, the only path that can reach SME2 on this CPU"
        )
    return "ggml's own Arm CPU kernels -- the controlled baseline"


def generate(
    host: HostProfile,
    model_paths: list[str | Path],
    runtimes: list[RuntimeSpec],
    workload: Workload,
    *,
    iterations: int = 5,
    thread_counts: list[int] | None = None,
) -> CandidatePlan:
    """Build the measurement plan for this host, these models and runtimes."""
    cpu = host.cpu
    candidates: list[Candidate] = []
    pruned: list[Pruned] = []
    notes: list[str] = []

    if not cpu.is_arm64:
        notes.append(
            "Host is not Arm64. Candidates are still generated so the workflow "
            "can be exercised, but no Arm-specific reasoning applies."
        )

    threads = (
        [(n, "explicitly requested") for n in thread_counts]
        if thread_counts
        else thread_candidates(cpu)
    )

    models: list[GGUFModel] = []
    for path in model_paths:
        try:
            models.append(read_gguf(path))
        except GGUFError as exc:
            pruned.append(Pruned(label=str(path), reason=f"unreadable: {exc}"))

    if not models:
        notes.append("No readable models; nothing to measure.")
        return CandidatePlan((), tuple(pruned), tuple(notes))

    available = host.available_memory_bytes

    for model in models:
        quant_note = _quantization_note(model, cpu)

        # A model that does not fit in memory will page, and a paging
        # measurement describes the disk, not the CPU.
        if available and model.file_size_bytes > available:
            pruned.append(
                Pruned(
                    label=f"{model.quantization or model.path.name}",
                    reason=(
                        f"model is {model.file_size_bytes / 1024**3:.1f} GB but only "
                        f"{available / 1024**3:.1f} GB is available; measuring it "
                        "would measure paging"
                    ),
                )
            )
            continue

        model_ref = ModelRef(
            path=str(Path(model.path).resolve()),
            name=model.name or Path(model.path).stem,
            size_bytes=model.file_size_bytes,
            quantization=model.quantization,
            n_params=model.parameter_count,
        )

        for runtime in runtimes:
            runtime_note = _runtime_note(runtime, cpu)

            # KleidiAI's micro-kernels target int8 matrix and SME paths. On a
            # CPU with neither, the build cannot do anything the baseline
            # cannot, so measuring it is a waste of bench time.
            if runtime.build_flags.get("kleidiai") and not (cpu.has("i8mm") or cpu.has("sme")):
                pruned.append(
                    Pruned(
                        label=f"{model_ref.quantization} · {runtime.label}",
                        reason=(
                            "KleidiAI kernels need FEAT_I8MM or FEAT_SME; this CPU "
                            "reports neither"
                        ),
                    )
                )
                continue

            for count, thread_note in threads:
                config = BenchConfig(
                    model=model_ref,
                    runtime=runtime,
                    workload=workload,
                    threads=count,
                    iterations=iterations,
                )
                candidates.append(
                    Candidate(
                        config=config,
                        rationale=(
                            f"{quant_note}; {runtime_note}; "
                            f"{count} threads: {thread_note}"
                        ),
                    )
                )

    if cpu.is_heterogeneous:
        notes.append(
            f"CPU is heterogeneous ({cpu.performance_cores} performance + "
            f"{cpu.physical_cores - cpu.performance_cores} efficiency cores). "
            "Prefill and decode are expected to prefer different thread counts, "
            "so they are scored separately."
        )

    return CandidatePlan(tuple(candidates), tuple(pruned), tuple(notes))
