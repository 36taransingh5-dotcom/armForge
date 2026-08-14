"""Rebuild result objects from the JSON ArmForge writes.

``to_dict`` is lossy on purpose -- it drops Python types and keeps the shape a
reader needs. This module is its inverse, so a report can be regenerated
months later from a committed artifact without rerunning a benchmark.

Deserialisation is strict about structure and forgiving about additions: an
unknown key is ignored, but a missing required one raises rather than being
silently defaulted, because a result that quietly invents a field is exactly
the failure mode this project exists to avoid.
"""

from __future__ import annotations

from typing import Any

from ..hardware.types import CoreCluster, CoreKind, CpuProfile, HostProfile
from .types import (
    BenchConfig,
    BenchmarkResult,
    MetricStats,
    ModelRef,
    RuntimeSpec,
    Status,
    Workload,
)


class DeserializationError(ValueError):
    """Raised when a document is not a result ArmForge could have written."""


def _require(data: dict[str, Any], key: str, context: str) -> Any:
    if key not in data:
        raise DeserializationError(f"{context} is missing required key {key!r}")
    return data[key]


def core_cluster(data: dict[str, Any]) -> CoreCluster:
    try:
        kind = CoreKind(data.get("kind", "unknown"))
    except ValueError:
        kind = CoreKind.UNKNOWN
    return CoreCluster(
        name=data.get("name", "unknown"),
        kind=kind,
        physical_cores=_require(data, "physical_cores", "cluster"),
        logical_cores=data.get("logical_cores", data.get("physical_cores", 0)),
        cpu_ids=tuple(data.get("cpu_ids", ())),
        max_freq_mhz=data.get("max_freq_mhz"),
        l2_cache_bytes=data.get("l2_cache_bytes"),
        core_name=data.get("core_name"),
    )


def cpu_profile(data: dict[str, Any]) -> CpuProfile:
    return CpuProfile(
        architecture=_require(data, "architecture", "cpu"),
        model=data.get("model", "unknown"),
        clusters=tuple(core_cluster(c) for c in data.get("clusters", ())),
        features=frozenset(data.get("features", ())),
        raw_features=tuple(data.get("raw_features", ())),
        implementer=data.get("implementer"),
        sve_vector_bits=data.get("sve_vector_bits"),
        sme_vector_bits=data.get("sme_vector_bits"),
    )


def host_profile(data: dict[str, Any]) -> HostProfile:
    return HostProfile(
        cpu=cpu_profile(_require(data, "cpu", "host")),
        os_name=data.get("os_name", "unknown"),
        os_release=data.get("os_release", ""),
        total_memory_bytes=data.get("total_memory_bytes", 0),
        available_memory_bytes=data.get("available_memory_bytes"),
        page_size_bytes=data.get("page_size_bytes"),
        detector=data.get("detector", "unknown"),
        warnings=tuple(data.get("warnings", ())),
    )


def metric_stats(data: dict[str, Any] | None) -> MetricStats | None:
    """``None`` stays ``None`` -- an unmeasured metric must not become zero."""
    if not data:
        return None
    return MetricStats(
        mean=_require(data, "mean", "metric"),
        median=data.get("median", data["mean"]),
        minimum=data.get("min", data["mean"]),
        maximum=data.get("max", data["mean"]),
        stddev=data.get("stddev", 0.0),
        samples=data.get("samples", 1),
        unit=data.get("unit", ""),
    )


def bench_config(data: dict[str, Any]) -> BenchConfig:
    model = _require(data, "model", "config")
    runtime = _require(data, "runtime", "config")
    workload = _require(data, "workload", "config")

    return BenchConfig(
        model=ModelRef(
            path=_require(model, "path", "model"),
            name=model.get("name", ""),
            size_bytes=model.get("size_bytes", 0),
            quantization=model.get("quantization"),
            content_hash=model.get("content_hash"),
            n_params=model.get("n_params"),
        ),
        runtime=RuntimeSpec(
            name=runtime.get("name", "unknown"),
            version=runtime.get("version", "unknown"),
            binary_path=runtime.get("binary_path", ""),
            build_flags=dict(runtime.get("build_flags", {})),
        ),
        workload=Workload(
            name=workload.get("name", "unknown"),
            prompt_tokens=workload.get("prompt_tokens", 0),
            generate_tokens=workload.get("generate_tokens", 0),
            description=workload.get("description", ""),
        ),
        threads=_require(data, "threads", "config"),
        warmup_iterations=data.get("warmup_iterations", 1),
        iterations=data.get("iterations", 1),
        extra=dict(data.get("extra", {})),
    )


def benchmark_result(data: dict[str, Any], host: HostProfile) -> BenchmarkResult:
    metrics = data.get("metrics", {})
    try:
        status = Status(_require(data, "status", "result"))
    except ValueError as exc:
        raise DeserializationError(f"unknown status {data.get('status')!r}") from exc

    return BenchmarkResult(
        config=bench_config(_require(data, "config", "result")),
        host=host,
        status=status,
        prefill_tps=metric_stats(metrics.get("prefill_tps")),
        decode_tps=metric_stats(metrics.get("decode_tps")),
        peak_memory_bytes=metrics.get("peak_memory_bytes"),
        model_load_ms=metrics.get("model_load_ms"),
        wall_time_s=metrics.get("wall_time_s"),
        error=data.get("error"),
        raw=dict(data.get("raw", {})),
        timestamp=data.get("timestamp", ""),
    )


def sweep_report(data: dict[str, Any]):
    """Rebuild a :class:`SweepReport` from ``optimize --output`` JSON."""
    from ..optimize.candidates import CandidatePlan, Pruned
    from ..optimize.sweep import SweepReport

    if "results" not in data or "host" not in data:
        raise DeserializationError(
            "not an ArmForge sweep: expected 'host' and 'results' keys. "
            "A single-benchmark artifact has neither."
        )

    host = host_profile(data["host"])
    plan_data = data.get("plan", {})

    # Candidates are not rebuilt: they carry BenchConfig objects already
    # present in the results, and nothing downstream of a completed sweep
    # reads them. Pruned entries are kept because they are part of the record.
    plan = CandidatePlan(
        candidates=(),
        pruned=tuple(
            Pruned(label=p.get("label", "?"), reason=p.get("reason", ""))
            for p in plan_data.get("pruned", ())
        ),
        notes=tuple(plan_data.get("notes", ())),
    )

    report = SweepReport(
        host=host,
        plan=plan,
        results=[benchmark_result(r, host) for r in data["results"]],
        started_at=data.get("started_at", ""),
    )
    report.finished_at = data.get("finished_at")
    return report
