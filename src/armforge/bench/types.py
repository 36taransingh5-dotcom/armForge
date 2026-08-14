"""Value types for benchmark configuration and results.

Every result carries enough provenance to be reproduced or challenged: which
model file (by content hash), which runtime build (by commit and build flags),
which host, which workload shape, how many iterations. A number without that
context is not evidence.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..hardware.types import HostProfile


class Status(str, enum.Enum):
    """Outcome of an attempted benchmark.

    ArmForge never silently pretends a configuration succeeded. A candidate
    that could not run is reported with the reason it could not run.
    """

    OK = "ok"
    UNSUPPORTED = "unsupported"
    """The configuration cannot work here, e.g. a runtime lacking a feature."""
    SKIPPED = "skipped"
    """Deliberately not attempted, e.g. pruned by the capability model."""
    FAILED = "failed"
    """Attempted and errored."""
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class Workload:
    """A benchmark shape.

    ``prompt_tokens`` drives the prefill (compute-bound) phase and
    ``generate_tokens`` drives the decode (memory-bandwidth-bound) phase.
    These are token counts rather than real text: llama-bench synthesises
    tokens, which keeps the workload reproducible and free of any dataset
    licensing question. It measures throughput at a given shape, not output
    quality.
    """

    name: str
    prompt_tokens: int
    generate_tokens: int
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prompt_tokens": self.prompt_tokens,
            "generate_tokens": self.generate_tokens,
            "description": self.description,
        }


@dataclass(frozen=True)
class MetricStats:
    """Summary of a repeated measurement.

    A single timing is noise. ArmForge reports the distribution so a reader
    can judge whether two configurations are actually different.
    """

    mean: float
    median: float
    minimum: float
    maximum: float
    stddev: float
    samples: int
    unit: str

    @property
    def relative_stddev(self) -> float:
        """Coefficient of variation. Above ~5% the machine was probably noisy."""
        return (self.stddev / self.mean) if self.mean else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "median": self.median,
            "min": self.minimum,
            "max": self.maximum,
            "stddev": self.stddev,
            "relative_stddev": round(self.relative_stddev, 4),
            "samples": self.samples,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class ModelRef:
    """Identity of the model file under test."""

    path: str
    name: str
    size_bytes: int
    quantization: str | None = None
    content_hash: str | None = None
    """Truncated SHA-256 of the file, so a result names an exact artifact."""
    n_params: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "size_bytes": self.size_bytes,
            "size_gb": round(self.size_bytes / (1024**3), 3),
            "quantization": self.quantization,
            "content_hash": self.content_hash,
            "n_params": self.n_params,
        }


@dataclass(frozen=True)
class RuntimeSpec:
    """Identity of the inference runtime build.

    ``build_flags`` matters as much as the version: the same llama.cpp commit
    built with and without KleidiAI produces different kernels on the same
    silicon, so the two are not interchangeable results.
    """

    name: str
    version: str
    binary_path: str
    build_flags: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        enabled = sorted(k for k, v in self.build_flags.items() if v is True)
        return f"{self.name}+{'+'.join(enabled)}" if enabled else self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "version": self.version,
            "binary_path": self.binary_path,
            "build_flags": dict(self.build_flags),
        }


@dataclass(frozen=True)
class BenchConfig:
    """One point in the configuration space."""

    model: ModelRef
    runtime: RuntimeSpec
    workload: Workload
    threads: int
    warmup_iterations: int = 1
    iterations: int = 5
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        quant = self.model.quantization or "unknown"
        return f"{quant} · {self.runtime.label} · {self.threads}t"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model": self.model.to_dict(),
            "runtime": self.runtime.to_dict(),
            "workload": self.workload.to_dict(),
            "threads": self.threads,
            "warmup_iterations": self.warmup_iterations,
            "iterations": self.iterations,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """A measured (or explicitly unmeasured) configuration.

    Prefill and decode are reported separately throughout. They are bound by
    different resources -- prefill by integer matrix throughput, decode by
    memory bandwidth -- so a single headline number hides the effect ArmForge
    exists to expose.
    """

    config: BenchConfig
    host: HostProfile
    status: Status
    prefill_tps: MetricStats | None = None
    decode_tps: MetricStats | None = None
    peak_memory_bytes: int | None = None
    model_load_ms: float | None = None
    wall_time_s: float | None = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    """Unmodified runtime output, so a reader can re-derive our numbers."""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def ok(self) -> bool:
        return self.status is Status.OK

    @property
    def ttft_ms(self) -> float | None:
        """Time to first token, derived from prefill throughput.

        This is a derived figure, not a directly observed one: prefill
        throughput times the prompt length gives the time to process the
        prompt, which is what a user perceives as the wait before the first
        token appears. It excludes model load time, which is reported
        separately.
        """
        if self.prefill_tps is None or not self.prefill_tps.mean:
            return None
        return (self.config.workload.prompt_tokens / self.prefill_tps.mean) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "config": self.config.to_dict(),
            "host": self.host.to_dict(),
            "metrics": {
                "prefill_tps": self.prefill_tps.to_dict() if self.prefill_tps else None,
                "decode_tps": self.decode_tps.to_dict() if self.decode_tps else None,
                "ttft_ms": round(self.ttft_ms, 2) if self.ttft_ms is not None else None,
                "peak_memory_bytes": self.peak_memory_bytes,
                "model_load_ms": self.model_load_ms,
                "wall_time_s": self.wall_time_s,
            },
            "error": self.error,
            "timestamp": self.timestamp,
            "raw": self.raw,
        }
