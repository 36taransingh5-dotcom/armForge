"""Core data types describing the host machine.

These types are deliberately free of any detection logic so that they can be
serialised, round-tripped through JSON, and constructed by tests without
touching the real machine.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class CoreKind(str, enum.Enum):
    """Role a CPU cluster plays in a (potentially) heterogeneous topology."""

    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency"
    UNIFORM = "uniform"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CoreCluster:
    """A group of identical cores.

    On a uniform server part (Graviton, most Neoverse designs) there is exactly
    one cluster. On a big.LITTLE / Apple-style part there are two or more, and
    the distinction matters enormously for thread-count selection.
    """

    name: str
    kind: CoreKind
    physical_cores: int
    logical_cores: int
    cpu_ids: tuple[int, ...] = ()
    max_freq_mhz: int | None = None
    l2_cache_bytes: int | None = None
    core_name: str | None = None
    """Decoded microarchitecture, e.g. "Neoverse-V2" or "Cortex-A520"."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "physical_cores": self.physical_cores,
            "logical_cores": self.logical_cores,
            "cpu_ids": list(self.cpu_ids),
            "max_freq_mhz": self.max_freq_mhz,
            "l2_cache_bytes": self.l2_cache_bytes,
            "core_name": self.core_name,
        }


@dataclass(frozen=True)
class CpuProfile:
    """Everything ArmForge knows about the CPU."""

    architecture: str
    model: str
    clusters: tuple[CoreCluster, ...]
    features: frozenset[str] = frozenset()
    """Normalised feature keys (see ``features.py``), e.g. ``{"i8mm", "sme2"}``."""
    raw_features: tuple[str, ...] = ()
    """Feature names exactly as the OS reported them, so nothing is lost."""
    implementer: str | None = None
    sve_vector_bits: int | None = None
    sme_vector_bits: int | None = None

    @property
    def is_arm64(self) -> bool:
        return self.architecture in ("arm64", "aarch64")

    @property
    def physical_cores(self) -> int:
        return sum(c.physical_cores for c in self.clusters)

    @property
    def logical_cores(self) -> int:
        return sum(c.logical_cores for c in self.clusters)

    @property
    def is_heterogeneous(self) -> bool:
        """True when the machine has more than one kind of core.

        This is the single most under-considered variable in CPU inference
        tuning: on a heterogeneous part, ``nproc`` is the wrong thread count.
        """
        kinds = {c.kind for c in self.clusters}
        return len([c for c in self.clusters if c.physical_cores > 0]) > 1 and len(kinds) > 1

    @property
    def performance_cores(self) -> int:
        """Physical cores in performance clusters, or all cores if uniform."""
        perf = sum(c.physical_cores for c in self.clusters if c.kind is CoreKind.PERFORMANCE)
        return perf if perf else self.physical_cores

    def has(self, *feature_keys: str) -> bool:
        """True only if every named feature is present."""
        return all(key in self.features for key in feature_keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "is_arm64": self.is_arm64,
            "model": self.model,
            "implementer": self.implementer,
            "physical_cores": self.physical_cores,
            "logical_cores": self.logical_cores,
            "performance_cores": self.performance_cores,
            "is_heterogeneous": self.is_heterogeneous,
            "clusters": [c.to_dict() for c in self.clusters],
            "features": sorted(self.features),
            "raw_features": list(self.raw_features),
            "sve_vector_bits": self.sve_vector_bits,
            "sme_vector_bits": self.sme_vector_bits,
        }


@dataclass(frozen=True)
class HostProfile:
    """CPU plus the surrounding machine and OS context."""

    cpu: CpuProfile
    os_name: str
    os_release: str
    total_memory_bytes: int
    available_memory_bytes: int | None = None
    page_size_bytes: int | None = None
    detector: str = "unknown"
    """Which backend produced this profile, e.g. ``darwin-sysctl``."""
    warnings: tuple[str, ...] = field(default_factory=tuple)
    """Things we could not determine. Surfaced rather than guessed."""

    @property
    def total_memory_gb(self) -> float:
        return self.total_memory_bytes / (1024**3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu": self.cpu.to_dict(),
            "os_name": self.os_name,
            "os_release": self.os_release,
            "total_memory_bytes": self.total_memory_bytes,
            "total_memory_gb": round(self.total_memory_gb, 2),
            "available_memory_bytes": self.available_memory_bytes,
            "page_size_bytes": self.page_size_bytes,
            "detector": self.detector,
            "warnings": list(self.warnings),
        }
