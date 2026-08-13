"""Hardware detection on macOS via ``sysctl``.

Apple silicon is a first-class ArmForge target: it is one of the most widely
available Armv9 development machines, and on M4 it exposes SME2 -- a matrix
engine that most Arm server parts do not have.
"""

from __future__ import annotations

import platform
import re
import subprocess

from .features import DARWIN_BARE_FEATURE_MAP, DARWIN_FEATURE_MAP
from .types import CoreCluster, CoreKind, CpuProfile, HostProfile

_SYSCTL_TIMEOUT = 5.0


def _sysctl_raw(names: list[str]) -> dict[str, str]:
    """Read sysctl leaves as a name -> value mapping. Missing leaves are omitted."""
    try:
        proc = subprocess.run(
            ["sysctl", *names],
            capture_output=True,
            text=True,
            timeout=_SYSCTL_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}

    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        # sysctl prints "name: value" on macOS.
        name, sep, value = line.partition(": ")
        if sep:
            out[name.strip()] = value.strip()
    return out


def _sysctl_int(values: dict[str, str], key: str) -> int | None:
    raw = values.get(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _detect_features(warnings: list[str]) -> tuple[frozenset[str], tuple[str, ...], int | None]:
    """Read ``hw.optional.arm.*`` and normalise to ArmForge feature keys."""
    values = _sysctl_raw(["-a"])
    if not values:
        warnings.append("sysctl -a returned nothing; CPU features could not be read")
        return frozenset(), (), None

    keys: set[str] = set()
    raw_present: list[str] = []

    for name, value in values.items():
        if not name.startswith("hw.optional."):
            continue
        if value.strip() != "1":
            continue
        leaf = name.rsplit(".", 1)[-1]
        if name.startswith("hw.optional.arm."):
            raw_present.append(leaf)
            mapped = DARWIN_FEATURE_MAP.get(leaf) or DARWIN_BARE_FEATURE_MAP.get(leaf)
        else:
            mapped = DARWIN_BARE_FEATURE_MAP.get(leaf)
            if mapped:
                raw_present.append(leaf)
        if mapped:
            keys.add(mapped)

    # Streaming vector length is reported in bytes.
    svl_bytes = _sysctl_int(values, "hw.optional.arm.sme_max_svl_b")
    sme_bits = svl_bytes * 8 if svl_bytes else None

    return frozenset(keys), tuple(sorted(raw_present)), sme_bits


def _detect_clusters(warnings: list[str]) -> tuple[CoreCluster, ...]:
    """Build the cluster list from ``hw.perflevel*``.

    macOS reports performance levels explicitly, which makes Apple silicon the
    easiest platform on which to observe heterogeneous topology. Level 0 is
    always the fastest.
    """
    base = _sysctl_raw(
        ["hw.nperflevels", "hw.physicalcpu", "hw.logicalcpu", "machdep.cpu.brand_string"]
    )
    nlevels = _sysctl_int(base, "hw.nperflevels") or 0

    if nlevels <= 1:
        physical = _sysctl_int(base, "hw.physicalcpu") or 0
        logical = _sysctl_int(base, "hw.logicalcpu") or physical
        return (
            CoreCluster(
                name="General",
                kind=CoreKind.UNIFORM,
                physical_cores=physical,
                logical_cores=logical,
            ),
        )

    clusters: list[CoreCluster] = []
    for level in range(nlevels):
        prefix = f"hw.perflevel{level}"
        values = _sysctl_raw(
            [
                f"{prefix}.name",
                f"{prefix}.physicalcpu",
                f"{prefix}.logicalcpu",
                f"{prefix}.l2cachesize",
            ]
        )
        physical = _sysctl_int(values, f"{prefix}.physicalcpu") or 0
        if physical == 0:
            continue
        name = values.get(f"{prefix}.name", f"Level {level}")
        # Apple names these "Performance" and "Efficiency"; level 0 is fastest.
        kind = CoreKind.PERFORMANCE if level == 0 else CoreKind.EFFICIENCY
        clusters.append(
            CoreCluster(
                name=name,
                kind=kind,
                physical_cores=physical,
                logical_cores=_sysctl_int(values, f"{prefix}.logicalcpu") or physical,
                l2_cache_bytes=_sysctl_int(values, f"{prefix}.l2cachesize"),
            )
        )

    if not clusters:
        warnings.append("hw.perflevel* reported no usable clusters")
    return tuple(clusters)


def _available_memory(warnings: list[str], page_size: int | None) -> int | None:
    """Approximate available memory from ``vm_stat``.

    macOS has no single "available" counter, so we sum the page classes that
    can be handed to a new allocation without swapping.
    """
    if not page_size:
        return None
    try:
        proc = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=_SYSCTL_TIMEOUT, check=False
        )
    except (OSError, subprocess.SubprocessError):
        warnings.append("vm_stat unavailable; free memory not measured")
        return None

    counts: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        match = re.match(r"Pages\s+([\w\s-]+):\s+(\d+)", line)
        if match:
            counts[match.group(1).strip()] = int(match.group(2))

    if not counts:
        return None
    reclaimable = (
        counts.get("free", 0) + counts.get("inactive", 0) + counts.get("speculative", 0)
    )
    return reclaimable * page_size


def detect() -> HostProfile:
    """Build a :class:`HostProfile` for the current macOS machine."""
    warnings: list[str] = []

    base = _sysctl_raw(["machdep.cpu.brand_string", "hw.memsize", "hw.pagesize"])
    model = base.get("machdep.cpu.brand_string", platform.processor() or "unknown")
    total_memory = _sysctl_int(base, "hw.memsize") or 0
    page_size = _sysctl_int(base, "hw.pagesize")

    if total_memory == 0:
        warnings.append("hw.memsize unavailable; total memory unknown")

    features, raw_features, sme_bits = _detect_features(warnings)
    clusters = _detect_clusters(warnings)

    architecture = platform.machine()
    implementer = "Apple" if architecture == "arm64" else None

    if architecture == "arm64":
        warnings.append(
            "macOS does not expose CPU affinity, so ArmForge cannot pin threads "
            "to specific clusters here; thread-count tuning is still available"
        )

    cpu = CpuProfile(
        architecture=architecture,
        model=model,
        clusters=clusters,
        features=features,
        raw_features=raw_features,
        implementer=implementer,
        sve_vector_bits=None,  # Apple silicon implements SME, not SVE.
        sme_vector_bits=sme_bits,
    )

    return HostProfile(
        cpu=cpu,
        os_name="macOS",
        os_release=platform.mac_ver()[0] or platform.release(),
        total_memory_bytes=total_memory,
        available_memory_bytes=_available_memory(warnings, page_size),
        page_size_bytes=page_size,
        detector="darwin-sysctl",
        warnings=tuple(warnings),
    )
