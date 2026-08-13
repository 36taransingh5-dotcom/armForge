"""Hardware detection on Linux via procfs, sysfs and ``prctl``.

This is the path that runs on Arm64 cloud instances (AWS Graviton, Ampere,
Azure Cobalt) and on Arm SBCs. Unlike macOS, Linux exposes ``MIDR_EL1`` per
CPU, which lets us identify the actual microarchitecture and detect
heterogeneous topology precisely rather than by inference from clock speed.
"""

from __future__ import annotations

import ctypes
import platform
import re
from pathlib import Path

from .features import EFFICIENCY_PARTS, LINUX_FEATURE_MAP, decode_midr
from .types import CoreCluster, CoreKind, CpuProfile, HostProfile

_CPU_ROOT = Path("/sys/devices/system/cpu")
_PROC_CPUINFO = Path("/proc/cpuinfo")
_PROC_MEMINFO = Path("/proc/meminfo")

# From <linux/prctl.h>
_PR_SVE_GET_VL = 51
_PR_SME_GET_VL = 64
_PR_VL_LEN_MASK = 0xFFFF


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def _read_int(path: Path) -> int | None:
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None


def _vector_length_bits(option: int) -> int | None:
    """Query an implemented vector length through ``prctl``.

    Returns ``None`` when the extension is absent -- prctl returns ``-EINVAL``,
    which ctypes surfaces as a negative result.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        result = libc.prctl(option, 0, 0, 0, 0)
    except (OSError, AttributeError):
        return None
    if result < 0:
        return None
    length_bytes = result & _PR_VL_LEN_MASK
    return length_bytes * 8 if length_bytes else None


def _parse_cpuinfo(text: str | None = None) -> tuple[list[str], dict[int, int]]:
    """Return HWCAP flag names and a ``{cpu_id: midr}`` mapping from cpuinfo.

    ``text`` may be supplied directly so the parser can be tested against
    captured output from machines we do not have to hand.
    """
    if text is None:
        text = _read_text(_PROC_CPUINFO)
    if not text:
        return [], {}

    flags: list[str] = []
    midrs: dict[int, int] = {}
    current_cpu: int | None = None
    implementer: int | None = None
    part: int | None = None

    def flush() -> None:
        if current_cpu is not None and implementer is not None and part is not None:
            midrs[current_cpu] = (implementer << 24) | (part << 4)

    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()

        if key == "processor":
            flush()
            implementer = part = None
            try:
                current_cpu = int(value)
            except ValueError:
                current_cpu = None
        elif key in ("Features", "flags") and not flags:
            # Arm kernels label the HWCAP line "Features"; x86 uses "flags".
            flags = value.split()
        elif key == "CPU implementer":
            try:
                implementer = int(value, 16)
            except ValueError:
                implementer = None
        elif key == "CPU part":
            try:
                part = int(value, 16)
            except ValueError:
                part = None

    flush()
    return flags, midrs


def _online_cpus() -> list[int]:
    """List online logical CPU ids from sysfs, falling back to a scan."""
    raw = _read_text(_CPU_ROOT / "online")
    cpus: list[int] = []
    if raw:
        for part in raw.strip().split(","):
            if "-" in part:
                start, _, end = part.partition("-")
                try:
                    cpus.extend(range(int(start), int(end) + 1))
                except ValueError:
                    continue
            elif part:
                try:
                    cpus.append(int(part))
                except ValueError:
                    continue
    if not cpus:
        cpus = sorted(
            int(p.name[3:]) for p in _CPU_ROOT.glob("cpu[0-9]*") if p.name[3:].isdigit()
        )
    return sorted(set(cpus))


def _midr_from_sysfs(cpu: int) -> int | None:
    raw = _read_text(_CPU_ROOT / f"cpu{cpu}" / "regs" / "identification" / "midr_el1")
    if raw is None:
        return None
    try:
        return int(raw.strip(), 16)
    except ValueError:
        return None


def _physical_core_key(cpu: int) -> tuple[int, int] | None:
    """``(package_id, core_id)`` identifying the physical core behind a thread."""
    topo = _CPU_ROOT / f"cpu{cpu}" / "topology"
    package = _read_int(topo / "physical_package_id")
    core = _read_int(topo / "core_id")
    if package is None or core is None:
        return None
    return (package, core)


def _build_clusters(cpus: list[int], cpuinfo_midrs: dict[int, int]) -> tuple[CoreCluster, ...]:
    """Group logical CPUs into clusters of identical cores.

    Cores are grouped by microarchitecture first (via MIDR) and by maximum
    frequency second, which covers SoCs that ship the same part at different
    clocks.
    """
    groups: dict[tuple[int | None, int | None], list[int]] = {}
    freqs: dict[int, int | None] = {}
    midrs: dict[int, int | None] = {}

    for cpu in cpus:
        midr = _midr_from_sysfs(cpu) or cpuinfo_midrs.get(cpu)
        freq = _read_int(_CPU_ROOT / f"cpu{cpu}" / "cpufreq" / "cpuinfo_max_freq")
        midrs[cpu] = midr
        freqs[cpu] = freq
        # Bucket frequency to 100 MHz to absorb reporting jitter.
        freq_bucket = (freq // 100_000) if freq else None
        groups.setdefault((midr, freq_bucket), []).append(cpu)

    if not groups:
        return ()

    # Physical core counts, so SMT threads are not double-counted.
    def physical_count(members: list[int]) -> int:
        keys = {_physical_core_key(c) for c in members}
        keys.discard(None)
        return len(keys) if keys else len(members)

    # Rank clusters fastest-first so cluster 0 is always the performance tier.
    def sort_key(item: tuple[tuple[int | None, int | None], list[int]]) -> tuple[int, int]:
        (_, freq_bucket), members = item
        return (-(freq_bucket or 0), min(members))

    ordered = sorted(groups.items(), key=sort_key)
    heterogeneous = len(ordered) > 1

    clusters: list[CoreCluster] = []
    for index, ((midr, _), members) in enumerate(ordered):
        implementer, core_name = decode_midr(midr) if midr else (None, None)

        if not heterogeneous:
            kind = CoreKind.UNIFORM
        elif core_name and core_name in EFFICIENCY_PARTS:
            kind = CoreKind.EFFICIENCY
        elif index == 0:
            kind = CoreKind.PERFORMANCE
        else:
            kind = CoreKind.EFFICIENCY

        max_freq = freqs[members[0]]
        clusters.append(
            CoreCluster(
                name=core_name or (f"Cluster {index}" if heterogeneous else "General"),
                kind=kind,
                physical_cores=physical_count(members),
                logical_cores=len(members),
                cpu_ids=tuple(sorted(members)),
                max_freq_mhz=(max_freq // 1000) if max_freq else None,
                core_name=core_name,
            )
        )

    return tuple(clusters)


def _distro_name() -> str:
    """Distribution name from ``/etc/os-release``, or a plain fallback."""
    try:
        return platform.freedesktop_os_release().get("NAME", "Linux")
    except (OSError, AttributeError):
        return "Linux"


def _meminfo(warnings: list[str]) -> tuple[int, int | None]:
    text = _read_text(_PROC_MEMINFO)
    if not text:
        warnings.append("/proc/meminfo unreadable; memory not measured")
        return 0, None

    values: dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"(\w+):\s+(\d+)\s+kB", line)
        if match:
            values[match.group(1)] = int(match.group(2)) * 1024
    return values.get("MemTotal", 0), values.get("MemAvailable")


def detect() -> HostProfile:
    """Build a :class:`HostProfile` for the current Linux machine."""
    warnings: list[str] = []

    flags, cpuinfo_midrs = _parse_cpuinfo()
    features = frozenset(LINUX_FEATURE_MAP[flag] for flag in flags if flag in LINUX_FEATURE_MAP)
    if not flags:
        warnings.append("/proc/cpuinfo exposed no feature flags")

    cpus = _online_cpus()
    clusters = _build_clusters(cpus, cpuinfo_midrs)
    if not clusters:
        warnings.append("could not enumerate CPUs from sysfs")

    architecture = platform.machine()
    implementer = None
    model = platform.processor() or architecture
    first_midr = next(
        (m for m in (_midr_from_sysfs(c) or cpuinfo_midrs.get(c) for c in cpus) if m),
        None,
    )
    if first_midr:
        implementer, core_name = decode_midr(first_midr)
        names = sorted({c.core_name for c in clusters if c.core_name})
        if names:
            model = f"{implementer or 'Arm'} {' + '.join(names)}"
        elif core_name:
            model = f"{implementer or 'Arm'} {core_name}"
        else:
            warnings.append(f"unrecognised MIDR 0x{first_midr:08x}; core model not identified")

    total_memory, available_memory = _meminfo(warnings)

    cpu = CpuProfile(
        architecture=architecture,
        model=model,
        clusters=clusters,
        features=features,
        raw_features=tuple(flags),
        implementer=implementer,
        sve_vector_bits=_vector_length_bits(_PR_SVE_GET_VL) if "sve" in features else None,
        sme_vector_bits=_vector_length_bits(_PR_SME_GET_VL) if "sme" in features else None,
    )

    return HostProfile(
        cpu=cpu,
        os_name=_distro_name(),
        os_release=platform.release(),
        total_memory_bytes=total_memory,
        available_memory_bytes=available_memory,
        page_size_bytes=None,
        detector="linux-sysfs",
        warnings=tuple(warnings),
    )
