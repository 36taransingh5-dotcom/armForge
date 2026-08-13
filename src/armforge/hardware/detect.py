"""Platform dispatch for hardware detection."""

from __future__ import annotations

import os
import platform
import sys
from functools import lru_cache

from .types import CoreCluster, CoreKind, CpuProfile, HostProfile


def _fallback() -> HostProfile:
    """Minimal profile for platforms without a dedicated backend.

    We report what the standard library can tell us and flag the rest as
    undetermined rather than filling in plausible values.
    """
    logical = os.cpu_count() or 1
    cpu = CpuProfile(
        architecture=platform.machine(),
        model=platform.processor() or platform.machine() or "unknown",
        clusters=(
            CoreCluster(
                name="General",
                kind=CoreKind.UNKNOWN,
                physical_cores=logical,
                logical_cores=logical,
            ),
        ),
    )
    return HostProfile(
        cpu=cpu,
        os_name=platform.system() or "unknown",
        os_release=platform.release(),
        total_memory_bytes=0,
        detector="fallback",
        warnings=(
            f"no ArmForge detection backend for platform {sys.platform!r}; "
            "CPU features, topology and memory are unknown",
        ),
    )


@lru_cache(maxsize=1)
def detect_host() -> HostProfile:
    """Detect the current machine.

    Cached: detection shells out to ``sysctl``/``vm_stat`` and reads a few
    dozen sysfs files, and the answer cannot change within a process.
    """
    if sys.platform == "darwin":
        from . import _darwin

        return _darwin.detect()
    if sys.platform.startswith("linux"):
        from . import _linux

        return _linux.detect()
    return _fallback()
