"""Shared test fixtures and builders."""

from __future__ import annotations

import pytest

from armforge.bench.types import BenchConfig, ModelRef, RuntimeSpec
from armforge.bench.workloads import SHORT
from armforge.hardware.types import CoreCluster, CoreKind, CpuProfile, HostProfile


@pytest.fixture
def apple_host() -> HostProfile:
    """A heterogeneous Arm host modelled on Apple M4: SME2, no SVE."""
    return HostProfile(
        cpu=CpuProfile(
            architecture="arm64",
            model="Apple M4",
            clusters=(
                CoreCluster("Performance", CoreKind.PERFORMANCE, 4, 4),
                CoreCluster("Efficiency", CoreKind.EFFICIENCY, 6, 6),
            ),
            features=frozenset(
                {"neon", "fp16", "dotprod", "i8mm", "bf16", "sme", "sme2", "sme_i8i32"}
            ),
            sme_vector_bits=512,
        ),
        os_name="macOS",
        os_release="26.5",
        total_memory_bytes=16 * 1024**3,
        detector="darwin-sysctl",
    )


@pytest.fixture
def graviton_host() -> HostProfile:
    """A uniform Arm server host modelled on Graviton3: SVE, no SME."""
    return HostProfile(
        cpu=CpuProfile(
            architecture="aarch64",
            model="Arm Neoverse-V1",
            clusters=(CoreCluster("Neoverse-V1", CoreKind.UNIFORM, 16, 16),),
            features=frozenset(
                {"neon", "fp16", "dotprod", "i8mm", "bf16", "sve", "svei8mm", "svebf16"}
            ),
            sve_vector_bits=256,
        ),
        os_name="Ubuntu",
        os_release="6.8.0-aws",
        total_memory_bytes=32 * 1024**3,
        detector="linux-sysfs",
    )


@pytest.fixture
def bench_config() -> BenchConfig:
    return BenchConfig(
        model=ModelRef(
            path="/models/test.gguf",
            name="test",
            size_bytes=400 * 1024**2,
            quantization="Q4_0",
        ),
        runtime=RuntimeSpec(
            name="llama.cpp", version="abc1234", binary_path="/bin/llama-bench"
        ),
        workload=SHORT,
        threads=4,
    )
