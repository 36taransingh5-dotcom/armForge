"""Shared test fixtures and builders."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pytest

from armforge.bench.types import BenchConfig, ModelRef, RuntimeSpec
from armforge.bench.workloads import SHORT
from armforge.hardware.types import CoreCluster, CoreKind, CpuProfile, HostProfile

_STRING, _ARRAY, _UINT32 = 8, 9, 4


def _encode_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _encode_value(value: Any) -> bytes:
    """Encode a metadata value with its type tag."""
    if isinstance(value, str):
        return struct.pack("<I", _STRING) + _encode_string(value)
    if isinstance(value, int):
        return struct.pack("<I", _UINT32) + struct.pack("<I", value)
    if isinstance(value, list):
        body = b"".join(_encode_string(v) for v in value)
        return (
            struct.pack("<I", _ARRAY)
            + struct.pack("<I", _STRING)
            + struct.pack("<Q", len(value))
            + body
        )
    raise TypeError(f"test builder cannot encode {type(value)}")


def build_gguf(
    path: Path,
    *,
    metadata: dict[str, Any],
    tensors: list[tuple[str, tuple[int, ...]]],
    version: int = 3,
    magic: bytes = b"GGUF",
    padding_bytes: int = 0,
) -> Path:
    """Write a syntactically valid GGUF header with no weight data.

    Enough for the analyzer, which never reads past the tensor index.
    """
    out = bytearray()
    out += magic
    out += struct.pack("<I", version)
    out += struct.pack("<Q", len(tensors))
    out += struct.pack("<Q", len(metadata))

    for key, value in metadata.items():
        out += _encode_string(key)
        out += _encode_value(value)

    for name, dims in tensors:
        out += _encode_string(name)
        out += struct.pack("<I", len(dims))
        for dim in dims:
            out += struct.pack("<Q", dim)
        out += struct.pack("<I", 0)  # ggml type
        out += struct.pack("<Q", 0)  # offset

    out += b"\x00" * padding_bytes
    path.write_bytes(bytes(out))
    return path


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
