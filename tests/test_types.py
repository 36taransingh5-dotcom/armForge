"""Tests for the host/CPU profile value types."""

from __future__ import annotations

import json

from armforge.hardware.types import CoreCluster, CoreKind, CpuProfile, HostProfile


def _cluster(kind: CoreKind, cores: int, name: str = "c") -> CoreCluster:
    return CoreCluster(name=name, kind=kind, physical_cores=cores, logical_cores=cores)


def _cpu(*clusters: CoreCluster, features: set[str] | None = None) -> CpuProfile:
    return CpuProfile(
        architecture="arm64",
        model="test",
        clusters=clusters,
        features=frozenset(features or set()),
    )


def test_uniform_cpu_is_not_heterogeneous():
    cpu = _cpu(_cluster(CoreKind.UNIFORM, 16))
    assert not cpu.is_heterogeneous
    assert cpu.physical_cores == 16


def test_big_little_cpu_is_heterogeneous():
    cpu = _cpu(
        _cluster(CoreKind.PERFORMANCE, 4, "Performance"),
        _cluster(CoreKind.EFFICIENCY, 6, "Efficiency"),
    )
    assert cpu.is_heterogeneous
    assert cpu.physical_cores == 10
    assert cpu.performance_cores == 4


def test_performance_cores_falls_back_to_all_cores_when_uniform():
    """On Graviton there is no P/E split, so every core is a performance core."""
    cpu = _cpu(_cluster(CoreKind.UNIFORM, 16))
    assert cpu.performance_cores == 16


def test_empty_clusters_are_not_counted_as_a_second_kind():
    cpu = _cpu(
        _cluster(CoreKind.PERFORMANCE, 8),
        _cluster(CoreKind.EFFICIENCY, 0),
    )
    assert not cpu.is_heterogeneous


def test_smt_threads_do_not_inflate_physical_core_count():
    cpu = _cpu(
        CoreCluster(name="General", kind=CoreKind.UNIFORM, physical_cores=8, logical_cores=16)
    )
    assert cpu.physical_cores == 8
    assert cpu.logical_cores == 16


def test_is_arm64_accepts_both_spellings():
    for arch in ("arm64", "aarch64"):
        assert CpuProfile(architecture=arch, model="m", clusters=()).is_arm64
    assert not CpuProfile(architecture="x86_64", model="m", clusters=()).is_arm64


def test_has_requires_every_named_feature():
    cpu = _cpu(_cluster(CoreKind.UNIFORM, 4), features={"i8mm", "dotprod"})
    assert cpu.has("i8mm")
    assert cpu.has("i8mm", "dotprod")
    assert not cpu.has("i8mm", "sve")


def test_host_profile_round_trips_through_json():
    host = HostProfile(
        cpu=_cpu(_cluster(CoreKind.PERFORMANCE, 4), features={"i8mm"}),
        os_name="Linux",
        os_release="6.8.0",
        total_memory_bytes=16 * 1024**3,
        detector="test",
        warnings=("something was undetermined",),
    )
    payload = json.loads(json.dumps(host.to_dict()))
    assert payload["cpu"]["features"] == ["i8mm"]
    assert payload["cpu"]["is_arm64"] is True
    assert payload["total_memory_gb"] == 16.0
    assert payload["warnings"] == ["something was undetermined"]
