"""Tests for rebuilding results from saved JSON.

The important property is that a round trip changes nothing: a report
regenerated from a committed artifact must report exactly the numbers that
were measured, on a machine that may not resemble the one that measured them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from armforge.bench.deserialize import (
    DeserializationError,
    benchmark_result,
    host_profile,
    metric_stats,
    sweep_report,
)
from armforge.bench.stats import summarize
from armforge.bench.types import BenchConfig, BenchmarkResult, ModelRef, RuntimeSpec, Status
from armforge.bench.workloads import LONG_CONTEXT
from armforge.hardware.types import CoreKind
from armforge.optimize.candidates import CandidatePlan, Pruned
from armforge.optimize.sweep import SweepReport

REPO_RESULTS = Path(__file__).resolve().parent.parent / "results"


def _report(host) -> SweepReport:
    model = ModelRef(path="/m/q4_0.gguf", name="q", size_bytes=428867584, quantization="Q4_0")
    runtime = RuntimeSpec(
        name="llama.cpp",
        version="a94d563",
        binary_path="/b/llama-bench",
        build_flags={"variant": "cpu", "kleidiai": False},
    )
    results = [
        BenchmarkResult(
            config=BenchConfig(
                model=model,
                runtime=runtime,
                workload=LONG_CONTEXT,
                threads=threads,
                iterations=5,
            ),
            host=host,
            status=Status.OK,
            prefill_tps=summarize([prefill - 1, prefill + 1], "tok/s"),
            decode_tps=summarize([decode - 1, decode + 1], "tok/s"),
            peak_memory_bytes=794_000_000,
        )
        for threads, prefill, decode in ((2, 277.6, 99.5), (6, 424.3, 78.9))
    ]
    results.append(
        BenchmarkResult(
            config=BenchConfig(model=model, runtime=runtime, workload=LONG_CONTEXT, threads=10),
            host=host,
            status=Status.FAILED,
            error="segfault",
        )
    )
    plan = CandidatePlan(
        candidates=(), pruned=(Pruned(label="F16", reason="too large"),), notes=("n",)
    )
    return SweepReport(host=host, plan=plan, results=results)


def test_round_trip_preserves_every_measurement(apple_host):
    original = _report(apple_host)
    restored = sweep_report(json.loads(json.dumps(original.to_dict())))

    assert len(restored.results) == len(original.results)
    for before, after in zip(original.results, restored.results, strict=True):
        assert after.status is before.status
        assert after.config.threads == before.config.threads
        assert after.config.model.quantization == before.config.model.quantization
        if before.prefill_tps:
            assert after.prefill_tps.mean == pytest.approx(before.prefill_tps.mean)
            assert after.prefill_tps.stddev == pytest.approx(before.prefill_tps.stddev)
            assert after.prefill_tps.samples == before.prefill_tps.samples
        assert after.peak_memory_bytes == before.peak_memory_bytes


def test_round_trip_preserves_the_hardware_profile(apple_host):
    restored = sweep_report(json.loads(json.dumps(_report(apple_host).to_dict())))
    cpu = restored.host.cpu

    assert cpu.model == "Apple M4"
    assert cpu.is_heterogeneous
    assert cpu.performance_cores == 4
    assert cpu.physical_cores == 10
    assert "i8mm" in cpu.features
    assert cpu.sme_vector_bits == 512
    assert cpu.clusters[0].kind is CoreKind.PERFORMANCE
    assert cpu.clusters[1].kind is CoreKind.EFFICIENCY


def test_round_trip_keeps_failures_and_pruned_candidates(apple_host):
    restored = sweep_report(json.loads(json.dumps(_report(apple_host).to_dict())))

    assert len(restored.failed) == 1
    assert restored.failed[0].error == "segfault"
    assert restored.plan.pruned[0].reason == "too large"


def test_unmeasured_metric_stays_none_rather_than_zero():
    assert metric_stats(None) is None
    assert metric_stats({}) is None


def test_metric_stats_reconstructs_the_distribution():
    stats = metric_stats(
        {
            "mean": 424.3,
            "median": 424.0,
            "min": 410.0,
            "max": 440.0,
            "stddev": 9.5,
            "samples": 5,
            "unit": "tok/s",
        }
    )
    assert stats.mean == 424.3
    assert stats.minimum == 410.0
    assert stats.samples == 5
    assert stats.relative_stddev > 0.02


def test_unknown_core_kind_degrades_instead_of_crashing():
    host = host_profile(
        {
            "cpu": {
                "architecture": "aarch64",
                "model": "Future Core",
                "clusters": [{"physical_cores": 8, "kind": "quantum"}],
            }
        }
    )
    assert host.cpu.clusters[0].kind is CoreKind.UNKNOWN
    assert host.cpu.physical_cores == 8


def test_missing_required_key_raises_rather_than_defaulting():
    with pytest.raises(DeserializationError, match="architecture"):
        host_profile({"cpu": {"model": "x"}})


def test_unknown_status_is_rejected(apple_host):
    with pytest.raises(DeserializationError, match="unknown status"):
        benchmark_result(
            {"status": "invented", "config": {}},
            apple_host,
        )


def test_a_single_benchmark_artifact_is_rejected_with_a_useful_message():
    """`benchmark --output` writes a different shape; say so plainly."""
    with pytest.raises(DeserializationError, match="not an ArmForge sweep"):
        sweep_report({"status": "ok", "config": {}, "metrics": {}})


def test_unknown_extra_keys_are_ignored(apple_host):
    """A newer ArmForge adding a field must not break an older reader."""
    payload = json.loads(json.dumps(_report(apple_host).to_dict()))
    payload["some_future_field"] = {"anything": 1}
    payload["host"]["cpu"]["new_extension"] = True

    restored = sweep_report(payload)
    assert len(restored.results) == 3


# -- against the committed artifacts ---------------------------------------


@pytest.mark.parametrize(
    "name",
    ["neoverse-n2-sweep-longcontext.json", "m4-sweep-longcontext.json"],
)
def test_committed_artifacts_round_trip(name):
    """The real files in results/ must remain readable by this code."""
    path = REPO_RESULTS / name
    if not path.is_file():
        pytest.skip(f"{name} not present")

    original = json.loads(path.read_text())
    restored = sweep_report(original)

    assert len(restored.results) == len(original["results"])
    assert restored.host.cpu.model == original["host"]["cpu"]["model"]

    # Every prefill mean must survive unchanged; this is the whole point.
    before = sorted(
        (r["config"]["label"], r["metrics"]["prefill_tps"]["mean"])
        for r in original["results"]
        if r["status"] == "ok" and r["metrics"]["prefill_tps"]
    )
    after = sorted(
        (r.config.label, r.prefill_tps.mean) for r in restored.results if r.ok and r.prefill_tps
    )
    assert before == after


def test_neoverse_artifact_regenerates_its_own_recommendation():
    """Re-scoring the CI sweep must reproduce what CI concluded."""
    path = REPO_RESULTS / "neoverse-n2-sweep-longcontext.json"
    if not path.is_file():
        pytest.skip("Neoverse artifact not present")

    from armforge.optimize import Objective, recommend

    payload = json.loads(path.read_text())
    report = sweep_report(payload)
    rec = recommend(report.results, report.host, Objective.BEST_BALANCE)

    assert rec is not None
    assert rec.deployment_command == payload["recommendation"]["deployment_command"]
    # The uniform machine's phases agree; the split collapses to one number.
    assert rec.prefill.threads == rec.decode.threads == 4
