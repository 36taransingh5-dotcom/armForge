"""Tests for the runner contract and result types."""

from __future__ import annotations

import json
from dataclasses import replace

from armforge.bench.mock import MockRunner
from armforge.bench.types import Status


def test_successful_run_reports_both_phases(bench_config, apple_host):
    result = MockRunner().run(bench_config, apple_host)

    assert result.status is Status.OK
    assert result.ok
    assert result.prefill_tps is not None
    assert result.decode_tps is not None
    assert result.error is None


def test_unavailable_runner_yields_unsupported_not_an_exception(bench_config, apple_host):
    result = MockRunner(available=False).run(bench_config, apple_host)

    assert result.status is Status.UNSUPPORTED
    assert not result.ok
    assert "disabled" in result.error
    assert result.prefill_tps is None


def test_unsupported_configuration_explains_itself(bench_config, apple_host):
    runner = MockRunner(unsupported_quantizations=frozenset({"Q4_0"}))
    result = runner.run(bench_config, apple_host)

    assert result.status is Status.UNSUPPORTED
    assert "Q4_0" in result.error


def test_a_crashing_runner_becomes_a_failed_result(bench_config, apple_host):
    """A runner that raises must not abort the whole sweep."""
    runner = MockRunner(fail_on_threads=frozenset({4}))
    result = runner.run(bench_config, apple_host)

    assert result.status is Status.FAILED
    assert "RuntimeError" in result.error
    assert "mock failure" in result.error


def test_ttft_is_derived_from_prefill_throughput(bench_config, apple_host):
    result = MockRunner().run(bench_config, apple_host)

    expected_ms = (bench_config.workload.prompt_tokens / result.prefill_tps.mean) * 1000
    assert result.ttft_ms == expected_ms


def test_ttft_is_none_when_prefill_was_not_measured(bench_config, apple_host):
    result = MockRunner(available=False).run(bench_config, apple_host)
    assert result.ttft_ms is None


def test_result_serialises_to_json_with_full_provenance(bench_config, apple_host):
    result = MockRunner().run(bench_config, apple_host)
    payload = json.loads(json.dumps(result.to_dict()))

    assert payload["status"] == "ok"
    assert payload["config"]["model"]["quantization"] == "Q4_0"
    assert payload["config"]["model"]["content_hash"] is None
    assert payload["config"]["runtime"]["version"] == "abc1234"
    assert payload["config"]["threads"] == 4
    assert payload["config"]["workload"]["prompt_tokens"] == 128
    assert payload["host"]["cpu"]["model"] == "Apple M4"
    assert payload["metrics"]["prefill_tps"]["unit"] == "tok/s"
    assert payload["timestamp"]


def test_mock_results_are_marked_synthetic(bench_config, apple_host):
    """Fake numbers must be identifiable as fake wherever they surface."""
    result = MockRunner().run(bench_config, apple_host)
    assert result.raw["synthetic"] is True
    assert result.to_dict()["raw"]["synthetic"] is True


def test_config_label_describes_the_configuration(bench_config):
    assert bench_config.label == "Q4_0 · llama.cpp · 4t"


def test_runtime_label_includes_enabled_build_flags(bench_config):
    """A KleidiAI build is not the same result as a stock build."""
    stock = bench_config.runtime
    kleidi = replace(stock, build_flags={"kleidiai": True, "openmp": False})

    assert stock.label == "llama.cpp"
    assert kleidi.label == "llama.cpp+kleidiai"


def test_thread_oversubscription_hurts_decode_on_heterogeneous_hosts(
    bench_config, apple_host, graviton_host
):
    """Exercises the sweep logic the mock exists to support.

    On the 4P+6E host, asking for 10 threads must score worse for decode than
    asking for 4. On the uniform 16-core host it must not. The numbers are
    synthetic; the branch being tested is real.
    """
    runner = MockRunner()

    apple_4 = runner.run(replace(bench_config, threads=4), apple_host)
    apple_10 = runner.run(replace(bench_config, threads=10), apple_host)
    assert apple_10.decode_tps.mean < apple_4.decode_tps.mean

    graviton_4 = runner.run(replace(bench_config, threads=4), graviton_host)
    graviton_16 = runner.run(replace(bench_config, threads=16), graviton_host)
    assert graviton_16.decode_tps.mean > graviton_4.decode_tps.mean
