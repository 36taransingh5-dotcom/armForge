"""Tests for the llama.cpp benchmark backend.

The JSON fixtures below are trimmed from real ``llama-bench -o json`` output
produced by commit a94d563 on an Apple M4, so the parser is tested against the
schema the tool actually emits rather than against an idealised one.
"""

from __future__ import annotations

import json

import pytest

from armforge.bench.llamacpp import (
    LlamaCppRunner,
    _cmake_flag,
    _find_row,
    discover_runtimes,
    parse_ggml_features,
)
from armforge.bench.types import RuntimeSpec, Status


def _row(**overrides):
    base = {
        "build_commit": "a94d563",
        "build_number": 1,
        "cpu_info": "Apple M4",
        "gpu_info": "",
        "backends": "CPU",
        "model_filename": "/models/qwen2.5-0.5b-instruct-q4_0.gguf",
        "model_type": "qwen2 1B Q4_0",
        "model_size": 422782464,
        "model_n_params": 630167424,
        "n_threads": 4,
        "n_prompt": 0,
        "n_gen": 0,
        "test_time": "2026-08-14T00:23:09Z",
        "avg_ns": 86343312,
        "stddev_ns": 730058,
        "avg_ts": 741.253552,
        "stddev_ts": 6.266503,
        "samples_ns": [86859458, 85827167],
        "samples_ts": [736.822, 745.685],
    }
    base.update(overrides)
    return base


PREFILL_ROW = _row(n_prompt=64, n_gen=0, samples_ts=[736.822, 745.685])
DECODE_ROW = _row(n_prompt=0, n_gen=16, samples_ts=[91.2, 89.8], avg_ts=90.5)


def test_finds_the_prefill_and_decode_rows():
    rows = [PREFILL_ROW, DECODE_ROW]
    assert _find_row(rows, prompt=True)["n_prompt"] == 64
    assert _find_row(rows, prompt=False)["n_gen"] == 16


def test_missing_row_is_none_not_an_error():
    assert _find_row([PREFILL_ROW], prompt=False) is None


def test_runner_parses_a_real_result(monkeypatch, bench_config, apple_host, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    binary = tmp_path / "llama-bench"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)

    runtime = RuntimeSpec(name="llama.cpp", version="a94d563", binary_path=str(binary))
    runner = LlamaCppRunner(runtime)

    from dataclasses import replace

    from armforge.bench import process
    from armforge.bench.types import ModelRef

    config = replace(
        bench_config,
        runtime=runtime,
        model=ModelRef(path=str(model), name="m", size_bytes=4, quantization="Q4_0"),
    )

    def fake_run(argv, **kwargs):
        return process.ProcessResult(
            argv=tuple(argv),
            returncode=0,
            stdout=json.dumps([PREFILL_ROW, DECODE_ROW]),
            stderr="",
            wall_time_s=12.5,
            peak_memory_bytes=900 * 1024**2,
            timed_out=False,
        )

    monkeypatch.setattr(process, "run", fake_run)
    result = runner.run(config, apple_host)

    assert result.status is Status.OK
    assert result.prefill_tps.mean == pytest.approx(741.2535, rel=1e-4)
    assert result.prefill_tps.samples == 2
    assert result.decode_tps.mean == pytest.approx(90.5, rel=1e-3)
    assert result.peak_memory_bytes == 900 * 1024**2
    assert result.raw["build_commit"] == "a94d563"
    assert result.raw["model_n_params"] == 630167424


def test_nonzero_exit_becomes_a_failed_result(monkeypatch, bench_config, apple_host, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    binary = tmp_path / "llama-bench"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)

    from dataclasses import replace

    from armforge.bench import process
    from armforge.bench.types import ModelRef

    runtime = RuntimeSpec(name="llama.cpp", version="x", binary_path=str(binary))
    config = replace(
        bench_config,
        runtime=runtime,
        model=ModelRef(path=str(model), name="m", size_bytes=4, quantization="Q4_0"),
    )

    def fake_run(argv, **kwargs):
        return process.ProcessResult(
            argv=tuple(argv),
            returncode=1,
            stdout="",
            stderr="llama_model_load: error loading model: unknown model architecture",
            wall_time_s=0.4,
            peak_memory_bytes=1024,
            timed_out=False,
        )

    monkeypatch.setattr(process, "run", fake_run)
    result = LlamaCppRunner(runtime).run(config, apple_host)

    assert result.status is Status.FAILED
    assert "unknown model architecture" in result.error


def test_missing_binary_is_unsupported_with_a_useful_reason(bench_config, apple_host):
    runtime = RuntimeSpec(name="llama.cpp", version="x", binary_path="/nonexistent/llama-bench")
    result = LlamaCppRunner(runtime).run(bench_config, apple_host)

    assert result.status is Status.UNSUPPORTED
    assert "setup-llama-cpp.sh" in result.error


def test_non_gguf_model_is_unsupported(bench_config, apple_host, tmp_path):
    from dataclasses import replace

    from armforge.bench.types import ModelRef

    binary = tmp_path / "llama-bench"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"x")

    runtime = RuntimeSpec(name="llama.cpp", version="x", binary_path=str(binary))
    config = replace(
        bench_config,
        runtime=runtime,
        model=ModelRef(path=str(model), name="m", size_bytes=1),
    )
    result = LlamaCppRunner(runtime).run(config, apple_host)

    assert result.status is Status.UNSUPPORTED
    assert "GGUF" in result.error


def test_argv_never_uses_a_shell(bench_config):
    runtime = RuntimeSpec(name="llama.cpp", version="x", binary_path="/bin/llama-bench")
    argv = LlamaCppRunner(runtime)._argv(bench_config)

    assert argv[0] == "/bin/llama-bench"
    assert "--threads" in argv
    assert argv[argv.index("--threads") + 1] == "4"
    assert all(isinstance(part, str) for part in argv)


# -- ggml feature probing --------------------------------------------------

SYSTEM_INFO = (
    "system_info: n_threads = 4 (n_threads_batch = 4) / 10 | "
    "CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | DOTPROD = 1 | "
    "MATMUL_INT8 = 1 | SME = 1 | LLAMAFILE = 1 | OPENMP = 0 | "
)


def test_parses_compiled_in_ggml_features():
    features = parse_ggml_features(SYSTEM_INFO)
    assert features["NEON"] is True
    assert features["MATMUL_INT8"] is True
    assert features["SME"] is True
    assert features["DOTPROD"] is True


def test_ignores_irrelevant_flags():
    """LLAMAFILE and OPENMP are not Arm capability claims."""
    features = parse_ggml_features(SYSTEM_INFO)
    assert "LLAMAFILE" not in features
    assert "OPENMP" not in features


def test_records_a_feature_that_was_compiled_out():
    """The case that matters: capable CPU, incapable build."""
    features = parse_ggml_features("CPU : NEON = 1 | MATMUL_INT8 = 0 | SME = 0 |")
    assert features["NEON"] is True
    assert features["MATMUL_INT8"] is False
    assert features["SME"] is False


def test_empty_output_yields_no_claims():
    """An unreadable probe must mean "not established", not "no features"."""
    assert parse_ggml_features("") == {}


# -- runtime discovery -----------------------------------------------------


def _make_build(root, name, **flags):
    build = root / f"build-{name}"
    (build / "bin").mkdir(parents=True)
    binary = build / "bin" / "llama-bench"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    lines = [f"{key}:BOOL={'ON' if value else 'OFF'}" for key, value in flags.items()]
    (build / "CMakeCache.txt").write_text("\n".join(lines) + "\n")
    return build


def test_cmake_flag_is_read_from_the_cache_not_the_directory_name(tmp_path):
    """A build must never claim KleidiAI just because it is named that way."""
    build = _make_build(tmp_path, "kleidiai", GGML_CPU_KLEIDIAI=False)
    assert _cmake_flag(build, "GGML_CPU_KLEIDIAI") is False


def test_cmake_flag_missing_from_cache_is_none(tmp_path):
    build = _make_build(tmp_path, "cpu", GGML_METAL=False)
    assert _cmake_flag(build, "GGML_CPU_KLEIDIAI") is None


def test_discover_finds_every_variant_with_its_real_flags(tmp_path):
    _make_build(tmp_path, "cpu", GGML_CPU_KLEIDIAI=False, GGML_ACCELERATE=False)
    _make_build(tmp_path, "kleidiai", GGML_CPU_KLEIDIAI=True, GGML_ACCELERATE=False)
    _make_build(tmp_path, "accelerate", GGML_CPU_KLEIDIAI=False, GGML_ACCELERATE=True)

    runtimes = {r.build_flags["variant"]: r for r in discover_runtimes(tmp_path)}

    assert set(runtimes) == {"cpu", "kleidiai", "accelerate"}
    assert runtimes["kleidiai"].build_flags["kleidiai"] is True
    assert runtimes["cpu"].build_flags["kleidiai"] is False
    assert runtimes["accelerate"].build_flags["accelerate"] is True


def test_discover_labels_distinguish_the_builds(tmp_path):
    """Two builds of the same commit must not be silently interchangeable."""
    _make_build(tmp_path, "cpu", GGML_CPU_KLEIDIAI=False, GGML_ACCELERATE=False)
    _make_build(tmp_path, "kleidiai", GGML_CPU_KLEIDIAI=True, GGML_ACCELERATE=False)

    labels = {r.label for r in discover_runtimes(tmp_path)}
    assert labels == {"llama.cpp", "llama.cpp+kleidiai"}


def test_discover_skips_directories_without_a_binary(tmp_path):
    (tmp_path / "build-broken").mkdir()
    assert discover_runtimes(tmp_path) == []


def test_discover_on_missing_cache_dir_is_empty(tmp_path):
    assert discover_runtimes(tmp_path / "absent") == []
