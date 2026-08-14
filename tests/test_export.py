"""Tests for the deployment package exporter and HTML report."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import replace

import pytest

from armforge.bench.stats import summarize
from armforge.bench.types import BenchConfig, BenchmarkResult, ModelRef, RuntimeSpec, Status
from armforge.bench.workloads import LONG_CONTEXT
from armforge.optimize.candidates import CandidatePlan, Pruned
from armforge.optimize.recommend import recommend
from armforge.optimize.scoring import Objective
from armforge.optimize.sweep import SweepReport
from armforge.report import content_hash, export_package, render_html

STOCK = RuntimeSpec(
    name="llama.cpp",
    version="a94d563",
    binary_path="/b/cpu/llama-bench",
    build_flags={"variant": "cpu", "kleidiai": False, "accelerate": False},
)

#: The real M4 shape: prefill peaks at 6, decode at 2, 10 collapses.
CURVE = {
    1: (138.7, 68.5),
    2: (277.6, 99.5),
    4: (353.1, 92.3),
    6: (424.3, 78.9),
    8: (418.7, 51.8),
    10: (169.0, 3.4),
}


@pytest.fixture
def sweep(apple_host, tmp_path):
    model_file = tmp_path / "qwen2.5-0.5b-instruct-q4_0.gguf"
    model_file.write_bytes(b"GGUF" + b"\x00" * 4096)
    model = ModelRef(
        path=str(model_file),
        name="qwen2.5-0.5b-instruct",
        size_bytes=428867584,
        quantization="Q4_0",
    )

    results = []
    for threads, (prefill, decode) in CURVE.items():
        results.append(
            BenchmarkResult(
                config=BenchConfig(
                    model=model,
                    runtime=STOCK,
                    workload=LONG_CONTEXT,
                    threads=threads,
                    iterations=5,
                ),
                host=apple_host,
                status=Status.OK,
                prefill_tps=summarize([prefill - 2, prefill + 2], "tok/s"),
                decode_tps=summarize([decode - 1, decode + 1], "tok/s"),
                peak_memory_bytes=831_000_000,
            )
        )

    plan = CandidatePlan(
        candidates=(),
        pruned=(Pruned(label="F16", reason="would not fit in memory"),),
        notes=(),
    )
    report = SweepReport(host=apple_host, plan=plan, results=results)
    report.finished_at = "2026-08-14T00:00:00+00:00"
    return report


@pytest.fixture
def rec(sweep, apple_host):
    return recommend(sweep.results, apple_host, Objective.BEST_BALANCE)


def test_export_writes_every_expected_file(sweep, rec, tmp_path):
    out = tmp_path / "package"
    written = export_package(sweep, rec, out)

    names = {p.name for p in written}
    assert names == {
        "config.json",
        "benchmark.json",
        "report.html",
        "README.md",
        "run.sh",
        "Dockerfile",
    }
    assert all(p.is_file() for p in written)


def test_run_script_is_executable_and_valid_bash(sweep, rec, tmp_path):
    out = tmp_path / "package"
    export_package(sweep, rec, out)
    script = out / "run.sh"

    assert script.stat().st_mode & 0o111, "run.sh must be executable"
    # bash -n parses without executing; a syntax error fails the build.
    result = subprocess.run(
        ["bash", "-n", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_run_script_carries_the_per_phase_thread_split(sweep, rec, tmp_path):
    out = tmp_path / "package"
    export_package(sweep, rec, out)
    text = (out / "run.sh").read_text()

    assert f"ARMFORGE_THREADS_GENERATION:-{rec.decode.threads}" in text
    assert f"ARMFORGE_THREADS_BATCH:-{rec.prefill.threads}" in text
    assert rec.prefill.threads != rec.decode.threads


def test_blank_thread_override_falls_back_to_the_measured_value(sweep, rec, tmp_path):
    """A blank environment variable must not discard the tuning.

    ``${VAR:-default}`` substitutes on empty as well as unset, so exporting an
    empty string gives back the measured thread count rather than dropping the
    flag. This also exercises the `set -e` hazard: an if-block is used instead
    of `[[ test ]] && cmd`, which would return non-zero and exit the script
    silently whenever the test failed.
    """
    out = tmp_path / "package"
    export_package(sweep, rec, out)

    result = subprocess.run(
        ["bash", str(out / "run.sh")],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": "/usr/bin:/bin",
            "ARMFORGE_MODEL": str(sweep.results[0].config.model.path),
            "LLAMA_BIN": "/bin/echo",
            "ARMFORGE_THREADS_GENERATION": "",
            "ARMFORGE_THREADS_BATCH": "",
        },
    )

    assert result.returncode == 0, result.stderr
    assert f"-t {rec.decode.threads}" in result.stdout
    assert f"-tb {rec.prefill.threads}" in result.stdout


def test_unmeasured_phase_omits_its_flag(sweep, rec):
    """When ArmForge could not determine a thread count, it must not invent one."""
    from armforge.report.package import _run_script

    script = _run_script(replace(rec, prefill=None))

    assert "ARMFORGE_THREADS_BATCH:-}" in script, "no default should be baked in"
    assert f"ARMFORGE_THREADS_GENERATION:-{rec.decode.threads}" in script


def test_run_script_invokes_the_binary_with_both_flags(sweep, rec, tmp_path):
    """End-to-end: the script builds the exact command line it documents."""
    out = tmp_path / "package"
    export_package(sweep, rec, out)

    result = subprocess.run(
        ["bash", str(out / "run.sh"), "-p", "hello"],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": "/usr/bin:/bin",
            "ARMFORGE_MODEL": str(sweep.results[0].config.model.path),
            "LLAMA_BIN": "/bin/echo",
        },
    )

    assert result.returncode == 0, result.stderr
    assert f"-t {rec.decode.threads}" in result.stdout
    assert f"-tb {rec.prefill.threads}" in result.stdout
    assert "-p hello" in result.stdout, "user arguments must be forwarded"


def test_run_script_fails_clearly_when_the_model_is_missing(sweep, rec, tmp_path):
    out = tmp_path / "package"
    export_package(sweep, rec, out)

    result = subprocess.run(
        ["bash", str(out / "run.sh")],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": "/usr/bin:/bin",
            "ARMFORGE_MODEL": "/nowhere/absent.gguf",
            "LLAMA_BIN": "/bin/echo",
        },
    )

    assert result.returncode == 1
    assert "model not found" in result.stderr


def test_config_json_is_machine_readable_and_complete(sweep, rec, tmp_path):
    out = tmp_path / "package"
    export_package(sweep, rec, out)
    config = json.loads((out / "config.json").read_text())

    assert config["model"]["quantization"] == "Q4_0"
    assert config["threads"]["generation"] == rec.decode.threads
    assert config["threads"]["batch"] == rec.prefill.threads
    assert config["measured_on"]["cpu"] == "Apple M4"
    assert config["measured_on"]["heterogeneous"] is True
    assert "i8mm" in config["measured_on"]["arm_features"]
    assert config["measured"]["prefill_tok_s"] > 0
    assert config["command"].startswith("llama-completion")


def test_benchmark_json_keeps_full_provenance(sweep, rec, tmp_path):
    out = tmp_path / "package"
    export_package(sweep, rec, out)
    payload = json.loads((out / "benchmark.json").read_text())

    assert len(payload["results"]) == len(CURVE)
    assert payload["recommendation"]["deployment_command"]
    assert payload["host"]["cpu"]["model"] == "Apple M4"
    # Pruned candidates are part of the record, not dropped.
    assert payload["plan"]["pruned"][0]["reason"] == "would not fit in memory"


def test_html_report_is_entirely_self_contained(sweep, rec, tmp_path):
    """No external fetches: the file must work with no network at all."""
    out = tmp_path / "package"
    export_package(sweep, rec, out)
    page = (out / "report.html").read_text()

    assert "<svg" in page, "charts should be inline SVG"
    assert "@import" not in page
    assert "<script" not in page
    assert re.search(r"(?:src|href)\s*=\s*[\"']https?://", page) is None

    # The SVG namespace is an identifier, not a request -- it is never fetched.
    # Anything else pointing off-host would be.
    external = [
        url
        for url in re.findall(r"https?://[^\s\"'<>]+", page)
        if not url.startswith("http://www.w3.org/")
    ]
    assert external == [], f"report reaches off-host: {external}"


def test_html_report_states_the_recommendation_and_caveats(sweep, rec, tmp_path):
    out = tmp_path / "package"
    export_package(sweep, rec, out)
    page = (out / "report.html").read_text()

    assert "Recommended configuration" in page
    assert "Q4_0" in page
    assert "What this report is not" in page
    # Every caveat the engine raised must reach the reader.
    for warning in rec.warnings:
        assert warning[:40] in page


def test_html_escapes_hostile_model_paths(sweep, rec):
    """A crafted filename must not become markup in the report.

    The model path reaches the page through the deployment command, so a file
    named to look like a tag is the realistic injection route.
    """
    nasty = replace(
        rec.winner.result.config.model,
        path="/models/<script>alert(1)</script>.gguf",
    )
    poisoned = replace(rec.winner.result.config, model=nasty)
    result = replace(rec.winner.result, config=poisoned)
    score = replace(rec.winner, result=result)
    page = render_html(sweep, replace(rec, winner=score))

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_readme_documents_the_split_and_the_baseline(sweep, rec, tmp_path):
    out = tmp_path / "package"
    export_package(sweep, rec, out)
    readme = (out / "README.md").read_text()

    assert "Prompt-processing threads" in readme
    assert "Generation threads" in readme
    assert "naive default" in readme
    assert "Not a quality claim" in readme


def test_dockerfile_targets_arm64_and_pins_the_build(sweep, rec, tmp_path):
    out = tmp_path / "package"
    export_package(sweep, rec, out)
    dockerfile = (out / "Dockerfile").read_text()

    assert "--platform=linux/arm64" in dockerfile
    assert "a94d563" in dockerfile, "llama.cpp commit must be pinned"
    assert "GGML_CPU_KLEIDIAI=OFF" in dockerfile
    # Containerising is not itself an optimisation, and the file should say so.
    assert "does not make anything faster" in dockerfile


def test_dockerfile_has_no_shell_redirection_in_copy(sweep, rec, tmp_path):
    """COPY is not a shell; redirection there is a syntax error."""
    out = tmp_path / "package"
    export_package(sweep, rec, out)
    for line in (out / "Dockerfile").read_text().splitlines():
        if line.startswith("COPY"):
            assert "2>" not in line and "||" not in line


def test_model_is_referenced_not_copied(sweep, rec, tmp_path):
    """Multi-gigabyte weights must not be duplicated into every package."""
    out = tmp_path / "package"
    export_package(sweep, rec, out)

    assert not list(out.glob("*.gguf"))
    config = json.loads((out / "config.json").read_text())
    assert config["model"]["content_hash"], "must identify the weights by hash"


def test_content_hash_is_stable_and_reports_truncation(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"x" * 4096)

    full = content_hash(path)
    assert full == content_hash(path), "hashing must be deterministic"
    assert full.startswith("sha256:")

    partial = content_hash(path, max_bytes=1024)
    assert partial.startswith("sha256-first1024:")
    assert partial != full


def test_content_hash_of_missing_file_is_none(tmp_path):
    assert content_hash(tmp_path / "absent.bin") is None


def test_export_creates_nested_directories(sweep, rec, tmp_path):
    out = tmp_path / "a" / "b" / "package"
    export_package(sweep, rec, out)
    assert (out / "config.json").is_file()
