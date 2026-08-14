"""Benchmark backend driving llama.cpp's ``llama-bench``.

``llama-bench`` is llama.cpp's own benchmarking tool. Using it rather than
timing ``llama-cli`` output ourselves matters for credibility: it is the
instrument Arm's own engineers use, it performs its own warmup, it repeats
each measurement, and it reports prompt processing and token generation as
separate tests -- which is exactly the split ArmForge is built around.

We add three things on top of it:

* peak resident memory, which llama-bench does not report;
* verification that the features the CPU advertises were actually compiled
  into the ggml backend, parsed from its own startup log;
* provenance binding each result to a specific build, including whether
  KleidiAI was enabled.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..hardware.types import HostProfile
from . import process
from .runner import BenchmarkRunner
from .stats import summarize
from .types import BenchConfig, BenchmarkResult, RuntimeSpec, Status

#: Where ``scripts/setup-llama-cpp.sh`` puts its builds.
DEFAULT_CACHE = Path.home() / ".cache" / "armforge" / "llama.cpp"

#: llama.cpp prints a system_info line at startup, e.g.
#: ``NEON = 1 | ARM_FMA = 1 | MATMUL_INT8 = 1 | SME = 1 |``.
#: These flags prove a code path was compiled in, as opposed to the CPU merely
#: being capable of it. ``llama-bench`` does not print it -- only ``llama-cli``
#: does, and only once a model has loaded -- which is why ArmForge probes the
#: two binaries separately.
_GGML_FEATURE_RE = re.compile(r"\b([A-Z_0-9]{3,})\s*=\s*([01])\b")

#: ggml feature names that correspond to ArmForge capability keys.
#: ``REPACK`` has no CPU-feature counterpart -- it is ggml's own weight
#: repacking pass, the thing that rearranges Q4_0 into a layout SMMLA can
#: consume -- so it maps to a pseudo-key. It is arguably the single most
#: important flag here, because without it a CPU's i8mm goes unused.
GGML_FEATURE_TO_KEY: dict[str, str] = {
    "NEON": "neon",
    "DOTPROD": "dotprod",
    "MATMUL_INT8": "i8mm",
    "SVE": "sve",
    "SME": "sme",
    "FP16_VA": "fp16",
    "BF16": "bf16",
    "ARM_FMA": "neon",
    "REPACK": "repack",
}


class LlamaCppRunner(BenchmarkRunner):
    """Runs one configuration through ``llama-bench``."""

    name = "llama.cpp"

    def __init__(self, runtime: RuntimeSpec, *, timeout: float = 900.0) -> None:
        self.runtime = runtime
        self.timeout = timeout
        # Probed once per runtime and reused: the answer is a property of the
        # build, not of any individual benchmark.
        self._ggml_features: dict[str, bool] | None = None

    def ggml_features(self, model_path: str) -> dict[str, bool]:
        """Which Arm code paths this build compiled in, probed on first use."""
        if self._ggml_features is None:
            self._ggml_features = probe_ggml_features(self.runtime, model_path)
        return self._ggml_features

    # -- availability -----------------------------------------------------

    def is_available(self) -> tuple[bool, str]:
        binary = Path(self.runtime.binary_path)
        if not binary.is_file():
            return False, (
                f"llama-bench not found at {binary}. "
                "Run scripts/setup-llama-cpp.sh to build it."
            )
        return True, "ok"

    def supports(self, config: BenchConfig) -> tuple[bool, str]:
        model = Path(config.model.path)
        if not model.is_file():
            return False, f"model file not found: {model}"
        if model.suffix != ".gguf":
            return False, f"llama.cpp needs a GGUF model, got {model.suffix or 'no'} suffix"
        if config.threads < 1:
            return False, f"thread count must be >= 1, got {config.threads}"
        return True, "supported"

    # -- execution --------------------------------------------------------

    def _argv(self, config: BenchConfig) -> list[str]:
        """Build the llama-bench command line.

        Prompt and generation are requested in one invocation; llama-bench
        runs them as two separate tests and reports them as separate rows.
        """
        return [
            self.runtime.binary_path,
            "--model", config.model.path,
            "--n-prompt", str(config.workload.prompt_tokens),
            "--n-gen", str(config.workload.generate_tokens),
            "--threads", str(config.threads),
            "--repetitions", str(config.iterations),
            "--output", "json",
        ]

    def execute(self, config: BenchConfig, host: HostProfile) -> BenchmarkResult:
        argv = self._argv(config)
        # Keep the runtime's own thread pool from being overridden by an
        # inherited OMP setting; -t is the variable under test.
        env = {"OMP_NUM_THREADS": str(config.threads)}

        result = process.run(argv, timeout=self.timeout, env=env)

        if result.timed_out:
            return BenchmarkResult(
                config=config,
                host=host,
                status=Status.TIMEOUT,
                error=f"llama-bench exceeded {self.timeout:.0f}s",
                wall_time_s=result.wall_time_s,
            )

        if result.returncode != 0:
            return BenchmarkResult(
                config=config,
                host=host,
                status=Status.FAILED,
                error=_first_error_line(result.stderr) or f"exit code {result.returncode}",
                wall_time_s=result.wall_time_s,
                raw={"stderr_tail": result.stderr[-4000:]},
            )

        try:
            rows = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            return BenchmarkResult(
                config=config,
                host=host,
                status=Status.FAILED,
                error=f"could not parse llama-bench JSON: {exc}",
                wall_time_s=result.wall_time_s,
                raw={"stdout_tail": result.stdout[-4000:]},
            )

        if not isinstance(rows, list) or not rows:
            return BenchmarkResult(
                config=config,
                host=host,
                status=Status.FAILED,
                error="llama-bench returned no measurements",
                wall_time_s=result.wall_time_s,
            )

        prefill_row = _find_row(rows, prompt=True)
        decode_row = _find_row(rows, prompt=False)

        return BenchmarkResult(
            config=config,
            host=host,
            status=Status.OK,
            prefill_tps=_throughput(prefill_row),
            decode_tps=_throughput(decode_row),
            peak_memory_bytes=result.peak_memory_bytes,
            wall_time_s=result.wall_time_s,
            raw={
                "build_commit": _first(rows, "build_commit"),
                "backends": _first(rows, "backends"),
                "cpu_info": _first(rows, "cpu_info"),
                "model_type": _first(rows, "model_type"),
                "model_size": _first(rows, "model_size"),
                "model_n_params": _first(rows, "model_n_params"),
                # llama-bench prints nothing on stderr in JSON mode, so the
                # feature set comes from a separate one-off probe.
                "ggml_features": self.ggml_features(config.model.path),
                "rows": rows,
            },
        )


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------


def _find_row(rows: list[dict[str, Any]], *, prompt: bool) -> dict[str, Any] | None:
    """Pick the prompt-processing or token-generation row.

    llama-bench emits one row per test: prompt processing has ``n_prompt > 0``
    and ``n_gen == 0``; token generation is the reverse.
    """
    for row in rows:
        n_prompt = row.get("n_prompt") or 0
        n_gen = row.get("n_gen") or 0
        if prompt and n_prompt > 0 and n_gen == 0:
            return row
        if not prompt and n_gen > 0 and n_prompt == 0:
            return row
    return None


def _throughput(row: dict[str, Any] | None):
    """Summarise a row's per-repetition throughputs.

    ``samples_ts`` holds one figure per repetition, which lets us report a
    real distribution. Older builds omit it, in which case we fall back to the
    single reported average -- and the resulting stddev of zero is honest,
    because we genuinely only have one number.
    """
    if row is None:
        return None
    samples = row.get("samples_ts")
    if isinstance(samples, list) and samples:
        return summarize(samples, "tok/s")
    average = row.get("avg_ts")
    if average is None:
        return None
    return summarize([average], "tok/s")


def parse_ggml_features(text: str) -> dict[str, bool]:
    """Extract ggml's compiled-in feature flags from its startup log.

    This is how ArmForge separates *capability* from *usage*: the CPU may
    advertise ``FEAT_I8MM``, but if ggml reports ``MATMUL_INT8 = 0`` then the
    build cannot exploit it and any speedup must have come from elsewhere.
    """
    found: dict[str, bool] = {}
    for name, value in _GGML_FEATURE_RE.findall(text):
        if name in GGML_FEATURE_TO_KEY:
            # A later "1" wins: the CPU backend line is printed after any
            # generic capability listing.
            found[name] = found.get(name, False) or value == "1"
    return found


def probe_ggml_features(
    runtime: RuntimeSpec, model_path: str, *, timeout: float = 120.0
) -> dict[str, bool]:
    """Ask a llama.cpp build which Arm code paths it was compiled with.

    Runs ``llama-cli`` for a single token purely to trigger its ``system_info``
    line, then parses the flags out of it. This is the evidence that links
    hardware capability to runtime usage: a CPU reporting ``FEAT_I8MM`` while
    ggml reports ``MATMUL_INT8 = 0`` means the build cannot use it, and any
    measured difference came from somewhere else.

    Returns an empty mapping if the probe could not run. An empty result means
    "not established", never "no features".
    """
    cli = Path(runtime.binary_path).with_name("llama-completion")
    if not cli.is_file():
        return {}

    argv = [
        str(cli),
        "--model", model_path,
        "--prompt", "x",
        "--n-predict", "1",
        "--no-warmup",
        "--threads", "1",
    ]
    try:
        result = process.run(argv, timeout=timeout)
    except process.ProcessError:
        return {}

    return parse_ggml_features(result.stderr + result.stdout)


def _first(rows: list[dict[str, Any]], key: str) -> Any:
    for row in rows:
        if key in row:
            return row[key]
    return None


def _first_error_line(stderr: str) -> str | None:
    """Pull the most informative line out of a failed run's stderr."""
    for line in reversed(stderr.strip().splitlines()):
        stripped = line.strip()
        if stripped and "error" in stripped.lower():
            return stripped[:400]
    tail = stderr.strip().splitlines()
    return tail[-1].strip()[:400] if tail else None


# --------------------------------------------------------------------------
# Runtime discovery
# --------------------------------------------------------------------------


def _cmake_flag(build_dir: Path, name: str) -> bool | None:
    """Read a boolean from a build's CMakeCache.

    We verify build flags from the cache rather than trusting the directory
    name, so a result can never claim KleidiAI was enabled when it was not.
    """
    cache = build_dir / "CMakeCache.txt"
    try:
        text = cache.read_text(errors="replace")
    except OSError:
        return None
    match = re.search(rf"^{re.escape(name)}:BOOL=(\w+)$", text, re.MULTILINE)
    if not match:
        return None
    return match.group(1).upper() in ("ON", "TRUE", "1", "YES")


def discover_runtimes(cache_dir: Path | None = None) -> list[RuntimeSpec]:
    """Find every llama.cpp build ArmForge has available.

    Returns one :class:`RuntimeSpec` per build directory containing a
    ``llama-bench`` binary, with build flags read from its CMake cache.
    """
    root = cache_dir or DEFAULT_CACHE
    specs: list[RuntimeSpec] = []

    for build_dir in sorted(root.glob("build-*")):
        binary = build_dir / "bin" / "llama-bench"
        if not binary.is_file():
            continue

        specs.append(
            RuntimeSpec(
                name="llama.cpp",
                version=_build_commit(root),
                binary_path=str(binary),
                build_flags={
                    "variant": build_dir.name.removeprefix("build-"),
                    "kleidiai": bool(_cmake_flag(build_dir, "GGML_CPU_KLEIDIAI")),
                    # Accelerate routes prefill through Apple's BLAS instead of
                    # ggml's Arm kernels, so a build with it on is measuring a
                    # different thing and must be labelled as such.
                    "accelerate": bool(_cmake_flag(build_dir, "GGML_ACCELERATE")),
                    "blas": bool(_cmake_flag(build_dir, "GGML_BLAS")),
                    "metal": bool(_cmake_flag(build_dir, "GGML_METAL")),
                },
            )
        )
    return specs


def _build_commit(repo: Path) -> str:
    """Short commit of the llama.cpp checkout, for provenance."""
    head = repo / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
        if ref.startswith("ref: "):
            ref = (repo / ".git" / ref[5:]).read_text().strip()
        return ref[:12]
    except OSError:
        return "unknown"
