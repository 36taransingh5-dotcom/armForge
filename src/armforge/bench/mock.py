"""A deterministic fake runner, for tests only.

This exists so the orchestration, scoring and reporting layers can be tested
without a multi-gigabyte model or a compiled inference engine.

It is deliberately quarantined:

* it lives in its own module and is never imported by the CLI or any real
  code path;
* :meth:`MockRunner.execute` marks every result it produces with
  ``raw["synthetic"] = True``;
* the numbers it invents come from a toy formula that is not a performance
  model of anything.

Nothing it produces may ever appear in a report presented as a measurement.
"""

from __future__ import annotations

from ..hardware.types import HostProfile
from .runner import BenchmarkRunner
from .stats import summarize
from .types import BenchConfig, BenchmarkResult, Status

#: Rough relative cost of each quantisation, used only to make the fake
#: numbers vary in a plausible direction. Not a performance model.
_TOY_QUANT_FACTOR = {
    "F16": 1.0,
    "Q8_0": 1.6,
    "Q4_K_M": 2.4,
    "Q4_0": 2.6,
}


class MockRunner(BenchmarkRunner):
    """Synthetic runner producing repeatable, clearly-marked fake results."""

    name = "mock"

    def __init__(
        self,
        *,
        available: bool = True,
        unsupported_quantizations: frozenset[str] = frozenset(),
        fail_on_threads: frozenset[int] = frozenset(),
    ) -> None:
        self._available = available
        self._unsupported = unsupported_quantizations
        self._fail_on_threads = fail_on_threads

    def is_available(self) -> tuple[bool, str]:
        if not self._available:
            return False, "mock runner disabled for this test"
        return True, "mock runner is always available"

    def supports(self, config: BenchConfig) -> tuple[bool, str]:
        quant = config.model.quantization
        if quant in self._unsupported:
            return False, f"mock runner does not support {quant}"
        return True, "supported"

    def execute(self, config: BenchConfig, host: HostProfile) -> BenchmarkResult:
        if config.threads in self._fail_on_threads:
            raise RuntimeError(f"mock failure at {config.threads} threads")

        factor = _TOY_QUANT_FACTOR.get(config.model.quantization or "F16", 1.0)

        # Prefill scales with threads; decode saturates then degrades. This
        # mimics the *shape* of real behaviour so orchestration logic can be
        # tested, but the magnitudes are invented.
        prefill = 20.0 * factor * config.threads
        useful_threads = min(config.threads, max(host.cpu.performance_cores, 1))
        decode = 8.0 * factor * useful_threads / (1 + 0.15 * (config.threads - useful_threads))

        samples = config.iterations
        return BenchmarkResult(
            config=config,
            host=host,
            status=Status.OK,
            prefill_tps=summarize([prefill] * samples, "tok/s"),
            decode_tps=summarize([decode] * samples, "tok/s"),
            peak_memory_bytes=int(config.model.size_bytes * 1.3),
            model_load_ms=100.0,
            wall_time_s=0.0,
            raw={"synthetic": True, "runner": self.name},
        )
