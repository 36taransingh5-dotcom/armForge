"""The benchmark runner interface.

A runner knows how to execute one :class:`BenchConfig` against one inference
engine and return a :class:`BenchmarkResult`. Adding support for a new engine
means writing one subclass; nothing else in ArmForge changes.

Runners must never raise on a failed measurement. A configuration that cannot
run is a *result* -- with a status and a reason -- because "this did not work
here, and here is why" is information the report should carry.
"""

from __future__ import annotations

import abc

from ..hardware.types import HostProfile
from .types import BenchConfig, BenchmarkResult, Status


class BenchmarkRunner(abc.ABC):
    """Base class for inference engine benchmark backends."""

    #: Short identifier used in results and reports, e.g. ``"llama.cpp"``.
    name: str = "unknown"

    @abc.abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """Whether this runner can run at all on this machine.

        Returns ``(available, reason)``. The reason is shown to the user when
        unavailable, so it should say what is missing and how to supply it.
        """

    @abc.abstractmethod
    def supports(self, config: BenchConfig) -> tuple[bool, str]:
        """Whether this specific configuration is runnable.

        Returns ``(supported, reason)``. Used to mark candidates
        :attr:`Status.UNSUPPORTED` before wasting time attempting them.
        """

    @abc.abstractmethod
    def execute(self, config: BenchConfig, host: HostProfile) -> BenchmarkResult:
        """Run the benchmark. Only called when :meth:`supports` returned True."""

    def run(self, config: BenchConfig, host: HostProfile) -> BenchmarkResult:
        """Run ``config``, converting any failure into a result.

        This is the method callers should use; :meth:`execute` is the part
        subclasses implement.
        """
        available, reason = self.is_available()
        if not available:
            return self.unsupported(config, host, reason)

        supported, reason = self.supports(config)
        if not supported:
            return self.unsupported(config, host, reason)

        try:
            return self.execute(config, host)
        except Exception as exc:  # noqa: BLE001 - a crashed runner is a data point
            return BenchmarkResult(
                config=config,
                host=host,
                status=Status.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )

    # -- helpers for subclasses -------------------------------------------

    @staticmethod
    def unsupported(config: BenchConfig, host: HostProfile, reason: str) -> BenchmarkResult:
        return BenchmarkResult(
            config=config, host=host, status=Status.UNSUPPORTED, error=reason
        )

    @staticmethod
    def skipped(config: BenchConfig, host: HostProfile, reason: str) -> BenchmarkResult:
        return BenchmarkResult(config=config, host=host, status=Status.SKIPPED, error=reason)

    @staticmethod
    def failed(config: BenchConfig, host: HostProfile, reason: str) -> BenchmarkResult:
        return BenchmarkResult(config=config, host=host, status=Status.FAILED, error=reason)
