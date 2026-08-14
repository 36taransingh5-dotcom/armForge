"""Isolated subprocess execution for benchmark runs.

Benchmarking means running other people's binaries, so this layer is the
security boundary:

* commands are always ``argv`` lists -- there is no shell, ever, so no user
  string can be interpreted as a command;
* every run has a hard timeout enforced by killing the whole process group;
* peak resident memory is measured exactly, not sampled.

Peak RSS comes from ``os.wait4``, which returns ``rusage`` for one specific
child. The more common ``getrusage(RUSAGE_CHILDREN)`` reports a high-water
mark across every child the process has ever reaped, which would attribute the
largest run's memory to all subsequent ones.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass


class ProcessError(RuntimeError):
    """Raised when a command could not be launched at all."""


@dataclass(frozen=True)
class ProcessResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    wall_time_s: float
    peak_memory_bytes: int | None
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def _maxrss_to_bytes(ru_maxrss: int) -> int:
    """Normalise ``ru_maxrss`` to bytes.

    macOS reports bytes; Linux reports kibibytes. Getting this wrong is a
    silent factor-of-1024 error in every memory number.
    """
    return ru_maxrss if sys.platform == "darwin" else ru_maxrss * 1024


def _kill_group(pid: int) -> None:
    """Terminate a process group, escalating to SIGKILL."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.5)
        try:
            os.killpg(os.getpgid(pid), 0)
        except (ProcessLookupError, PermissionError):
            return


def run(
    argv: list[str],
    *,
    timeout: float,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> ProcessResult:
    """Run ``argv`` to completion under a timeout, measuring peak memory.

    ``env`` is merged over the current environment, which is how thread and
    allocator settings (``OMP_NUM_THREADS`` and friends) reach the child.
    """
    if not argv or not all(isinstance(part, str) for part in argv):
        raise ProcessError("argv must be a non-empty list of strings")

    child_env = {**os.environ, **(env or {})}
    started = time.perf_counter()

    try:
        proc = subprocess.Popen(  # noqa: S603 - argv list, shell is never used
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=child_env,
            cwd=cwd,
            # Own process group, so a timeout can kill the whole tree.
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        raise ProcessError(f"could not launch {argv[0]!r}: {exc}") from exc

    # Drain both pipes concurrently; a full pipe buffer would otherwise
    # deadlock against our wait4 below.
    captured: dict[str, str] = {}

    def drain(stream, key: str) -> None:
        try:
            captured[key] = stream.read()
        except (OSError, ValueError):
            captured[key] = ""

    readers = [
        threading.Thread(target=drain, args=(proc.stdout, "stdout"), daemon=True),
        threading.Thread(target=drain, args=(proc.stderr, "stderr"), daemon=True),
    ]
    for reader in readers:
        reader.start()

    timed_out = threading.Event()

    def on_timeout() -> None:
        timed_out.set()
        _kill_group(proc.pid)

    watchdog = threading.Timer(timeout, on_timeout)
    watchdog.start()

    try:
        _, status, rusage = os.wait4(proc.pid, 0)
    except ChildProcessError:
        # Already reaped by something else; fall back to Popen's own wait.
        proc.wait()
        status, rusage = 0, None
    finally:
        watchdog.cancel()
        for reader in readers:
            reader.join(timeout=5)
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()

    if proc.returncode is None:
        try:
            proc.returncode = os.waitstatus_to_exitcode(status)
        except ValueError:
            proc.returncode = -1

    return ProcessResult(
        argv=tuple(argv),
        returncode=proc.returncode if proc.returncode is not None else -1,
        stdout=captured.get("stdout", ""),
        stderr=captured.get("stderr", ""),
        wall_time_s=time.perf_counter() - started,
        peak_memory_bytes=_maxrss_to_bytes(rusage.ru_maxrss) if rusage else None,
        timed_out=timed_out.is_set(),
    )
