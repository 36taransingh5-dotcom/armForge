"""Tests for isolated subprocess execution.

These run real processes: the point of this layer is that it behaves correctly
against the operating system, which a mock cannot demonstrate.
"""

from __future__ import annotations

import sys

import pytest

from armforge.bench.process import ProcessError, run


def test_captures_stdout_and_exit_code():
    result = run([sys.executable, "-c", "print('hello')"], timeout=30)
    assert result.ok
    assert result.returncode == 0
    assert result.stdout.strip() == "hello"


def test_captures_stderr_separately():
    result = run(
        [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"],
        timeout=30,
    )
    assert not result.ok
    assert result.returncode == 3
    assert "boom" in result.stderr
    assert result.stdout == ""


def test_large_output_does_not_deadlock():
    """A child that fills the pipe buffer must not hang the parent.

    llama-bench is quiet, but a verbose runtime can emit megabytes of loading
    chatter on stderr; if we waited on the child before draining the pipes,
    both sides would block forever.
    """
    result = run(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('x' * 5_000_000); sys.stderr.write('y' * 5_000_000)",
        ],
        timeout=60,
    )
    assert result.ok
    assert len(result.stdout) == 5_000_000
    assert len(result.stderr) == 5_000_000


def test_timeout_kills_the_process():
    result = run([sys.executable, "-c", "import time; time.sleep(60)"], timeout=2)
    assert result.timed_out
    assert not result.ok
    assert result.wall_time_s < 30


def test_timeout_kills_the_whole_process_group():
    """A runner that spawns children must not leave orphans behind."""
    script = (
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "time.sleep(60)"
    )
    result = run([sys.executable, "-c", script], timeout=3)
    assert result.timed_out


def test_environment_is_passed_to_the_child():
    result = run(
        [sys.executable, "-c", "import os; print(os.environ['OMP_NUM_THREADS'])"],
        timeout=30,
        env={"OMP_NUM_THREADS": "7"},
    )
    assert result.stdout.strip() == "7"


def test_peak_memory_is_measured_and_plausible():
    """Allocate ~200 MB and check it shows up in peak RSS."""
    result = run(
        [sys.executable, "-c", "buf = bytearray(200 * 1024 * 1024); buf[-1] = 1"],
        timeout=60,
    )
    assert result.ok
    assert result.peak_memory_bytes is not None
    assert result.peak_memory_bytes > 150 * 1024**2
    # Sanity bound: if the KiB/bytes unit conversion were wrong this would be
    # off by a factor of 1024 in one direction or the other.
    assert result.peak_memory_bytes < 4 * 1024**3


def test_peak_memory_is_per_child_not_cumulative():
    """A small run after a large one must not inherit the large peak."""
    big = run(
        [sys.executable, "-c", "buf = bytearray(300 * 1024 * 1024); buf[-1] = 1"],
        timeout=60,
    )
    small = run([sys.executable, "-c", "pass"], timeout=30)

    assert big.peak_memory_bytes > 200 * 1024**2
    assert small.peak_memory_bytes < 100 * 1024**2


def test_wall_time_is_recorded():
    result = run([sys.executable, "-c", "import time; time.sleep(0.5)"], timeout=30)
    assert result.wall_time_s >= 0.5


def test_missing_binary_raises_process_error():
    with pytest.raises(ProcessError, match="could not launch"):
        run(["/nonexistent/armforge-definitely-not-here"], timeout=10)


def test_empty_argv_is_rejected():
    with pytest.raises(ProcessError, match="non-empty list"):
        run([], timeout=10)


def test_non_string_argv_is_rejected():
    """Guards against a config value leaking into a command line unvalidated."""
    with pytest.raises(ProcessError, match="list of strings"):
        run(["echo", 42], timeout=10)  # type: ignore[list-item]


def test_shell_metacharacters_are_not_interpreted():
    """There is no shell, so this must be a literal argument, not a command."""
    result = run(
        [sys.executable, "-c", "import sys; print(sys.argv[1])", "; rm -rf /"], timeout=30
    )
    assert result.ok
    assert result.stdout.strip() == "; rm -rf /"
