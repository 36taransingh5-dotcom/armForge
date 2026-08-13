"""End-to-end tests for the CLI and live detection on the current host."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from armforge import __version__
from armforge.cli import app
from armforge.hardware import detect_host

runner = CliRunner()


def test_hardware_command_succeeds():
    result = runner.invoke(app, ["hardware"])
    assert result.exit_code == 0, result.output
    assert "Architecture" in result.output
    assert "Cores" in result.output


def test_hardware_json_is_valid_and_complete():
    result = runner.invoke(app, ["hardware", "--json"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    assert payload["cpu"]["architecture"]
    assert payload["cpu"]["logical_cores"] >= 1
    assert isinstance(payload["cpu"]["features"], list)
    assert isinstance(payload["warnings"], list)
    assert payload["detector"] in {"darwin-sysctl", "linux-sysfs", "fallback"}


def test_hardware_explain_describes_features():
    result = runner.invoke(app, ["hardware", "--explain"])
    assert result.exit_code == 0, result.output


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_detection_is_internally_consistent():
    """Whatever this host is, the profile it produces must not contradict itself."""
    host = detect_host()
    cpu = host.cpu

    assert cpu.logical_cores >= cpu.physical_cores >= 0
    assert cpu.performance_cores <= cpu.physical_cores or cpu.physical_cores == 0
    assert sum(c.physical_cores for c in cpu.clusters) == cpu.physical_cores

    # Every normalised feature key must be one the registry knows about.
    from armforge.hardware.features import FEATURES

    assert set(cpu.features) <= set(FEATURES)

    # An Arm64 host must at minimum report NEON; it is architecturally mandatory.
    if cpu.is_arm64 and host.detector != "fallback":
        assert "neon" in cpu.features


def test_detect_host_is_cached():
    assert detect_host() is detect_host()
