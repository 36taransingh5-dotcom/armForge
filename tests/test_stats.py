"""Tests for measurement statistics."""

from __future__ import annotations

import pytest

from armforge.bench.stats import is_meaningfully_different, percent_change, summarize
from armforge.bench.types import MetricStats


def test_summarize_computes_the_distribution():
    stats = summarize([10.0, 12.0, 14.0, 16.0, 18.0], "tok/s")
    assert stats is not None
    assert stats.mean == 14.0
    assert stats.median == 14.0
    assert stats.minimum == 10.0
    assert stats.maximum == 18.0
    assert stats.samples == 5
    assert stats.unit == "tok/s"
    assert stats.stddev == pytest.approx(3.1623, rel=1e-3)


def test_summarize_of_empty_samples_is_none_not_zero():
    """ "Not measured" must never render as "measured zero"."""
    assert summarize([], "tok/s") is None


def test_single_sample_has_zero_spread_and_does_not_raise():
    stats = summarize([42.0], "tok/s")
    assert stats is not None
    assert stats.stddev == 0.0
    assert stats.samples == 1


def test_relative_stddev_flags_a_noisy_measurement():
    quiet = summarize([100.0, 101.0, 99.0], "tok/s")
    noisy = summarize([100.0, 150.0, 50.0], "tok/s")
    assert quiet.relative_stddev < 0.05
    assert noisy.relative_stddev > 0.05


def test_relative_stddev_of_zero_mean_does_not_divide_by_zero():
    stats = summarize([0.0, 0.0], "tok/s")
    assert stats.relative_stddev == 0.0


def test_percent_change_signs():
    assert percent_change(100.0, 174.7) == pytest.approx(74.7)
    assert percent_change(100.0, 49.0) == pytest.approx(-51.0)
    assert percent_change(100.0, 100.0) == 0.0


def test_percent_change_from_zero_baseline_is_undefined():
    assert percent_change(0.0, 10.0) is None


def _stats(mean: float, stddev: float) -> MetricStats:
    return MetricStats(
        mean=mean,
        median=mean,
        minimum=mean,
        maximum=mean,
        stddev=stddev,
        samples=5,
        unit="tok/s",
    )


def test_large_gap_with_tight_variance_is_meaningful():
    assert is_meaningfully_different(_stats(18.0, 0.2), _stats(31.0, 0.3))


def test_small_gap_swamped_by_noise_is_not_meaningful():
    """Thermal jitter on a laptop must not be reported as an optimisation."""
    assert not is_meaningfully_different(_stats(18.0, 3.0), _stats(19.0, 3.0))


def test_identical_noiseless_measurements_are_not_different():
    assert not is_meaningfully_different(_stats(20.0, 0.0), _stats(20.0, 0.0))


def test_noiseless_but_unequal_measurements_are_different():
    assert is_meaningfully_different(_stats(20.0, 0.0), _stats(25.0, 0.0))


def test_sigma_threshold_is_adjustable():
    baseline, candidate = _stats(100.0, 2.0), _stats(105.0, 2.0)
    assert is_meaningfully_different(baseline, candidate, sigma=1.0)
    assert not is_meaningfully_different(baseline, candidate, sigma=3.0)
