"""Tests of rates and binning, including a case worked out by hand."""

from pathlib import Path

import numpy as np
import pytest

from lib_muon import binning, io, rates

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def synthetic_run(timestamps_ms, deadtime_us, coincident=None):
    """Minimal run for checking arithmetic against a hand-computed answer."""
    n = len(timestamps_ms)
    return io.Run(
        angle_deg=0,
        path="<synthetic>",
        event=np.arange(1, n + 1, dtype=np.int64),
        timestamp_ms=np.asarray(timestamps_ms, dtype=float),
        adc1=np.zeros(n, dtype=np.int64),
        adc2=np.zeros(n, dtype=np.int64),
        sipm_mv=np.zeros(n),
        pressure_pa=np.full(n, 93600.0),
        temperature_c=np.full(n, 28.0),
        deadtime_us=np.asarray(deadtime_us, dtype=float),
        coincident=(np.ones(n, dtype=np.int64) if coincident is None
                    else np.asarray(coincident, dtype=np.int64)),
        sensor_startup=np.zeros(n, dtype=bool),
        quality=io.QualityReport(path="<synthetic>"),
    )


class TestHandComputedCase:
    """11 events at one per second, with 0.1 s of total dead time."""

    def test_full_arithmetic(self):
        run = synthetic_run(
            timestamps_ms=np.arange(11) * 1000.0,
            deadtime_us=np.full(11, 100_000.0 / 11),  # 100 ms spread out
        )
        rate = rates.compute_rate(run)

        # By hand: window = t11 - t1 = 10000 ms = exactly 10 s.
        assert rate.window_s == pytest.approx(10.0)
        # 11 events, 10 intervals.
        assert rate.n_events == 11
        assert rate.n_intervals == 10
        # Dead time = 100000 us = 0.1 s. Live = 10 - 0.1 = 9.9 s.
        assert rate.dead_time_s == pytest.approx(0.1)
        assert rate.live_time_s == pytest.approx(9.9)
        # Wall rate = 10/10 = exactly 1 Hz.
        assert rate.wall_rate == pytest.approx(1.0)
        # Live rate = 10/9.9 = 1.0101... Hz.
        assert rate.live_rate == pytest.approx(10.0 / 9.9)
        # Relative error = 1/sqrt(10).
        assert rate.live_rate_error == pytest.approx(rate.live_rate / np.sqrt(10))

    def test_the_estimator_uses_n_minus_one(self):
        """For N events in [t1, tN] there are N-1 intervals, not N. The
        original script divided by N and overstated the rate by 1/N."""
        run = synthetic_run(timestamps_ms=np.arange(101) * 1000.0,
                            deadtime_us=np.zeros(101))
        rate = rates.compute_rate(run)
        assert rate.wall_rate == pytest.approx(1.0)      # 100/100, correct
        biased = 101 / 100.0                              # what the original did
        assert biased == pytest.approx(1.01)
        assert rate.wall_rate != pytest.approx(biased)

    def test_live_time_raises_the_rate(self):
        run = synthetic_run(timestamps_ms=np.arange(11) * 1000.0,
                            deadtime_us=np.full(11, 1e6 / 11))
        rate = rates.compute_rate(run)
        assert rate.live_time_s == pytest.approx(9.0)
        assert rate.live_rate > rate.wall_rate
        assert rate.live_rate == pytest.approx(10.0 / 9.0)

    def test_too_few_events_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            rates.compute_rate(synthetic_run([0.0], [0.0]))


class TestNormalisation:
    def test_the_reference_angle_gives_exactly_one(self):
        runs = io.load_all(RAW)
        computed = {a: rates.compute_rate(r) for a, r in runs.items()}
        assert rates.normalise(computed, reference_angle=0)[0][0] == pytest.approx(1.0)

    def test_missing_reference_raises(self):
        runs = io.load_all(RAW)
        computed = {a: rates.compute_rate(r) for a, r in runs.items()}
        with pytest.raises(ValueError, match="reference"):
            rates.normalise(computed, reference_angle=77)


class TestBinning:
    def test_drops_the_final_partial_bin(self):
        """10.5 s of data with 1 s bins give 10 bins, not 11."""
        centres, _, widths = binning.complete_bins(
            np.arange(0, 10.5, 0.1), bin_s=1.0, start_s=0.0, end_s=10.5
        )
        assert len(centres) == 10
        assert np.all(widths == 1.0)

    def test_correct_counts_in_a_trivial_case(self):
        centres, counts, _ = binning.complete_bins(
            np.array([0.5, 1.5, 1.6, 2.5]), bin_s=1.0, start_s=0.0, end_s=3.0
        )
        np.testing.assert_array_equal(counts, [1.0, 2.0, 1.0])
        np.testing.assert_allclose(centres, [0.5, 1.5, 2.5])

    def test_a_bin_wider_than_the_data_yields_no_bins(self):
        centres, _, _ = binning.complete_bins(np.array([0.5]), bin_s=100.0,
                                              start_s=0.0, end_s=1.0)
        assert len(centres) == 0

    def test_non_positive_bin_raises(self):
        with pytest.raises(ValueError, match="positive"):
            binning.complete_bins(np.array([0.5]), bin_s=0.0)

    def test_the_error_of_an_empty_bin_is_not_zero(self):
        _, _, errors = binning.binned_rates(np.array([0.5, 2.5]), bin_s=1.0,
                                            start_s=0.0, end_s=3.0)
        assert errors[1] > 0

    def test_moving_average_of_a_constant_is_the_constant(self):
        means, deviations = binning.moving_average(np.full(20, 3.0), 5)
        np.testing.assert_allclose(means, 3.0)
        np.testing.assert_allclose(deviations, 0.0, atol=1e-12)


class TestBinningDoesNotAffectThePhysics:
    """The defect being fixed: --bin-seconds used to move the published rate."""

    def test_the_rate_does_not_depend_on_bin_size(self):
        run = io.load_run(RAW / "000deg")
        rate = rates.compute_rate(run)
        # The rate is computed over the full window and takes no bin parameter:
        # there is no way for a cosmetic setting to move it.
        assert rate.window_s == pytest.approx(run.window_s)
        assert "bin" not in rates.compute_rate.__code__.co_varnames
