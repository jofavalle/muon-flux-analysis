"""Tests of the altitude functions against closed forms and known values.

The numbers of the 2026 comparison itself are pinned in test_pipeline.py,
on the product of step 09, which is what the repository publishes.
"""

import numpy as np
import pytest

from lib_muon import altitude as alt


class TestAtmosphere:
    def test_no_height_between_equal_pressures(self):
        assert alt.hypsometric(91_290.0, 91_290.0) == 0.0

    def test_hypsometric_is_additive(self):
        a, b, c = 97_820.0, 95_000.0, 91_290.0
        assert alt.hypsometric(a, c) == pytest.approx(alt.hypsometric(a, b) + alt.hypsometric(b, c))

    def test_hypsometric_scales_with_the_column_temperature(self):
        warm = alt.hypsometric(97_820.0, 91_290.0, 300.0)
        cold = alt.hypsometric(97_820.0, 91_290.0, 270.0)
        assert warm / cold == pytest.approx(300.0 / 270.0)

    def test_standard_atmosphere_at_sea_level_and_at_1000_m(self):
        assert alt.standard_altitude(alt.P_STANDARD) == pytest.approx(0.0, abs=1e-9)
        # U.S. Standard Atmosphere 1976: 89 874.6 Pa at 1000 m.
        assert alt.standard_altitude(89_874.6) == pytest.approx(1000.0, abs=0.5)


class TestDecay:
    def test_survival_is_one_at_zero_distance(self):
        assert alt.survival(0.0, 1.0) == 1.0

    def test_classical_survival_over_one_decay_length(self):
        ctau = alt.C_LIGHT * alt.TAU_MU
        assert alt.survival(ctau, 1.0) == pytest.approx(np.exp(-1.0))
        assert ctau == pytest.approx(658.6, abs=0.1)

    @pytest.mark.parametrize("gamma", [1.0, 4.8, 37.9])
    def test_gamma_min_inverts_the_survival(self, gamma):
        dh = 603.0
        assert alt.gamma_min(dh, float(alt.survival(dh, gamma))) == pytest.approx(gamma)


class TestCounting:
    def test_only_complete_minutes_are_counted(self):
        t = np.array([0.5, 30.0, 61.0, 119.0, 150.0])
        assert list(alt.counts_per_minute(t, 0.0, 150.0)) == [2, 2]

    def test_dispersion_of_poisson_counts_is_near_one(self):
        rng = np.random.default_rng(3)
        d, z = alt.dispersion(rng.poisson(100.0, 2000))
        assert d == pytest.approx(1.0, abs=0.1)
        assert abs(z) < 4

    def test_chi2_poisson_accepts_poisson_and_rejects_overdispersion(self):
        rng = np.random.default_rng(4)
        _, _, p_ok, _ = alt.chi2_poisson(rng.poisson(90.0, 1000))
        assert p_ok > 0.001
        wide = rng.poisson(rng.choice([60.0, 120.0], 1000))
        _, _, p_bad, _ = alt.chi2_poisson(wide)
        assert p_bad < 1e-6

    def test_hourly_rates_skip_hours_not_fully_covered(self):
        t = np.arange(1800.0, 4 * 3600.0, 1.0)
        rates, _, _ = alt.hourly_rates(t, np.full(len(t), 500), np.zeros(len(t)),
                                       np.full(len(t), 30.0))
        assert len(rates) == 2 and np.allclose(rates, 1.0)

    def test_hour_excess_is_zero_when_the_spread_is_poisson(self):
        rates = np.array([1.0, 1.0, 1.0])
        assert alt.hour_excess(rates, np.full(3, 0.01)) == 0.0
