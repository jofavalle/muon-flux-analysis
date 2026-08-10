"""Tests of the models and the inference.

The first class is a REGRESSION test: it pins the defect that motivated
rebuilding the whole analysis. If it ever fails again, the repository has gone
back to publishing an exponent that is double-precision rounding.
"""

import numpy as np
import pytest

from lib_muon import models


class TestCosine90Regression:
    """cos(deg2rad(90)) is not zero, and it very nearly does not show."""

    def test_numpy_does_not_give_zero_at_90_degrees(self):
        raw = np.cos(np.deg2rad(90.0))
        assert raw != 0.0
        assert raw == pytest.approx(6.123233995736766e-17, rel=1e-12)

    def test_clamping_does_give_exact_zero(self):
        clamped = models.clamped_cosine(np.array([90.0]))
        assert clamped[0] == 0.0

    def test_the_original_bug_reproduced(self):
        """Unclamped, cos(90)^n with small n is of order one."""
        raw = np.cos(np.deg2rad(90.0))
        assert raw**0.01 == pytest.approx(0.688, abs=1e-3)
        assert raw**0.006 == pytest.approx(0.799, abs=1e-3)

    def test_clamped_any_positive_exponent_gives_zero(self):
        cos_t = models.clamped_cosine(np.array([0.0, 45.0, 90.0]))
        for n in (0.006, 0.01, 0.5, 1.0, 2.0, 5.0):
            shape = models.cosn_shape(cos_t, n)
            assert shape[-1] == 0.0, f"cos^{n}(90 degrees) should be exactly zero"
        assert models.cosn_shape(cos_t, 2.0)[0] == pytest.approx(1.0)
        assert models.cosn_shape(cos_t, 2.0)[1] == pytest.approx(0.5)

    def test_ordinary_angles_are_untouched(self):
        angles = np.array([0.0, 5.0, 15.0, 25.0, 35.0, 45.0, 89.9])
        np.testing.assert_allclose(
            models.clamped_cosine(angles), np.cos(np.deg2rad(angles)),
            rtol=0, atol=0,
        )

    def test_pure_cosn_is_impossible_with_counts_at_90_degrees(self):
        """The consequence of clamping: rigorously null likelihood."""
        cos_t = models.clamped_cosine(np.array([0.0, 45.0, 90.0]))
        times = np.array([300.0, 300.0, 300.0])
        counts = np.array([600.0, 550.0, 450.0])
        fit = models.fit_pure_cosn(counts, times, cos_t)
        assert not np.isfinite(fit.nll)
        assert not fit.converged


class TestLikelihood:
    def test_nll_infinite_when_mu_zero_with_counts(self):
        assert models.poisson_nll(np.array([0.0]), np.array([5.0])) == np.inf

    def test_nll_finite_when_mu_zero_without_counts(self):
        assert np.isfinite(models.poisson_nll(np.array([0.0]), np.array([0.0])))

    def test_likelihood_minimised_at_the_true_value(self):
        """For a Poisson, -logL is minimised at mu = N."""
        counts = np.array([100.0])
        candidates = np.array([80.0, 95.0, 100.0, 105.0, 120.0])
        nll = [models.poisson_nll(np.array([m]), counts) for m in candidates]
        assert np.argmin(nll) == 2

    def test_constant_recovers_the_exact_rate(self):
        times = np.array([100.0, 200.0, 300.0])
        counts = np.array([200.0, 400.0, 600.0])
        fit = models.fit_constant(counts, times)
        assert fit.parameters["R_hz"] == pytest.approx(2.0)
        assert models.deviance(counts, fit.mu) == pytest.approx(0.0, abs=1e-9)


class TestParameterRecovery:
    """On synthetic data whose answer is known beforehand."""

    @pytest.mark.parametrize("true_n", [0.5, 1.0, 2.0, 3.0])
    def test_recovers_a_known_n(self, true_n):
        generator = np.random.default_rng(20260810)
        angles = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        cos_t = models.clamped_cosine(angles)
        times = np.full(len(angles), 20000.0)
        true_rate = 2.0
        mu = true_rate * times * models.cosn_shape(cos_t, true_n)
        counts = generator.poisson(mu).astype(float)

        fit = models.fit_pure_cosn(counts, times, cos_t)
        assert fit.converged
        assert fit.parameters["n"] == pytest.approx(true_n, abs=0.05)
        assert fit.parameters["R_hz"] == pytest.approx(true_rate, rel=0.02)

    def test_recovers_known_signal_and_background(self):
        generator = np.random.default_rng(11)
        angles = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
        cos_t = models.clamped_cosine(angles)
        times = np.full(len(angles), 40000.0)
        true_s, true_b, true_n = 1.0, 0.5, 2.0
        mu = times * (true_s * models.cosn_shape(cos_t, true_n) + true_b)
        counts = generator.poisson(mu).astype(float)

        fit = models.fit_signal_background(counts, times, cos_t, n=true_n)
        assert fit.parameters["S_hz"] == pytest.approx(true_s, rel=0.06)
        assert fit.parameters["B_hz"] == pytest.approx(true_b, rel=0.10)

    def test_the_profile_brackets_the_true_value(self):
        generator = np.random.default_rng(7)
        angles = np.array([0.0, 20.0, 40.0, 60.0, 80.0])
        cos_t = models.clamped_cosine(angles)
        times = np.full(len(angles), 30000.0)
        true_n = 2.0
        mu = 2.0 * times * models.cosn_shape(cos_t, true_n)
        counts = generator.poisson(mu).astype(float)
        fit = models.fit_pure_cosn(counts, times, cos_t)

        def nll_of_n(n):
            shape = models.cosn_shape(cos_t, n)
            r = float(counts.sum() / float(np.sum(times * shape)))
            return models.poisson_nll(r * times * shape, counts)

        low, high = models.profile_likelihood(
            nll_of_n, fit.parameters["n"], nll_of_n(fit.parameters["n"])
        )
        assert low < true_n < high


class TestModelComparison:
    def test_aic_penalises_parameters(self):
        counts = np.array([100.0, 100.0, 100.0])
        times = np.array([50.0, 50.0, 50.0])
        constant = models.fit_constant(counts, times)
        cos_t = models.clamped_cosine(np.array([0.0, 10.0, 20.0]))
        signal = models.fit_signal_background(counts, times, cos_t, n=2.0)
        # The data are flat: the extra parameter buys nothing and AIC must
        # penalise it.
        assert constant.aic < signal.aic

    def test_likelihood_ratio_finds_no_signal_where_there_is_none(self):
        d, p = models.likelihood_ratio(nll_null=100.0, nll_alternative=99.9,
                                       extra_dof=1)
        assert d == pytest.approx(0.2, abs=1e-9)
        assert p > 0.05

    def test_likelihood_ratio_finds_a_large_signal(self):
        _, p = models.likelihood_ratio(nll_null=110.0, nll_alternative=100.0,
                                       extra_dof=1)
        assert p < 1e-4


class TestSystematic:
    def test_a_systematic_relaxes_the_chi2(self):
        observed = np.array([2.0, 1.8, 2.2, 1.9, 2.1])
        errors = np.full(5, 0.05)
        _, chi2_without, p_without = models.chi2_with_systematic(observed, errors, 0.0)
        _, chi2_with, p_with = models.chi2_with_systematic(observed, errors, 0.10)
        assert chi2_with < chi2_without
        assert p_with > p_without

    def test_required_systematic_is_zero_when_already_consistent(self):
        observed = np.array([2.0, 2.01, 1.99])
        errors = np.full(3, 0.5)
        assert models.required_systematic(observed, errors) == 0.0

    def test_required_systematic_reaches_the_target_p(self):
        observed = np.array([2.0, 1.6, 2.4, 1.8, 2.2])
        errors = np.full(5, 0.05)
        f = models.required_systematic(observed, errors, target_p=0.05)
        assert f > 0
        _, _, p = models.chi2_with_systematic(observed, errors, f)
        assert p == pytest.approx(0.05, abs=1e-6)
