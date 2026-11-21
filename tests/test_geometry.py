"""Tests of the single-slab reference against closed forms.

For a cos^2 flux the projected-area integral has exact values: a thin plate
gives g(alpha) = cos^2(alpha) + sin^2(alpha) / 2, so 3/4 at 45 degrees and 1/2
at 90; a 5 x 5 x 1 cm box gives 5/6 at 45 degrees and 2/3 at 90.
"""

import numpy as np
import pytest

from lib_muon import geometry

THIN = (5.0, 5.0, 1e-9)


class TestSlabResponse:
    def test_horizontal_is_the_reference(self):
        assert geometry.slab_response([0.0])[0] == 1.0

    @pytest.mark.parametrize("alpha", [5.0, 15.0, 25.0, 35.0, 45.0, 90.0])
    def test_thin_plate_closed_form(self, alpha):
        a = np.deg2rad(alpha)
        expected = np.cos(a) ** 2 + np.sin(a) ** 2 / 2
        assert geometry.slab_response([alpha], THIN)[0] == pytest.approx(expected, abs=1e-5)

    def test_box_5x5x1_closed_form(self):
        g = geometry.slab_response([45.0, 90.0])
        assert g[0] == pytest.approx(5 / 6, abs=1e-5)
        assert g[1] == pytest.approx(2 / 3, abs=1e-5)

    def test_thicker_slab_falls_less(self):
        g = [geometry.slab_response([90.0], (5.0, 5.0, c))[0] for c in (0.5, 1.0, 2.0)]
        assert g[0] < g[1] < g[2]

    def test_response_decreases_with_tilt(self):
        g = geometry.slab_response(np.arange(0, 91, 5))
        assert np.all(np.diff(g) < 0)

    def test_isotropic_flux_does_not_see_the_tilt_of_a_plate(self):
        """With n = 0 the flux is the same from every direction of the upper
        hemisphere. A horizontal plate collects integral |cos theta| dOmega = pi,
        and a vertical one integral sin^2 theta |cos phi| dtheta dphi = pi: the
        same. It is why a background that arrives from all directions is
        modelled as independent of orientation (methodology 10.3)."""
        g = geometry.slab_response([90.0], THIN, n=0.0)[0]
        assert g == pytest.approx(1.0, abs=1e-4)


class TestMuonFractionFit:
    def test_recovers_the_fraction_from_exact_data(self):
        g = geometry.slab_response([0, 5, 15, 25, 35, 45, 90])
        rates = 2.0 * geometry.two_component(g, 0.3)
        fit = geometry.fit_muon_fraction(rates, np.full(7, 0.05), g)
        assert fit["muon_fraction"] == pytest.approx(0.3, abs=1e-3)
        assert fit["scale"] == pytest.approx(2.0, rel=1e-6)
        assert fit["chi2"] == pytest.approx(0.0, abs=1e-9)
        assert fit["interval"][0] < 0.3 < fit["interval"][1]

    def test_a_flat_profile_gives_zero_muons(self):
        g = geometry.slab_response([0, 5, 15, 25, 35, 45, 90])
        fit = geometry.fit_muon_fraction(np.full(7, 1.8), np.full(7, 0.07), g)
        assert fit["muon_fraction"] == 0.0
