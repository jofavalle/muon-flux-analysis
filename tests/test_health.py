"""Tests of the detector-state classifier.

The synthetic tests fix what each rule does. The real-data tests fix what the
rules find: every one of the ten fault episodes falls under the veto, the
cleaned Tacachico rate is constant to about 1 %, and no 2025 run carries
either fault signature. If a threshold is changed, these say what moved.
"""

from pathlib import Path

import numpy as np
import pytest

from lib_muon import acquisition, health, io

ROOT = Path(__file__).resolve().parents[1]
ALTITUDE = ROOT / "data" / "raw" / "altitude_2026"
RAW = ROOT / "data" / "raw"


def minutes_of(n_per_minute, adc1, adc2, start_minute=0):
    """Synthetic events: n per minute, evenly spread, fixed ADC values per minute."""
    t, a1, a2 = [], [], []
    for i, (n, x1, x2) in enumerate(zip(n_per_minute, adc1, adc2)):
        m = start_minute + i
        t += list(60.0 * m + (np.arange(n) + 0.5) * 60.0 / n)
        a1 += [x1] * n
        a2 += [x2] * n
    return np.array(t), np.array(a1), np.array(a2)


class TestRules:
    def test_clean_minutes_are_not_vetoed(self):
        t, a1, a2 = minutes_of([90] * 5, [110] * 5, [800] * 5)
        state = health.minute_state(t, a1, a2)
        assert not state.vetoed.any()
        assert np.allclose(state.pedestal, 110)

    def test_a_high_pedestal_vetoes_its_minute_and_one_on_each_side(self):
        t, a1, a2 = minutes_of([90] * 5, [110, 110, 190, 110, 110], [800] * 5)
        state = health.minute_state(t, a1, a2)
        assert list(state.vetoed) == [False, True, True, True, False]

    def test_a_burst_is_vetoed_whatever_its_pedestal(self):
        t, a1, a2 = minutes_of([90, 90, 90], [110, 110, 110], [800, 800, 800])
        t_b, a1_b, a2_b = minutes_of([500], [339], [230], start_minute=5)
        state = health.minute_state(np.r_[t, t_b], np.r_[a1, a1_b], np.r_[a2, a2_b])
        assert state.vetoed[5] and state.vetoed[4] and not state.vetoed[3]
        assert state.burst_fraction[5] == 1.0

    def test_a_near_threshold_pile_up_is_vetoed_under_the_pedestal_limit(self):
        t, a1, a2 = minutes_of([90, 500, 90], [110, 165, 110], [800, 215, 800])
        state = health.minute_state(t, a1, a2, guard=0)
        assert list(state.vetoed) == [False, True, False]

    def test_pedestal_ignores_large_pulses(self):
        t, a1, a2 = minutes_of([60], [110], [800])
        t_big, a1_big, a2_big = minutes_of([30], [600], [4095])
        state = health.minute_state(np.r_[t, t_big], np.r_[a1, a1_big], np.r_[a2, a2_big])
        assert state.pedestal[0] == 110

    def test_events_map_to_their_minute(self):
        t, a1, a2 = minutes_of([90, 90, 90], [110, 190, 110], [800] * 3, start_minute=10)
        state = health.minute_state(t, a1, a2, guard=0)
        assert state.event_vetoed(t).sum() == 90

    def test_episodes_are_runs_of_high_minutes(self):
        counts = np.array([90, 90, 900, 950, 90, 90, 800, 90])
        assert health.episodes(counts) == [(2, 3), (6, 6)]

    def test_episodes_are_absolute_minutes_when_the_data_start_late(self):
        """A file whose first event falls in minute 5 and whose episode is
        minute 8: the episode must come back as minute 8, not 3."""
        t, a1, a2 = minutes_of([90, 90, 90, 900, 90], [110] * 5, [800] * 5, start_minute=5)
        state = health.minute_state(t, a1, a2)
        found = health.episodes(state.n_events, first_minute=int(state.minute[0]))
        assert found == [(8, 8)]
        assert state.n_events[found[0][0] - state.minute[0]] == 900

    def test_sipm_as_a_function_of_the_pair_has_zero_spread(self):
        rng = np.random.default_rng(1)
        a1 = rng.integers(80, 120, 5000)
        a2 = rng.integers(200, 260, 5000)
        derived = health.sipm_is_derived(a1, a2, 0.01 * a1 + 0.001 * a2)
        assert derived["max_spread_mv"] == pytest.approx(0.0, abs=1e-12)
        noisy = health.sipm_is_derived(a1, a2, rng.normal(size=5000))
        assert noisy["max_spread_mv"] > 0.1

    def test_constant_test_on_poisson_counts(self):
        rng = np.random.default_rng(2)
        n = rng.poisson(5000, 40)
        result = health.constant_test(n / 3600.0, np.sqrt(n) / 3600.0)
        assert result["dof"] == 39
        assert 0.001 < result["p_value"] < 0.999


@pytest.fixture(scope="module")
def tacachico():
    return acquisition.load_acquisitions(ALTITUDE / "tacachico").acquisitions[0]


@pytest.fixture(scope="module")
def tacachico_state(tacachico):
    return health.minute_state(tacachico.time_s, tacachico.adc1, tacachico.adc2)


class TestOnTheRealData:
    def test_sipm_is_derived_in_tacachico(self, tacachico):
        derived = health.sipm_is_derived(tacachico.adc1, tacachico.adc2, tacachico.sipm_mv)
        assert derived["max_spread_mv"] == 0.0
        assert derived["events_in_repeated_pairs"] > 300_000

    def test_ten_episodes_all_under_the_veto(self, tacachico_state):
        offset = int(tacachico_state.minute[0])
        found = health.episodes(tacachico_state.n_events, first_minute=offset)
        assert len(found) == 10
        for first, last in found:
            assert tacachico_state.vetoed[first - offset:last - offset + 1].all(), (first, last)

    def test_the_burst_reads_more_on_the_low_gain_channel(self, tacachico):
        burst = tacachico.window(4200.0, 4380.0)
        assert np.mean(burst.adc1 > burst.adc2) > 0.95
        clean = tacachico.window(93_600.0, 144_000.0)
        assert np.mean(clean.adc1 > clean.adc2) < 0.001

    def test_cleaned_hourly_rate_is_constant_to_about_one_percent(self, tacachico,
                                                                  tacachico_state):
        first, last = int(np.ceil(tacachico.time_s[0] / 3600)), int(tacachico.time_s[-1] // 3600)
        hourly = health.hourly_clean_rates(
            tacachico.time_s, tacachico.adc2, tacachico.deadtime_s, tacachico.temperature_c,
            tacachico_state, health.ADC2_CLEAN, first, last)
        test = health.constant_test(hourly["rate"], hourly["error"])
        assert len(hourly["hour"]) >= 35
        assert test["excess_fraction"] < 0.02
        assert test["chi2"] / test["dof"] < 2.0

    def test_no_2025_run_carries_a_fault_signature(self):
        for angle, run in io.load_all(RAW).items():
            d = health.run_diagnostics(run.timestamp_ms / 1e3, run.adc1, run.adc2,
                                       run.deadtime_us / 1e6)
            assert d["vetoed_minutes"] == 0, angle
            assert d["burst_events"] == 0, angle
            assert d["near_threshold_rate_hz"] < 0.2, angle

    def test_same_trigger_threshold_in_both_campaigns(self, tacachico):
        runs = io.load_all(RAW)
        a2_2025 = np.concatenate([r.adc2 for r in runs.values()])
        assert tacachico.adc2.min() == a2_2025.min() == 201
