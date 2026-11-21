"""Detector state: telling the fault apart from normal operation.

The 2026 detector had a mechanical fault: a board loose inside its case that,
in some positions, sends the detector into a high-rate state. The raw files
carry no flag for it. What they carry is enough to recognise it, and this
module does so with two observables only, because there are only two:

- **SiPM[mV] carries no information of its own.** It is an exact function of
  the pair (ADC1, ADC2): every event sharing a pair shares its SiPM value
  (`sipm_is_derived`). The pulse is described by ADC1 and ADC2 alone.
- **ADC1 is almost always the pedestal of the low-gain channel.** For all but
  the largest pulses the low-gain channel barely moves off its baseline, so the
  median ADC1 of a minute (taken over events with ADC2 < 1500, below the bend
  where ADC1 starts to follow the pulse) measures the baseline of the
  electronics, not the pulses. That makes it a state monitor.

The fault shows two signatures (docs/detector_fault.md):

1. **Burst**: ADC1 > ADC2 in nearly every event, events every 6 to 10 ms. A
   real pulse never reads more on the low-gain channel than on the high-gain
   one; in clean data this happens in 0 to 0.1 % of events.
2. **Slow fault**: the pedestal rises (from about 100-150 to 170-200) and the
   extra events pile up just above the ADC2 trigger threshold. Above
   ADC2 = 400 the rate does not change at all while the total rate grows by
   a factor of four.

Cleaning is therefore a veto on the state of each minute (high pedestal,
burst, or a pile-up just above threshold) plus an amplitude cut. **The thresholds below were chosen by looking at the Tacachico file
itself**, so the validation on that file is not blind (methodology 9.6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

# ADC2 below which ADC1 sits on its pedestal (the bend is near 3000).
PEDESTAL_ADC2_MAX = 1500
# A minute whose pedestal exceeds this is in the slow-fault state.
PEDESTAL_VETO = 170.0
# A minute where more than this fraction of events has ADC1 > ADC2 is a burst.
BURST_FRACTION_VETO = 0.01
# A minute with more events than this in the near-threshold band is in the
# slow-fault state even if its pedestal stays under PEDESTAL_VETO. Clean minutes
# hold 7 to 16 such events; the episode at minute 319 of Tacachico, with a
# pedestal of 167, holds hundreds.
NEAR_THRESHOLD_VETO = 60
# Minutes vetoed on each side of a vetoed minute: transitions are not instantaneous.
GUARD_MINUTES = 1
# Amplitude cut that removes the near-threshold excess of the slow fault.
ADC2_CLEAN = 260
# Band just above the trigger threshold, where the slow fault adds its events.
NEAR_THRESHOLD = (200, 260)
# ADC2 bands for the rate-versus-pedestal table. The last one is saturation.
BANDS = ((200, 260), (260, 400), (400, 700), (700, 1200), (1200, 2500),
         (2500, 4095), (4095, 4096))
# Dead time pattern of the firmware: a fixed part plus whole display refreshes.
DEADTIME_BASE_S = 5.6e-3
DEADTIME_STEP_S = 41.0e-3


@dataclass
class MinuteState:
    """State of each minute counted from the detector's zero."""

    minute: np.ndarray          # minute index, floor(t / 60)
    n_events: np.ndarray
    pedestal: np.ndarray        # median ADC1 of events with ADC2 < 1500, NaN if none
    burst_fraction: np.ndarray  # fraction of events with ADC1 > ADC2
    near_threshold: np.ndarray  # events with ADC2 in NEAR_THRESHOLD
    vetoed: np.ndarray          # bool, after adding the guard minutes

    def event_vetoed(self, time_s: np.ndarray) -> np.ndarray:
        index = np.floor(time_s / 60.0).astype(np.int64) - int(self.minute[0])
        return self.vetoed[index]


def minute_state(time_s: np.ndarray, adc1: np.ndarray, adc2: np.ndarray,
                 pedestal_veto: float = PEDESTAL_VETO,
                 burst_veto: float = BURST_FRACTION_VETO,
                 near_veto: float = NEAR_THRESHOLD_VETO,
                 guard: int = GUARD_MINUTES) -> MinuteState:
    """Pedestal, burst fraction and veto of every minute spanned by the events."""
    k = np.floor(time_s / 60.0).astype(np.int64)
    first, last = int(k.min()), int(k.max())
    n = last - first + 1
    idx = k - first
    counts = np.bincount(idx, minlength=n)
    burst = np.bincount(idx, weights=(adc1 > adc2).astype(float), minlength=n)
    in_band = (adc2 >= NEAR_THRESHOLD[0]) & (adc2 < NEAR_THRESHOLD[1])
    near = np.bincount(idx, weights=in_band.astype(float), minlength=n).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        burst_fraction = np.where(counts > 0, burst / counts, 0.0)

    pedestal = np.full(n, np.nan)
    low = adc2 < PEDESTAL_ADC2_MAX
    order = np.argsort(idx[low], kind="stable")
    values, minutes = adc1[low][order], idx[low][order]
    bounds = np.flatnonzero(np.diff(minutes)) + 1
    for chunk_m, chunk_v in zip(np.split(minutes, bounds), np.split(values, bounds)):
        if len(chunk_m):
            pedestal[chunk_m[0]] = np.median(chunk_v)

    bad = ((np.nan_to_num(pedestal, nan=-np.inf) > pedestal_veto)
           | (burst_fraction > burst_veto) | (near > near_veto))
    vetoed = bad.copy()
    for shift in range(1, guard + 1):
        vetoed[shift:] |= bad[:-shift]
        vetoed[:-shift] |= bad[shift:]
    return MinuteState(np.arange(first, last + 1), counts, pedestal, burst_fraction, near,
                       vetoed)


def episodes(counts: np.ndarray, factor: float = 5.0) -> list[tuple[int, int]]:
    """Runs of consecutive minutes with more than `factor` times the median count.

    This is the criterion the fault report used to list its ten episodes. It
    only sees the fault when the rate jumps; the state veto also sees it when
    the rate creeps up.
    """
    high = np.flatnonzero(counts > factor * np.median(counts))
    if len(high) == 0:
        return []
    groups = np.split(high, np.flatnonzero(np.diff(high) > 1) + 1)
    return [(int(g[0]), int(g[-1])) for g in groups]


def sipm_is_derived(adc1: np.ndarray, adc2: np.ndarray, sipm: np.ndarray) -> dict:
    """Spread of SiPM among events that share the same (ADC1, ADC2) pair."""
    key = adc1.astype(np.int64) * 100_000 + adc2.astype(np.int64)
    order = np.argsort(key, kind="stable")
    k, s = key[order], sipm[order]
    starts = np.r_[0, np.flatnonzero(np.diff(k)) + 1]
    ends = np.r_[starts[1:], len(k)]
    repeated = (ends - starts) > 1
    spread = np.maximum.reduceat(s, starts) - np.minimum.reduceat(s, starts)
    return {
        "repeated_pairs": int(repeated.sum()),
        "events_in_repeated_pairs": int((ends - starts)[repeated].sum()),
        "events_total": int(len(k)),
        "max_spread_mv": float(spread[repeated].max()) if repeated.any() else 0.0,
        "correlation_with_adc1": float(np.corrcoef(sipm, adc1)[0, 1]),
        "correlation_with_adc2": float(np.corrcoef(sipm, adc2)[0, 1]),
    }


def firmware_fingerprint(adc2: np.ndarray, deadtime_s: np.ndarray) -> dict:
    """Trigger threshold and dead-time pattern, to compare detectors and firmware."""
    steps = np.round((deadtime_s - DEADTIME_BASE_S) / DEADTIME_STEP_S)
    residual = deadtime_s - DEADTIME_BASE_S - steps * DEADTIME_STEP_S
    return {
        "adc2_min": int(adc2.min()),
        "adc2_max": int(adc2.max()),
        "deadtime_on_pattern_fraction": float(np.mean(np.abs(residual) < 1e-3)),
        "deadtime_step_fractions": [float(np.mean(steps == s)) for s in range(4)],
    }


def band_rates(time_s: np.ndarray, adc1: np.ndarray, adc2: np.ndarray,
               deadtime_s: np.ndarray, bin_s: float = 600.0,
               min_events: int = 20) -> np.ndarray:
    """Pedestal and live rate per ADC2 band in complete bins from the first event.

    Returns an array with one row per bin: pedestal, total live rate, then one
    live rate per entry of BANDS. The dead time of every event counts, whatever
    its band: every trigger blinds the detector.
    """
    k = np.floor((time_s - time_s[0]) / bin_s).astype(np.int64)
    rows = []
    for b in range(int(k.max())):           # the last, partial bin is dropped
        m = k == b
        if m.sum() < min_events:
            continue
        live = bin_s - deadtime_s[m].sum()
        low = m & (adc2 < PEDESTAL_ADC2_MAX)
        pedestal = float(np.median(adc1[low])) if low.any() else np.nan
        row = [pedestal, m.sum() / live]
        row += [np.sum(m & (adc2 >= lo) & (adc2 < hi)) / live for lo, hi in BANDS]
        rows.append(row)
    return np.array(rows)


def hourly_clean_rates(time_s: np.ndarray, adc2: np.ndarray, deadtime_s: np.ndarray,
                       temperature_c: np.ndarray, state: MinuteState,
                       adc2_min: int, first_hour: int, last_hour: int,
                       min_good_minutes: int = 30) -> dict:
    """Live rate per hour using only minutes that pass the state veto.

    Live time of an hour is 60 s per good minute minus the dead time of every
    event in those minutes. Hours with fewer than `min_good_minutes` good
    minutes are skipped.
    """
    minute = np.floor(time_s / 60.0).astype(np.int64)
    good_event = ~state.event_vetoed(time_s)
    hours, rate, error, temp, good_minutes = [], [], [], [], []
    for h in range(first_hour, last_hour):
        mins = np.arange(60 * h, 60 * (h + 1)) - int(state.minute[0])
        mins = mins[(mins >= 0) & (mins < len(state.vetoed))]
        n_good = int((~state.vetoed[mins]).sum())
        if n_good < min_good_minutes:
            continue
        in_hour = (minute >= 60 * h) & (minute < 60 * (h + 1)) & good_event
        live = 60.0 * n_good - float(deadtime_s[in_hour].sum())
        n = int((in_hour & (adc2 >= adc2_min)).sum())
        hours.append(h)
        rate.append(n / live)
        error.append(np.sqrt(n) / live)
        temp.append(float(temperature_c[in_hour].mean()))
        good_minutes.append(n_good)
    return {"hour": np.array(hours), "rate": np.array(rate), "error": np.array(error),
            "temperature": np.array(temp), "good_minutes": np.array(good_minutes)}


def constant_test(rate: np.ndarray, error: np.ndarray) -> dict:
    """Weighted mean, chi2 against a constant and excess spread over Poisson."""
    w = 1.0 / error**2
    mean = float(np.sum(w * rate) / w.sum())
    chi2 = float(np.sum((rate - mean) ** 2 * w))
    dof = len(rate) - 1
    excess = np.sqrt(max(0.0, rate.var(ddof=1) - np.mean(error**2)))
    return {"mean": mean, "chi2": chi2, "dof": dof, "p_value": float(stats.chi2.sf(chi2, dof)),
            "excess_fraction": float(excess / mean)}


def run_diagnostics(time_s: np.ndarray, adc1: np.ndarray, adc2: np.ndarray,
                    deadtime_s: np.ndarray) -> dict:
    """Every fault signature evaluated on one short run.

    Time intervals are tested against an exponential only above 60 ms, out of
    the region shaped by the dead time. Counts in 10 s bins are tested for the
    Poisson index of dispersion D = var / mean, which a burst would push far
    above 1.
    """
    state = minute_state(time_s, adc1, adc2)
    pedestals = state.pedestal[~np.isnan(state.pedestal) & (state.n_events > 5)]
    duration = time_s[-1] - time_s[0]
    live = duration - deadtime_s[1:].sum()
    dt = np.diff(time_s)
    rate = len(dt) / duration
    tail = dt[dt > 0.060] - 0.060
    ks_p = float(stats.kstest(tail, "expon", args=(0.0, tail.mean())).pvalue)
    counts = np.bincount(np.floor((time_s - time_s[0]) / 10.0).astype(np.int64))[:-1]
    d = counts.var(ddof=1) / counts.mean()
    d_p = float(stats.chi2.sf(d * (len(counts) - 1), len(counts) - 1))
    near = (adc2 >= NEAR_THRESHOLD[0]) & (adc2 < NEAR_THRESHOLD[1])
    return {
        "n_events": int(len(time_s)),
        "pedestal_median": float(np.median(pedestals)),
        "pedestal_min": float(pedestals.min()),
        "pedestal_max": float(pedestals.max()),
        "vetoed_minutes": int(state.vetoed.sum()),
        "minutes": int(len(state.vetoed)),
        "burst_events": int((adc1 > adc2).sum()),
        "near_threshold_rate_hz": float(near.sum() / live),
        "fraction_dt_below_10ms": float(np.mean(dt <= 0.010)),
        "poisson_fraction_dt_below_10ms": float(1.0 - np.exp(-rate * 0.010)),
        "close_triplets_below_20ms": int(np.sum((dt[:-1] < 0.020) & (dt[1:] < 0.020))),
        "ks_p_exponential_tail": ks_p,
        "dispersion_10s": float(d),
        "dispersion_10s_p": d_p,
    }
