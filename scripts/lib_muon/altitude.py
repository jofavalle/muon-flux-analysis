"""Rate against altitude: two sites, one detector, the same setup.

Ported from the analysis presented at the International Cosmic Day 2026, with
the same definitions so that the published numbers come out identical:

- live rate N / (T - sum of DeadTime) over a fixed window [t0, t1), with
  sigma = sqrt(N) / T_live;
- altitude difference from the pressure the detector itself records, through
  the hypsometric equation with an assumed air column of 25 +- 5 C (the
  detector's own temperature is that of its case, not of the air);
- survival of a muon over a height Delta h, exp(-Delta h / (gamma c tau)),
  classical (gamma = 1) and relativistic (gamma = E / m for E = 4 GeV);
- gamma_min, the smallest Lorentz factor compatible with a measured ratio R.

The interpretation of a ratio of trigger rates as a ratio of muon rates holds
only if every trigger is a muon, and with a single detector it does not
(methodology 11.6). The functions do not know that; the reports say it.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# Physical constants (PDG 2024).
C_LIGHT = 299_792_458.0          # m/s, exact
TAU_MU = 2.1969811e-6            # s, muon lifetime at rest
M_MU = 105.6583755               # MeV/c2
E_MEAN = 4000.0                  # MeV, mean muon energy at ground level (PDG 2020)

# Detector and flux.
AREA_CM2 = 25.0                  # cm2, 5 x 5 cm slab (assumed, dataset.md 4.1)
FLUX_PDG = 1.0                   # muons per cm2 per minute, horizontal detector (PDG 2020)
FLUX_AXANI = 0.756               # cm-2 min-1, coincidences measured by Axani et al. (2025)

# Atmosphere.
G = 9.80665                      # m/s2, standard gravity
R_AIR = 287.05                   # J/(kg K), dry air
T_COLUMN = 298.15                # K, 25 C: air between 300 and 900 m in El Salvador (assumed)
T_COLUMN_UNC = 5.0               # K, declared interval of that assumption
DP_WEATHER = 200.0               # Pa, assumed synoptic change between the two days
P_ABS_UNC = 100.0                # Pa, typical absolute accuracy of the BMP280
P_STANDARD = 101_325.0           # Pa, standard sea-level pressure
# Published altitude of each town (Spanish Wikipedia, read 2026-11-18).
ALT_PUBLISHED_M = {"Tacachico": (305.0, 306.0), "Santa Tecla": (920.0, 920.0)}


def live_rate(n: int, duration_s: float, dead_s: float) -> tuple[float, float]:
    """N / T_live and its Poisson error."""
    live = duration_s - dead_s
    return n / live, np.sqrt(n) / live


def hypsometric(p_low: float, p_high: float, t_k: float = T_COLUMN) -> float:
    """Height difference between two pressures for an isothermal column."""
    return R_AIR * t_k / G * np.log(p_low / p_high)


def standard_altitude(p: float) -> float:
    """U.S. Standard Atmosphere 1976, troposphere: h = T0/L (1 - (P/P0)^(R L / g M))."""
    return 44_330.8 * (1.0 - (p / P_STANDARD) ** 0.190263)


def decay_length(gamma) -> np.ndarray:
    return np.asarray(gamma, dtype=np.float64) * C_LIGHT * TAU_MU


def survival(distance_m, gamma: float) -> np.ndarray:
    """Fraction of muons surviving decay over a distance, at fixed gamma."""
    return np.exp(-np.asarray(distance_m, dtype=np.float64) / decay_length(gamma))


def gamma_min(dh_m: float, ratio: float) -> float:
    """Lorentz factor for which survival over dh equals the measured ratio."""
    return dh_m / (C_LIGHT * TAU_MU * np.log(1.0 / ratio))


def counts_per_minute(time_s: np.ndarray, t0_s: float, t1_s: float) -> np.ndarray:
    """Counts in the complete minutes of [t0, t1), counted from t0."""
    k = np.floor((time_s - t0_s) / 60.0).astype(np.int64)
    complete = int(np.floor((t1_s - t0_s) / 60.0))
    return np.bincount(k[k < complete], minlength=complete)


def dispersion(counts: np.ndarray) -> tuple[float, float]:
    """Index of dispersion var / mean and its distance from 1 in sigmas."""
    d = counts.var(ddof=1) / counts.mean()
    return float(d), float((d - 1.0) / np.sqrt(2.0 / (len(counts) - 1)))


def chi2_poisson(counts: np.ndarray) -> tuple[float, int, float, int]:
    """Goodness of fit of the counts-per-minute histogram to a Poisson of its mean.

    Values of k are grouped from below until each group expects at least 5
    minutes; the high tail goes into the last group. Degrees of freedom:
    groups minus 2 (normalisation and mean, which come from the data). Returns
    chi2, dof, p value and number of groups. A high p does not prove the counts
    are Poisson, only that this test finds no disagreement.
    """
    n, mean = len(counts), float(counts.mean())
    k = np.arange(0, int(counts.max()) + 60)
    expected = n * stats.poisson.pmf(k, mean)
    observed = np.bincount(counts, minlength=len(k))[:len(k)]
    groups, o_acc, e_acc = [], 0, 0.0
    for o, e in zip(observed, expected):
        o_acc, e_acc = o_acc + int(o), e_acc + float(e)
        if e_acc >= 5.0:
            groups.append([o_acc, e_acc])
            o_acc, e_acc = 0, 0.0
    groups[-1][0] += o_acc
    groups[-1][1] += e_acc
    o_g = np.array([g[0] for g in groups], dtype=float)
    e_g = np.array([g[1] for g in groups])
    e_g *= n / e_g.sum()
    chi2 = float(((o_g - e_g) ** 2 / e_g).sum())
    dof = len(groups) - 2
    return chi2, dof, float(stats.chi2.sf(chi2, dof)), len(groups)


def hourly_rates(time_s: np.ndarray, adc2: np.ndarray, deadtime_s: np.ndarray,
                 temperature_c: np.ndarray, first_hour: int | None = None,
                 last_hour: int | None = None, threshold: float | None = None):
    """Live rate, Poisson error and mean temperature of every complete hour.

    Only hours the data cover entirely are used: an hour missing its start
    would give a low rate that comes from the counting, not from the detector.
    With `threshold`, only events with ADC2 at or above it are counted, over
    the live time of all of them: every trigger blinds the detector, whether
    or not it passes the cut.
    """
    first = int(np.ceil(time_s[0] / 3600.0)) if first_hour is None else first_hour
    last = int(time_s[-1] // 3600) if last_hour is None else last_hour
    rates, errors, temps = [], [], []
    for h in range(first, last):
        s = (time_s >= 3600 * h) & (time_s < 3600 * (h + 1))
        live = 3600.0 - float(deadtime_s[s].sum())
        n = int(s.sum()) if threshold is None else int((s & (adc2 >= threshold)).sum())
        rates.append(n / live)
        errors.append(np.sqrt(n) / live)
        temps.append(float(temperature_c[s].mean()))
    return np.array(rates), np.array(errors), np.array(temps)


def hour_excess(rates: np.ndarray, errors: np.ndarray) -> float:
    """Spread between hours that Poisson does not explain: sqrt(var - <sigma^2>), or 0."""
    return float(np.sqrt(max(0.0, rates.var(ddof=1) - np.mean(errors**2))))


def hour_excess_error(rates: np.ndarray, excess: float) -> float:
    """Error of the excess, from that of the sample variance, var sqrt(2 / (n - 1))."""
    return float(rates.var(ddof=1) * np.sqrt(2.0 / (len(rates) - 1)) / (2.0 * excess))
