#!/usr/bin/env python3
"""
09_altitude.py

Rate against altitude with the 2026 campaign: Tacachico (about 307 m) against
Santa Tecla (about 911 m), the same detector and the same setup at both.

Two analyses, reported side by side and never combined (methodology 11.7):

  (a) As presented at the International Cosmic Day 2026. Tacachico uses only
      hours 26 to 40 of its first acquisition, the reference window of the
      fault report; Santa Tecla its whole file; every trigger counts. Every
      number here is the one quoted in the presentation.
  (b) After cleaning. Both sites keep every minute that passes the state veto
      of step 07 and only events with ADC2 >= 260, so Tacachico contributes
      its whole first acquisition. A fixed ADC2 cut is a different energy
      threshold at a different case temperature, so this ratio is a different
      observable from (a), not a correction of it.

Steps of (a):
  1. Live rates N / (T - sum DeadTime) and their Poisson errors.
  2. Counts per minute: index of dispersion and chi2 against a Poisson.
  3. Ratio R = R_Tacachico / R_SantaTecla. Its systematic is the hour-to-hour
     spread of Tacachico beyond Poisson, which follows the case temperature
     (methodology 11.3).
  4. Delta h from the two mean pressures (hypsometric equation, 25 +- 5 C
     column, 200 Pa of weather between the two days), and each altitude from
     sea level.
  5. Predictions exp(-Delta h / gamma c tau), classical and relativistic, and
     gamma_min with its one-sided 95 % bound taken in log scale.
  6. Composition: how many triggers can be muons, and what that does to the
     classical prediction. Ratios above ADC2 thresholds 1000 and 2000.

Exact outputs:
  - results/tables/09_altitude_vN.json and .txt
  - results/figures/09_ratio_vN.pdf/png
  - results/figures/09_ratio_vs_threshold_vN.pdf/png
  - results/figures/09_counts_per_minute_vN.pdf/png

Exit code: 0 if every internal check passes; 1 otherwise.

Typical use, from the repository root with .venv active:
  python scripts/09_altitude.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_muon import acquisition, altitude as alt, health, plotting, provenance  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

WINDOW_S = (93_600.0, 144_000.0)     # hours 26 to 40 of Tacachico's acquisition 1
HALVES_H = ((26, 36), (36, 40))
THRESHOLDS = (1000, 2000)            # ADC2 cuts quoted in the presentation
SMALL_PULSE = 600                    # ADC2 below which mostly SiPM noise
THRESHOLD_SCAN = np.arange(200, 3001, 50)


class Site:
    """A fixed window [t0, t1) of one acquisition."""

    def __init__(self, data: acquisition.Acquisition, t0: float, t1: float):
        self.data, self.t0, self.t1 = data.window(t0, t1), t0, t1

    @property
    def n(self) -> int:
        return len(self.data)

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    @property
    def dead(self) -> float:
        return float(self.data.deadtime_s.sum())

    @property
    def live(self) -> float:
        return self.duration - self.dead

    @property
    def rate(self) -> float:
        return self.n / self.live

    @property
    def sigma(self) -> float:
        return np.sqrt(self.n) / self.live

    def rate_above(self, threshold: float) -> tuple[float, float]:
        n = int((self.data.adc2 >= threshold).sum())
        return n / self.live, np.sqrt(n) / self.live

    def per_minute(self) -> np.ndarray:
        return alt.counts_per_minute(self.data.time_s, self.t0, self.t1)


def ratio(a: float, sa: float, b: float, sb: float) -> tuple[float, float]:
    q = a / b
    return q, q * np.hypot(sa / a, sb / b)


def cleaned_rate(d: acquisition.Acquisition, adc2_min: int) -> dict:
    """Live rate over the complete minutes that pass the state veto."""
    state = health.minute_state(d.time_s, d.adc1, d.adc2)
    inside = ((state.minute * 60.0 >= d.time_s[0])
              & ((state.minute + 1) * 60.0 <= d.time_s[-1]))
    good = ~state.vetoed & inside
    minute = np.floor(d.time_s / 60.0).astype(np.int64) - int(state.minute[0])
    keep = good[minute]
    live = 60.0 * good.sum() - float(d.deadtime_s[keep].sum())
    n = int((keep & (d.adc2 >= adc2_min)).sum())
    return {"minutes_kept": int(good.sum()), "minutes_spanned": int(inside.sum()),
            "n_events": n, "live_time_s": live, "rate_hz": n / live,
            "sigma_hz": np.sqrt(n) / live, "state": state}


def main() -> int:
    ap = argparse.ArgumentParser(description="Rate against altitude, 2026 campaign.")
    ap.add_argument("--altitude-dir", default="data/raw/altitude_2026")
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--out-figures", default="results/figures")
    args = ap.parse_args()

    directory = Path(args.altitude_dir)
    tac_file = acquisition.load_acquisitions(directory / "tacachico")
    st_file = acquisition.load_acquisitions(directory / "santa-tecla.txt")
    acq1 = tac_file.acquisitions[0]
    st_acq = st_file.acquisitions[0]

    # (a) As presented.
    tac = Site(acq1, *WINDOW_S)
    st = Site(st_acq, float(st_acq.time_s[0]), float(st_acq.time_s[-1]))
    r, r_sigma = ratio(tac.rate, tac.sigma, st.rate, st.sigma)

    h0, h1 = int(WINDOW_S[0] // 3600), int(WINDOW_S[1] // 3600)
    th, te, tt = alt.hourly_rates(acq1.time_s, acq1.adc2, acq1.deadtime_s, acq1.temperature_c,
                                  h0, h1)
    excess_tac = alt.hour_excess(th, te)
    r_sys = r * excess_tac / tac.rate
    r_tot = float(np.hypot(r_sigma, r_sys))
    corr_t = float(np.corrcoef(th, tt)[0, 1])
    z, nc = np.arctanh(corr_t), len(th)
    corr_ci = [float(np.tanh(z - 1.96 / np.sqrt(nc - 3))), float(np.tanh(z + 1.96 / np.sqrt(nc - 3)))]
    sh, se, _ = alt.hourly_rates(st_acq.time_s, st_acq.adc2, st_acq.deadtime_s,
                                 st_acq.temperature_c)
    excess_st = alt.hour_excess(sh, se)
    halves = []
    for a, b in HALVES_H:
        half = Site(acq1, 3600.0 * a, 3600.0 * b)
        q, sq = ratio(half.rate, half.sigma, st.rate, st.sigma)
        halves.append({"hours": [a, b], "ratio": q, "sigma": sq,
                       "mean_temperature_c": float(half.data.temperature_c.mean())})

    p_tac, p_st = float(tac.data.pressure_pa.mean()), float(st.data.pressure_pa.mean())
    dh = alt.hypsometric(p_tac, p_st)
    dh_t = abs(alt.hypsometric(p_tac, p_st, alt.T_COLUMN + alt.T_COLUMN_UNC) - dh)
    dh_p = alt.R_AIR * alt.T_COLUMN / alt.G * alt.DP_WEATHER * np.sqrt(2.0) / np.sqrt(p_tac * p_st)
    dh_unc = float(np.hypot(dh_t, dh_p))
    k = alt.R_AIR * alt.T_COLUMN / alt.G

    def altitude_unc(p: float, h: float) -> float:
        return float(np.sqrt((h * alt.T_COLUMN_UNC / alt.T_COLUMN) ** 2
                             + (k * alt.DP_WEATHER / p) ** 2 + (k * alt.P_ABS_UNC / p) ** 2))

    hh_tac, hh_st = alt.hypsometric(alt.P_STANDARD, p_tac), alt.hypsometric(alt.P_STANDARD, p_st)
    ctau = alt.C_LIGHT * alt.TAU_MU
    gamma = alt.E_MEAN / alt.M_MU
    r_clas, r_rel = float(alt.survival(dh, 1.0)), float(alt.survival(dh, gamma))
    r_clas_unc, r_rel_unc = r_clas * dh_unc / ctau, r_rel * dh_unc / (gamma * ctau)
    gmin = alt.gamma_min(dh, r)
    gmin_unc = gmin * np.hypot(dh_unc / dh, r_tot / (r * np.log(1.0 / r)))
    gmin95 = gmin * np.exp(-1.645 * gmin_unc / gmin)
    nsig_clas = (r - r_clas) / np.hypot(r_tot, r_clas * dh_unc / ctau)
    dx = (p_tac - p_st) / alt.G / 10.0                     # g/cm2

    def threshold_systematic(threshold: float, q: float) -> float:
        hu, eu, _ = alt.hourly_rates(acq1.time_s, acq1.adc2, acq1.deadtime_s,
                                     acq1.temperature_c, h0, h1, threshold=threshold)
        return q * alt.hour_excess(hu, eu) / tac.rate_above(threshold)[0]

    thresholds = []
    for u in THRESHOLDS:
        q, sq = ratio(*tac.rate_above(u), *st.rate_above(u))
        sys_u = threshold_systematic(u, q)
        thresholds.append({"adc2_min": u, "ratio": q, "stat": sq, "sys": sys_u,
                           "total": float(np.hypot(sq, sys_u))})
    scan = [ratio(*tac.rate_above(u), *st.rate_above(u)) for u in THRESHOLD_SCAN]

    mu_pdg = alt.FLUX_PDG * alt.AREA_CM2 / 60.0
    mu_axani = alt.FLUX_AXANI * alt.AREA_CM2 / 60.0
    r_clas_bg = [1 - (1 - r_clas) * mu_pdg / st.rate, 1 - (1 - r_clas) * mu_axani / st.rate]

    sites = {}
    for name, s in (("tacachico", tac), ("santa_tecla", st)):
        counts = s.per_minute()
        d, zd = alt.dispersion(counts)
        chi2, dof, p, groups = alt.chi2_poisson(counts)
        small = int((s.data.adc2 < SMALL_PULSE).sum())
        sites[name] = {
            "window_s": [s.t0, s.t1], "n_events": s.n, "duration_s": s.duration,
            "dead_fraction": s.dead / s.duration, "raw_rate_hz": s.n / s.duration,
            "raw_sigma_hz": float(np.sqrt(s.n) / s.duration),
            "live_rate_hz": s.rate, "live_sigma_hz": s.sigma,
            "minutes": int(len(counts)), "counts_per_minute_mean": float(counts.mean()),
            "counts_per_minute_var": float(counts.var(ddof=1)),
            "dispersion": d, "dispersion_sigmas": zd,
            "dispersion_unc": float(np.sqrt(2.0 / (len(counts) - 1))),
            "poisson_chi2": chi2, "poisson_dof": dof, "poisson_p": p, "poisson_groups": groups,
            "pressure_mean_pa": float(s.data.pressure_pa.mean()),
            "pressure_range_pa": [float(s.data.pressure_pa.min()), float(s.data.pressure_pa.max())],
            "temperature_c": {"min": float(s.data.temperature_c.min()),
                              "max": float(s.data.temperature_c.max()),
                              "mean": float(s.data.temperature_c.mean()),
                              "sd": float(s.data.temperature_c.std())},
            "coincident_fraction": float(s.data.coincident.mean()),
            "small_pulse_rate_hz": small / s.live,
            "small_pulse_sigma_hz": float(np.sqrt(small) / s.live),
            "muon_fraction_range": [mu_axani / s.rate, mu_pdg / s.rate],
        }

    presented = {
        "ratio": r, "ratio_stat": r_sigma, "ratio_sys_hours": r_sys, "ratio_total": r_tot,
        "hour_excess_tacachico_fraction": excess_tac / tac.rate,
        "hour_excess_tacachico_fraction_unc": alt.hour_excess_error(th, excess_tac) / tac.rate,
        "hour_excess_santa_tecla_fraction": excess_st / st.rate,
        "hour_excess_santa_tecla_fraction_unc": alt.hour_excess_error(sh, excess_st) / st.rate,
        "rate_temperature_correlation": corr_t, "rate_temperature_correlation_ci95": corr_ci,
        "halves": halves,
        "pressure_difference_pa": p_tac - p_st,
        "pressure_difference_unc_pa": alt.DP_WEATHER * np.sqrt(2.0),
        "column_density_difference_g_cm2": dx,
        "column_density_g_cm2": [p_tac / alt.G / 10.0, p_st / alt.G / 10.0],
        "delta_h_m": dh, "delta_h_unc_m": dh_unc, "delta_h_unc_temperature_m": dh_t,
        "delta_h_unc_weather_m": dh_p,
        "altitude_hypsometric_m": [hh_tac, hh_st],
        "altitude_hypsometric_unc_m": [altitude_unc(p_tac, hh_tac), altitude_unc(p_st, hh_st)],
        "altitude_standard_atmosphere_m": [alt.standard_altitude(p_tac),
                                           alt.standard_altitude(p_st)],
        "altitude_published_m": alt.ALT_PUBLISHED_M,
        "decay_length_at_rest_m": ctau, "gamma_typical": gamma,
        "prediction_classical": r_clas, "prediction_classical_unc": r_clas_unc,
        "prediction_relativistic": r_rel, "prediction_relativistic_unc": r_rel_unc,
        "classical_distance_sigmas": nsig_clas,
        "gamma_min": gmin, "gamma_min_unc": gmin_unc, "gamma_min_95_lower": gmin95,
        "attenuation_length_g_cm2": dx / np.log(1.0 / r),
        "muons_expected_hz": {"pdg": mu_pdg, "axani": mu_axani},
        "classical_with_background_range": r_clas_bg,
        "classical_with_background_sigmas": (r_clas_bg[1] - r) / r_tot,
        "max_muon_fraction_classical": (1 - r) / (1 - r_clas),
        "max_muon_fraction_classical_unc": r_tot / (1 - r_clas),
        "thresholds": thresholds,
    }

    # (b) After cleaning.
    tac_clean = cleaned_rate(acq1, health.ADC2_CLEAN)
    st_clean = cleaned_rate(st_acq, health.ADC2_CLEAN)
    rc, rc_sigma = ratio(tac_clean["rate_hz"], tac_clean["sigma_hz"],
                         st_clean["rate_hz"], st_clean["sigma_hz"])
    first_h, last_h = int(np.ceil(acq1.time_s[0] / 3600)), int(acq1.time_s[-1] // 3600)
    hourly = health.hourly_clean_rates(acq1.time_s, acq1.adc2, acq1.deadtime_s,
                                       acq1.temperature_c, tac_clean["state"],
                                       health.ADC2_CLEAN, first_h, last_h)
    excess_clean = alt.hour_excess(hourly["rate"], hourly["error"])
    rc_sys = rc * excess_clean / tac_clean["rate_hz"]
    rc_tot = float(np.hypot(rc_sigma, rc_sys))
    window_only = Site(acq1, *WINDOW_S)
    rw, rw_sigma = ratio(*window_only.rate_above(health.ADC2_CLEAN),
                         *st.rate_above(health.ADC2_CLEAN))
    gmin_c = alt.gamma_min(dh, rc)
    gmin_c_unc = gmin_c * np.hypot(dh_unc / dh, rc_tot / (rc * np.log(1.0 / rc)))
    cleaned = {
        "adc2_min": health.ADC2_CLEAN,
        "tacachico": {k2: v for k2, v in tac_clean.items() if k2 != "state"},
        "santa_tecla": {k2: v for k2, v in st_clean.items() if k2 != "state"},
        "tacachico_hours_used": int(len(hourly["hour"])),
        "hour_excess_tacachico_fraction": excess_clean / tac_clean["rate_hz"],
        "rate_temperature_correlation": float(np.corrcoef(hourly["rate"],
                                                          hourly["temperature"])[0, 1]),
        "ratio": rc, "ratio_stat": rc_sigma, "ratio_sys_hours": rc_sys, "ratio_total": rc_tot,
        "ratio_window_only_same_cut": rw, "ratio_window_only_same_cut_stat": rw_sigma,
        "classical_distance_sigmas": (rc - r_clas) / np.hypot(rc_tot, r_clas * dh_unc / ctau),
        "gamma_min": gmin_c, "gamma_min_unc": gmin_c_unc,
        "gamma_min_95_lower": gmin_c * np.exp(-1.645 * gmin_c_unc / gmin_c),
    }

    checks = [
        ("72 755 events in hours 26 to 40 (fault report 6.1)", tac.n == 72_755),
        ("Santa Tecla: one acquisition", len(st_file.acquisitions) == 1),
        ("Santa Tecla counts per minute compatible with Poisson (|z| < 3)",
         abs(sites["santa_tecla"]["dispersion_sigmas"]) < 3.0),
        ("hour systematic with threshold 0 equals that of all triggers",
         abs(threshold_systematic(0.0, r) - r_sys) < 1e-12),
        ("gamma_min inverted reproduces the ratio",
         abs(float(alt.survival(dh, gmin)) - r) < 1e-12),
        ("classical ratio = exp(-dh / c tau)", abs(np.log(r_clas) + dh / ctau) < 1e-12),
        ("hypsometric dh matches the standard atmosphere to 1 m",
         abs(alt.hypsometric(p_tac, p_st, 288.15 - 0.0065 * 0.5 * (
             alt.standard_altitude(p_tac) + alt.standard_altitude(p_st)))
             - (alt.standard_altitude(p_st) - alt.standard_altitude(p_tac))) < 1.0),
        ("cleaned Tacachico keeps most of its first acquisition",
         tac_clean["minutes_kept"] > 0.8 * tac_clean["minutes_spanned"]),
    ]

    t_ = sites["tacachico"]
    s_ = sites["santa_tecla"]
    lines = [
        "(a) AS PRESENTED AT THE INTERNATIONAL COSMIC DAY 2026",
        "    Tacachico: hours 26 to 40 of acquisition 1. Santa Tecla: whole file. All triggers.",
        f"  {'':<26}{'Tacachico':>14}{'Santa Tecla':>14}",
        f"  {'events N':<26}{t_['n_events']:>14}{s_['n_events']:>14}",
        f"  {'window (s)':<26}{t_['duration_s']:>14.0f}{s_['duration_s']:>14.0f}",
        f"  {'dead fraction':<26}{t_['dead_fraction']:>14.3f}{s_['dead_fraction']:>14.3f}",
        f"  {'raw rate (1/s)':<26}{t_['raw_rate_hz']:>14.3f}{s_['raw_rate_hz']:>14.3f}",
        f"  {'live rate (1/s)':<26}{t_['live_rate_hz']:>9.3f}+-{t_['live_sigma_hz']:.3f}"
        f"{s_['live_rate_hz']:>9.3f}+-{s_['live_sigma_hz']:.3f}",
        f"  {'counts/min mean, var':<26}{t_['counts_per_minute_mean']:>8.1f},{t_['counts_per_minute_var']:>6.1f}"
        f"{s_['counts_per_minute_mean']:>8.1f},{s_['counts_per_minute_var']:>6.1f}",
        f"  {'dispersion D (sigmas)':<26}{t_['dispersion']:>9.2f} ({t_['dispersion_sigmas']:.1f})"
        f"{s_['dispersion']:>9.2f} ({s_['dispersion_sigmas']:.1f})",
        f"  {'Poisson chi2 / dof, p':<26}{t_['poisson_chi2']:>7.1f}/{t_['poisson_dof']}, "
        f"{t_['poisson_p']:.2f}{s_['poisson_chi2']:>7.1f}/{s_['poisson_dof']}, {s_['poisson_p']:.2f}",
        f"  {'mean pressure (hPa)':<26}{t_['pressure_mean_pa'] / 100:>14.1f}"
        f"{s_['pressure_mean_pa'] / 100:>14.1f}",
        f"  {'case temperature (C)':<26}{t_['temperature_c']['mean']:>14.1f}"
        f"{s_['temperature_c']['mean']:>14.1f}",
        f"  {'altitude from P (m)':<26}{hh_tac:>9.0f}+-{presented['altitude_hypsometric_unc_m'][0]:.0f}"
        f"{hh_st:>11.0f}+-{presented['altitude_hypsometric_unc_m'][1]:.0f}",
        "",
        f"  R = {r:.3f} +- {r_sigma:.3f} (stat) +- {r_sys:.3f} (hours of Tacachico) "
        f"= {r:.3f} +- {r_tot:.3f}",
        f"  Tacachico hour-to-hour excess over Poisson {100 * excess_tac / tac.rate:.1f} %, "
        f"correlation with the case temperature {corr_t:.2f} "
        f"[{corr_ci[0]:.2f}, {corr_ci[1]:.2f}]; Santa Tecla {100 * excess_st / st.rate:.1f} %.",
        f"  R in hours 26-36: {halves[0]['ratio']:.3f} +- {halves[0]['sigma']:.3f} "
        f"({halves[0]['mean_temperature_c']:.1f} C); hours 36-40: {halves[1]['ratio']:.3f} +- "
        f"{halves[1]['sigma']:.3f} ({halves[1]['mean_temperature_c']:.1f} C).",
        f"  Delta h = {dh:.0f} +- {dh_unc:.0f} m (column temperature {dh_t:.0f} m, weather "
        f"{dh_p:.0f} m); Delta X = {dx:.1f} g/cm2.",
        f"  Predictions: classical {r_clas:.2f} +- {r_clas_unc:.2f}, relativistic (gamma = "
        f"{gamma:.0f}) {r_rel:.3f} +- {r_rel_unc:.3f}.",
        f"  The classical prediction sits {nsig_clas:.0f} sigma from R.",
        f"  gamma_min = {gmin:.1f} +- {gmin_unc:.1f}; one-sided 95 % lower bound {gmin95:.1f}.",
        "  All of this assumes every trigger is a muon. It is not. Counting the 25 cm2 face",
        "  only, as presented (the edges of a 1 cm slab add a fifth, step 08),",
        f"  the slab expects {mu_axani:.2f} to {mu_pdg:.2f} muons/s, "
        f"{100 * t_['muon_fraction_range'][0]:.0f} to {100 * t_['muon_fraction_range'][1]:.0f} % "
        f"of Tacachico's triggers and {100 * s_['muon_fraction_range'][0]:.0f} to "
        f"{100 * s_['muon_fraction_range'][1]:.0f} % of Santa Tecla's. With the rest "
        "independent of altitude,",
        f"  the classical prediction becomes {r_clas_bg[0]:.2f} to {r_clas_bg[1]:.2f}, "
        f"{presented['classical_with_background_sigmas']:.1f} sigma above R.",
        "  Ratios above ADC2 thresholds (stat, sys of the hours, total):",
    ]
    for u in thresholds:
        lines.append(f"    ADC2 >= {u['adc2_min']}: {u['ratio']:.2f} +- {u['stat']:.3f} +- "
                     f"{u['sys']:.3f} = +- {u['total']:.3f}")
    lines += [
        "  These data cannot decide between the two predictions.",
        "",
        f"(b) AFTER CLEANING: state veto and ADC2 >= {health.ADC2_CLEAN} at both sites",
        f"  Tacachico: {tac_clean['minutes_kept']} of {tac_clean['minutes_spanned']} minutes kept, "
        f"{tac_clean['n_events']} events, {tac_clean['rate_hz']:.4f} +- {tac_clean['sigma_hz']:.4f} /s",
        f"  Santa Tecla: {st_clean['minutes_kept']} of {st_clean['minutes_spanned']} minutes kept, "
        f"{st_clean['n_events']} events, {st_clean['rate_hz']:.4f} +- {st_clean['sigma_hz']:.4f} /s",
        f"  Tacachico hours used {len(hourly['hour'])}, hour-to-hour excess "
        f"{100 * cleaned['hour_excess_tacachico_fraction']:.1f} %, correlation with T "
        f"{cleaned['rate_temperature_correlation']:+.2f}.",
        f"  R_clean = {rc:.3f} +- {rc_sigma:.3f} (stat) +- {rc_sys:.3f} (hours) = +- {rc_tot:.3f}",
        f"  Same cut, ICD window only: {rw:.3f} +- {rw_sigma:.3f}.",
        f"  gamma_min = {gmin_c:.1f} +- {gmin_c_unc:.1f} with the same caveat on composition.",
        "  Cleaning removes the fault and lets the whole acquisition in, which shrinks",
        "  the hour systematic; it does not turn triggers into muons, and a fixed",
        "  ADC2 cut does not mean the same energy at 38 C as at 35 C.",
        "",
        "INTERNAL CHECKS",
    ]
    lines += [f"  [{'OK' if ok else 'FAIL'}] {name}" for name, ok in checks]
    text = "\n".join(lines)
    print(text)

    json_path, text_path = provenance.write_product(
        args.out, "09_altitude",
        data={"files": {"tacachico": tac_file.summary(), "santa_tecla": st_file.summary()},
              "sites": sites, "presented": presented, "cleaned": cleaned,
              "threshold_scan": {"adc2_min": THRESHOLD_SCAN.tolist(),
                                 "ratio": [q for q, _ in scan], "stat": [e for _, e in scan]},
              "checks": [{"name": n, "ok": bool(ok)} for n, ok in checks]},
        text=text,
        parameters={"altitude_dir": args.altitude_dir, "window_s": list(WINDOW_S),
                    "adc2_clean": health.ADC2_CLEAN},
    )
    print(f"[09] wrote {json_path}")
    print(f"[09] wrote {text_path}")

    # Figures.
    plotting.apply_style()
    figures = Path(args.out_figures)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.linspace(0, 1000, 200)
    ax.plot(x, alt.survival(x, 1.0), color=plotting.COLOUR_MODEL, label="classical (gamma = 1)")
    ax.plot(x, alt.survival(x, gamma), color=plotting.COLOUR_ALTERNATE,
            label=f"relativistic (gamma = {gamma:.0f})")
    points = [("all triggers, ICD window", r, r_tot, "o", plotting.COLOUR_DATA),
              (f"ADC2 >= {THRESHOLDS[0]}", thresholds[0]["ratio"], thresholds[0]["total"], "s",
               plotting.COLOUR_BAND),
              (f"ADC2 >= {THRESHOLDS[1]}", thresholds[1]["ratio"], thresholds[1]["total"], "D",
               plotting.COLOUR_BAND),
              (f"cleaned, ADC2 >= {health.ADC2_CLEAN}", rc, rc_tot, "^", "k")]
    for i, (label, q, e, marker, colour) in enumerate(points):
        ax.errorbar(dh + (i - 1.5) * 12, q, yerr=e, xerr=dh_unc if i == 0 else None,
                    fmt=marker, color=colour, ms=5, capsize=2, label=label)
    ax.set(xlabel="Delta h (m)", ylabel="R = rate(low site) / rate(high site)", ylim=(0, 1.05),
           title="Tacachico / Santa Tecla against the two predictions")
    ax.legend(fontsize=7, loc="lower left")
    plotting.save(fig, figures, "09_ratio")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    q = np.array([v for v, _ in scan])
    e = np.array([v for _, v in scan])
    ax.fill_between(THRESHOLD_SCAN, q - e, q + e, color=plotting.COLOUR_BAND, alpha=0.5)
    ax.plot(THRESHOLD_SCAN, q, color=plotting.COLOUR_DATA, label="ICD window, statistical band")
    ax.axhline(r_rel, color=plotting.COLOUR_ALTERNATE, ls="--", label="relativistic, decay only")
    ax.axhline(r_clas, color=plotting.COLOUR_MODEL, ls="--", label="classical")
    ax.set(xlabel="ADC2 threshold", ylabel="R above the threshold", ylim=(0, 1.05),
           title="The ratio falls as the pulses get larger")
    ax.legend(fontsize=8)
    plotting.save(fig, figures, "09_ratio_vs_threshold")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for a, (name, s) in zip(axes, (("Tacachico, hours 26-40", tac),
                                   ("Santa Tecla, whole file", st))):
        counts = s.per_minute()
        edges = np.arange(counts.min() - 0.5, counts.max() + 1.5)
        a.hist(counts, bins=edges, color=plotting.COLOUR_BAND, label="minutes")
        kk = np.arange(counts.min(), counts.max() + 1)
        a.plot(kk, len(counts) * stats.poisson.pmf(kk, counts.mean()),
               color=plotting.COLOUR_MODEL, label="Poisson, same mean")
        a.set(title=name, xlabel="counts per minute", ylabel="minutes")
        a.legend(fontsize=7)
    plotting.save(fig, figures, "09_counts_per_minute")

    failed = [name for name, ok in checks if not ok]
    if failed:
        print("[09] failed checks: " + "; ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
