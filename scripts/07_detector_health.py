#!/usr/bin/env python3
"""
07_detector_health.py

Recognise the detector fault in the 2026 data, validate a cleaning recipe on
the file that has the fault, and check whether the 2025 angular runs carry it.

Steps:
  1. Integrity of the 2026 files against their documented state: checksums,
     acquisitions, rejected lines, glued comments. Compared in both directions,
     like the quality gate (methodology 7.3).
  2. SiPM[mV] is an exact function of (ADC1, ADC2): it adds nothing to separate
     fault from signal (methodology 9.1).
  3. State of every minute of the first Tacachico acquisition: pedestal of the
     low-gain channel, fraction of ADC1 > ADC2, veto (methodology 9.2 to 9.4).
     The ten high-rate episodes of the fault report are listed with their
     signature, and every one of their minutes must fall under the veto.
  4. Live rate per ADC2 band against the pedestal, in 10 minute bins: the
     fault adds events just above the trigger threshold and nowhere else.
  5. Validation: hourly live rate of Tacachico before and after cleaning
     (veto plus ADC2 >= 260), against a constant; the same on Santa Tecla,
     where nothing should change.
  6. The same signatures on each of the eight 2025 runs, and the firmware
     fingerprint of the three data sets.

The thresholds were chosen by looking at Tacachico, so step 5 is not a blind
test (methodology 9.6). The conclusion about 2025 does not depend on them:
no 2025 event has ADC1 > ADC2 and the near-threshold rate is that of the
cleanest state, whatever the thresholds.

Exact outputs:
  - results/tables/07_detector_health_vN.json and .txt
  - results/figures/07_adc_plane_vN.pdf/png
  - results/figures/07_band_rates_vN.pdf/png
  - results/figures/07_tacachico_cleaning_vN.pdf/png
  - results/figures/07_state_timeline_vN.pdf/png

Exit code: 0 if the data match their documented state; 1 otherwise, naming
what changed (for example a 2025 run that is no longer clean).

Typical use, from the repository root with .venv active:
  python scripts/07_detector_health.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_muon import acquisition, health, io, plotting, provenance  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (after plotting fixes the backend)
from matplotlib.colors import LogNorm  # noqa: E402

# Documented state of the 2026 files (docs/dataset.md, section 7).
DOCUMENTED = {
    "tacachico": {
        "sha256": "0e6d19ef771f526ea3496246dad9a70a8a8e8f7bafb217a77c8e89dfe9a67d3e",
        "rows_per_acquisition": [382_709, 4_330, 2_491],
        "rejected_lines": [1, 385_764],
        "glued_comment_lines": [382_717],
    },
    "santa-tecla.txt": {
        "sha256": "f255bfeb496e12f30fb2f2cc7b93cb78909ec5e07dfa91c1eac7fc413e35019c",
        "rows_per_acquisition": [200_859],
        "rejected_lines": [2],
        "glued_comment_lines": [],
    },
}
# The reference window of the fault report: hours 26 to 40 of acquisition 1.
WINDOW_S = (93_600.0, 144_000.0)
# The burst quoted in the fault report: minutes 70 to 73.
BURST_S = (4_200.0, 4_380.0)
CUTS = (0, health.ADC2_CLEAN, 400, 1000)
PEDESTAL_CLASSES = ((90, 110), (110, 130), (130, 150), (150, 170), (170, 185),
                    (185, 200), (200, 400))


def integrity(name: str, report: acquisition.AcquisitionFile) -> list[str]:
    expected = DOCUMENTED[name]
    observed = {
        "sha256": report.sha256,
        "rows_per_acquisition": [len(a) for a in report.acquisitions],
        "rejected_lines": [n for n, _ in report.rejected_texts],
        "glued_comment_lines": report.glued_comment_lines,
    }
    return [f"{name}: {key} is {observed[key]}, documented {value}"
            for key, value in expected.items() if observed[key] != value]


def signature(d: acquisition.Acquisition, duration_s: float) -> dict:
    dt = np.diff(d.time_s)
    return {
        "n_events": int(len(d)),
        "raw_rate_hz": float(len(d) / duration_s),
        "coincident_fraction": float(d.coincident.mean()),
        "adc1_median": float(np.median(d.adc1)),
        "adc2_median": float(np.median(d.adc2)),
        "sipm_median_mv": float(np.median(d.sipm_mv)),
        "adc1_above_adc2_fraction": float(np.mean(d.adc1 > d.adc2)),
        "dt_median_ms": float(np.median(np.rint(dt * 1e3))) if len(dt) else None,
        "dt_at_most_10ms_fraction": float(np.mean(np.rint(dt * 1e3) <= 10)) if len(dt) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Detector state and fault cleaning.")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--altitude-dir", default="data/raw/altitude_2026")
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--out-figures", default="results/figures")
    args = ap.parse_args()

    altitude_dir = Path(args.altitude_dir)
    files = {name: acquisition.load_acquisitions(altitude_dir / name) for name in DOCUMENTED}
    problems = []
    for name, report in files.items():
        problems += integrity(name, report)

    tac = files["tacachico"].acquisitions[0]
    st = files["santa-tecla.txt"].acquisitions[0]
    runs = io.load_all(args.data_dir)

    # 2. SiPM is derived from the two ADC channels.
    sipm = health.sipm_is_derived(
        np.concatenate([tac.adc1, st.adc1] + [r.adc1 for r in runs.values()]),
        np.concatenate([tac.adc2, st.adc2] + [r.adc2 for r in runs.values()]),
        np.concatenate([tac.sipm_mv, st.sipm_mv] + [r.sipm_mv for r in runs.values()]),
    )
    if sipm["max_spread_mv"] != 0.0:
        problems.append(f"SiPM is not a function of (ADC1, ADC2): spread {sipm['max_spread_mv']}")

    # 3. State of every minute of Tacachico.
    state = health.minute_state(tac.time_s, tac.adc1, tac.adc2)
    m0 = int(state.minute[0])
    episode_list = health.episodes(state.n_events, first_minute=m0)
    episode_rows = []
    for first, last in episode_list:
        seg = tac.window(60.0 * first, 60.0 * (last + 1))
        row = {"first_minute": first, "last_minute": last}
        row.update(signature(seg, 60.0 * (last - first + 1)))
        row["all_minutes_vetoed"] = bool(state.vetoed[first - m0:last - m0 + 1].all())
        episode_rows.append(row)
    in_episode = np.zeros(len(state.minute), bool)
    for first, last in episode_list:
        in_episode[first - m0:last - m0 + 1] = True
    unvetoed_episode_minutes = int((in_episode & ~state.vetoed).sum())
    if unvetoed_episode_minutes:
        problems.append(f"{unvetoed_episode_minutes} episode minutes escape the state veto")
    event_vetoed = state.event_vetoed(tac.time_s)
    episode_events = in_episode[np.floor(tac.time_s / 60).astype(int) - m0]

    st_state = health.minute_state(st.time_s, st.adc1, st.adc2)
    states = {
        "santa_tecla_all": signature(st, st.time_s[-1] - st.time_s[0]),
        "tacachico_window_h26_40": signature(tac.window(*WINDOW_S), WINDOW_S[1] - WINDOW_S[0]),
        "tacachico_burst_min70_73": signature(tac.window(*BURST_S), BURST_S[1] - BURST_S[0]),
        "tacachico_episode_minutes": signature(tac.select(episode_events),
                                               60.0 * in_episode.sum()),
        "tacachico_vetoed_minutes": signature(tac.select(event_vetoed),
                                              60.0 * state.vetoed.sum()),
        "tacachico_kept_minutes": signature(tac.select(~event_vetoed),
                                            60.0 * (~state.vetoed).sum()),
    }

    # The Coincident flag: when it changes in acquisition 1.
    hour_of = np.floor(tac.time_s / 3600).astype(int)

    def preamble_messages(path: Path) -> int:
        with open(path, "r", encoding="ascii", errors="replace") as handle:
            return sum("Coincidence detector found" in line for line in handle)

    coincident = {
        "preamble_messages": {
            **{name: preamble_messages(altitude_dir / name) for name in DOCUMENTED},
            "runs_2025": sum(preamble_messages(Path(args.data_dir) / f"{a:03d}deg")
                             for a in runs),
        },
        "flagged_fraction_per_acquisition_tacachico": [
            float(a.coincident.mean()) for a in files["tacachico"].acquisitions],
        "acquisition1_fraction": float(tac.coincident.mean()),
        "flagged_fraction_hours_0_to_4": [float(tac.coincident[hour_of == h].mean())
                                          for h in range(5)],
        "flagged_fraction_from_hour_4": float(tac.coincident[hour_of >= 4].mean()),
        "santa_tecla_fraction": float(st.coincident.mean()),
        "runs_2025_fraction": float(np.concatenate([r.coincident for r in runs.values()]).mean()),
    }

    # 4. Band rates against the pedestal.
    tac_bins = health.band_rates(tac.time_s, tac.adc1, tac.adc2, tac.deadtime_s)
    st_bins = health.band_rates(st.time_s, st.adc1, st.adc2, st.deadtime_s)
    band_names = [f"[{lo},{hi})" for lo, hi in health.BANDS]
    by_pedestal = []
    for lo, hi in PEDESTAL_CLASSES:
        m = (tac_bins[:, 0] >= lo) & (tac_bins[:, 0] < hi)
        if m.any():
            by_pedestal.append({"pedestal": [lo, hi], "bins": int(m.sum()),
                                "total_hz": float(np.median(tac_bins[m, 1])),
                                "bands_hz": [float(np.median(tac_bins[m, 2 + j]))
                                             for j in range(len(band_names))]})
    live_2025 = sum(r.live_time_s for r in runs.values())
    adc2_2025 = np.concatenate([r.adc2[1:] for r in runs.values()])
    bands_2025 = [float(np.sum((adc2_2025 >= lo) & (adc2_2025 < hi)) / live_2025)
                  for lo, hi in health.BANDS]
    bands_st = [float(np.median(st_bins[:, 2 + j])) for j in range(len(band_names))]

    # 5. Validation of the cleaning.
    first_h, last_h = int(np.ceil(tac.time_s[0] / 3600)), int(tac.time_s[-1] // 3600)
    no_veto = health.minute_state(tac.time_s, tac.adc1, tac.adc2, pedestal_veto=np.inf,
                                  burst_veto=np.inf, near_veto=np.inf)
    validation = {"raw": None, "cuts": {}}
    raw = health.hourly_clean_rates(tac.time_s, tac.adc2, tac.deadtime_s, tac.temperature_c,
                                    no_veto, 0, first_h, last_h)
    validation["raw"] = health.constant_test(raw["rate"], raw["error"])
    validation["raw"]["hours"] = int(len(raw["hour"]))
    hourly = {}
    for cut in CUTS:
        h = health.hourly_clean_rates(tac.time_s, tac.adc2, tac.deadtime_s, tac.temperature_c,
                                      state, cut, first_h, last_h)
        hourly[cut] = h
        test = health.constant_test(h["rate"], h["error"])
        in_window = (h["hour"] >= WINDOW_S[0] / 3600) & (h["hour"] < WINDOW_S[1] / 3600)
        rw, ro = h["rate"][in_window], h["rate"][~in_window]
        spread = np.sqrt(rw.var(ddof=1) / len(rw) + ro.var(ddof=1) / len(ro))
        test.update({
            "hours": int(len(h["hour"])),
            "correlation_with_temperature": float(np.corrcoef(h["rate"], h["temperature"])[0, 1]),
            "window_mean_hz": float(rw.mean()),
            "outside_window_mean_hz": float(ro.mean()),
            "window_minus_outside_sigmas": float((rw.mean() - ro.mean()) / spread),
        })
        validation["cuts"][str(cut)] = test
    st_first, st_last = int(np.ceil(st.time_s[0] / 3600)), int(st.time_s[-1] // 3600)
    control = {}
    for cut in (0, health.ADC2_CLEAN):
        h = health.hourly_clean_rates(st.time_s, st.adc2, st.deadtime_s, st.temperature_c,
                                      st_state, cut, st_first, st_last)
        test = health.constant_test(h["rate"], h["error"])
        test.update({"hours": int(len(h["hour"])),
                     "correlation_with_temperature":
                         float(np.corrcoef(h["rate"], h["temperature"])[0, 1])})
        control[str(cut)] = test

    # 6. The 2025 runs and the firmware fingerprint.
    diagnostics_2025 = {}
    for angle, run in sorted(runs.items()):
        diagnostics_2025[str(angle)] = health.run_diagnostics(
            run.timestamp_ms / 1e3, run.adc1, run.adc2, run.deadtime_us / 1e6)
        diag = diagnostics_2025[str(angle)]
        if diag["vetoed_minutes"] or diag["burst_events"]:
            problems.append(f"2025 run {angle} deg is no longer clean: "
                            f"{diag['vetoed_minutes']} vetoed minutes, "
                            f"{diag['burst_events']} events with ADC1 > ADC2")
    fingerprint = {
        "tacachico": health.firmware_fingerprint(tac.adc2, tac.deadtime_s),
        "santa_tecla": health.firmware_fingerprint(st.adc2, st.deadtime_s),
        "runs_2025": health.firmware_fingerprint(
            np.concatenate([r.adc2 for r in runs.values()]),
            np.concatenate([r.deadtime_us for r in runs.values()]) / 1e6),
    }

    # Text report.
    lines = ["INTEGRITY OF THE 2026 FILES"]
    for name, report in files.items():
        lines.append(f"  {name}: {report.n_rows} rows in {len(report.acquisitions)} "
                     f"acquisition(s) {[len(a) for a in report.acquisitions]}, "
                     f"{report.rejected_lines} rejected, glued comments at "
                     f"{report.glued_comment_lines}, sha256 {report.sha256[:12]}...")
    lines += ["", "SiPM[mV] AGAINST (ADC1, ADC2)",
              f"  {sipm['events_in_repeated_pairs']} of {sipm['events_total']} events share an "
              f"(ADC1, ADC2) pair with another; the largest SiPM spread inside a pair is "
              f"{sipm['max_spread_mv']:.2f} mV.",
              "  SiPM is a function of the two channels and carries no information of its own.",
              "", "STATES (Tacachico acquisition 1 unless named)",
              f"  {'state':<28}{'N':>8}{'rate/s':>9}{'coinc':>7}{'ADC1':>6}{'ADC2':>6}"
              f"{'SiPM':>6}{'A1>A2':>7}{'dt<=10ms':>9}"]
    for name, s in states.items():
        lines.append(f"  {name:<28}{s['n_events']:>8}{s['raw_rate_hz']:>9.3f}"
                     f"{s['coincident_fraction']:>7.3f}{s['adc1_median']:>6.0f}"
                     f"{s['adc2_median']:>6.0f}{s['sipm_median_mv']:>6.1f}"
                     f"{s['adc1_above_adc2_fraction']:>7.3f}{s['dt_at_most_10ms_fraction']:>9.3f}")
    lines += ["", f"EPISODES (minutes above 5 times the median count): {len(episode_rows)}",
              f"  {'minutes':>11}{'N':>7}{'rate/s':>8}{'coinc':>6}{'A1>A2':>6}{'ADC1':>6}"
              f"{'ADC2':>6}{'dt med':>7}  vetoed"]
    for e in episode_rows:
        lines.append(f"  {e['first_minute']:>5}-{e['last_minute']:<5}{e['n_events']:>7}"
                     f"{e['raw_rate_hz']:>8.1f}{e['coincident_fraction']:>6.2f}"
                     f"{e['adc1_above_adc2_fraction']:>6.2f}{e['adc1_median']:>6.0f}"
                     f"{e['adc2_median']:>6.0f}{e['dt_median_ms']:>7.0f}  "
                     f"{'all' if e['all_minutes_vetoed'] else 'NOT ALL'}")
    lines += [f"  Events in episodes: {episode_events.mean():.1%} of acquisition 1.",
              f"  The state veto (pedestal > {health.PEDESTAL_VETO:.0f} or more than "
              f"{health.BURST_FRACTION_VETO:.0%} of ADC1 > ADC2 or more than "
              f"{health.NEAR_THRESHOLD_VETO} events in [200, 260), {health.GUARD_MINUTES} guard "
              f"minute) removes {state.vetoed.sum()} of {len(state.vetoed)} minutes "
              f"({state.vetoed.mean():.1%}); episode minutes it misses: "
              f"{unvetoed_episode_minutes}.",
              "", "COINCIDENT FLAG",
              f"  Acquisition 1: {coincident['acquisition1_fraction']:.1%} of events flagged. "
              "Per hour, hours 0 to 4: "
              + ", ".join(f"{x:.1%}" for x in coincident["flagged_fraction_hours_0_to_4"])
              + f"; from hour 4 on, {coincident['flagged_fraction_from_hour_4']:.3%}.",
              f"  Santa Tecla {coincident['santa_tecla_fraction']:.2%}; 2025 runs "
              f"{coincident['runs_2025_fraction']:.0%}.",
              "  Every preamble in the files prints 'Coincidence detector found.' ("
              + ", ".join(f"{k}: {v}" for k, v in coincident["preamble_messages"].items())
              + "; santa-tecla.txt starts",
              "  after its preamble), yet acquisitions 2 and 3",
              "  of Tacachico print it and flag "
              + " and ".join(f"{x:.2%}" for x in coincident[
                  "flagged_fraction_per_acquisition_tacachico"][1:])
              + " of their events. Neither the message nor the flag",
              "  follows a partner. The group's report says no second detector was connected",
              "  in 2026; the flag changed on its own and is not a coincidence.",
              "", "LIVE RATE PER ADC2 BAND AGAINST THE PEDESTAL (median of 10 min bins, 1/s)",
              f"  {'pedestal':<10}{'bins':>5}{'total':>7}" + "".join(f"{b:>12}" for b in band_names)]
    for row in by_pedestal:
        lo, hi = row["pedestal"]
        lines.append(f"  {f'{lo}-{hi}':<10}{row['bins']:>5}{row['total_hz']:>7.3f}"
                     + "".join(f"{v:>12.3f}" for v in row["bands_hz"]))
    lines.append(f"  {'Sta Tecla':<10}{len(st_bins):>5}{np.median(st_bins[:, 1]):>7.3f}"
                 + "".join(f"{v:>12.3f}" for v in bands_st))
    lines.append(f"  {'2025 all':<10}{'':>5}{len(adc2_2025) / live_2025:>7.3f}"
                 + "".join(f"{v:>12.3f}" for v in bands_2025))
    low_cls, high_cls = by_pedestal[0], [r for r in by_pedestal if r["pedestal"][1] <= 200][-1]
    near_factor = high_cls["bands_hz"][0] / low_cls["bands_hz"][0]
    upper = [high_cls["bands_hz"][j] / low_cls["bands_hz"][j] - 1
             for j, (lo, _) in enumerate(health.BANDS) if lo >= 400]
    lines += [f"  From pedestal class {low_cls['pedestal']} to {high_cls['pedestal']} the band"
              f" just above threshold grows by a factor of {near_factor:.0f}, the bands",
              f"  from ADC2 = 400 up change by {100 * min(upper):+.0f} % to {100 * max(upper):+.0f} %."
              " The slow fault adds its events just above",
              "  the trigger threshold. In the burst state (pedestal above 200) the [400, 700)",
              "  band rises as well.",
              "", "VALIDATION ON TACACHICO: HOURLY LIVE RATE AGAINST A CONSTANT",
              f"  raw (no veto, no cut): {validation['raw']['hours']} hours, chi2 "
              f"{validation['raw']['chi2']:.1f} / {validation['raw']['dof']}",
              f"  {'cut':<10}{'hours':>6}{'rate/s':>9}{'chi2/dof':>14}{'p':>10}{'excess':>8}"
              f"{'corr(T)':>9}{'window':>9}{'outside':>9}"]
    for cut, v in validation["cuts"].items():
        lines.append(f"  ADC2>={cut:<4}{v['hours']:>6}{v['mean']:>9.4f}"
                     f"{v['chi2']:>9.1f}/{v['dof']:<4}{v['p_value']:>10.2g}"
                     f"{v['excess_fraction']:>8.1%}{v['correlation_with_temperature']:>+9.2f}"
                     f"{v['window_mean_hz']:>9.4f}{v['outside_window_mean_hz']:>9.4f}")
    v260 = validation["cuts"][str(health.ADC2_CLEAN)]
    lines += [f"  With ADC2 >= {health.ADC2_CLEAN} the ICD window and the hours outside it still differ by "
              f"{100 * (v260['window_mean_hz'] / v260['outside_window_mean_hz'] - 1):.1f} %",
              f"  ({v260['window_minus_outside_sigmas']:.1f} sigma from the hourly spread): the cleaned"
              " rate is close to constant, not",
              f"  constant. The residual excess of {100 * v260['excess_fraction']:.1f} % is the systematic"
              " the cleaned ratio carries (step 09).",
              "  Santa Tecla, same procedure (control):"]
    for cut, v in control.items():
        lines.append(f"  ADC2>={cut:<4}{v['hours']:>6}{v['mean']:>9.4f}"
                     f"{v['chi2']:>9.1f}/{v['dof']:<4}{v['p_value']:>10.2g}"
                     f"{v['excess_fraction']:>8.1%}{v['correlation_with_temperature']:>+9.2f}")
    lines += [f"  Santa Tecla minutes vetoed: {st_state.vetoed.sum()} of {len(st_state.vetoed)}.",
              "  Larger cuts drift with the gain, which falls as the case warms up: a fixed",
              "  ADC2 cut is a different energy threshold at a different temperature.",
              "", "THE 2025 ANGULAR RUNS",
              f"  {'run':>5}{'N':>6}{'ped':>6}{'range':>10}{'veto':>6}{'A1>A2':>7}"
              f"{'R[200,260)':>12}{'dt<=10ms':>9}{'expect':>8}{'triplets':>9}"
              f"{'KS p':>7}{'D(10s)':>8}{'p_D':>6}"]
    for angle, d in diagnostics_2025.items():
        lines.append(f"  {angle:>5}{d['n_events']:>6}{d['pedestal_median']:>6.0f}"
                     f"{d['pedestal_min']:>5.0f}-{d['pedestal_max']:<4.0f}"
                     f"{d['vetoed_minutes']:>6}{d['burst_events']:>7}"
                     f"{d['near_threshold_rate_hz']:>12.3f}{d['fraction_dt_below_10ms']:>9.3f}"
                     f"{d['expected_fraction_dt_below_10ms']:>8.3f}"
                     f"{d['close_triplets_below_20ms']:>9}{d['ks_p_live_waits']:>7.2f}"
                     f"{d['dispersion_10s']:>8.2f}{d['dispersion_10s_p']:>6.2f}")
    lines += ["  No 2025 run shows either signature: they were not taken in the fault state.",
              "", "FIRMWARE FINGERPRINT",
              f"  {'data set':<14}{'ADC2 min':>9}{'ADC2 max':>9}{'on 5.6+k*41 ms':>16}"]
    for name, f in fingerprint.items():
        lines.append(f"  {name:<14}{f['adc2_min']:>9}{f['adc2_max']:>9}"
                     f"{f['deadtime_on_pattern_fraction']:>16.3f}")
    lines += ["  Same trigger threshold and dead-time pattern: same firmware and settings.",
              "  Whether it is the same physical unit is not recorded."]
    if problems:
        lines += ["", "DEVIATIONS FROM THE DOCUMENTED STATE"] + [f"  {p}" for p in problems]
    text = "\n".join(lines)
    print(text)

    json_path, text_path = provenance.write_product(
        args.out, "07_detector_health",
        data={
            "integrity": {n: r.summary() for n, r in files.items()},
            "sipm": sipm,
            "thresholds": {"pedestal_adc2_max": health.PEDESTAL_ADC2_MAX,
                           "pedestal_veto": health.PEDESTAL_VETO,
                           "burst_fraction_veto": health.BURST_FRACTION_VETO,
                           "near_threshold_veto": health.NEAR_THRESHOLD_VETO,
                           "guard_minutes": health.GUARD_MINUTES,
                           "adc2_clean": health.ADC2_CLEAN},
            "states": states,
            "episodes": episode_rows,
            "episode_event_fraction": float(episode_events.mean()),
            "vetoed_minutes": int(state.vetoed.sum()),
            "minutes": int(len(state.vetoed)),
            "unvetoed_episode_minutes": unvetoed_episode_minutes,
            "coincident_flag": coincident,
            "band_names": band_names,
            "band_rates_by_pedestal": by_pedestal,
            "band_rates_santa_tecla": bands_st,
            "band_rates_2025": bands_2025,
            "validation_tacachico": validation,
            "control_santa_tecla": control,
            "santa_tecla_vetoed_minutes": int(st_state.vetoed.sum()),
            "runs_2025": diagnostics_2025,
            "firmware": fingerprint,
            "deviations": problems,
        },
        text=text,
        parameters={"data_dir": args.data_dir, "altitude_dir": args.altitude_dir},
    )
    print(f"[07] wrote {json_path}")
    print(f"[07] wrote {text_path}")

    # Figures.
    plotting.apply_style()
    figures = Path(args.out_figures)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    bx = np.logspace(np.log10(50), np.log10(4200), 90)
    by = np.logspace(np.log10(190), np.log10(4200), 90)
    bad, _, _ = np.histogram2d(tac.adc1[event_vetoed], tac.adc2[event_vetoed], bins=[bx, by])
    mesh = ax.pcolormesh(bx, by, np.where(bad.T > 0, bad.T, np.nan), norm=LogNorm(), cmap="Reds")
    good, _, _ = np.histogram2d(st.adc1, st.adc2, bins=[bx, by])
    ax.contour(np.sqrt(bx[1:] * bx[:-1]), np.sqrt(by[1:] * by[:-1]), good.T,
               levels=[3, 30, 300], colors="k", linewidths=0.8)
    a1 = np.concatenate([r.adc1 for r in runs.values()])
    a2 = np.concatenate([r.adc2 for r in runs.values()])
    ax.scatter(a1, a2, s=2, color=plotting.COLOUR_ALTERNATE, alpha=0.5,
               label="2025 angular runs (all events)")
    ax.plot([50, 4200], [50, 4200], "--", color=plotting.COLOUR_DATA, lw=0.8,
            label="ADC2 = ADC1")
    ax.plot([], [], color="k", lw=0.8, label="Santa Tecla 2026 (contours)")
    fig.colorbar(mesh, ax=ax, label="Tacachico events in vetoed minutes")
    ax.set(xscale="log", yscale="log", xlabel="ADC1 (low gain)", ylabel="ADC2 (high gain)",
           title="Pulse plane: fault (red), clean 2026 (black), 2025 (green)")
    ax.legend(loc="lower right", fontsize=8)
    plotting.save(fig, figures, "07_adc_plane")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    near = 2
    above = [2 + j for j, (lo, _) in enumerate(health.BANDS) if lo >= 400]
    ax.scatter(tac_bins[:, 0], tac_bins[:, near], s=6, color=plotting.COLOUR_MODEL,
               label="Tacachico, ADC2 in [200, 260)")
    ax.scatter(tac_bins[:, 0], tac_bins[:, above].sum(axis=1), s=6, color=plotting.COLOUR_DATA,
               label="Tacachico, ADC2 >= 400")
    ax.scatter(st_bins[:, 0], st_bins[:, near], s=6, color=plotting.COLOUR_MODEL, alpha=0.3,
               marker="s", label="Santa Tecla, ADC2 in [200, 260)")
    ax.scatter(st_bins[:, 0], st_bins[:, above].sum(axis=1), s=6, color=plotting.COLOUR_BAND,
               marker="s", label="Santa Tecla, ADC2 >= 400")
    for angle, d in diagnostics_2025.items():
        ax.scatter(d["pedestal_median"], d["near_threshold_rate_hz"], marker="*", s=70,
                   color=plotting.COLOUR_ALTERNATE, zorder=5)
    ax.scatter([], [], marker="*", s=70, color=plotting.COLOUR_ALTERNATE,
               label="2025 runs, ADC2 in [200, 260)")
    ax.axvline(health.PEDESTAL_VETO, color="k", ls=":", lw=0.8)
    ax.set(yscale="log", xlabel="pedestal of ADC1 (median, ADC2 < 1500)",
           ylabel="live rate (1/s), 10 min bins",
           title="The fault only adds events just above the trigger threshold")
    ax.legend(fontsize=7)
    plotting.save(fig, figures, "07_band_rates")

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.errorbar(raw["hour"] + 0.5, raw["rate"], raw["error"], fmt="o", ms=3,
                color=plotting.COLOUR_MODEL, label="raw: every trigger, every minute")
    h = hourly[health.ADC2_CLEAN]
    ax.errorbar(h["hour"] + 0.5, h["rate"], h["error"], fmt="o", ms=3,
                color=plotting.COLOUR_DATA, label=f"state veto and ADC2 >= {health.ADC2_CLEAN}")
    ax.axvspan(WINDOW_S[0] / 3600, WINDOW_S[1] / 3600, color=plotting.COLOUR_BAND, alpha=0.15,
               label="window used at the ICD (hours 26 to 40)")
    ax.set(yscale="log", xlabel="hour of acquisition 1", ylabel="live rate (1/s)",
           title="Tacachico: after cleaning, within about 1 % of constant")
    ax.legend(fontsize=8)
    plotting.save(fig, figures, "07_tacachico_cleaning")

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.6), sharex=True)
    hours = state.minute / 60.0
    axes[0].semilogy(hours, state.n_events, lw=0.5, color=plotting.COLOUR_DATA)
    axes[0].set_ylabel("events / min")
    axes[1].plot(hours, state.pedestal, lw=0.5, color=plotting.COLOUR_DATA)
    axes[1].axhline(health.PEDESTAL_VETO, color="k", ls=":", lw=0.8)
    axes[1].set(ylabel="ADC1 pedestal", ylim=(60, 260))
    minute_temp = np.bincount(np.floor(tac.time_s / 60).astype(int) - state.minute[0],
                              weights=tac.temperature_c, minlength=len(state.minute))
    with np.errstate(invalid="ignore"):
        axes[2].plot(hours, minute_temp / state.n_events, lw=0.8, color=plotting.COLOUR_DATA)
    axes[2].set_ylabel("case T (C)")
    minute_coinc = np.bincount(np.floor(tac.time_s / 60).astype(int) - state.minute[0],
                               weights=tac.coincident.astype(float), minlength=len(state.minute))
    with np.errstate(invalid="ignore"):
        axes[3].plot(hours, minute_coinc / state.n_events, lw=0.8, color=plotting.COLOUR_DATA)
    axes[3].set(ylabel="Coincident flag", xlabel="hour of acquisition 1")
    for a in axes:
        a.fill_between(hours, 0, 1, where=state.vetoed, transform=a.get_xaxis_transform(),
                       color=plotting.COLOUR_MODEL, alpha=0.2, lw=0)
    axes[0].set_title("Tacachico acquisition 1 (red: vetoed minutes)")
    plotting.save(fig, figures, "07_state_timeline")

    if problems:
        print("[07] the data deviate from their documented state", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
