#!/usr/bin/env python3
"""
08_angular_reading.py

A second reading of the 2025 angular runs, with what the 2026 campaign taught.

The first reading (steps 00 to 06) compared the profile with cos^2(theta) and
found it flat: I(90)/I(0) = 0.806 +- 0.049 against the 0 of cos^2. Step 07
showed the runs were not taken in the fault state. This step asks what a
single detector should have seen, and finds the flat profile is what one gets.

Steps:
  1. The record holds singles, not coincidences. Two 5 x 5 cm slabs stacked
     in coincidence see at most 0.32 muons/s (Axani et al., 2025); the runs
     record about 2 events/s with Coincident = 1 on all of them. The 2026
     detector, of the same firmware, raised the same flag for three hours
     without a partner (according to the group that took the data).
     Whatever the flag means, these are the triggers of one detector.
  2. The reference for a single slab (methodology 10): g(alpha), the rate of
     a tilted box relative to the horizontal one under a cos^2 flux. For the
     assumed 5 x 5 x 1 cm slab, g(90) = 2/3 even with a pure muon sample.
  3. Two components: a fraction f of the triggers at 0 degrees are muons and
     follow g; the rest does not depend on orientation. Expected f from the
     known flux, collected over the effective area of the same box (its
     edges add a fifth to the 25 cm2 face at 0 degrees), and the I(90)/I(0)
     that implies.
  4. The profile again above ADC2 cuts 0, 260, 1000, 2500 and 4095
     (saturation). g counts every crossing, which holds when nearly every
     muon passes the cut: f is fitted only at cuts 0 and 260. Above larger
     cuts the efficiency depends on the path length, which grows with the
     tilt for near-vertical muons (1 cm / cos), so the profiles are shown
     and not fitted (methodology 10.5).

Exact outputs:
  - results/tables/08_angular_reading_vN.json and .txt
  - results/figures/08_angular_reading_vN.pdf/png

Exit code: 0 if the cut-0 profile reproduces step 02 exactly; 1 otherwise.

Typical use, from the repository root with .venv active, after step 06:
  python scripts/08_angular_reading.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_muon import altitude as alt, geometry, io, plotting, provenance, rates  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

PHYSICAL = [0, 5, 15, 25, 35, 45, 90]
REFERENCE_ANGLE = 0
CONTROL_ANGLE = 360
CUTS = (0, 260, 1000, 2500, 4095)
FIT_CUTS = (0, 260)               # cuts that keep nearly every muon crossing
THICKNESSES_CM = (0.5, 1.0, 2.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Second reading of the angular runs.")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--out", default=None, help="defaults to --tables")
    ap.add_argument("--out-figures", default="results/figures")
    args = ap.parse_args()
    out = args.out or args.tables

    runs = io.load_all(args.data_dir)
    published = provenance.read_product(args.tables, "02_rates")["rates"]
    repeatability = float(provenance.read_product(args.tables, "06_systematics")
                          ["dominant_systematic"])

    # 1. Singles, not coincidences.
    live = {a: rates.compute_rate(run) for a, run in runs.items()}
    r0 = live[REFERENCE_ANGLE].live_rate
    coincidence_max = alt.FLUX_AXANI * alt.AREA_CM2 / 60.0
    singles = {
        "rate_at_0deg_hz": r0,
        "max_true_coincidence_rate_hz": coincidence_max,
        "max_true_coincidence_fraction": coincidence_max / r0,
        "coincident_flag_fraction": float(np.mean(np.concatenate(
            [run.coincident for run in runs.values()]))),
    }

    # 2. Single-slab reference.
    g = {f"{c}": geometry.slab_response(PHYSICAL, (5.0, 5.0, c)).tolist()
         for c in THICKNESSES_CM}
    g["thin"] = geometry.slab_response(PHYSICAL, (5.0, 5.0, 1e-9)).tolist()
    g_ref = np.array(g["1.0"])

    # 3. Expected muon fraction at 0 degrees and what it implies.
    # The flux references are per cm2 of a horizontal plate; the box of the
    # reference curve collects more at 0 degrees, through its edges.
    edge_factor = (geometry.slab_rate(0.0, geometry.DEFAULT_DIMS_CM)
                   / geometry.slab_rate(0.0, (5.0, 5.0, 1e-9)))
    f_expected = [flux * alt.AREA_CM2 * edge_factor / 60.0 / r0
                  for flux in (alt.FLUX_AXANI, alt.FLUX_PDG)]
    i90_expected = [float(geometry.two_component(g_ref[-1], f)) for f in f_expected]

    # 4. Profiles above each cut.
    profiles = {}
    mismatch = []
    for cut in CUTS:
        rows = {}
        for angle, run in sorted(runs.items()):
            n = int((run.adc2[1:] >= cut).sum())
            t_live = live[angle].live_time_s
            rows[angle] = (n / t_live, np.sqrt(n) / t_live, n)
            if cut == 0 and abs(rows[angle][0] - published[str(angle)]["live_rate_hz"]) > 1e-12:
                mismatch.append(angle)
        rr = np.array([rows[a][0] for a in PHYSICAL])
        ee = np.array([rows[a][1] for a in PHYSICAL])
        ref, ref_e = rows[REFERENCE_ANGLE][0], rows[REFERENCE_ANGLE][1]
        relative = {str(a): [rows[a][0] / ref,
                             rows[a][0] / ref * np.hypot(rows[a][1] / rows[a][0], ref_e / ref)]
                    for a in rows}
        w = 1.0 / ee**2
        mean = np.sum(w * rr) / w.sum()
        chi2_const = float(np.sum(w * (rr - mean) ** 2))
        if cut in FIT_CUTS:
            fit = geometry.fit_muon_fraction(rr, ee, g_ref)
            fit["p_value"] = float(stats.chi2.sf(fit["chi2"], fit["dof"]))
        else:
            fit = None
        profiles[str(cut)] = {
            "counts": {str(a): rows[a][2] for a in rows},
            "rate_hz": {str(a): rows[a][0] for a in rows},
            "rate_error_hz": {str(a): rows[a][1] for a in rows},
            "relative": relative,
            "chi2_constant": chi2_const,
            "p_constant": float(stats.chi2.sf(chi2_const, len(PHYSICAL) - 1)),
            "fit": fit,
        }
    base = profiles["0"]["relative"]
    i90, i90_e = base["90"]
    combined = np.hypot(i90_e, repeatability * i90)
    comparison = {
        "measured_i90_over_i0": i90, "statistical_error": i90_e,
        "repeatability_systematic": repeatability, "combined_error": float(combined),
        "expected_range": i90_expected,
        "sigmas_from_nearest_expected_statistical": float((min(i90_expected) - i90) / i90_e),
        "sigmas_from_nearest_expected_combined": float((min(i90_expected) - i90) / combined),
    }

    lines = [
        "1. THE RECORD HOLDS SINGLES",
        f"  Rate at 0 degrees: {r0:.3f} /s. Two 5 x 5 cm slabs stacked in coincidence see at",
        f"  most {coincidence_max:.2f} muons/s, {100 * coincidence_max / r0:.0f} % of that. Yet "
        f"{100 * singles['coincident_flag_fraction']:.0f} % of events carry Coincident = 1.",
        "  The 2026 detector of the same firmware raised the same flag for three hours",
        "  without a partner (group's report; step 07). Whatever the flag means, these are",
        "  the triggers of one detector, and cos^2 is the wrong reference for them.",
        "",
        "2. WHAT ONE SLAB SEES UNDER A cos^2 FLUX, RELATIVE TO HORIZONTAL",
        f"  {'angle':>6}" + "".join(f"{a:>8}" for a in PHYSICAL),
    ]
    for name, values in g.items():
        label = "thin" if name == "thin" else f"{name} cm"
        lines.append(f"  {label:>6}" + "".join(f"{v:>8.3f}" for v in values))
    lines += [
        "  The 5 x 5 x 1 cm slab is assumed (dataset.md 4.1). At 90 degrees its edges",
        "  still face the sky: even pure muons fall only to 2/3, never to 0.",
        "",
        "3. TWO COMPONENTS: f g(alpha) + 1 - f",
        f"  The box collects {edge_factor:.2f} times the muons of its 25 cm2 face at 0 degrees.",
        f"  Known flux: {100 * f_expected[0]:.0f} to {100 * f_expected[1]:.0f} % of the "
        f"triggers at 0 degrees are muons, so I(90)/I(0) should be {i90_expected[1]:.3f} to "
        f"{i90_expected[0]:.3f}.",
        f"  Measured {i90:.3f} +- {i90_e:.3f}: {comparison['sigmas_from_nearest_expected_statistical']:.1f} "
        f"sigma below with the statistical error alone, "
        f"{comparison['sigmas_from_nearest_expected_combined']:.1f} sigma with the "
        f"{100 * repeatability:.2f} % run-to-run systematic of step 06.",
        "",
        "4. THE PROFILE ABOVE EACH ADC2 CUT (relative to 0 degrees)",
        f"  {'cut':>6}{'N(0)':>6}" + "".join(f"{a:>8}" for a in PHYSICAL)
        + f"{'360':>8}{'chi2 const':>12}{'f fitted':>20}{'chi2 fit':>12}",
    ]
    for cut, prof in profiles.items():
        rel = prof["relative"]
        fit = prof["fit"]
        fitted = (f"   {fit['muon_fraction']:.2f} [{fit['interval'][0]:.2f}, "
                  f"{fit['interval'][1]:.2f}]{fit['chi2']:>8.1f}/{fit['dof']}"
                  if fit else "   not fitted (path length)")
        lines.append(
            f"  {cut:>6}{prof['counts']['0']:>6}"
            + "".join(f"{rel[str(a)][0]:>8.3f}" for a in PHYSICAL)
            + f"{rel['360'][0]:>8.3f}{prof['chi2_constant']:>8.1f}/{len(PHYSICAL) - 1}"
            + fitted
        )
    lines += [
        "  At cuts 0 and 260 the fitted f overlaps the range the known flux gives. Above",
        "  larger cuts the profile is not a muon fraction: a tilted slab gives near-vertical",
        "  muons a longer path, so more of them pass a high cut, and the saturated pulses",
        "  rise above 1 at intermediate angles. The dips at 5 and 15 degrees fit no model;",
        "  they are of the size of the run-to-run spread the 360 degree control measures.",
        "",
        "READING. The 2025 runs were taken with a healthy detector that recorded singles.",
        "Their flat profile is what one slab with a muon fraction of a few tens of percent",
        "should show. Measuring cos^2 needs a working coincidence, which this record lacks.",
    ]
    if mismatch:
        lines += ["", f"MISMATCH with step 02 at angles {mismatch}"]
    text = "\n".join(lines)
    print(text)

    json_path, text_path = provenance.write_product(
        out, "08_angular_reading",
        data={"singles": singles, "slab_response": {"angles_deg": PHYSICAL, **g},
              "edge_factor_at_0deg": edge_factor,
              "expected_muon_fraction": f_expected, "comparison": comparison,
              "profiles": profiles, "mismatch_with_step_02": mismatch},
        text=text,
        parameters={"data_dir": args.data_dir, "tables": args.tables,
                    "assumed_dims_cm": list(geometry.DEFAULT_DIMS_CM)},
    )
    print(f"[08] wrote {json_path}")
    print(f"[08] wrote {text_path}")

    plotting.apply_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    x = np.linspace(0, 90, 91)
    gx = geometry.slab_response(x, n_theta=100)    # a drawing grid; the fits use n_theta=400
    styles = [("0", "all triggers (as published)", plotting.COLOUR_DATA, "o"),
              ("260", "ADC2 >= 260", plotting.COLOUR_BAND, "s"),
              ("2500", "ADC2 >= 2500", plotting.COLOUR_MODEL, "D")]
    for i, (cut, label, colour, marker) in enumerate(styles):
        rel = profiles[cut]["relative"]
        y = np.array([rel[str(a)][0] for a in PHYSICAL])
        e = np.array([rel[str(a)][1] for a in PHYSICAL])
        ax.errorbar(np.array(PHYSICAL) + (i - 1) * 1.0, y, e, fmt=marker, ms=4, capsize=2,
                    color=colour, label=label)
        ax.errorbar([97 + i], [rel["360"][0]], [rel["360"][1]], fmt=marker, ms=4, mfc="none",
                    color=colour)
    for f, ls in ((f_expected[1], "-"), (1.0, "--")):
        ax.plot(x, geometry.two_component(gx, f), ls, color=plotting.COLOUR_ALTERNATE, lw=1,
                label=f"one 5x5x1 cm slab, {100 * f:.0f} % muons")
    ax.plot(x, np.cos(np.deg2rad(x)) ** 2, ":", color="k", lw=1,
            label="cos^2 (two detectors in coincidence)")
    ax.text(98, 0.62, "360\n(control)", fontsize=7, ha="center")
    ax.set(xlabel="nominal angle (degrees; convention not recorded)",
           ylabel="rate / rate at 0 degrees", ylim=(0, 1.2),
           title="2025 runs: the profile of one slab, not of a telescope")
    ax.legend(fontsize=7, loc="lower left")
    plotting.save(fig, args.out_figures, "08_angular_reading")

    return 1 if mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
