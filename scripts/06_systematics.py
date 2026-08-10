#!/usr/bin/env python3
"""
06_systematics.py

Systematic uncertainties and reassessment of the significance of the result.

This is the step that settles the conclusion of the analysis. 05_fits performs
the inference under the assumption that counting statistics is the only source
of uncertainty, and under that assumption the modulated component comes out
significant. This script shows that the assumption is false for this data set,
and by how wide a margin.

Sources evaluated:
  1. RUN-TO-RUN REPEATABILITY. The 360 degree file is a second measurement of
     the same angle as the 0 degree one. The discrepancy between them measures
     directly how much the apparatus moves between runs for reasons that are
     not statistical. It is the dominant systematic and the only one this data
     set allows to be measured rather than assumed.
  2. LIVE TIME. Its effect on the absolute rate is separated from its effect on
     the ratios between angles, which is second order because the dead fraction
     is nearly identical across the eight runs.
  3. TIME WINDOW. How much the ratios move when only the first half of each run
     is analysed. Reported as the EXCESS over the Poisson expectation, since
     half a run holds half the events and fluctuates accordingly.
  4. REFERENCE ANGLE. How much the result depends on having normalised to 0
     degrees rather than another angle.
  5. BIN SIZE. In the previous analysis this moved the published rate by 0.4 %;
     here it is verified to enter no physical number at all.

Every candidate source is compared against its own Poisson expectation before
entering the table. Without that comparison a purely statistical fluctuation
enters the systematic budget dressed as a systematic, which for the time window
would have contributed a spurious 5.45 %.

With the dominant systematic measured, the model comparison is redone and the
upper bound on the modulated fraction is recomputed.

Exact outputs:
  - results/tables/06_systematics_vN.json and .txt
  - results/figures/06_systematics_vN.pdf and .png

Exit code: 0 if the assessment completes; 1 if the repeatability control is
missing, without which the dominant systematic is not measurable.

Typical use, from the repository root with .venv active:
  python scripts/06_systematics.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_muon import io, models, plotting, provenance, rates  # noqa: E402
from lib_muon.plotting import plt  # noqa: E402

REFERENCE_ANGLE = 0
CONTROL_ANGLE = 360


def weighted_fit(observed, sigmas, shape):
    """Weighted least squares of R = S*shape + B. Returns (S, B, chi2)."""
    matrix = np.vstack([shape, np.ones_like(shape)]).T
    weights = np.diag(1.0 / sigmas**2)
    coefficients = np.linalg.solve(matrix.T @ weights @ matrix,
                                   matrix.T @ weights @ observed)
    residual = observed - matrix @ coefficients
    return coefficients[0], coefficients[1], float(np.sum((residual / sigmas) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser(description="Systematics and reassessment.")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--fits", default="results/fits")
    ap.add_argument("--out-figures", default="results/figures")
    args = ap.parse_args()

    plotting.apply_style()
    product = provenance.read_product(args.tables, "02_rates")
    rate_rows = {int(a): r for a, r in product["rates"].items()}

    if CONTROL_ANGLE not in rate_rows:
        print(f"[06] ERROR: the {CONTROL_ANGLE} degree control run is missing; "
              "the dominant systematic is not measurable", file=sys.stderr)
        return 1

    lines = []
    sources = []

    # --- 1. Run-to-run repeatability ---------------------------------------
    r_ref = rate_rows[REFERENCE_ANGLE]["live_rate_hz"]
    e_ref = rate_rows[REFERENCE_ANGLE]["live_rate_error_hz"]
    r_ctl = rate_rows[CONTROL_ANGLE]["live_rate_hz"]
    e_ctl = rate_rows[CONTROL_ANGLE]["live_rate_error_hz"]

    ratio = r_ctl / r_ref
    ratio_error = np.sqrt((e_ctl / r_ref) ** 2 + (r_ctl * e_ref / r_ref**2) ** 2)
    discrepancy = abs(1.0 - ratio)
    discrepancy_sigmas = discrepancy / ratio_error

    normalisation = provenance.read_product(args.tables, "04_normalization")
    row_45 = next(r for r in normalisation["rows"] if r["angle_deg"] == 45)
    drop_0_45 = 1.0 - row_45["normalised"]

    lines += [
        "=" * 78,
        "1. RUN-TO-RUN REPEATABILITY (the dominant systematic)",
        "=" * 78,
        f"The {CONTROL_ANGLE} degree file and the {REFERENCE_ANGLE} degree file "
        "measure the SAME angle.",
        f"  R({REFERENCE_ANGLE:>3} deg) = {r_ref:.4f} +- {e_ref:.4f} 1/s",
        f"  R({CONTROL_ANGLE:>3} deg) = {r_ctl:.4f} +- {e_ctl:.4f} 1/s",
        f"  ratio = {ratio:.4f} +- {ratio_error:.4f}, against the expected 1",
        f"  discrepancy = {100 * discrepancy:.1f} % "
        f"({discrepancy_sigmas:.2f} sigma)",
        "",
        "Honest reading: the discrepancy is not established as an effect "
        f"({discrepancy_sigmas:.2f} sigma is not",
        "enough to assert that the two runs really differ). But as the SCALE of "
        "the run-to-run",
        "systematic it is the best information available, and the only one this "
        "data set allows",
        "to be measured rather than assumed.",
        "",
        "And that scale is large next to what the analysis is trying to "
        "interpret:",
        f"  total drop from 0 to 45 degrees:            {100 * drop_0_45:.1f} %",
        f"  discrepancy between two measurements of 0:  {100 * discrepancy:.1f} %",
        "",
        "The systematic between two runs at the SAME angle is of the same size "
        "as the entire",
        "drop from 0 to 45 degrees. Put plainly: it eats the whole signal the "
        "fit is trying to",
        "interpret. That single sentence is the result of the analysis.",
    ]
    sources.append({
        "source": "run-to-run repeatability",
        "relative_effect": float(discrepancy),
        "measured_or_estimated": "measured",
        "detail": f"{CONTROL_ANGLE} degree control against {REFERENCE_ANGLE} "
                  f"degrees, {discrepancy_sigmas:.2f} sigma",
    })

    # --- 2. Live time --------------------------------------------------------
    runs = io.load_all(args.data_dir)
    computed = {a: rates.compute_rate(r) for a, r in runs.items()}
    fractions = np.array([r.dead_fraction for r in computed.values()])
    absolute_effect = float(fractions.mean() / (1 - fractions.mean()))

    wall_norm = {a: r.wall_rate / computed[REFERENCE_ANGLE].wall_rate
                 for a, r in computed.items()}
    live_norm = {a: r.live_rate / computed[REFERENCE_ANGLE].live_rate
                 for a, r in computed.items()}
    ratio_effect = max(abs(live_norm[a] - wall_norm[a]) / wall_norm[a]
                       for a in live_norm)

    lines += [
        "",
        "=" * 78,
        "2. LIVE TIME",
        "=" * 78,
        f"Dead fraction between {100 * fractions.min():.2f} % and "
        f"{100 * fractions.max():.2f} %, spread {100 * fractions.std():.3f} points.",
        f"  effect on the ABSOLUTE rate:  {100 * absolute_effect:.2f} %",
        f"  effect on the RATIOS:         {100 * ratio_effect:.2f} %",
        "",
        "Correcting for live time is mandatory and the previous analysis did not",
        "do it, but it is worth not overstating what it fixes: since the dead",
        "fraction is nearly identical across the eight runs, it cancels in the",
        "ratios and leaves the angular shape essentially where it was. It matters",
        "for the absolute rate, not for the angular dependence.",
    ]
    sources.append({
        "source": "live time (on ratios)",
        "relative_effect": float(ratio_effect),
        "measured_or_estimated": "measured",
        "detail": f"effect on the absolute rate: {100 * absolute_effect:.2f} %",
    })

    # --- 3. Time window ------------------------------------------------------
    observed_shifts = []
    expected_shifts = []
    for angle, run in runs.items():
        times = run.elapsed_s[run.coincident == 1]
        half = times[-1] / 2.0
        first = times[times <= half]
        if len(first) < 2:
            continue
        half_rate = (len(first) - 1) / (first[-1] - first[0])
        observed_shifts.append(
            abs(half_rate - computed[angle].wall_rate) / computed[angle].wall_rate
        )
        # Half against whole share the events of the first half, so the variance
        # of the relative difference is 1/N_1 - 1/N.
        n_half, n_total = len(first) - 1, len(times) - 1
        expected_shifts.append(np.sqrt(max(1 / n_half - 1 / n_total, 0.0)))

    observed_shifts = np.array(observed_shifts)
    expected_shifts = np.array(expected_shifts)
    window_observed = float(np.max(observed_shifts))
    window_expected = float(np.mean(expected_shifts))
    window_excess = float(np.sqrt(max(
        np.mean(observed_shifts**2) - np.mean(expected_shifts**2), 0.0
    )))

    lines += [
        "",
        "=" * 78,
        "3. TIME WINDOW",
        "=" * 78,
        "Comparing the first half of each run against the whole run:",
        f"  largest observed shift:        {100 * window_observed:.2f} %",
        f"  median observed:               {100 * np.median(observed_shifts):.2f} %",
        f"  expected from statistics:      {100 * window_expected:.2f} %",
        f"  excess over statistics:        {100 * window_excess:.2f} %",
        "",
        "THIS ROW IS NOT A SYSTEMATIC and must not be read as one. Half a run",
        "holds half the events, so its rate fluctuates by whatever Poisson",
        "dictates; the observed number is the size of that fluctuation. What",
        "enters the systematic budget is the EXCESS over it, which here is",
        f"{100 * window_excess:.2f} %. This matches 03_timeseries, where the bin",
        "spread within each run is consistent with Poisson: the instability is",
        "BETWEEN runs, not within them.",
    ]
    sources.append({
        "source": "time window (excess over statistics)",
        "relative_effect": window_excess,
        "measured_or_estimated": "measured",
        "detail": f"observed {100 * window_observed:.2f} %, expected from "
                  f"statistics {100 * window_expected:.2f} %; only the excess "
                  f"counts as a systematic",
    })

    # --- 4. Reference angle --------------------------------------------------
    physical_angles = sorted(a for a in rate_rows if a != CONTROL_ANGLE)
    reference_variation = []
    baseline = {a: rate_rows[a]["live_rate_hz"] / r_ref for a in physical_angles}
    for alternative in physical_angles:
        r_alt = rate_rows[alternative]["live_rate_hz"]
        for a in physical_angles:
            own = rate_rows[a]["live_rate_hz"] / r_alt
            rescaled = own * (r_alt / r_ref)
            reference_variation.append(abs(rescaled - baseline[a]))
    reference_effect = float(np.max(reference_variation))

    lines += [
        "",
        "=" * 78,
        "4. REFERENCE ANGLE",
        "=" * 78,
        "Normalising to a different angle rescales every ratio by a common",
        "factor and does not change the SHAPE of the profile: the residual",
        f"variation is {reference_effect:.2e}, i.e. zero up to rounding.",
        "What does change is the uncertainty: the reference angle injects its own",
        f"error into all seven ratios ({100 * e_ref / r_ref:.1f} % here), which is",
        "why the inference in 05_fits works on raw counts rather than ratios,",
        "where that common error would be counted seven times as if independent.",
    ]
    sources.append({
        "source": "reference angle",
        "relative_effect": reference_effect,
        "measured_or_estimated": "measured",
        "detail": "rescaling only; the reference error is handled in 05_fits",
    })

    # --- 5. Bin size ---------------------------------------------------------
    lines += [
        "",
        "=" * 78,
        "5. BIN SIZE",
        "=" * 78,
        "In the previous analysis each run's duration was truncated to a whole",
        "multiple of --bin-seconds, so a plotting parameter moved the published",
        "rate: 1.8118 1/s with 1 s bins against 1.8194 1/s with 60 s bins, and",
        "the ratio I(90)/I(0) from 0.8079 to 0.8148.",
        "In this pipeline binning enters only the 03_timeseries figures. Rates",
        "are computed over the full [t1, tN] window and compute_rate takes no bin",
        "parameter at all, so the effect is exactly zero by construction, not",
        "merely small.",
    ]
    sources.append({
        "source": "bin size",
        "relative_effect": 0.0,
        "measured_or_estimated": "eliminated by construction",
        "detail": "previously 0.4 % on the rate; binning no longer enters the physics",
    })

    # --- 6. Reassessment -----------------------------------------------------
    angles = np.array(physical_angles, dtype=float)
    counts = np.array([rate_rows[int(a)]["n_intervals"] for a in angles], float)
    times = np.array([rate_rows[int(a)]["live_time_s"] for a in angles])
    observed_rate = counts / times
    statistical_sigma = observed_rate / np.sqrt(counts)
    shape2 = models.cosn_shape(models.clamped_cosine(angles), 2.0)

    f_sys = float(discrepancy)

    def delta_chi2(f):
        sigma = np.sqrt(statistical_sigma**2 + (f * observed_rate) ** 2)
        weights = 1.0 / sigma**2
        mean = np.sum(weights * observed_rate) / np.sum(weights)
        chi2_constant = float(np.sum(((observed_rate - mean) / sigma) ** 2))
        _, _, chi2_modulated = weighted_fit(observed_rate, sigma, shape2)
        return chi2_constant - chi2_modulated

    scan = []
    for f in (0.0, 0.02, 0.03, 0.05, f_sys, 0.10):
        sigma = np.sqrt(statistical_sigma**2 + (f * observed_rate) ** 2)
        weights = 1.0 / sigma**2
        mean = float(np.sum(weights * observed_rate) / np.sum(weights))
        chi2_constant = float(np.sum(((observed_rate - mean) / sigma) ** 2))
        s_fitted, _, chi2_modulated = weighted_fit(observed_rate, sigma, shape2)
        d = chi2_constant - chi2_modulated
        scan.append({
            "f_sys": float(f),
            "chi2_constant": chi2_constant,
            "p_constant": float(stats.chi2.sf(chi2_constant, len(angles) - 1)),
            "S_hz": float(s_fitted),
            "delta_chi2": float(d),
            "p_modulation": float(stats.chi2.sf(d, 1)),
        })

    f_critical = float(optimize.brentq(
        lambda f: delta_chi2(f) - stats.chi2.isf(0.05, 1), 1e-6, 0.5
    ))
    f_compatible = models.required_systematic(observed_rate, statistical_sigma, 0.05)

    lines += [
        "",
        "=" * 78,
        "6. REASSESSMENT WITH THE MEASURED SYSTEMATIC",
        "=" * 78,
        "05_fits finds the modulated component improving the fit at p = 0.008,",
        "but that assumes counting statistics is the only uncertainty. Adding the",
        "run-to-run systematic in quadrature:",
        "",
        f"{'f_sys':>8} {'chi2_const':>11} {'p_const':>9} {'S [1/s]':>9} "
        f"{'Dchi2':>8} {'p_mod':>8}  verdict",
        "-" * 76,
    ]
    for row in scan:
        mark = "  <- measured by the control" if abs(row["f_sys"] - f_sys) < 1e-12 else ""
        lines.append(
            f"{100 * row['f_sys']:>7.2f}% {row['chi2_constant']:>11.2f} "
            f"{row['p_constant']:>9.4f} {row['S_hz']:>9.4f} "
            f"{row['delta_chi2']:>8.2f} {row['p_modulation']:>8.4f}  "
            f"{'significant' if row['p_modulation'] < 0.05 else 'NOT significant'}"
            f"{mark}"
        )

    lines += [
        "-" * 76,
        "",
        f"The modulation stops being significant beyond a systematic of "
        f"{100 * f_critical:.2f} %.",
        f"The repeatability control measures {100 * f_sys:.1f} %, nearly twice that.",
        f"The constant-rate hypothesis becomes consistent at "
        f"{100 * f_compatible:.2f} %.",
        "",
        "CONCLUSION OF THE ANALYSIS:",
        "The preference for a modulated component that appears under pure",
        "counting statistics does NOT survive the run-to-run systematic that the",
        "data set itself measures. It cannot be claimed that this setup detected",
        "the angular dependence of the muon flux. What can be stated is an upper",
        "bound on how much modulation would be consistent with what was observed.",
    ]

    # --- 7. Upper bound with the systematic folded in ------------------------
    total_sigma = np.sqrt(statistical_sigma**2 + (f_sys * observed_rate) ** 2)
    s_best, _, chi2_min = weighted_fit(observed_rate, total_sigma, shape2)

    def chi2_of_s(s):
        return float(optimize.minimize_scalar(
            lambda b: float(np.sum(((observed_rate - (s * shape2 + b)) / total_sigma) ** 2)),
            bounds=(0.0, 5.0), method="bounded",
        ).fun)

    target = chi2_min + stats.chi2.isf(0.10, 1)  # one-sided 95 %
    s_bound = float(optimize.brentq(lambda s: chi2_of_s(s) - target,
                                    max(s_best, 0.0), 5.0))
    b_at_bound = float(optimize.minimize_scalar(
        lambda b: float(np.sum(((observed_rate - (s_bound * shape2 + b)) / total_sigma) ** 2)),
        bounds=(0.0, 5.0), method="bounded",
    ).x)
    bound_fraction = s_bound / (s_bound + b_at_bound)

    fits_product = provenance.read_product(args.fits, "05_fits")
    statistical_bound = fits_product["modulation_n2"]["modulated_fraction_bound_95"]

    lines += [
        "",
        "=" * 78,
        "7. UPPER BOUND ON THE MODULATED FRACTION",
        "=" * 78,
        f"  statistics only (05_fits):          < {100 * statistical_bound:.1f} %",
        f"  with the {100 * f_sys:.1f} % systematic:        "
        f"< {100 * bound_fraction:.1f} %",
        "",
        "The second is the one published: it includes the only systematic this",
        "data set allows to be measured. It is the quantitative result of the",
        "analysis.",
    ]

    text = "\n".join(lines)
    print(text)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4), layout="constrained")

    grid = np.linspace(0.0, 0.12, 121)
    p_values = [stats.chi2.sf(delta_chi2(f), 1) for f in grid]
    ax.plot(100 * grid, p_values, color=plotting.COLOUR_DATA, lw=2)
    ax.axhline(0.05, color=plotting.COLOUR_MODEL, ls="--", lw=1.4, label="p = 0.05")
    ax.axvline(100 * f_sys, color=plotting.COLOUR_ALTERNATE, lw=1.8,
               label=f"measured by the control: {100 * f_sys:.1f} %")
    ax.axvspan(100 * f_critical, 12, color=plotting.COLOUR_ALTERNATE, alpha=0.12)
    ax.set_xlabel("run-to-run systematic [%]")
    ax.set_ylabel("p-value of the modulated component")
    ax.set_title("The signal does not survive the measured systematic")
    ax.set_xlim(0, 12)
    ax.legend(fontsize=8)

    labels = [s["source"] for s in sources]
    values = [100 * s["relative_effect"] for s in sources]
    positions = np.arange(len(labels))
    colours = [plotting.COLOUR_MODEL if v == max(values) else plotting.COLOUR_DATA
               for v in values]
    ax2.barh(positions, values, color=colours, alpha=0.85)
    ax2.set_yticks(positions)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("relative effect [%]")
    ax2.set_title("Systematics budget")
    for i, v in enumerate(values):
        ax2.text(v + 0.1, i, f"{v:.2f} %", va="center", fontsize=8)
    ax2.set_xlim(0, max(values) * 1.35)

    paths = plotting.save(fig, args.out_figures, "06_systematics")
    print(f"[06] figure -> {paths[0].name}")

    json_path, _ = provenance.write_product(
        args.tables, "06_systematics",
        data={
            "sources": sources,
            "dominant_systematic": f_sys,
            "critical_systematic_for_significance": f_critical,
            "systematic_for_constant_compatibility": f_compatible,
            "scan": scan,
            "modulated_fraction_bound_statistics_only": statistical_bound,
            "modulated_fraction_bound_with_systematic": float(bound_fraction),
            "S_bound_hz": s_bound,
        },
        text=text,
        parameters={"control_angle": CONTROL_ANGLE,
                    "reference_angle": REFERENCE_ANGLE},
    )
    print(f"[06] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
