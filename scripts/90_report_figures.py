#!/usr/bin/env python3
"""
90_report_figures.py

Regenerates the figures embedded in the README, from the products already
written by steps 02 to 06. It runs no fit of its own: every number it draws is
read from results/, so a figure in the README cannot disagree with the table it
came from.

The numbering follows the convention of the pipeline: 90 and above are
presentation steps, run after the analysis and never before it.

Steps:
  1. Read results/tables/02_rates_vN.json, 04_normalization_vN.json,
     06_systematics_vN.json and results/fits/05_fits_vN.json.
  2. Summary figure with three panels: the measured angular profile against
     cos^2, the same profile against what the data actually admit, and the
     p-value of the modulated component as a function of the run-to-run
     systematic.
  3. Repeatability figure: the two measurements of the same angle, which is the
     single plot that carries the conclusion.

Exact outputs:
  - results/figures/90_summary_vN.pdf and .png
  - results/figures/90_repeatability_vN.pdf and .png

Exit code: 0 if every input product exists; 1 if any is missing, naming the
step that has to be run first.

Typical use, from the repository root with .venv active:
  python scripts/90_report_figures.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_muon import models, plotting, provenance  # noqa: E402
from lib_muon.plotting import plt  # noqa: E402

CONTROL_ANGLE = 360
REFERENCE_ANGLE = 0


def main() -> int:
    ap = argparse.ArgumentParser(description="README figures.")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--fits", default="results/fits")
    ap.add_argument("--out-figures", default="results/figures")
    args = ap.parse_args()

    plotting.apply_style()
    try:
        normalisation = provenance.read_product(args.tables, "04_normalization")
        fits = provenance.read_product(args.fits, "05_fits")
        systematics = provenance.read_product(args.tables, "06_systematics")
    except FileNotFoundError as error:
        print(f"[90] ERROR: {error}", file=sys.stderr)
        return 1

    rows = normalisation["rows"]
    physical = [r for r in rows if r["angle_deg"] <= 90]
    angles = np.array([r["angle_deg"] for r in physical], dtype=float)
    normalised = np.array([r["normalised"] for r in physical])
    normalised_error = np.array([r["normalised_error"] for r in physical])

    counts = np.array(fits["counts"], dtype=float)
    times = np.array(fits["live_times_s"], dtype=float)
    observed_rate = counts / times
    observed_error = observed_rate / np.sqrt(counts)

    constant = fits["constant"]["R_hz"]
    modulation = fits["modulation_n2"]
    f_sys = systematics["dominant_systematic"]
    f_critical = systematics["critical_systematic_for_significance"]
    bound = systematics["modulated_fraction_bound_with_systematic"]
    scan = systematics["scan"]

    fine = np.linspace(0, 90, 361)
    cos_fine = models.clamped_cosine(fine)
    cos2_fine = models.cosn_shape(cos_fine, 2.0)

    # --- Summary figure ------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), layout="constrained")

    ax = axes[0]
    ax.errorbar(angles, normalised, yerr=normalised_error, fmt="o", ms=6,
                capsize=3, color=plotting.COLOUR_DATA, label="measured")
    ax.plot(fine, cos2_fine, "--", color=plotting.COLOUR_MODEL, lw=1.8,
            label="cos^2(theta), as promised")
    ax.set_xlabel("nominal angle [degrees]")
    ax.set_ylabel("I(theta) / I(0)")
    ax.set_title("What was expected")
    ax.set_ylim(-0.05, 1.2)
    ax.legend(fontsize=8, loc="lower left")
    ax.text(0.96, 0.94,
            f"chi2 = {normalisation['chi2_against_cos2']:.0f}\n"
            f"I(90)/I(0) = {normalised[-1]:.2f} +- {normalised_error[-1]:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85})

    ax = axes[1]
    ax.errorbar(angles, observed_rate, yerr=observed_error, fmt="o", ms=6,
                capsize=3, color=plotting.COLOUR_DATA, label="measured")
    ax.axhline(constant, color=plotting.COLOUR_ALTERNATE, lw=1.8,
               label=f"constant, {constant:.2f} 1/s")
    ax.plot(fine, modulation["S_hz"] * cos2_fine + modulation["B_hz"], "-",
            color=plotting.COLOUR_MODEL, lw=1.8,
            label=f"S cos^2 + B, S = {modulation['S_hz']:.2f} 1/s")
    ax.set_xlabel("nominal angle [degrees]")
    ax.set_ylabel("live-time corrected rate [1/s]")
    ax.set_title("What the data admit")
    ax.set_ylim(0, 2.4)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[2]
    scan_f = np.array([s["f_sys"] for s in scan])
    scan_p = np.array([s["p_modulation"] for s in scan])
    order = np.argsort(scan_f)
    ax.plot(100 * scan_f[order], scan_p[order], "o-", color=plotting.COLOUR_DATA,
            lw=2, ms=5)
    ax.axhline(0.05, color=plotting.COLOUR_MODEL, ls="--", lw=1.4,
               label="p = 0.05")
    ax.axvline(100 * f_sys, color=plotting.COLOUR_ALTERNATE, lw=1.8,
               label=f"measured: {100 * f_sys:.1f} %")
    ax.axvspan(100 * f_critical, 100 * max(scan_f) * 1.05,
               color=plotting.COLOUR_ALTERNATE, alpha=0.12)
    ax.set_xlabel("run-to-run systematic [%]")
    ax.set_ylabel("p-value of the modulation")
    ax.set_title("Why it cannot be claimed")
    ax.set_xlim(0, 100 * max(scan_f) * 1.05)
    ax.legend(fontsize=8, loc="lower right")

    summary_paths = plotting.save(fig, args.out_figures, "90_summary")
    print(f"[90] summary -> {summary_paths[0].name}")

    # --- Repeatability figure ------------------------------------------------
    reference = next(r for r in rows if r["angle_deg"] == REFERENCE_ANGLE)
    control = next(r for r in rows if r["angle_deg"] == CONTROL_ANGLE)
    row_45 = next(r for r in rows if r["angle_deg"] == 45)

    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    positions = [0, 1]
    values = [reference["live_rate_hz"], control["live_rate_hz"]]
    errors = [reference["error_hz"], control["error_hz"]]
    ax.errorbar(positions, values, yerr=errors, fmt="o", ms=9, capsize=5,
                color=plotting.COLOUR_DATA, lw=2)
    ax.set_xticks(positions)
    ax.set_xticklabels(["file 000deg", "file 360deg"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("live-time corrected rate [1/s]")
    ax.set_title("Two measurements of the same angle")

    drop = reference["live_rate_hz"] * (1 - row_45["normalised"])
    ax.annotate(
        "", xy=(1.28, values[0]), xytext=(1.28, values[1]),
        arrowprops={"arrowstyle": "<->", "color": plotting.COLOUR_MODEL, "lw": 1.6},
    )
    ax.text(1.32, np.mean(values),
            f"{100 * abs(1 - control['normalised']):.1f} %\nbetween two runs\n"
            "at the SAME angle",
            fontsize=8, va="center", color=plotting.COLOUR_MODEL)
    ax.annotate(
        "", xy=(-0.28, values[0]), xytext=(-0.28, values[0] - drop),
        arrowprops={"arrowstyle": "<->", "color": plotting.COLOUR_ALTERNATE, "lw": 1.6},
    )
    ax.text(-0.24, values[0] - drop / 2,
            f"{100 * (1 - row_45['normalised']):.1f} %\nis the ENTIRE drop\n"
            "from 0 to 45 degrees",
            fontsize=8, va="center", color=plotting.COLOUR_ALTERNATE)
    margin = max(drop, abs(values[0] - values[1])) * 1.6
    ax.set_ylim(min(values) - margin, max(values) + margin * 0.5)

    repeat_paths = plotting.save(fig, args.out_figures, "90_repeatability")
    print(f"[90] repeatability -> {repeat_paths[0].name}")

    text = "\n".join([
        "README figures, regenerated from the products of steps 02 to 06.",
        "No fit is performed here: every number drawn is read from results/.",
        "",
        f"  90_summary:       {summary_paths[0].name}, {summary_paths[1].name}",
        f"  90_repeatability: {repeat_paths[0].name}, {repeat_paths[1].name}",
        "",
        "Headline numbers shown:",
        f"  chi2 against pure cos^2:                "
        f"{normalisation['chi2_against_cos2']:.1f}",
        f"  I(90)/I(0):                             "
        f"{normalised[-1]:.4f} +- {normalised_error[-1]:.4f}",
        f"  constant rate:                          {constant:.4f} 1/s",
        f"  run-to-run systematic (measured):       {100 * f_sys:.2f} %",
        f"  systematic that cancels significance:   {100 * f_critical:.2f} %",
        f"  bound on the modulated fraction:        < {100 * bound:.1f} %",
    ])

    json_path, _ = provenance.write_product(
        args.tables, "90_report_figures",
        data={
            "summary_figures": [p.name for p in summary_paths],
            "repeatability_figures": [p.name for p in repeat_paths],
            "headline": {
                "chi2_against_cos2": normalisation["chi2_against_cos2"],
                "constant_rate_hz": constant,
                "dominant_systematic": f_sys,
                "critical_systematic": f_critical,
                "modulated_fraction_bound": bound,
            },
        },
        text=text,
        parameters={"source_tables": args.tables, "source_fits": args.fits},
    )
    print(f"[90] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
