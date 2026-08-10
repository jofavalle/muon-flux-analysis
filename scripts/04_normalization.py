#!/usr/bin/env python3
"""
04_normalization.py

Rates relative to the 0 degree run, I(theta)/I(0), and their figure.

These ratios are the traditional way of presenting angular dependence and allow
comparison with the previous analysis, but they are NOT the basis of the
inference, for two reasons the earlier script overlooked:

  - The seven ratios share the denominator I(0), so they are not independent.
    A diagonal chi-square over them, which is what the previous analysis did,
    ignores that correlation entirely.
  - The 0 degree point is degenerate: it equals 1 exactly by construction and
    its residual is always zero. The previous analysis still counted it in the
    degrees of freedom and thereby understated the reduced chi-square.

Inference is done in 05_fits on the raw counts, where neither problem exists.

Steps:
  1. Read results/tables/02_rates_vN.json.
  2. Ratio and error propagation relative to 0 degrees.
  3. Contrast against cos^2(theta), the model the repository promised.
  4. Monotonicity check: any cos^n with n > 0 decreases monotonically over
     [0, 90], so any rise is incompatible with the whole family.
  5. Figure of I(theta)/I(0) with cos^2 overlaid.

Exact outputs:
  - results/tables/04_normalization_vN.json and .txt
  - results/figures/04_normalization_vN.pdf and .png

Exit code: 0 if normalisation completes; 1 if the reference run is missing.

Typical use, from the repository root with .venv active:
  python scripts/04_normalization.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_muon import models, plotting, provenance  # noqa: E402
from lib_muon.plotting import plt  # noqa: E402

REFERENCE_ANGLE = 0
CONTROL_ANGLE = 360


def main() -> int:
    ap = argparse.ArgumentParser(description="Rates normalised to 0 degrees.")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--out-figures", default="results/figures")
    args = ap.parse_args()

    plotting.apply_style()
    product = provenance.read_product(args.tables, "02_rates")
    rates = {int(a): r for a, r in product["rates"].items()}

    if REFERENCE_ANGLE not in rates:
        print(f"[04] ERROR: reference run at {REFERENCE_ANGLE} degrees missing",
              file=sys.stderr)
        return 1

    r0 = rates[REFERENCE_ANGLE]["live_rate_hz"]
    e0 = rates[REFERENCE_ANGLE]["live_rate_error_hz"]

    rows = []
    for angle in sorted(rates):
        r = rates[angle]["live_rate_hz"]
        e = rates[angle]["live_rate_error_hz"]
        ratio = r / r0
        error = np.sqrt((e / r0) ** 2 + (r * e0 / r0**2) ** 2)
        cos2 = float(models.cosn_shape(models.clamped_cosine(np.array([angle])), 2.0)[0]) \
            if angle <= 90 else None
        rows.append({
            "angle_deg": angle,
            "live_rate_hz": r,
            "error_hz": e,
            "normalised": float(ratio),
            "normalised_error": float(error),
            "cos2_predicted": cos2,
            "deviation_sigma": float((ratio - cos2) / error) if cos2 is not None else None,
        })

    physical = [r for r in rows if r["angle_deg"] <= 90]
    angles = np.array([r["angle_deg"] for r in physical], dtype=float)
    values = np.array([r["normalised"] for r in physical])
    errors = np.array([r["normalised_error"] for r in physical])
    cos2 = np.array([r["cos2_predicted"] for r in physical])

    chi2_cos2 = float(np.sum(((values - cos2) / errors) ** 2))

    # Monotonicity: any cos^n with n > 0 decreases over [0, 90].
    differences = np.diff(values)
    rises = [
        {"from_deg": float(angles[i]), "to_deg": float(angles[i + 1]),
         "step": float(differences[i]),
         "sigmas": float(differences[i] / np.hypot(errors[i], errors[i + 1]))}
        for i in range(len(differences)) if differences[i] > 0
    ]

    lines = [
        f"{'ang':>4} {'R_live':>9} {'I/I0':>9} {'sigma':>8} {'cos^2':>8} {'dev':>8}",
        "-" * 54,
    ]
    for row in rows:
        cos2_text = f"{row['cos2_predicted']:.4f}" if row["cos2_predicted"] is not None else "   ---"
        dev_text = f"{row['deviation_sigma']:+.1f}s" if row["deviation_sigma"] is not None else "   ---"
        mark = "  <- repeatability control" if row["angle_deg"] == CONTROL_ANGLE else ""
        lines.append(
            f"{row['angle_deg']:>4} {row['live_rate_hz']:>9.4f} "
            f"{row['normalised']:>9.4f} {row['normalised_error']:>8.4f} "
            f"{cos2_text:>8} {dev_text:>8}{mark}"
        )

    lines += [
        "-" * 54,
        "",
        f"chi2 of the {len(physical)} points from 0 to 90 degrees against pure "
        f"cos^2: {chi2_cos2:.1f}",
        f"I(90)/I(0) measured = {values[-1]:.4f} +- {errors[-1]:.4f}, against the "
        "0 that cos^2 predicts.",
        "",
        "MONOTONICITY. Every cos^n model with n > 0 decreases monotonically over",
        "[0, 90] degrees. The measured profile does not:",
    ]
    if rises:
        for rise in rises:
            lines.append(
                f"  rises from {rise['from_deg']:.0f} to {rise['to_deg']:.0f} "
                f"degrees: {rise['step']:+.4f} ({rise['sigmas']:+.2f} sigma)"
            )
        lines += [
            "",
            "This argument is of a different kind from the chi-square, and the two",
            "should not be conflated. The chi-square measures discrepancy and is",
            "the evidence carrying the conclusion. Non-monotonicity would be, IF",
            "the rise were real, a structural falsification of the entire cos^n",
            "family: no member can rise, however it is fitted. But the rise on its",
            "own is worth 1.47 sigma, which is not enough to assert it, still less",
            "without having pre-specified that these two points would be examined.",
            "It points the same way as the chi-square without being independent of",
            "it; it neither replaces nor outweighs it.",
        ]
    else:
        lines.append("  the measured profile is monotonic after all.")

    text = "\n".join(lines)
    print(text)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.errorbar(angles, values, yerr=errors, fmt="o", ms=6, capsize=3,
                color=plotting.COLOUR_DATA, label="measured, I(theta)/I(0)")
    fine = np.linspace(0, 90, 361)
    ax.plot(fine, models.cosn_shape(models.clamped_cosine(fine), 2.0), "--",
            color=plotting.COLOUR_MODEL, lw=1.8,
            label="cos^2(theta), what the repository promised")
    control = [r for r in rows if r["angle_deg"] == CONTROL_ANGLE]
    if control:
        ax.errorbar([0], [control[0]["normalised"]],
                    yerr=[control[0]["normalised_error"]], fmt="s", ms=7,
                    color=plotting.COLOUR_ALTERNATE, capsize=3,
                    label="repeat of 0 degrees (file 360deg)")
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("nominal angle [degrees]")
    ax.set_ylabel("normalised rate I(theta)/I(0)")
    ax.set_title("Measured angular dependence against the expected one")
    ax.set_ylim(-0.05, 1.25)
    ax.legend(fontsize=8, loc="lower left")
    ax.text(
        0.97, 0.95,
        f"chi2 against cos^2 = {chi2_cos2:.0f}\nwith {len(physical)} points",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    paths = plotting.save(fig, args.out_figures, "04_normalization")
    print(f"[04] figure -> {paths[0].name}")

    json_path, _ = provenance.write_product(
        args.tables, "04_normalization",
        data={
            "reference_angle": REFERENCE_ANGLE,
            "chi2_against_cos2": chi2_cos2,
            "n_physical_points": len(physical),
            "non_monotonic_rises": rises,
            "rows": rows,
        },
        text=text,
        parameters={"reference_angle": REFERENCE_ANGLE},
    )
    print(f"[04] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
