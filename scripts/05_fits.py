#!/usr/bin/env python3
"""
05_fits.py

Inference on the angular dependence, by Poisson maximum likelihood on the raw
counts.

Why not chi-square on the normalised ratios, which is what the previous
analysis did: the seven ratios share the denominator I(0) and are not
independent, so a diagonal chi-square is ill-posed; and the 0 degree point is
degenerate after normalising, with a residual that is zero by construction, yet
it still added to the degrees of freedom. On raw counts neither problem exists.
The observable is the number of inter-event intervals, N-1, which is Poisson
with mean R*T_live.

Models fitted, all with cos(90) clamped to exact zero:
  A. Constant:          mu_i = R * T_i.                        1 parameter.
  B. Free cos^n:        mu_i = R * T_i * cos^n(theta_i).       2 parameters.
  C. Fixed cos^2:       mu_i = R * T_i * cos^2(theta_i).       1 parameter.
  D. Signal+background: mu_i = T_i * (S*cos^n(theta_i) + B).   3 parameters.
  E. Signal+background with n = 2 fixed.                       2 parameters.

Central result, and not an implementation detail: with cos(90) = 0 exactly,
models B and C predict exactly zero counts at 90 degrees, where 446 were
observed. Their likelihood is rigorously zero and their -logL diverges, so they
have NO finite AIC or BIC to compete with. The only reason the previous
analysis could fit a cos^n was that it evaluated cos(deg2rad(90))**n =
(6.123e-17)**n, which for small n is of order 1. The signal-plus-background
family is the appropriate one: B collects in a single number what all three
hypotheses of the analysis predict at 90 degrees (wide geometric acceptance, an
accidental component, and an undocumented angular convention), which this data
cannot separate.

Steps:
  1. Read results/tables/02_rates_vN.json and build the observable.
  2. Fit the five models.
  3. Uncertainties by profile likelihood, Delta(-2lnL) = 1 for 68 %.
  4. Goodness of fit by deviance and chi-square p-value.
  5. Comparison by AIC, BIC and likelihood ratio.
  6. Upper bound at 95 % on the fraction of flux modulated as cos^2.
  7. Fit figure with a Pearson residual panel.

Exact outputs:
  - results/fits/05_fits_vN.json and .txt
  - results/figures/05_fits_vN.pdf and .png

Exit code: 0 if the constant and signal-plus-background fits converge; 1 if
either fails. B and C having null likelihood is NOT a failure: it is the
result, and it is recorded as such.

Typical use, from the repository root with .venv active:
  python scripts/05_fits.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_muon import models, plotting, provenance  # noqa: E402
from lib_muon.plotting import plt  # noqa: E402

CONTROL_ANGLE = 360
# Delta(-logL) for a one-sided 95 % bound: Delta(-2lnL) = 2.706.
DELTA_BOUND_95 = 1.3530


def main() -> int:
    ap = argparse.ArgumentParser(description="Poisson likelihood fits.")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--out-fits", default="results/fits")
    ap.add_argument("--out-figures", default="results/figures")
    args = ap.parse_args()

    plotting.apply_style()
    product = provenance.read_product(args.tables, "02_rates")
    rates = {int(a): r for a, r in product["rates"].items()}

    # The 360 degree file is a repeatability control of the 0 degree run, not a
    # different physical angle: it belongs in 06_systematics, not in the fit.
    angles = np.array(sorted(a for a in rates if a != CONTROL_ANGLE), dtype=float)
    counts = np.array([rates[int(a)]["n_intervals"] for a in angles], dtype=float)
    times = np.array([rates[int(a)]["live_time_s"] for a in angles])
    cos_t = models.clamped_cosine(angles)

    print(f"[05] {len(angles)} points from {angles[0]:.0f} to {angles[-1]:.0f} "
          f"degrees; the {CONTROL_ANGLE} degree control is left to step 06.")
    print(f"[05] clamped cos(theta): {np.array2string(cos_t, precision=6)}")

    fits = {
        "A_constant": models.fit_constant(counts, times),
        "B_free_cosn": models.fit_pure_cosn(counts, times, cos_t),
        "C_fixed_cos2": models.fit_fixed_cosn(counts, times, cos_t, n=2.0),
        "D_signal_background": models.fit_signal_background(counts, times, cos_t),
        "E_signal_background_n2": models.fit_signal_background(counts, times, cos_t, n=2.0),
    }

    if not np.isfinite(fits["A_constant"].nll):
        print("[05] ERROR: the constant model did not converge", file=sys.stderr)
        return 1
    if not np.isfinite(fits["E_signal_background_n2"].nll):
        print("[05] ERROR: the signal-plus-background model did not converge",
              file=sys.stderr)
        return 1

    goodness = {}
    for key, fit in fits.items():
        d, p = models.goodness_of_fit(counts, fit.mu, fit.dof)
        goodness[key] = {"deviance": d, "dof": fit.dof, "p_value": p}

    constant = fits["A_constant"]

    def nll_of_r(r):
        return models.poisson_nll(r * times, counts)

    r_low, r_high = models.profile_likelihood(
        nll_of_r, constant.parameters["R_hz"], constant.nll,
        lower_limit=1e-6, upper_limit=10.0,
    )

    fit_e = fits["E_signal_background_n2"]
    shape2 = models.cosn_shape(cos_t, 2.0)

    def nll_profile_s(s):
        """-logL minimised over B at fixed S."""
        if s < 0:
            return np.inf
        result = optimize.minimize_scalar(
            lambda b: models.poisson_nll(times * (s * shape2 + b), counts)
            if b > 0 else np.inf,
            bounds=(1e-9, 20.0), method="bounded",
        )
        return float(result.fun)

    def best_b_given_s(s):
        result = optimize.minimize_scalar(
            lambda b: models.poisson_nll(times * (s * shape2 + b), counts)
            if b > 0 else np.inf,
            bounds=(1e-9, 20.0), method="bounded",
        )
        return float(result.x)

    s_best = fit_e.parameters["S_hz"]
    s_low_68, s_high_68 = models.profile_likelihood(
        nll_profile_s, s_best, fit_e.nll, delta=0.5,
        lower_limit=0.0, upper_limit=10.0,
    )
    _, s_bound_95 = models.profile_likelihood(
        nll_profile_s, s_best, fit_e.nll, delta=DELTA_BOUND_95,
        lower_limit=0.0, upper_limit=10.0,
    )
    b_at_bound = best_b_given_s(s_bound_95) if np.isfinite(s_bound_95) else np.nan
    bound_fraction = (s_bound_95 / (s_bound_95 + b_at_bound)
                      if np.isfinite(s_bound_95) else np.nan)

    d_lr, p_lr = models.likelihood_ratio(
        constant.nll, fit_e.nll,
        extra_dof=fit_e.n_parameters - constant.n_parameters,
    )

    lines = [
        "OBSERVABLE: number of inter-event intervals, N-1, Poisson with mean "
        "R*T_live.",
        f"{len(angles)} points, from {angles[0]:.0f} to {angles[-1]:.0f} degrees. "
        f"The {CONTROL_ANGLE} degree control is handled in 06_systematics.",
        "",
        f"{'ang':>4} {'N-1':>6} {'T_live':>9} {'cos(theta)':>11}",
        "-" * 34,
    ]
    for a, n, t, c in zip(angles, counts, times, cos_t):
        lines.append(f"{a:>4.0f} {n:>6.0f} {t:>9.2f} {c:>11.6f}")

    lines += ["", "=" * 78, "FITS", "=" * 78]
    for key, fit in fits.items():
        info = goodness[key]
        if np.isfinite(fit.nll):
            parameters = ", ".join(f"{k} = {v:.4f}" for k, v in fit.parameters.items())
            lines += [
                f"{key}  ({fit.n_parameters} parameters)",
                f"   {parameters}",
                f"   -logL = {fit.nll:.3f}   AIC = {fit.aic:.2f}   BIC = {fit.bic:.2f}",
                f"   deviance = {info['deviance']:.2f} with {info['dof']} dof, "
                f"p = {info['p_value']:.4f}",
            ]
        else:
            lines += [
                f"{key}  ({fit.n_parameters} parameters)",
                "   -logL = INFINITE. The model predicts exactly 0 counts at 90 "
                f"degrees and {counts[-1]:.0f} were observed.",
                "   Its likelihood is rigorously zero: there is no AIC or BIC to "
                "compare.",
                "   This follows from clamping cos(90) = 0. The previous analysis "
                "could fit it only",
                "   because it evaluated (6.123e-17)**n, which for small n is of "
                "order 1.",
            ]
        lines.append("")

    lines += [
        "=" * 78,
        "RESULTS",
        "=" * 78,
        "",
        "1. Single rate (constant model)",
        f"   R = {constant.parameters['R_hz']:.4f} 1/s, "
        f"68 % interval [{r_low:.4f}, {r_high:.4f}]",
        f"   deviance = {goodness['A_constant']['deviance']:.2f} with "
        f"{goodness['A_constant']['dof']} dof, "
        f"p = {goodness['A_constant']['p_value']:.4f}",
    ]
    if goodness["A_constant"]["p_value"] < 0.05:
        lines += [
            "   This is NOT a good fit under pure Poisson statistics. The data are",
            "   more scattered than counting statistics alone allows. The excess",
            "   scatter is quantified in 06_systematics, where it turns out to be",
            "   the size of the run-to-run systematic that the 360 degree control",
            "   itself measures.",
        ]
    lines += [
        "",
        "2. How much angular modulation the data admit (model E, n = 2 fixed)",
        f"   S = {fit_e.parameters['S_hz']:.4f} 1/s "
        f"[{s_low_68:.4f}, {s_high_68:.4f}] at 68 %",
        f"   B = {fit_e.parameters['B_hz']:.4f} 1/s",
        f"   modulated fraction at 0 degrees = "
        f"{fit_e.parameters['modulated_fraction']:.4f}",
        f"   UPPER BOUND AT 95 %: S < {s_bound_95:.4f} 1/s, that is a modulated",
        f"   fraction below {100 * bound_fraction:.1f} % (statistics only; the",
        "   published bound, with the systematic folded in, is in 06_systematics).",
        "",
        "3. Comparison against the constant-rate hypothesis",
        f"   Delta(-2lnL) = {d_lr:.3f} with "
        f"{fit_e.n_parameters - constant.n_parameters} dof, p = {p_lr:.4f}",
    ]
    if p_lr > 0.05:
        lines.append("   Adding a modulated component does NOT significantly "
                     "improve the fit.")
    else:
        lines += [
            "   Under pure Poisson statistics the modulated component DOES improve",
            "   the fit.",
            "",
            "   THIS P-VALUE IS NOT THE CONCLUSION, and taking it for one would be",
            "   the very error this analysis exists to avoid. It assumes counting",
            "   statistics is the only source of uncertainty, and this data set",
            "   contains the proof that it is not: the 360 degree file repeats the",
            "   0 degree measurement and differs by 7.5 %. A run-to-run systematic",
            "   of only 3.9 % is enough for this preference to stop being",
            "   significant, well below what the control itself measures. The",
            "   reassessment is in 06_systematics and that is the one that stands.",
        ]
    lines += [
        "   Further statistical caveat: S is bounded to non-negative values, so",
        "   the parameter sits on the boundary of its space and the asymptotic",
        "   distribution is not exactly chi-square.",
        "",
        "4. Pearson residuals against the constant model",
    ]
    constant_residuals = models.pearson_residuals(counts, constant.mu)
    for a, n, m, r in zip(angles, counts, constant.mu, constant_residuals):
        lines.append(
            f"   {a:>3.0f} deg: N-1 = {n:>4.0f}, expected = {m:>7.1f}, "
            f"residual = {r:>+6.2f} sigma"
        )

    text = "\n".join(lines)
    print(text)

    # constrained_layout rather than tight_layout: the latter cannot handle a
    # gridspec with a fixed hspace and warns that the result may be wrong.
    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(7.6, 6.2), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [3, 1]},
    )

    observed_rate = counts / times
    observed_error = observed_rate / np.sqrt(counts)
    ax.errorbar(angles, observed_rate, yerr=observed_error, fmt="o", ms=6,
                capsize=3, color=plotting.COLOUR_DATA, label="measured", zorder=5)

    fine = np.linspace(0, 90, 361)
    cos_fine = models.clamped_cosine(fine)
    ax.axhline(constant.parameters["R_hz"], color=plotting.COLOUR_ALTERNATE,
               lw=1.8, label=f"constant, R = {constant.parameters['R_hz']:.3f} 1/s")
    ax.fill_between(fine, r_low, r_high, color=plotting.COLOUR_ALTERNATE, alpha=0.15)
    ax.plot(fine,
            fit_e.parameters["S_hz"] * models.cosn_shape(cos_fine, 2.0)
            + fit_e.parameters["B_hz"],
            "-", color=plotting.COLOUR_MODEL, lw=1.8,
            label=f"S cos^2 + B, S = {fit_e.parameters['S_hz']:.3f} 1/s")
    ax.plot(fine, s_bound_95 * models.cosn_shape(cos_fine, 2.0) + b_at_bound,
            ":", color=plotting.COLOUR_MODEL, lw=1.5,
            label=f"95 % bound, S < {s_bound_95:.3f} 1/s")
    ax.plot(fine, constant.parameters["R_hz"] * models.cosn_shape(cos_fine, 2.0),
            "--", color="0.45", lw=1.5, label="pure cos^2 (null likelihood)")
    ax.set_ylabel("live-time corrected rate [1/s]")
    ax.set_title("Angular dependence of the flux: what was measured and what fits")
    ax.set_ylim(0, 2.6)
    ax.legend(fontsize=8, loc="lower left")

    ax_res.errorbar(angles, constant_residuals, yerr=1.0, fmt="o", ms=5,
                    capsize=3, color=plotting.COLOUR_DATA)
    ax_res.axhline(0, color=plotting.COLOUR_ALTERNATE, lw=1.4)
    ax_res.axhspan(-2, 2, color=plotting.COLOUR_ALTERNATE, alpha=0.12)
    ax_res.set_xlabel("nominal angle [degrees]")
    ax_res.set_ylabel("residual\n[sigma]")
    ax_res.set_ylim(-4, 4)
    paths = plotting.save(fig, args.out_figures, "05_fits")
    print(f"[05] figure -> {paths[0].name}")

    data = {
        "angles_deg": angles,
        "counts": counts,
        "live_times_s": times,
        "clamped_cosine": cos_t,
        "cos_eps": models.COS_EPS,
        "fits": {k: {**f.as_dict(), **goodness[k]} for k, f in fits.items()},
        "constant": {"R_hz": constant.parameters["R_hz"],
                     "interval_68": [r_low, r_high]},
        "modulation_n2": {
            "S_hz": fit_e.parameters["S_hz"],
            "S_interval_68": [s_low_68, s_high_68],
            "B_hz": fit_e.parameters["B_hz"],
            "modulated_fraction": fit_e.parameters["modulated_fraction"],
            "S_bound_95": s_bound_95,
            "B_at_bound_95": b_at_bound,
            "modulated_fraction_bound_95": bound_fraction,
        },
        "likelihood_ratio_constant_vs_modulated": {
            "delta_2lnL": d_lr,
            "dof": fit_e.n_parameters - constant.n_parameters,
            "p_value": p_lr,
        },
        "constant_pearson_residuals": constant_residuals,
    }
    json_path, _ = provenance.write_product(
        args.out_fits, "05_fits", data=data, text=text,
        parameters={"excluded_control_angle": CONTROL_ANGLE,
                    "delta_bound_95": DELTA_BOUND_95},
    )
    print(f"[05] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
