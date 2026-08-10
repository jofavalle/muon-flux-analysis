#!/usr/bin/env python3
"""
03_timeseries.py

Instantaneous rate time series, one figure per run plus a combined one.

They serve the only purpose binning is good for: seeing whether the rate held
steady through a run or whether there were anomalous stretches. No physical
number of the analysis comes from here; rates are computed in 02_rates over the
full window and without bins.

Steps:
  1. Read the runs and bin them in intervals of --bin-seconds, dropping the
     final partial bin, whose apparent rate is inflated.
  2. Centred moving average and a band of one local standard deviation.
  3. Contrast of the observed spread against the Poisson expectation: for a
     constant rate R and bins of width b, sigma/mean equals 1/sqrt(R*b). A
     ratio well above 1 would indicate an unstable run.
  4. Combined figure with all eight runs to compare levels.

Exact outputs:
  - results/figures/03_timeseries_{angle:03d}deg_vN.pdf and .png  (eight pairs)
  - results/figures/03_timeseries_all_vN.pdf and .png
  - results/tables/03_timeseries_vN.json and .txt

Exit code: 0 as long as the runs are read; 1 if any admits not a single
complete bin.

Typical use, from the repository root with .venv active:
  python scripts/03_timeseries.py
  python scripts/03_timeseries.py --bin-seconds 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_muon import binning, io, plotting, provenance  # noqa: E402
from lib_muon.plotting import plt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Rate time series.")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-figures", default="results/figures")
    ap.add_argument("--out-tables", default="results/tables")
    ap.add_argument("--bin-seconds", type=float, default=10.0,
                    help="bin width for the series (default 10 s). Purely a "
                         "plotting parameter: it enters no physical result.")
    ap.add_argument("--rolling-seconds", type=float, default=60.0,
                    help="moving-average window (default 60 s)")
    args = ap.parse_args()

    plotting.apply_style()
    runs = io.load_all(args.data_dir)
    summary = []

    for angle in sorted(runs):
        run = runs[angle]
        times = run.elapsed_s[run.coincident == 1]
        centres, series, errors = binning.binned_rates(
            times, bin_s=args.bin_seconds, start_s=0.0, end_s=float(times[-1]),
        )
        if len(centres) == 0:
            print(f"[03] ERROR: {angle} degrees admits not a single complete "
                  f"bin of {args.bin_seconds} s", file=sys.stderr)
            return 1

        window_bins = max(int(round(args.rolling_seconds / args.bin_seconds)), 1)
        means, deviations = binning.moving_average(series, window_bins)

        mean = float(np.mean(series))
        spread = float(np.std(series))
        poisson_expected = np.sqrt(mean / args.bin_seconds)
        ratio = spread / poisson_expected if poisson_expected > 0 else np.nan

        fig, ax = plt.subplots(figsize=(8.0, 3.4))
        ax.errorbar(centres, series, yerr=errors, fmt="o", ms=3, lw=0.8,
                    color=plotting.COLOUR_DATA, alpha=0.8,
                    label=f"rate per {args.bin_seconds:g} s bin")
        ax.plot(centres, means, color=plotting.COLOUR_MODEL, lw=1.6,
                label=f"moving average ({args.rolling_seconds:g} s)")
        ax.fill_between(centres, means - deviations, means + deviations,
                        color=plotting.COLOUR_BAND, alpha=0.35,
                        label="+-1 local sigma")
        ax.set_xlabel("time since first event [s]")
        ax.set_ylabel("rate [1/s]")
        ax.set_title(f"Run at {angle} degrees")
        ax.legend(fontsize=8, loc="upper right")
        ax.text(
            0.02, 0.05,
            f"N = {len(times)}\nmean = {mean:.2f} 1/s\n"
            f"spread/Poisson = {ratio:.2f}",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        fig.tight_layout()
        paths = plotting.save(fig, args.out_figures,
                              f"03_timeseries_{angle:03d}deg")
        print(f"[03] {angle:>3} deg: {len(centres)} bins, "
              f"spread/Poisson = {ratio:.2f} -> {paths[0].name}")

        summary.append({
            "angle_deg": angle,
            "n_bins": len(centres),
            "mean_hz": mean,
            "spread_hz": spread,
            "poisson_expected_spread_hz": float(poisson_expected),
            "spread_ratio": float(ratio),
        })

    # Combined figure
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colours = plt.get_cmap("viridis")
    angles = sorted(runs)
    for i, angle in enumerate(angles):
        run = runs[angle]
        times = run.elapsed_s[run.coincident == 1]
        centres, series, _ = binning.binned_rates(
            times, bin_s=args.bin_seconds, start_s=0.0, end_s=float(times[-1])
        )
        window_bins = max(int(round(args.rolling_seconds / args.bin_seconds)), 1)
        means, _ = binning.moving_average(series, window_bins)
        ax.plot(centres, means, lw=1.5, color=colours(i / max(len(angles) - 1, 1)),
                label=f"{angle} deg")
    ax.set_xlabel("time since first event [s]")
    ax.set_ylabel(f"smoothed rate ({args.rolling_seconds:g} s) [1/s]")
    ax.set_title("Temporal stability of the eight runs")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    all_paths = plotting.save(fig, args.out_figures, "03_timeseries_all")
    print(f"[03] combined -> {all_paths[0].name}")

    ratios = np.array([s["spread_ratio"] for s in summary])
    text = "\n".join([
        f"Time series with {args.bin_seconds:g} s bins and a "
        f"{args.rolling_seconds:g} s moving average.",
        "",
        f"{'ang':>4} {'bins':>5} {'mean':>9} {'spread':>9} {'Poisson':>9} {'ratio':>6}",
        "-" * 50,
        *[
            f"{s['angle_deg']:>4} {s['n_bins']:>5} {s['mean_hz']:>9.4f} "
            f"{s['spread_hz']:>9.4f} {s['poisson_expected_spread_hz']:>9.4f} "
            f"{s['spread_ratio']:>6.2f}"
            for s in summary
        ],
        "-" * 50,
        f"spread/Poisson ratio between {ratios.min():.2f} and {ratios.max():.2f}.",
        "A value near 1 means the run is consistent with a constant rate plus",
        "Poisson fluctuation, that is, there are no anomalous stretches within a",
        "run. The spread BETWEEN runs is a different matter and is handled in",
        "06_systematics, where it turns out to be the dominant systematic.",
    ])

    json_path, _ = provenance.write_product(
        args.out_tables, "03_timeseries",
        data={"series": summary},
        text=text,
        parameters={"bin_seconds": args.bin_seconds,
                    "rolling_seconds": args.rolling_seconds},
    )
    print(f"[03] wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
