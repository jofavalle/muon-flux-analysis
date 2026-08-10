#!/usr/bin/env python3
"""
02_rates.py

Count rate of each run, corrected for live time.

Steps:
  1. Read the eight raw runs.
  2. Observation window as tN - t1, not truncated to a multiple of the bin
     size. In the previous analysis the duration was truncated to multiples of
     --bin-seconds, so a cosmetic parameter moved the published rate:
     1.8118 1/s with 1 s bins against 1.8194 1/s with 60 s bins.
  3. Dead time from the DeadTime[us] column, previously parsed and never used.
     The dead fraction runs from 9.09 % to 9.28 %.
  4. Rate through the unbiased estimator (N-1)/T_live. For N events in
     [t1, tN] there are N-1 intervals, not N.
  5. Explicit comparison against the wall-clock rate, to show how much each
     correction moves.

A caveat the report repeats: since the dead fraction is nearly identical across
the eight runs, the live-time correction cancels almost entirely in the
NORMALISED rates. It matters for the absolute rate, not for the angular shape.
Correcting is right, but it does not fix anything that was broken in the
comparison between angles.

Exact outputs:
  - results/tables/02_rates_vN.json   (consumed by steps 04, 05 and 06)
  - results/tables/02_rates_vN.txt

Exit code: 0 if all eight rates are computed; 1 if any run lacks enough events.

Typical use, from the repository root with .venv active:
  python scripts/02_rates.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_muon import io, provenance, rates  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Live-time corrected rates.")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()

    runs = io.load_all(args.data_dir)
    computed = {}
    for angle in sorted(runs):
        try:
            computed[angle] = rates.compute_rate(runs[angle])
        except ValueError as error:
            print(f"[02] ERROR at {angle} degrees: {error}", file=sys.stderr)
            return 1

    lines = [
        f"{'ang':>4} {'N':>5} {'N-1':>5} {'window':>9} {'live':>9} {'dead%':>8} "
        f"{'R_wall':>9} {'R_live':>9} {'sigma':>8} {'raise%':>8}",
        "-" * 88,
    ]
    for angle, rate in computed.items():
        raised = 100 * (rate.live_rate / rate.wall_rate - 1)
        lines.append(
            f"{angle:>4} {rate.n_events:>5} {rate.n_intervals:>5} "
            f"{rate.window_s:>9.2f} {rate.live_time_s:>9.2f} "
            f"{100 * rate.dead_fraction:>8.2f} {rate.wall_rate:>9.4f} "
            f"{rate.live_rate:>9.4f} {rate.live_rate_error:>8.4f} {raised:>8.2f}"
        )

    fractions = np.array([r.dead_fraction for r in computed.values()])
    lines += [
        "-" * 88,
        f"Dead fraction between {100 * fractions.min():.2f} % and "
        f"{100 * fractions.max():.2f} %, with a spread of "
        f"{100 * fractions.std():.3f} percentage points.",
        "It is nearly identical across the eight runs, so the live-time",
        "correction cancels almost entirely in the ratios between angles: it",
        "lifts every rate by about 10 % and leaves the angular shape where it",
        "was. It matters for the absolute rate, not for the angular dependence.",
    ]
    text = "\n".join(lines)
    print(text)

    json_path, text_path = provenance.write_product(
        args.out, "02_rates",
        data={"rates": {str(a): r.as_dict() for a, r in computed.items()}},
        text=text,
        parameters={"data_dir": args.data_dir, "coincident_only": True},
    )
    print(f"[02] wrote {json_path}")
    print(f"[02] wrote {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
