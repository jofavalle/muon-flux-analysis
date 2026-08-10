#!/usr/bin/env python3
"""
00_ingest.py

Reading and inventory of the eight raw CosmicWatch runs, first step of the
muon-angle-analysis pipeline.

For each file in data/raw:
  1. Line-by-line parsing, counting and keeping the text of every line that is
     not readable. The previous script dropped them silently with
     `except Exception: continue`, and one really exists: data/raw/015deg is
     truncated at the top and its first line is a fragment of the preamble.
  2. Decision of the actual order of the pressure and temperature columns,
     which is NOT the one the header declares. It is decided once per file
     against physical ranges using the median, not row by row by magnitude.
  3. Flagging of the rows carrying the BMP280 startup reading
     (73157.02 Pa / 23.13 C), emitted before the sensor settles.
  4. Count of coincident events against the total.
  5. Time accounting: window [t1, tN], accumulated dead time from the
     DeadTime[us] column, and the resulting live time.

Exact outputs:
  - results/tables/00_ingest_vN.json   (per-file inventory, read by step 01)
  - results/tables/00_ingest_vN.txt    (the same inventory as a table)

Exit code: 0 if all eight files are read; 1 if any cannot be read or its column
order cannot be decided.

Typical use, from the repository root with .venv active:
  python scripts/00_ingest.py
  python scripts/00_ingest.py --data-dir data/raw --out results/tables
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_muon import io, provenance  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Read and inventory the raw runs.")
    ap.add_argument("--data-dir", default="data/raw",
                    help="directory holding the NNNdeg files (default data/raw)")
    ap.add_argument("--out", default="results/tables",
                    help="output directory (default results/tables)")
    args = ap.parse_args()

    print(f"[00] reading runs from {args.data_dir}")
    try:
        runs = io.load_all(args.data_dir)
    except (ValueError, OSError) as error:
        print(f"[00] ERROR: {error}", file=sys.stderr)
        return 1

    inventory = []
    alerts = []
    for angle in sorted(runs):
        run = runs[angle]
        quality = run.quality
        good = ~run.sensor_startup
        inventory.append({
            "angle_deg": angle,
            "file": Path(run.path).name,
            "n_events": len(run),
            "n_coincident": int((run.coincident == 1).sum()),
            "window_s": run.window_s,
            "dead_time_s": run.dead_time_s,
            "live_time_s": run.live_time_s,
            "dead_fraction": run.dead_fraction,
            "mean_pressure_pa": float(run.pressure_pa[good].mean()) if good.any() else None,
            "mean_temperature_c": float(run.temperature_c[good].mean()) if good.any() else None,
            "quality": quality.as_dict(),
        })
        if quality.rejected_lines:
            alerts.append(
                f"{angle:>3} deg: {quality.rejected_lines} rejected line(s): "
                f"{quality.rejected_texts}"
            )

    total_events = sum(row["n_events"] for row in inventory)
    total_rejected = sum(row["quality"]["rejected_lines"] for row in inventory)
    total_startup = sum(row["quality"]["sensor_startup_rows"] for row in inventory)

    lines = [
        f"{'ang':>4} {'file':>8} {'N':>5} {'coinc':>6} {'window':>9} "
        f"{'dead':>8} {'live':>9} {'dead%':>8} {'bad':>5} {'startup':>8}",
        "-" * 86,
    ]
    for row in inventory:
        lines.append(
            f"{row['angle_deg']:>4} {row['file']:>8} {row['n_events']:>5} "
            f"{row['n_coincident']:>6} {row['window_s']:>9.2f} "
            f"{row['dead_time_s']:>8.2f} {row['live_time_s']:>9.2f} "
            f"{100 * row['dead_fraction']:>8.2f} "
            f"{row['quality']['rejected_lines']:>5} "
            f"{row['quality']['sensor_startup_rows']:>8}"
        )
    lines += [
        "-" * 86,
        f"total: {total_events} events in {len(inventory)} runs, "
        f"{total_rejected} rejected line(s), "
        f"{total_startup} row(s) carrying the sensor startup reading.",
        "",
        "Column order: the header declares",
        f"  {io.DECLARED_HEADER}",
        "and the data arrive as",
        f"  {io.ACTUAL_HEADER}",
        "that is, pressure before temperature. Verified file by file against "
        "physical ranges.",
    ]
    if alerts:
        lines += ["", "Alerts:"] + [f"  - {a}" for a in alerts]

    text = "\n".join(lines)
    print(text)

    json_path, text_path = provenance.write_product(
        args.out, "00_ingest",
        data={
            "n_runs": len(inventory),
            "n_events_total": total_events,
            "n_rejected_lines_total": total_rejected,
            "n_startup_rows_total": total_startup,
            "declared_header": io.DECLARED_HEADER,
            "actual_header": io.ACTUAL_HEADER,
            "inventory": inventory,
        },
        text=text,
        parameters={"data_dir": args.data_dir},
    )
    print(f"[00] wrote {json_path}")
    print(f"[00] wrote {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
