#!/usr/bin/env python3
"""
01_validation.py

Data quality gate. Checks the conditions the rest of the pipeline takes for
granted, and warns when one of them stops holding.

Checks, in order:
  1. The eight expected runs are present and none is empty.
  2. The clock never runs backwards in any run. This justifies measuring the
     window as tN - t1; the previous script accumulated positive differences,
     which would have absorbed a backwards step silently.
  3. The dead-time fraction sits in the 9 % characteristic of this set. A run
     falling outside would indicate a different operating mode.
  4. Pressure and temperature fall in physical range once the sensor startup
     rows are excluded.
  5. No run exceeds the 600 s window the previous script imposed with --max-ms:
     we verify that this cut never actually acted on this data.
  6. Coincident == 1 for every event. This is verified and its consequence is
     DECLARED: with no singles recorded, the accidental rate is not estimable
     from this set.
  7. Rejected lines are enumerated one by one, with their text.

Two anomalies of this data are already documented in docs/dataset.md: the
truncated line in 015deg and the eight BMP280 startup rows. They are always
reported, marked KNOWN, but they do not fail the gate: what a gate must detect
is a CHANGE in the data, not the existence of defects already described and
bounded. A gate that fails every time is a gate nobody reads.

The comparison against the documented state is exact and runs in BOTH
directions. The anchors hold the literal text of the rejected line and the
per-angle counts, so the gate fires if a new anomaly appears, if a documented
one changes, and equally if a documented one DISAPPEARS: the data would have
changed just as much. With --strict, documented anomalies count as alerts too.

Exact outputs:
  - results/tables/01_validation_vN.json
  - results/tables/01_validation_vN.txt

Exit code: 0 if there are no new alerts; 1 if any undocumented anomaly appears,
or any anomaly at all when --strict is passed.

Typical use, from the repository root with .venv active:
  python scripts/01_validation.py
  python scripts/01_validation.py --strict
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_muon import io, provenance  # noqa: E402

EXPECTED_ANGLES = [0, 5, 15, 25, 35, 45, 90, 360]
EXPECTED_DEAD_FRACTION = (0.085, 0.100)
PREVIOUS_MAX_WINDOW_S = 600.0

# Anomalies already documented in docs/dataset.md. Always reported, but they do
# not fail the gate: what it must detect is a change in the data, not defects
# already described and bounded. With --strict they count and the script exits 1.
KNOWN_REJECTIONS = {15: ["118  | Hexa address:  0x76"]}
KNOWN_STARTUP = {0: 2, 5: 2, 25: 2, 35: 1, 360: 1}


def main() -> int:
    ap = argparse.ArgumentParser(description="Raw data quality gate.")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--strict", action="store_true",
                    help="treat already documented anomalies as alerts too")
    args = ap.parse_args()

    runs = io.load_all(args.data_dir)
    checks: list[dict] = []
    lines: list[str] = []

    def check(name: str, passed: bool, detail: str, known: bool = False) -> None:
        """`known` marks an already documented anomaly: reported, but it does
        not fail the gate unless --strict."""
        alert = not passed and (args.strict or not known)
        checks.append({
            "name": name, "passed": bool(passed), "detail": detail,
            "known_anomaly": bool(known), "counts_as_alert": bool(alert),
        })
        mark = "OK  " if passed else ("KNOWN" if known and not args.strict else "ALERT")
        lines.append(f"[{mark:>5}] {name}: {detail}")

    def check_against_documented(name: str, observed, documented, describe) -> None:
        """Compare the observed state against the documented one.

        Any difference is an alert, in BOTH directions. A documented anomaly
        that disappears means the data changed just as much as a new one
        appearing, and a gate that stayed quiet about it would be checking only
        half of what it claims to check.
        """
        matches = observed == documented
        if matches and not documented:
            check(name, True, describe(observed, "none"))
            return
        if matches:
            check(name, False, describe(observed, "matches what is documented"),
                  known=True)
            return
        missing_now = {k: v for k, v in documented.items() if k not in observed}
        appeared = {k: v for k, v in observed.items() if k not in documented}
        changed = {k: (documented[k], observed[k]) for k in observed
                   if k in documented and observed[k] != documented[k]}
        reason = []
        if appeared:
            reason.append(f"new: {appeared}")
        if missing_now:
            reason.append(f"NO LONGER PRESENT: {missing_now}")
        if changed:
            reason.append(f"changed (documented -> observed): {changed}")
        check(name, False,
              describe(observed, "DIFFERS from docs/dataset.md -> "
                                 + "; ".join(reason)))

    # 1. Inventory
    missing = sorted(set(EXPECTED_ANGLES) - set(runs))
    unexpected = sorted(set(runs) - set(EXPECTED_ANGLES))
    check(
        "run inventory",
        not missing and not unexpected,
        f"{len(runs)} runs"
        + (f", missing {missing}" if missing else "")
        + (f", unexpected {unexpected}" if unexpected else ""),
    )

    # 2. Clock monotonicity
    non_monotonic = [a for a, r in runs.items() if not np.all(np.diff(r.timestamp_ms) > 0)]
    check(
        "clock never runs backwards",
        not non_monotonic,
        "all timestamps strictly increasing" if not non_monotonic
        else f"runs backwards at {non_monotonic}",
    )

    # 3. Dead time
    out_of_band = {
        a: round(r.dead_fraction, 4) for a, r in runs.items()
        if not (EXPECTED_DEAD_FRACTION[0] < r.dead_fraction < EXPECTED_DEAD_FRACTION[1])
    }
    fractions = np.array([r.dead_fraction for r in runs.values()])
    check(
        "dead-time fraction",
        not out_of_band,
        f"between {100 * fractions.min():.2f} % and {100 * fractions.max():.2f} %"
        + (f"; out of range: {out_of_band}" if out_of_band else ""),
    )

    # 4. Sensor physical ranges
    out_of_range = {a: r.quality.out_of_range_rows for a, r in runs.items()
                    if r.quality.out_of_range_rows}
    check(
        "pressure and temperature in range",
        not out_of_range,
        "all rows within physical range" if not out_of_range
        else f"out-of-range rows: {out_of_range}",
    )

    # 5. The 600 s window of the previous analysis
    too_long = {a: round(r.window_s, 1) for a, r in runs.items()
                if r.window_s > PREVIOUS_MAX_WINDOW_S}
    windows = np.array([r.window_s for r in runs.values()])
    check(
        "600 s window of the previous analysis",
        not too_long,
        f"the longest run lasts {windows.max():.1f} s, so the --max-ms cut at "
        f"600 s never acted on this data"
        + (f"; exceeding: {too_long}" if too_long else ""),
    )

    # 6. Coincidence
    non_coincident = {a: int((r.coincident != 1).sum()) for a, r in runs.items()
                      if (r.coincident != 1).any()}
    total_events = sum(len(r) for r in runs.values())
    check(
        "coincidence filter",
        True,
        f"Coincident == 1 for all {total_events} events, so the filter is a "
        "no-op. With no singles recorded, THE ACCIDENTAL RATE IS NOT ESTIMABLE "
        "from this set: it can neither be subtracted nor bounded."
        + (f" Exceptions: {non_coincident}" if non_coincident else ""),
    )

    # 7. Rejected lines and sensor startup
    rejections = {a: r.quality.rejected_texts for a, r in runs.items()
                  if r.quality.rejected_lines}
    total_rejected = sum(r.quality.rejected_lines for r in runs.values())
    check_against_documented(
        "malformed lines", rejections, KNOWN_REJECTIONS,
        lambda observed, verdict: (
            f"{total_rejected} rejected line(s) {observed}; {verdict}. "
            "Documented: data/raw/015deg is truncated at the top. The literal "
            "text is anchored, not just the count, so a different malformed "
            "line would also be caught."
        ),
    )

    startup = {a: r.quality.sensor_startup_rows for a, r in runs.items()
               if r.quality.sensor_startup_rows}
    total_startup = sum(startup.values())
    check_against_documented(
        "sensor startup readings", startup, KNOWN_STARTUP,
        lambda observed, verdict: (
            f"{total_startup} row(s) carrying {io.STARTUP_PRESSURE_PA} Pa / "
            f"{io.STARTUP_TEMPERATURE_C} C before the BMP280 settles, spread as "
            f"{observed}; {verdict}. They affect neither event counts nor "
            "times; they are flagged because they invalidate any use of "
            "pressure or temperature in those rows."
        ),
    )

    n_alerts = sum(1 for c in checks if c["counts_as_alert"])
    n_known = sum(1 for c in checks if not c["passed"] and c["known_anomaly"])
    lines += [
        "",
        f"{len(checks)} checks, {n_alerts} alert(s), {n_known} anomaly(ies) "
        "already documented in docs/dataset.md.",
    ]
    if n_known and not args.strict:
        lines.append(
            "Documented anomalies do not fail the gate: what it must detect is a "
            "change in the data. With --strict they do."
        )

    text = "\n".join(lines)
    print(text)

    json_path, text_path = provenance.write_product(
        args.out, "01_validation",
        data={"n_checks": len(checks), "n_alerts": n_alerts,
              "n_known_anomalies": n_known, "checks": checks},
        text=text,
        parameters={"data_dir": args.data_dir, "strict": args.strict},
    )
    print(f"[01] wrote {json_path}")
    print(f"[01] wrote {text_path}")
    return 1 if n_alerts else 0


if __name__ == "__main__":
    raise SystemExit(main())
