#!/usr/bin/env python3
"""
check_reproducibility.py

Compares the results versioned in the repository against those produced by a
fresh run, and fails if they disagree beyond floating-point noise.

Not a pipeline step, hence no number: steps 00 to 06 produce the analysis and
step 90 presents it, whereas this is a tool that checks the products. It is
what CI runs after executing the pipeline on a clean runner.

Why it exists: the results committed to `results/` were produced on a developer
machine. Publishing them is what makes the repository's claims checkable, but
it also means a reader is trusting numbers they did not generate. If a clean
runner with pinned dependencies cannot reproduce them, the numbers are not
reproducible and that has to fail loudly rather than sit in the repository
looking authoritative.

Steps:
  1. For every product base name, locate the oldest version (_v1, the published
     one) and the newest (_vN, the one just produced).
  2. Walk both JSON trees together and compare every numeric leaf.
  3. Ignore the provenance block, which is expected to differ: it carries
     timestamps, a commit hash and platform details.
  4. Report every mismatch beyond tolerance, with its path in the JSON tree.

Exact outputs: none written; the report goes to stdout.

Exit code: 0 if every number matches within tolerance; 1 if any differs, or if
there is nothing to compare against.

Typical use, from the repository root with .venv active:
  python scripts/check_reproducibility.py
  python scripts/check_reproducibility.py --rtol 1e-8
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_muon import provenance  # noqa: E402

# Keys whose difference between runs carries no information about
# reproducibility: timestamps, hashes, platform.
IGNORED_KEYS = {"provenance"}

# A product filename differing only in its _vN suffix is the versioning working
# as designed, not a result that changed. Comparing 90_summary_v1.pdf against
# 90_summary_v2.pdf must not be reported as a reproducibility failure.
VERSIONED_NAME = re.compile(r"^(?P<base>.+)_v\d+(?P<extension>\.[a-z]+)$")


def same_but_for_version(a: str, b: str) -> bool:
    """True if two strings are the same product name at different versions."""
    match_a, match_b = VERSIONED_NAME.match(a), VERSIONED_NAME.match(b)
    if not match_a or not match_b:
        return False
    return (match_a.group("base") == match_b.group("base")
            and match_a.group("extension") == match_b.group("extension"))

PRODUCTS = [
    ("results/tables", "00_ingest"),
    ("results/tables", "01_validation"),
    ("results/tables", "02_rates"),
    ("results/tables", "03_timeseries"),
    ("results/tables", "04_normalization"),
    ("results/fits", "05_fits"),
    ("results/tables", "06_systematics"),
    ("results/tables", "90_report_figures"),
]


def compare(published, fresh, rtol, atol, path=""):
    """Walk two JSON trees together and return a list of mismatches."""
    problems = []

    if isinstance(published, dict) and isinstance(fresh, dict):
        for key in sorted(set(published) | set(fresh)):
            if key in IGNORED_KEYS:
                continue
            if key not in published:
                problems.append(f"{path}/{key}: only in the fresh run")
            elif key not in fresh:
                problems.append(f"{path}/{key}: only in the published run")
            else:
                problems += compare(published[key], fresh[key], rtol, atol,
                                    f"{path}/{key}")
        return problems

    if isinstance(published, list) and isinstance(fresh, list):
        if len(published) != len(fresh):
            return [f"{path}: length {len(published)} against {len(fresh)}"]
        for i, (a, b) in enumerate(zip(published, fresh)):
            problems += compare(a, b, rtol, atol, f"{path}[{i}]")
        return problems

    if isinstance(published, bool) or isinstance(fresh, bool):
        if published != fresh:
            problems.append(f"{path}: {published} against {fresh}")
        return problems

    if isinstance(published, (int, float)) and isinstance(fresh, (int, float)):
        difference = abs(published - fresh)
        if difference > atol + rtol * abs(published):
            problems.append(
                f"{path}: {published!r} against {fresh!r} "
                f"(difference {difference:.3e})"
            )
        return problems

    if isinstance(published, str) and isinstance(fresh, str):
        if published != fresh and not same_but_for_version(published, fresh):
            problems.append(f"{path}: {published!r} against {fresh!r}")
        return problems

    if published != fresh:
        problems.append(f"{path}: {published!r} against {fresh!r}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Check published results reproduce.")
    ap.add_argument("--rtol", type=float, default=1e-6,
                    help="relative tolerance (default 1e-6). Loose enough for "
                         "the optimiser to land on a slightly different point "
                         "of a flat minimum, tight enough that any real change "
                         "in a result shows up.")
    ap.add_argument("--atol", type=float, default=1e-12,
                    help="absolute tolerance (default 1e-12)")
    args = ap.parse_args()

    total_problems = 0
    compared = 0

    for directory, base in PRODUCTS:
        directory = Path(directory)
        latest = provenance.latest_version(directory, base)
        if latest is None:
            print(f"[check] MISSING: no {base}_vN.json in {directory}")
            total_problems += 1
            continue
        if latest < 2:
            print(f"[check] SKIPPED: {base} has only one version, so there is "
                  "nothing to compare against. Run the pipeline first.")
            total_problems += 1
            continue

        published = json.loads((directory / f"{base}_v1.json").read_text())
        fresh = json.loads((directory / f"{base}_v{latest}.json").read_text())
        problems = compare(published, fresh, args.rtol, args.atol)
        compared += 1

        if problems:
            print(f"[check] {base}: {len(problems)} MISMATCH(ES) between _v1 "
                  f"(published) and _v{latest} (this run)")
            for problem in problems[:20]:
                print(f"          {problem}")
            if len(problems) > 20:
                print(f"          ... and {len(problems) - 20} more")
            total_problems += len(problems)
        else:
            print(f"[check] {base}: reproduces exactly (_v1 against _v{latest})")

    print()
    if total_problems:
        print(f"[check] FAILED: {total_problems} problem(s) across {compared} "
              "product(s).")
        print("[check] The results committed to the repository do not reproduce "
              "here. Either the")
        print("[check] analysis changed and results/ needs regenerating, or it "
              "depends on something")
        print("[check] about the machine that produced it. Both matter.")
        return 1

    print(f"[check] OK: {compared} product(s) reproduce within "
          f"rtol={args.rtol:g}, atol={args.atol:g}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
