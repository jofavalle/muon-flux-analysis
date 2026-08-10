#!/usr/bin/env python3
"""
check_reproducibility.py

Runs the whole pipeline into a scratch directory and compares its output
against the results committed to the repository, failing if any number differs
beyond floating-point noise.

Not a pipeline step, hence no number: steps 00 to 06 produce the analysis and
step 90 presents it, whereas this is a tool that checks the products. It is
self-contained on purpose: it needs no prior command, and it leaves `results/`
untouched. A checker that requires setup gets run wrong, and a checker that
writes into the directory it is checking makes the next check meaningless.

Why it exists: the results committed to `results/` were produced on a developer
machine. Publishing them is what makes the repository's claims checkable, but
it also means a reader is trusting numbers they did not generate. If a clean
machine with pinned dependencies cannot reproduce them, the numbers are not
reproducible and that has to fail loudly rather than sit in the repository
looking authoritative.

Steps:
  1. Run steps 00 to 06 and 90 into a temporary directory, reading the same
     `data/raw` and never touching `results/`.
  2. For each product, walk the committed JSON and the fresh one together and
     compare every numeric leaf.
  3. Ignore the provenance block, which is expected to differ: it carries
     timestamps, a commit hash and platform details. Product filenames that
     differ only in their _vN suffix are likewise not a disagreement.
  4. Report every mismatch beyond tolerance, with its path in the JSON tree.

Exact outputs: none written under `results/`; the report goes to stdout and the
scratch directory is removed on exit unless --keep is given.

Exit codes:
  0  every published number reproduces within tolerance
  1  some number disagrees, or the repository publishes no results to check
  2  the pipeline itself failed to run

Typical use, from the repository root with .venv active:
  python scripts/check_reproducibility.py
  python scripts/check_reproducibility.py --keep      (inspect the fresh run)
  python scripts/check_reproducibility.py --rtol 1e-8
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib_muon import provenance  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# Keys whose difference between runs carries no information about
# reproducibility: timestamps, hashes, platform.
IGNORED_KEYS = {"provenance"}

# A product filename differing only in its _vN suffix is the versioning working
# as designed, not a result that changed.
VERSIONED_NAME = re.compile(r"^(?P<base>.+)_v\d+(?P<extension>\.[a-z]+)$")

# Each step, with the arguments that redirect all of its output into the
# scratch directory. Order is the pipeline order.
def pipeline_steps(scratch: Path) -> list[tuple[str, list[str]]]:
    tables = str(scratch / "tables")
    fits = str(scratch / "fits")
    figures = str(scratch / "figures")
    data = str(ROOT / "data" / "raw")
    return [
        ("00_ingest.py", ["--data-dir", data, "--out", tables]),
        ("01_validation.py", ["--data-dir", data, "--out", tables]),
        ("02_rates.py", ["--data-dir", data, "--out", tables]),
        ("03_timeseries.py", ["--data-dir", data, "--out-tables", tables,
                              "--out-figures", figures]),
        ("04_normalization.py", ["--tables", tables, "--out-figures", figures]),
        ("05_fits.py", ["--tables", tables, "--out-fits", fits,
                        "--out-figures", figures]),
        ("06_systematics.py", ["--data-dir", data, "--tables", tables,
                               "--fits", fits, "--out-figures", figures]),
        ("90_report_figures.py", ["--tables", tables, "--fits", fits,
                                  "--out-figures", figures]),
    ]


# Committed product location against its location in the scratch run.
PRODUCTS = [
    ("results/tables", "tables", "00_ingest"),
    ("results/tables", "tables", "01_validation"),
    ("results/tables", "tables", "02_rates"),
    ("results/tables", "tables", "03_timeseries"),
    ("results/tables", "tables", "04_normalization"),
    ("results/fits", "fits", "05_fits"),
    ("results/tables", "tables", "06_systematics"),
    ("results/tables", "tables", "90_report_figures"),
]


def same_but_for_version(a: str, b: str) -> bool:
    """True if two strings are the same product name at different versions."""
    match_a, match_b = VERSIONED_NAME.match(a), VERSIONED_NAME.match(b)
    if not match_a or not match_b:
        return False
    return (match_a.group("base") == match_b.group("base")
            and match_a.group("extension") == match_b.group("extension"))


def compare(committed, fresh, rtol, atol, path=""):
    """Walk two JSON trees together and return a list of mismatches."""
    problems = []

    if isinstance(committed, dict) and isinstance(fresh, dict):
        for key in sorted(set(committed) | set(fresh)):
            if key in IGNORED_KEYS:
                continue
            if key not in committed:
                problems.append(f"{path}/{key}: only in the fresh run")
            elif key not in fresh:
                problems.append(f"{path}/{key}: only in the committed run")
            else:
                problems += compare(committed[key], fresh[key], rtol, atol,
                                    f"{path}/{key}")
        return problems

    if isinstance(committed, list) and isinstance(fresh, list):
        if len(committed) != len(fresh):
            return [f"{path}: length {len(committed)} against {len(fresh)}"]
        for i, (a, b) in enumerate(zip(committed, fresh)):
            problems += compare(a, b, rtol, atol, f"{path}[{i}]")
        return problems

    if isinstance(committed, bool) or isinstance(fresh, bool):
        if committed != fresh:
            problems.append(f"{path}: {committed} against {fresh}")
        return problems

    if isinstance(committed, (int, float)) and isinstance(fresh, (int, float)):
        difference = abs(committed - fresh)
        if difference > atol + rtol * abs(committed):
            problems.append(
                f"{path}: {committed!r} against {fresh!r} "
                f"(difference {difference:.3e})"
            )
        return problems

    if isinstance(committed, str) and isinstance(fresh, str):
        if committed != fresh and not same_but_for_version(committed, fresh):
            problems.append(f"{path}: {committed!r} against {fresh!r}")
        return problems

    if committed != fresh:
        problems.append(f"{path}: {committed!r} against {fresh!r}")
    return problems


def run_pipeline(scratch: Path) -> int:
    """Run the whole pipeline into `scratch`. Returns 0 on success."""
    print(f"[check] running the pipeline into {scratch}")
    for script, arguments in pipeline_steps(scratch):
        process = subprocess.run(
            [sys.executable, str(SCRIPTS / script), *arguments],
            cwd=ROOT, capture_output=True, text=True, timeout=600,
        )
        if process.returncode != 0:
            print(f"[check] {script} exited with {process.returncode}",
                  file=sys.stderr)
            print(process.stdout[-2000:], file=sys.stderr)
            print(process.stderr[-2000:], file=sys.stderr)
            return process.returncode
        print(f"[check]   {script} ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check that the committed results reproduce here."
    )
    ap.add_argument("--rtol", type=float, default=1e-6,
                    help="relative tolerance (default 1e-6). Loose enough for "
                         "the optimiser to land on a slightly different point "
                         "of a flat minimum, tight enough that any real change "
                         "in a result shows up.")
    ap.add_argument("--atol", type=float, default=1e-12,
                    help="absolute tolerance (default 1e-12)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the scratch directory instead of removing it")
    args = ap.parse_args()

    committed_products = [
        (Path(directory), base) for directory, _, base in PRODUCTS
    ]
    missing = [
        f"{directory}/{base}_v1.json"
        for directory, base in committed_products
        if provenance.latest_version(ROOT / directory, base) is None
    ]
    if missing:
        print("[check] This repository publishes no results to check against.")
        for name in missing:
            print(f"[check]   missing: {name}")
        print("[check] Nothing was compared, so nothing can be concluded about")
        print("[check] reproducibility. Run the pipeline and commit its output.")
        return 1

    scratch = Path(tempfile.mkdtemp(prefix="muon-repro-"))
    try:
        failure = run_pipeline(scratch)
        if failure:
            print()
            print("[check] The pipeline did not run to completion, so the "
                  "committed results")
            print("[check] were NOT checked. This says nothing about whether "
                  "they reproduce.")
            return 2

        print()
        total_problems = 0
        compared = 0

        for committed_directory, scratch_subdirectory, base in PRODUCTS:
            committed_path = ROOT / committed_directory / f"{base}_v1.json"
            fresh_directory = scratch / scratch_subdirectory
            fresh_version = provenance.latest_version(fresh_directory, base)
            if fresh_version is None:
                print(f"[check] {base}: the fresh run produced no product")
                total_problems += 1
                continue

            fresh_path = fresh_directory / f"{base}_v{fresh_version}.json"
            problems = compare(
                json.loads(committed_path.read_text()),
                json.loads(fresh_path.read_text()),
                args.rtol, args.atol,
            )
            compared += 1

            if problems:
                print(f"[check] {base}: {len(problems)} MISMATCH(ES)")
                for problem in problems[:20]:
                    print(f"          {problem}")
                if len(problems) > 20:
                    print(f"          ... and {len(problems) - 20} more")
                total_problems += len(problems)
            else:
                print(f"[check] {base}: reproduces exactly")

        print()
        if total_problems:
            print(f"[check] FAILED: {total_problems} disagreement(s) across "
                  f"{compared} product(s) compared.")
            print("[check] The results committed to the repository do not "
                  "reproduce here. Either")
            print("[check] the analysis changed and results/ needs "
                  "regenerating, or it depends on")
            print("[check] something about the machine that produced it. Both "
                  "matter.")
            return 1

        print(f"[check] OK: {compared} product(s) compared, all reproduce "
              f"within rtol={args.rtol:g}, atol={args.atol:g}.")
        print("[check] Every number published in results/ was regenerated here "
              "and matches.")
        return 0
    finally:
        if args.keep:
            print(f"[check] scratch directory kept at {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
