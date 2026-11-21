"""Smoke test for muon-flux-analysis.

Checks the only thing checkable without the pipeline: that the raw files are
where the pipeline expects them and are not empty. The real checks live in the
other files under tests/.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANGLES = [0, 5, 15, 25, 35, 45, 90, 360]


def test_the_eight_raw_files_exist():
    for angle in ANGLES:
        path = ROOT / "data" / "raw" / f"{angle:03d}deg"
        assert path.is_file(), f"missing raw file {path}"
        assert path.stat().st_size > 0, f"empty raw file: {path}"


def test_the_two_2026_files_exist():
    for name in ("tacachico", "santa-tecla.txt"):
        path = ROOT / "data" / "raw" / "altitude_2026" / name
        assert path.is_file(), f"missing raw file {path}"
        assert path.stat().st_size > 1_000_000, f"truncated raw file: {path}"
