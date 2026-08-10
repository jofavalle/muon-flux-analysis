"""Tests of raw file reading.

They run on the repository's real data: it is 256 kB and it is versioned
precisely so that this is possible.
"""

from pathlib import Path

import numpy as np
import pytest

from lib_muon import io

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ANGLES = [0, 5, 15, 25, 35, 45, 90, 360]


class TestLineParsing:
    def test_valid_line(self):
        line = "1\t83\t201\t3741\t2.6\t73157.02\t23.13\t1411\t1\tCOSMIC"
        fields = io.parse_line(line)
        assert fields[0] == 1
        assert fields[1] == 83
        assert fields[5] == pytest.approx(73157.02)
        assert fields[6] == pytest.approx(23.13)
        assert fields[8] == 1
        assert fields[9] == "COSMIC"

    def test_space_separation_also_works(self):
        line = "1 83 201 3741 2.6 93609.2 27.8 1411 1 COSMIC"
        assert io.parse_line(line)[0] == 1

    @pytest.mark.parametrize(
        "bad",
        [
            " 118  | Hexa address:  0x76",   # the one in data/raw/015deg, verbatim
            "1\t83\t201",
            "",
            "these are not numbers at all not one of them anywhere here zero",
        ],
    )
    def test_malformed_lines_raise(self, bad):
        with pytest.raises(ValueError):
            io.parse_line(bad)

    def test_extra_columns_also_raise(self):
        """The original script accepted >= 10 columns and kept the first ten,
        so a line with junk appended came through as good."""
        with pytest.raises(ValueError):
            io.parse_line("1 83 201 3741 2.6 93609.2 27.8 1411 1 COSMIC extra")


class TestReadingRealFiles:
    def test_all_eight_files_are_read(self):
        assert sorted(io.load_all(RAW)) == ANGLES

    def test_total_event_count(self):
        runs = io.load_all(RAW)
        assert sum(len(r) for r in runs.values()) == 4760

    def test_015deg_has_exactly_one_rejected_line(self):
        """The file is truncated at the top: it lost the first lines of the
        preamble and the fourth was cut, so its first line is the fragment
        ' 118  | Hexa address:  0x76'. The original script dropped it silently
        with `except Exception: continue`.

        The column-name line did survive the truncation, so the file reads
        unambiguously; what was lost is the sensor preamble.
        """
        run = io.load_run(RAW / "015deg")
        assert run.quality.rejected_lines == 1
        assert run.quality.rejected_texts == ["118  | Hexa address:  0x76"]
        assert run.quality.header_present

    def test_no_other_file_has_rejected_lines(self):
        for angle in ANGLES:
            if angle == 15:
                continue
            run = io.load_run(RAW / f"{angle:03d}deg")
            assert run.quality.rejected_lines == 0, (
                f"{angle} degrees has unexpected rejected lines"
            )

    def test_the_column_header_is_in_all_eight(self):
        for angle in ANGLES:
            assert io.load_run(RAW / f"{angle:03d}deg").quality.header_present


class TestColumnOrder:
    def test_columns_are_swapped_in_all_eight_files(self):
        """The header declares Temp[C] Pressure[Pa] and they arrive reversed."""
        for angle in ANGLES:
            run = io.load_run(RAW / f"{angle:03d}deg")
            assert run.quality.columns_swapped, (
                f"{angle} degrees: expected pressure before temperature"
            )

    def test_pressure_and_temperature_land_in_physical_range(self):
        for angle in ANGLES:
            run = io.load_run(RAW / f"{angle:03d}deg")
            assert io.PRESSURE_RANGE_PA[0] < np.median(run.pressure_pa) < io.PRESSURE_RANGE_PA[1]
            assert io.TEMPERATURE_RANGE_C[0] < np.median(run.temperature_c) < io.TEMPERATURE_RANGE_C[1]
            assert run.quality.out_of_range_rows == 0

    def test_it_refuses_to_guess_when_neither_column_fits(self):
        with pytest.raises(ValueError, match="cannot decide"):
            io._decide_column_order(np.array([5.0, 5.0]), np.array([7.0, 7.0]))


class TestSensorStartup:
    def test_count_per_file(self):
        """The BMP280 emits 73157.02 Pa / 23.13 C before settling."""
        expected = {0: 2, 5: 2, 15: 0, 25: 2, 35: 1, 45: 0, 90: 0, 360: 1}
        for angle, how_many in expected.items():
            run = io.load_run(RAW / f"{angle:03d}deg")
            assert run.quality.sensor_startup_rows == how_many, (
                f"{angle} degrees: expected {how_many} startup rows"
            )

    def test_always_at_the_start_of_the_run(self):
        for angle in ANGLES:
            run = io.load_run(RAW / f"{angle:03d}deg")
            indices = run.quality.sensor_startup_indices
            if indices:
                assert max(indices) < 10, (
                    f"{angle} degrees: sensor startup outside the beginning"
                )


class TestTimes:
    def test_the_clock_never_runs_backwards(self):
        """Justifies using the plain difference tN - t1 rather than
        accumulating positive steps, which is what the original script did."""
        for angle in ANGLES:
            run = io.load_run(RAW / f"{angle:03d}deg")
            assert np.all(np.diff(run.timestamp_ms) > 0), (
                f"{angle} degrees: the clock goes back or repeats"
            )

    def test_dead_fraction_in_the_expected_band(self):
        for angle in ANGLES:
            run = io.load_run(RAW / f"{angle:03d}deg")
            assert 0.085 < run.dead_fraction < 0.10, (
                f"{angle} degrees: dead fraction {run.dead_fraction:.4f} outside "
                "the 9 % characteristic of this set"
            )

    def test_every_run_fits_within_the_600_s_window(self):
        """The --max-ms of 600 s in the original script cut nothing: the
        longest run lasts 444 s."""
        for angle in ANGLES:
            assert io.load_run(RAW / f"{angle:03d}deg").window_s < 600.0


class TestCoincidence:
    def test_the_coincidence_filter_is_a_no_op(self):
        """Coincident == 1 for all 4760 events. With no singles recorded, the
        accidental rate is not estimable from this data."""
        for angle, run in io.load_all(RAW).items():
            assert np.all(run.coincident == 1), (
                f"{angle} degrees has non-coincident events"
            )
