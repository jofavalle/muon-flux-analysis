"""Tests of the reader for the long, multi-acquisition files of 2026.

Most run on the real files of data/raw/altitude_2026. The counts pinned here
are the ones quoted in the analysis presented at the International Cosmic Day
2026, so a change in the acceptance rule shows up as a failure, not as a
silent shift of every number downstream.
"""

from pathlib import Path

import numpy as np
import pytest

from lib_muon import acquisition

ROOT = Path(__file__).resolve().parents[1]
ALTITUDE = ROOT / "data" / "raw" / "altitude_2026"
SHA256 = {
    "tacachico": "0e6d19ef771f526ea3496246dad9a70a8a8e8f7bafb217a77c8e89dfe9a67d3e",
    "santa-tecla.txt": "f255bfeb496e12f30fb2f2cc7b93cb78909ec5e07dfa91c1eac7fc413e35019c",
}


@pytest.fixture(scope="module")
def tacachico():
    return acquisition.load_acquisitions(ALTITUDE / "tacachico")


@pytest.fixture(scope="module")
def santa_tecla():
    return acquisition.load_acquisitions(ALTITUDE / "santa-tecla.txt")


def write(tmp_path, text):
    path = tmp_path / "run"
    path.write_text(text)
    return path


ROW = "{n}\t{t}\t100\t900\t0.7\t97000.0\t30.0\t5600\t0\tCOSMIC"


class TestRulesOnSyntheticFiles:
    def test_a_counter_reset_starts_a_new_acquisition(self, tmp_path):
        rows = [ROW.format(n=n, t=1000 * n) for n in (1, 2, 3, 1, 2)]
        report = acquisition.load_acquisitions(write(tmp_path, "\n".join(rows)))
        assert [len(a) for a in report.acquisitions] == [3, 2]

    def test_a_glued_comment_is_cut_and_the_row_kept(self, tmp_path):
        rows = [ROW.format(n=1, t=1000), ROW.format(n=2, t=2000) + "# Temperature 23.13C"]
        report = acquisition.load_acquisitions(write(tmp_path, "\n".join(rows)))
        assert report.n_rows == 2
        assert report.glued_comment_lines == [2]
        assert report.rejected_lines == 0

    def test_incomplete_and_fused_rows_are_rejected_and_kept_as_text(self, tmp_path):
        rows = [ROW.format(n=1, t=1000), "7\t0\tCOSMIC",
                ROW.format(n=2, t=2000) + "\t" + ROW.format(n=3, t=3000),
                ROW.format(n=4, t=4000)]
        report = acquisition.load_acquisitions(write(tmp_path, "\n".join(rows)))
        assert report.n_rows == 2
        assert report.rejected_lines == 2
        assert [n for n, _ in report.rejected_texts] == [2, 3]

    def test_the_name_field_must_start_with_cosmic(self, tmp_path):
        rows = [ROW.format(n=1, t=1000), ROW.format(n=2, t=2000).replace("COSMIC", "X")]
        report = acquisition.load_acquisitions(write(tmp_path, "\n".join(rows)))
        assert report.n_rows == 1 and report.rejected_lines == 1

    def test_window_is_half_open(self, tmp_path):
        rows = [ROW.format(n=n, t=1000 * n) for n in (1, 2, 3, 4)]
        acq = acquisition.load_acquisitions(write(tmp_path, "\n".join(rows))).acquisitions[0]
        assert list(acq.window(2.0, 4.0).event) == [2, 3]


class TestTacachico:
    def test_checksum(self, tacachico):
        assert tacachico.sha256 == SHA256["tacachico"]

    def test_three_acquisitions_with_their_row_counts(self, tacachico):
        assert [len(a) for a in tacachico.acquisitions] == [382_709, 4_330, 2_491]
        assert tacachico.n_rows == 389_530

    def test_exactly_two_rejected_lines(self, tacachico):
        assert tacachico.rejected_lines == 2
        assert [n for n, _ in tacachico.rejected_texts] == [1, 385_764]
        assert tacachico.rejected_texts[1][1] == "7\t0\tCOSMIC"

    def test_the_last_event_of_acquisition_one_keeps_its_row(self, tacachico):
        assert tacachico.glued_comment_lines == [382_717]
        assert tacachico.acquisitions[0].event[-1] == 382_709

    def test_acquisition_two_lost_125_events(self, tacachico):
        assert tacachico.acquisitions[1].counter_jumps() == [(3036, 3162)]
        assert tacachico.acquisitions[0].counter_jumps() == []

    def test_columns_are_swapped_and_physical(self, tacachico):
        assert tacachico.columns_swapped
        a = tacachico.acquisitions[0]
        assert 97_000 < np.median(a.pressure_pa) < 99_000
        assert 25 < np.median(a.temperature_c) < 45

    def test_the_first_acquisition_lasts_40_9_hours(self, tacachico):
        assert tacachico.acquisitions[0].time_s[-1] / 3600 == pytest.approx(40.9, abs=0.05)


class TestSantaTecla:
    def test_checksum(self, santa_tecla):
        assert santa_tecla.sha256 == SHA256["santa-tecla.txt"]

    def test_one_acquisition_of_200859_rows(self, santa_tecla):
        assert len(santa_tecla.acquisitions) == 1
        assert santa_tecla.n_rows == 200_859

    def test_the_file_starts_in_the_middle_of_a_row(self, santa_tecla):
        """Its first line is the tail of event 484 (read as 84) and its second
        fuses events 485 and 486; the counter therefore jumps from 84 to 487."""
        assert santa_tecla.rejected_lines == 1
        assert santa_tecla.rejected_texts[0][0] == 2
        assert santa_tecla.acquisitions[0].counter_jumps() == [(84, 487)]
