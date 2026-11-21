"""End-to-end test: the whole pipeline on the real data.

It fits in CI because the raw data are versioned: 256 kB for the 2025 angular
runs and 33 MB for the 2026 altitude campaign, about half a minute of
pipeline. It checks that the ten analysis scripts run in order, that each
writes what its docstring promises, and that the numbers the repository
publishes come out of that run and not from somewhere else.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

PIPELINE = [
    ("00_ingest.py", 0),
    ("01_validation.py", 0),
    ("02_rates.py", 0),
    ("03_timeseries.py", 0),
    ("04_normalization.py", 0),
    ("05_fits.py", 0),
    ("06_systematics.py", 0),
    ("07_detector_health.py", 0),
    ("08_angular_reading.py", 0),
    ("09_altitude.py", 0),
]


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    """Run the whole pipeline in a clean directory."""
    workdir = tmp_path_factory.mktemp("pipeline")
    shutil.copytree(ROOT / "data", workdir / "data")
    shutil.copytree(ROOT / "scripts", workdir / "scripts")

    outputs = {}
    for script, expected in PIPELINE:
        process = subprocess.run(
            [PYTHON, f"scripts/{script}"],
            cwd=workdir, capture_output=True, text=True, timeout=300,
        )
        assert process.returncode == expected, (
            f"{script} exited with {process.returncode}, expected {expected}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
        outputs[script] = process.stdout
    return workdir, outputs


def read(workdir, subdirectory, base):
    path = workdir / "results" / subdirectory / f"{base}_v1.json"
    assert path.is_file(), f"missing {path}"
    return json.loads(path.read_text())


class TestThePipelineRunsThrough:
    def test_all_ten_scripts_finish_cleanly(self, full_run):
        _, outputs = full_run
        assert len(outputs) == len(PIPELINE)

    def test_each_step_writes_its_product(self, full_run):
        workdir, _ = full_run
        for base in ("00_ingest", "01_validation", "02_rates", "03_timeseries",
                     "04_normalization", "06_systematics", "07_detector_health",
                     "08_angular_reading", "09_altitude"):
            assert (workdir / "results" / "tables" / f"{base}_v1.json").is_file()
            assert (workdir / "results" / "tables" / f"{base}_v1.txt").is_file()
        assert (workdir / "results" / "fits" / "05_fits_v1.json").is_file()

    def test_figures_come_paired_in_pdf_and_png(self, full_run):
        workdir, _ = full_run
        figures = workdir / "results" / "figures"
        pdfs = sorted(p.stem for p in figures.glob("*.pdf"))
        pngs = sorted(p.stem for p in figures.glob("*.png"))
        assert pdfs == pngs, "every figure must exist as both PDF and PNG"
        assert len(pdfs) >= 12


class TestThePublishedNumbers:
    """Anchors the figures the README and the report cite."""

    def test_event_count(self, full_run):
        workdir, _ = full_run
        assert read(workdir, "tables", "00_ingest")["n_events_total"] == 4760

    def test_a_single_malformed_line(self, full_run):
        workdir, _ = full_run
        assert read(workdir, "tables", "00_ingest")["n_rejected_lines_total"] == 1

    def test_pure_cosn_has_null_likelihood(self, full_run):
        """The central result of the analysis: it does not fit badly, its
        likelihood is zero. In the JSON that is written as null."""
        workdir, _ = full_run
        fits = read(workdir, "fits", "05_fits")["fits"]
        for key in ("B_free_cosn", "C_fixed_cos2"):
            assert fits[key]["nll"] is None
            assert fits[key]["nll_infinite"] is True
            assert fits[key]["aic"] is None
        assert fits["A_constant"]["nll"] is not None

    def test_the_cosine_of_90_is_clamped_to_zero(self, full_run):
        workdir, _ = full_run
        assert read(workdir, "fits", "05_fits")["clamped_cosine"][-1] == 0.0

    def test_the_modulation_does_not_survive_the_systematic(self, full_run):
        """The published conclusion. If this fails, the conclusion changed."""
        workdir, _ = full_run
        systematics = read(workdir, "tables", "06_systematics")
        critical = systematics["critical_systematic_for_significance"]
        measured = systematics["dominant_systematic"]
        assert measured > critical, (
            "the measured systematic no longer exceeds the one that cancels the "
            "significance: the repository's conclusion would have changed"
        )

    def test_there_is_an_upper_bound_on_the_modulated_fraction(self, full_run):
        workdir, _ = full_run
        systematics = read(workdir, "tables", "06_systematics")
        bound = systematics["modulated_fraction_bound_with_systematic"]
        assert 0.0 < bound < 1.0
        assert bound > systematics["modulated_fraction_bound_statistics_only"], (
            "folding in a systematic cannot tighten the bound"
        )


class TestTheDetectorHealthNumbers:
    """Anchors for step 07: what the fault looks like and what it does not touch."""

    def test_every_fault_episode_is_vetoed(self, full_run):
        workdir, _ = full_run
        health = read(workdir, "tables", "07_detector_health")
        assert len(health["episodes"]) == 10
        assert all(e["all_minutes_vetoed"] for e in health["episodes"])
        assert health["unvetoed_episode_minutes"] == 0
        assert health["deviations"] == []

    def test_cleaning_makes_tacachico_constant(self, full_run):
        workdir, _ = full_run
        validation = read(workdir, "tables", "07_detector_health")["validation_tacachico"]
        assert validation["raw"]["chi2"] / validation["raw"]["dof"] > 1000
        clean = validation["cuts"]["260"]
        assert clean["chi2"] / clean["dof"] < 2.0
        assert clean["excess_fraction"] < 0.02

    def test_no_2025_run_was_taken_in_the_fault_state(self, full_run):
        """The published reading of the 2025 runs rests on this."""
        workdir, _ = full_run
        runs = read(workdir, "tables", "07_detector_health")["runs_2025"]
        assert len(runs) == 8
        for angle, d in runs.items():
            assert d["vetoed_minutes"] == 0 and d["burst_events"] == 0, angle

    def test_sipm_is_derived(self, full_run):
        workdir, _ = full_run
        assert read(workdir, "tables", "07_detector_health")["sipm"]["max_spread_mv"] == 0.0


class TestTheAngularReadingNumbers:
    def test_the_record_holds_singles(self, full_run):
        workdir, _ = full_run
        singles = read(workdir, "tables", "08_angular_reading")["singles"]
        assert singles["coincident_flag_fraction"] == 1.0
        assert singles["max_true_coincidence_fraction"] < 0.2

    def test_cut_zero_reproduces_the_published_profile(self, full_run):
        workdir, _ = full_run
        reading = read(workdir, "tables", "08_angular_reading")
        assert reading["mismatch_with_step_02"] == []
        assert reading["profiles"]["0"]["relative"]["90"][0] == pytest.approx(0.8056, abs=1e-4)

    def test_the_single_slab_reference(self, full_run):
        workdir, _ = full_run
        g = read(workdir, "tables", "08_angular_reading")["slab_response"]["1.0"]
        assert g[-1] == pytest.approx(2 / 3, abs=1e-5)


class TestTheAltitudeNumbers:
    """The numbers quoted at the International Cosmic Day 2026."""

    def test_counts_and_rates(self, full_run):
        workdir, _ = full_run
        sites = read(workdir, "tables", "09_altitude")["sites"]
        assert sites["tacachico"]["n_events"] == 72_755
        assert sites["santa_tecla"]["n_events"] == 200_858
        assert sites["tacachico"]["live_rate_hz"] == pytest.approx(1.587, abs=5e-4)
        assert sites["santa_tecla"]["live_rate_hz"] == pytest.approx(1.917, abs=5e-4)

    def test_ratio_and_its_errors(self, full_run):
        workdir, _ = full_run
        presented = read(workdir, "tables", "09_altitude")["presented"]
        assert presented["ratio"] == pytest.approx(0.828, abs=5e-4)
        assert presented["ratio_stat"] == pytest.approx(0.004, abs=5e-4)
        assert presented["ratio_sys_hours"] == pytest.approx(0.038, abs=5e-4)

    def test_height_difference_and_gamma_min(self, full_run):
        workdir, _ = full_run
        presented = read(workdir, "tables", "09_altitude")["presented"]
        assert presented["delta_h_m"] == pytest.approx(603, abs=0.5)
        assert presented["delta_h_unc_m"] == pytest.approx(28, abs=0.5)
        assert presented["gamma_min"] == pytest.approx(4.8, abs=0.05)
        assert presented["gamma_min_95_lower"] == pytest.approx(3.2, abs=0.05)
        assert presented["prediction_classical"] == pytest.approx(0.40, abs=5e-3)
        assert presented["prediction_relativistic"] == pytest.approx(0.976, abs=5e-4)

    def test_internal_checks_pass(self, full_run):
        workdir, _ = full_run
        assert all(c["ok"] for c in read(workdir, "tables", "09_altitude")["checks"])

    def test_cleaning_shrinks_the_hour_systematic(self, full_run):
        workdir, _ = full_run
        altitude = read(workdir, "tables", "09_altitude")
        assert (altitude["cleaned"]["ratio_sys_hours"]
                < altitude["presented"]["ratio_sys_hours"] / 3)


class TestReproducibility:
    def test_products_carry_a_provenance_stamp(self, full_run):
        """The pipeline runs here outside a git repository, so the commit is
        recorded as null. That is the correct behaviour: say it is unknown,
        do not invent a hash."""
        workdir, _ = full_run
        stamp = read(workdir, "fits", "05_fits")["provenance"]
        assert "commit" in stamp["git"]
        assert stamp["git"]["commit"] is None
        assert stamp["parameters"] != {}
        assert stamp["seed"] is not None
        assert stamp["numpy"] and stamp["python"]

    def test_inside_the_repository_the_commit_is_captured(self):
        """Complementary check: in the real repository the stamp carries the
        hash and the state of the tree."""
        raw = subprocess.run(
            [PYTHON, "-c",
             "import sys; sys.path.insert(0, 'scripts');"
             "from lib_muon import provenance; import json;"
             "print(json.dumps(provenance.commit_hash()))"],
            cwd=ROOT, capture_output=True, text=True, timeout=60, check=True,
        ).stdout
        data = json.loads(raw)
        assert data["commit"] is not None and len(data["commit"]) == 40
        assert data["inputs_clean"] in (True, False)

    def test_a_second_run_does_not_clobber_the_first(self, full_run):
        workdir, _ = full_run
        subprocess.run([PYTHON, "scripts/02_rates.py"], cwd=workdir,
                       capture_output=True, text=True, timeout=120, check=True)
        tables = workdir / "results" / "tables"
        assert (tables / "02_rates_v1.json").is_file()
        assert (tables / "02_rates_v2.json").is_file()
