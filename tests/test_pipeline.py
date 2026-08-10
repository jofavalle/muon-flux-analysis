"""End-to-end test: the whole pipeline on the real data.

It fits in CI because the data set is 256 kB, which is exactly why the raw data
are versioned. It checks that the seven scripts run in order, that each writes
what its docstring promises, and that the numbers the repository publishes come
out of that run and not from somewhere else.
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
    def test_all_seven_scripts_finish_cleanly(self, full_run):
        _, outputs = full_run
        assert len(outputs) == len(PIPELINE)

    def test_each_step_writes_its_product(self, full_run):
        workdir, _ = full_run
        for base in ("00_ingest", "01_validation", "02_rates", "03_timeseries",
                     "04_normalization", "06_systematics"):
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
