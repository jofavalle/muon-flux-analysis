"""Traceability tests: _vN versioning and product serialisation."""

import json
from pathlib import Path

import numpy as np
import pytest

from lib_muon import provenance

ROOT = Path(__file__).resolve().parents[1]


class TestVersioning:
    def test_the_first_version_is_1(self, tmp_path):
        assert provenance.next_version(tmp_path, "07_probe") == 1
        assert provenance.latest_version(tmp_path, "07_probe") is None

    def test_no_run_overwrites_the_previous_one(self, tmp_path):
        for expected in (1, 2, 3):
            json_path, text_path = provenance.write_product(
                tmp_path, "07_probe", data={"value": expected}, text="x"
            )
            assert json_path.name == f"07_probe_v{expected}.json"
            assert text_path.name == f"07_probe_v{expected}.txt"
        # All three still exist with their original contents.
        for expected in (1, 2, 3):
            content = json.loads((tmp_path / f"07_probe_v{expected}.json").read_text())
            assert content["value"] == expected

    def test_read_product_returns_the_most_recent(self, tmp_path):
        for value in (10, 20, 30):
            provenance.write_product(tmp_path, "07_probe", data={"value": value},
                                     text="x")
        assert provenance.read_product(tmp_path, "07_probe")["value"] == 30

    def test_a_missing_product_names_the_step_to_run(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Run pipeline step"):
            provenance.read_product(tmp_path, "02_rates")


class TestSerialisation:
    def test_numpy_types_are_written(self, tmp_path):
        data = {
            "integer": np.int64(7),
            "floating": np.float64(1.5),
            "array": np.array([1.0, 2.0, 3.0]),
            "boolean": np.bool_(True),
        }
        json_path, _ = provenance.write_product(tmp_path, "07_probe", data=data,
                                                text="x")
        loaded = json.loads(json_path.read_text())
        assert loaded["integer"] == 7
        assert loaded["floating"] == 1.5
        assert loaded["array"] == [1.0, 2.0, 3.0]
        assert loaded["boolean"] is True

    def test_infinity_and_nan_are_written_as_null(self, tmp_path):
        """JSON has no Infinity and no NaN. A fit with no finite value is null,
        not a zero: the difference matters, because an infinite -logL IS the
        result for the pure cos^n model."""
        data = {
            "infinite_nll": float("inf"),
            "no_value": float("nan"),
            "nested": {"list": [1.0, float("inf"), 3.0]},
            "array": np.array([np.inf, 1.0]),
        }
        json_path, _ = provenance.write_product(tmp_path, "07_probe", data=data,
                                                text="x")
        raw = json_path.read_text()
        assert "Infinity" not in raw and "NaN" not in raw
        loaded = json.loads(raw)
        assert loaded["infinite_nll"] is None
        assert loaded["no_value"] is None
        assert loaded["nested"]["list"] == [1.0, None, 3.0]
        assert loaded["array"] == [None, 1.0]


class TestPortablePaths:
    """Committed products must not carry absolute paths.

    Two reasons: an absolute path makes a product depend on where the
    repository sits, so two correct runs disagree for no reason; and these
    products are published, so an absolute path leaks the directory layout and
    username of whoever ran the pipeline.
    """

    def test_a_path_inside_the_repository_becomes_relative(self):
        absolute = ROOT / "data" / "raw" / "000deg"
        assert provenance.portable_path(absolute) == "data/raw/000deg"

    def test_an_outside_path_is_reduced_to_its_filename(self, tmp_path):
        outside = tmp_path / "secret_directory" / "000deg"
        outside.parent.mkdir(parents=True)
        outside.write_text("x")
        assert provenance.portable_path(outside) == "000deg"

    def test_a_relative_path_stays_relative(self):
        assert provenance.portable_path("data/raw/090deg") == "data/raw/090deg"

    def test_directory_parameters_are_normalised_in_the_stamp(self):
        stamp = provenance.stamp({"data_dir": str(ROOT / "data" / "raw")})
        assert stamp["parameters"]["data_dir"] == "data/raw"

    def test_non_path_parameters_are_left_alone(self):
        stamp = provenance.stamp({"bin_seconds": 10.0, "strict": False,
                                  "label": "not a path"})
        assert stamp["parameters"]["bin_seconds"] == 10.0
        assert stamp["parameters"]["strict"] is False
        assert stamp["parameters"]["label"] == "not a path"

    def test_no_committed_product_contains_an_absolute_path(self):
        """Guards the published files themselves, not just the helper."""
        results = ROOT / "results"
        if not results.is_dir():
            pytest.skip("no results/ committed in this working tree")
        offenders = []
        for path in sorted(results.rglob("*.json")):
            text = path.read_text()
            for marker in ('"/home/', '"/Users/', '"C:\\\\', '"/tmp/'):
                if marker in text:
                    offenders.append(f"{path.name} contains {marker}")
        assert not offenders, offenders


class TestProvenanceStamp:
    def test_every_product_carries_commit_and_parameters(self, tmp_path):
        json_path, text_path = provenance.write_product(
            tmp_path, "07_probe", data={"value": 1}, text="content",
            parameters={"bin_seconds": 10.0},
        )
        stamp = json.loads(json_path.read_text())["provenance"]
        assert "git" in stamp and "commit" in stamp["git"]
        assert "inputs_clean" in stamp["git"]
        assert stamp["parameters"]["bin_seconds"] == 10.0
        assert stamp["seed"] == provenance.DEFAULT_SEED
        assert "numpy" in stamp and "python" in stamp
        # The txt repeats the commit in its header so it can be read without json.
        assert "commit:" in text_path.read_text()
