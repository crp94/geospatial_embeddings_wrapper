"""Focused contract tests for the point-query CLI helpers."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts import get_embeddings
from wrappers.embedding_encoder import GeoEmbeddingEncoder


class DummyEncoder:
    """Small registry stand-in that avoids loading external model dependencies."""

    def __init__(self, device=None, data_root=None):
        self.device = device
        self.data_root = data_root
        self.last_coordinates = None
        self.last_year = None

    def encode(self, coordinates, year=None):
        self.last_coordinates = coordinates.clone()
        self.last_year = year
        return torch.cat((coordinates, coordinates[:, :1] + coordinates[:, 1:2]), dim=1)

    def get_metadata(self):
        return {"name": "dummy", "embedding_dim": 3, "input_coordinate_order": "lat_lon"}


class DummyYearEncoder(GeoEmbeddingEncoder):
    """Registry stand-in whose third feature is the year it was asked for."""

    def __init__(self, device=None, data_root=None):
        super().__init__(device="cpu")
        self.calls: list = []

    def encode(self, coordinates, year=None):
        self.calls.append(year)
        year_column = torch.full_like(coordinates[:, :1], 0.0 if year is None else float(year))
        return torch.cat((coordinates, year_column), dim=1)

    def get_embedding_dim(self):
        return 3

    def is_temporal(self):
        return True


class GetEmbeddingsTests(unittest.TestCase):
    def _generator(self, names=None, roots=None):
        with patch.object(get_embeddings, "get_encoder_class", return_value=DummyEncoder):
            return get_embeddings.EmbeddingGenerator(
                encoders=names or ["torchspatial_direct"],
                device="cpu",
                encoder_roots=roots,
            )

    def test_registry_alias_is_normalized_and_root_is_passed_to_encoder(self):
        generator = self._generator(names=["direct"], roots={"torchspatial_direct": "/tmp/data"})

        self.assertEqual(list(generator.encoders), ["torchspatial_direct"])
        self.assertEqual(generator.encoders["torchspatial_direct"].data_root, "/tmp/data")

    def test_generation_validates_coordinates_and_forwards_year(self):
        generator = self._generator()
        embeddings = generator.generate_embeddings([(40.0, -3.0)], year=2024)
        encoder = generator.encoders["torchspatial_direct"]

        self.assertEqual(tuple(embeddings["torchspatial_direct"].shape), (1, 3))
        self.assertEqual(encoder.last_year, 2024)
        self.assertEqual(encoder.last_coordinates.dtype, torch.float32)
        with self.assertRaisesRegex(ValueError, "Latitude"):
            generator.generate_embeddings([(91.0, 0.0)])
        with self.assertRaisesRegex(ValueError, "finite"):
            get_embeddings.validate_coordinates([(float("nan"), 0.0)])

    def test_save_npz_and_pt_share_coordinate_schema_and_preserve_encoder_keys(self):
        generator = self._generator()
        coordinates = [(40.0, -3.0), (34.0, -118.0)]
        embeddings = generator.generate_embeddings(coordinates)

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "nested" / "embeddings.npz"
            pt_path = Path(tmpdir) / "nested" / "embeddings.pt"
            generator.save_embeddings(embeddings, npz_path, coordinates)
            generator.save_embeddings(embeddings, pt_path, coordinates)

            with np.load(npz_path) as saved_npz:
                self.assertIn("torchspatial_direct", saved_npz.files)
                self.assertIn("coordinates", saved_npz.files)
                self.assertIn("coordinates_latlon", saved_npz.files)
                self.assertIn("coordinates_lonlat", saved_npz.files)
                metadata = json.loads(saved_npz["metadata_json"].item())
                self.assertEqual(metadata["coordinate_order"]["coordinates"], "lat_lon")
                np.testing.assert_allclose(
                    saved_npz["coordinates_lonlat"],
                    np.array([[-3.0, 40.0], [-118.0, 34.0]], dtype=np.float32),
                )

            saved_pt = torch.load(pt_path, weights_only=False)
            self.assertIsInstance(saved_pt["torchspatial_direct"], torch.Tensor)
            self.assertIn("metadata", saved_pt)
            self.assertIn("coordinates", saved_pt)
            self.assertEqual(saved_pt["metadata"]["n_points"], 2)
            torch.testing.assert_close(saved_pt["coordinates"], saved_pt["coordinates_latlon"])
            torch.testing.assert_close(
                saved_pt["coordinates_lonlat"],
                torch.tensor([[-3.0, 40.0], [-118.0, 34.0]]),
            )

    def test_coordinate_file_readers_accept_documented_forms_and_reject_bad_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            csv_path = base / "points.csv"
            csv_path.write_text("longitude,latitude\n-3.0,40.0\n-118.0,34.0\n", encoding="utf-8")
            json_path = base / "points.json"
            json_path.write_text(
                json.dumps({"coordinates": [{"lat": 40, "lon": -3}, [34, -118]]}),
                encoding="utf-8",
            )
            bad_path = base / "bad.txt"
            bad_path.write_text("40.0\n", encoding="utf-8")

            self.assertEqual(get_embeddings.read_coordinates(csv_path), [(40.0, -3.0), (34.0, -118.0)])
            self.assertEqual(get_embeddings.read_coordinates(json_path), [(40.0, -3.0), (34.0, -118.0)])
            with self.assertRaisesRegex(ValueError, "Line 1"):
                get_embeddings.read_coordinates(bad_path)

    def _year_generator(self):
        with patch.object(get_embeddings, "get_encoder_class", return_value=DummyYearEncoder):
            return get_embeddings.EmbeddingGenerator(encoders=["terranova"], device="cpu")

    def test_generation_accepts_one_year_per_coordinate(self):
        generator = self._year_generator()
        embeddings = generator.generate_embeddings([(40.0, -3.0), (34.0, -118.0)], year=[2000, 2010])
        np.testing.assert_allclose(embeddings["terranova"][:, 2].numpy(), [2000.0, 2010.0])
        with self.assertRaisesRegex(ValueError, "one year per coordinate"):
            generator.generate_embeddings([(40.0, -3.0), (34.0, -118.0)], year=[2000])

    def test_save_records_per_row_years_and_year_mode(self):
        generator = self._year_generator()
        coordinates = [(40.0, -3.0), (34.0, -118.0)]
        years = [2000, 2010]
        embeddings = generator.generate_embeddings(coordinates, year=years)
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "e.npz"
            pt_path = Path(tmpdir) / "e.pt"
            generator.save_embeddings(embeddings, npz_path, coordinates, years=years)
            generator.save_embeddings(embeddings, pt_path, coordinates, years=years)
            with np.load(npz_path) as saved:
                np.testing.assert_array_equal(saved["year"], np.array([2000, 2010]))
                metadata = json.loads(saved["metadata_json"].item())
            self.assertEqual(metadata["year_mode"], "per_row")
            self.assertEqual(metadata["year_range"], [2000, 2010])
            self.assertIsNone(metadata["year"])
            saved_pt = torch.load(pt_path, weights_only=False)
            torch.testing.assert_close(saved_pt["year"], torch.tensor([2000, 2010]))

            scalar_path = Path(tmpdir) / "s.npz"
            generator.save_embeddings(embeddings, scalar_path, coordinates, years=2015)
            with np.load(scalar_path) as saved:
                np.testing.assert_array_equal(saved["year"], np.array([2015, 2015]))
                metadata = json.loads(saved["metadata_json"].item())
            self.assertEqual(metadata["year_mode"], "scalar")
            self.assertEqual(metadata["year"], 2015)

            static_path = Path(tmpdir) / "n.npz"
            generator.save_embeddings(embeddings, static_path, coordinates)
            with np.load(static_path) as saved:
                self.assertNotIn("year", saved.files)
                self.assertEqual(json.loads(saved["metadata_json"].item())["year_mode"], "static")

    def test_coordinate_file_readers_return_optional_per_row_years(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            csv_path = base / "p.csv"
            csv_path.write_text("lat,lon,year\n40.0,-3.0,2000\n34.0,-118.0,2010\n", encoding="utf-8")
            json_rows = base / "rows.json"
            json_rows.write_text(
                json.dumps([{"lat": 40, "lon": -3, "year": 2000}, {"lat": 34, "lon": -118, "year": 2010}]),
                encoding="utf-8",
            )
            json_arrays = base / "arrays.json"
            json_arrays.write_text(
                json.dumps({"latitude": [40, 34], "longitude": [-3, -118], "year": [2000, 2010]}),
                encoding="utf-8",
            )
            plain = base / "plain.txt"
            plain.write_text("40.0 -3.0\n34.0 -118.0\n", encoding="utf-8")
            partial = base / "partial.json"
            partial.write_text(json.dumps([{"lat": 40, "lon": -3, "year": 2000}, [34, -118]]), encoding="utf-8")
            short = base / "short.json"
            short.write_text(json.dumps({"latitude": [40, 34], "longitude": [-3, -118], "year": [2000]}), encoding="utf-8")

            expected = ([(40.0, -3.0), (34.0, -118.0)], [2000, 2010])
            self.assertEqual(get_embeddings.read_coordinates_with_years(csv_path), expected)
            self.assertEqual(get_embeddings.read_coordinates_with_years(json_rows), expected)
            self.assertEqual(get_embeddings.read_coordinates_with_years(json_arrays), expected)
            self.assertEqual(get_embeddings.read_coordinates_with_years(plain), (expected[0], None))
            # the legacy reader keeps its pair-only contract
            self.assertEqual(get_embeddings.read_coordinates(csv_path), expected[0])
            with self.assertRaisesRegex(ValueError, "every row or none"):
                get_embeddings.read_coordinates_with_years(partial)
            with self.assertRaisesRegex(ValueError, "same length"):
                get_embeddings.read_coordinates_with_years(short)

    def _run_main(self, argv, tmpdir):
        output = Path(tmpdir) / "out.npz"
        argv = ["get_embeddings.py", *argv, "--encoders", "terranova", "--output", str(output)]
        with patch.object(get_embeddings, "get_encoder_class", return_value=DummyYearEncoder), patch(
            "sys.argv", argv
        ):
            code = get_embeddings.main()
        return code, output

    def test_cli_year_accepts_one_value_or_one_per_coordinate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code, output = self._run_main(["--lat", "40", "34", "--lon", "-3", "-118", "--year", "2000", "2010"], tmpdir)
            self.assertEqual(code, 0)
            with np.load(output) as saved:
                np.testing.assert_allclose(saved["terranova"][:, 2], [2000.0, 2010.0])
                np.testing.assert_array_equal(saved["year"], [2000, 2010])

            code, output = self._run_main(["--lat", "40", "34", "--lon", "-3", "-118", "--year", "2000"], tmpdir)
            self.assertEqual(code, 0)
            with np.load(output) as saved:
                np.testing.assert_allclose(saved["terranova"][:, 2], [2000.0, 2000.0])

            with self.assertRaises(SystemExit):
                self._run_main(["--lat", "40", "34", "--lon", "-3", "-118", "--year", "2000", "2010", "2020"], tmpdir)

    def test_cli_rejects_year_flag_when_input_file_carries_years(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "p.csv"
            csv_path.write_text("lat,lon,year\n40.0,-3.0,2000\n", encoding="utf-8")
            code, output = self._run_main(["--input", str(csv_path)], tmpdir)
            self.assertEqual(code, 0)
            with np.load(output) as saved:
                np.testing.assert_array_equal(saved["year"], [2000])
            with self.assertRaises(SystemExit):
                self._run_main(["--input", str(csv_path), "--year", "1999"], tmpdir)

    def test_encoder_root_parser_uses_registry_aliases(self):
        self.assertEqual(
            get_embeddings.parse_encoder_roots(["direct=/tmp/data"]),
            {"torchspatial_direct": "/tmp/data"},
        )
        with self.assertRaisesRegex(ValueError, "Expected encoder"):
            get_embeddings.parse_encoder_roots(["missing-separator"])


if __name__ == "__main__":
    unittest.main()
