"""Focused contract tests for the point-query CLI helpers."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts import get_embeddings


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

    def test_encoder_root_parser_uses_registry_aliases(self):
        self.assertEqual(
            get_embeddings.parse_encoder_roots(["direct=/tmp/data"]),
            {"torchspatial_direct": "/tmp/data"},
        )
        with self.assertRaisesRegex(ValueError, "Expected encoder"):
            get_embeddings.parse_encoder_roots(["missing-separator"])


if __name__ == "__main__":
    unittest.main()
