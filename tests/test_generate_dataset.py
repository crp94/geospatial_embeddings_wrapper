import tempfile
import unittest
from pathlib import Path
from unittest import mock

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.transform import from_bounds
from shapely.geometry import box

from scripts.generate_dataset import ANTARCTICA_LATITUDE_CUTOFF, GeospatialDatasetGenerator
from wrappers.embedding_encoder import GeoEmbeddingEncoder
from wrappers.registry import normalize_encoder_name
from wrappers.torchgeo_encoders import _sample_from_wgs84_file_index


class DummyStaticEncoder(GeoEmbeddingEncoder):
    def __init__(self, dim: int = 3):
        super().__init__(device="cpu")
        self.dim = dim

    def encode(self, coordinates: torch.Tensor, year: int | None = None) -> torch.Tensor:
        lat = coordinates[:, 0:1]
        lon = coordinates[:, 1:2]
        year_term = torch.zeros_like(lat) if year is None else torch.full_like(lat, year)
        features = [lat, lon, year_term]
        return torch.cat(features[: self.dim], dim=1)

    def get_embedding_dim(self) -> int:
        return self.dim


class DummyTemporalEncoder(DummyStaticEncoder):
    def __init__(self, years: list[int], dim: int = 3):
        super().__init__(dim=dim)
        self.years = years

    def is_temporal(self) -> bool:
        return True

    def get_available_years(self) -> list[int] | None:
        return self.years


class DummyInvalidEncoder(DummyStaticEncoder):
    def __init__(self):
        super().__init__(dim=3)
        self.encode_calls = 0

    def encode(self, coordinates: torch.Tensor, year: int | None = None) -> torch.Tensor:
        self.encode_calls += 1
        embeddings = super().encode(coordinates, year=year)
        if self.encode_calls == 1:
            embeddings[0] = torch.nan
        return embeddings


class DummyCoverageEncoder(DummyStaticEncoder):
    def __init__(self):
        super().__init__(dim=3)
        self.sample_calls = 0

    def supports_coverage_sampling(self) -> bool:
        return True

    def get_sampling_oversample_factor(self) -> float:
        return 1.0

    def sample_candidate_coordinates(self, n_points, year=None, rng=None):
        self.sample_calls += 1
        longitude = np.linspace(10.0, 10.0 + n_points - 1, num=n_points, dtype=np.float64)
        latitude = np.linspace(1.0, 1.0 + n_points - 1, num=n_points, dtype=np.float64)
        return longitude, latitude


class SequenceGenerator(GeospatialDatasetGenerator):
    def __init__(self, batches):
        self._cache_dir = tempfile.TemporaryDirectory()
        super().__init__(cache_dir=self._cache_dir.name)
        self._batches = batches
        self._batch_index = 0

    def __del__(self):
        self._cache_dir.cleanup()

    def initialize_encoders(self, encoder_names=None, device=None):
        return self.encoders

    def fibonacci_sphere_sampling(self, n_points: int):
        batch = self._batches[self._batch_index]
        self._batch_index += 1
        return batch

    def vectorized_land_filter(self, longitude, latitude, batch_size=10000):
        return longitude, latitude

    def plot_sampled_locations(self, longitude, latitude, save_path, year=None):
        return None

    def plot_ica_embeddings(
        self, longitude, latitude, embeddings, encoder_name, save_path, year=None
    ):
        return None


class DirectSamplingGenerator(GeospatialDatasetGenerator):
    def __init__(self):
        self._cache_dir = tempfile.TemporaryDirectory()
        super().__init__(cache_dir=self._cache_dir.name)

    def __del__(self):
        self._cache_dir.cleanup()

    def initialize_encoders(self, encoder_names=None, device=None):
        return self.encoders

    def fibonacci_sphere_sampling(self, n_points: int):
        raise AssertionError("Global Fibonacci sampling should not be used")

    def vectorized_land_filter(self, longitude, latitude, batch_size=10000):
        return longitude, latitude

    def plot_sampled_locations(self, longitude, latitude, save_path, year=None):
        return None

    def plot_ica_embeddings(
        self, longitude, latitude, embeddings, encoder_name, save_path, year=None
    ):
        return None


class GenerateDatasetTests(unittest.TestCase):
    def test_registry_aliases(self):
        self.assertEqual(normalize_encoder_name("clay"), "lgnd_clay")
        self.assertEqual(normalize_encoder_name("copernicus-embed"), "copernicus_embed")
        self.assertEqual(normalize_encoder_name("gse"), "google_satellite_embedding")
        self.assertEqual(normalize_encoder_name("range+"), "range_plus")
        self.assertEqual(normalize_encoder_name("gt-loc"), "gtloc")
        self.assertEqual(normalize_encoder_name("csp_inat"), "csp_inat")
        self.assertEqual(normalize_encoder_name("cartesian_3d"), "torchspatial_cartesian3d")

    def test_coordinate_validation_enforces_public_latlon_contract(self):
        valid = torch.tensor([[90.0, -180.0], [-90.0, 180.0]], dtype=torch.float32)
        self.assertIs(GeoEmbeddingEncoder.validate_coordinates(valid), valid)

        invalid_inputs = (
            torch.tensor([0.0, 0.0]),
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.tensor([[float("nan"), 0.0]]),
            torch.tensor([[90.1, 0.0]]),
            torch.tensor([[0.0, 180.1]]),
        )
        for coordinates in invalid_inputs:
            with self.subTest(coordinates=coordinates):
                with self.assertRaises(ValueError):
                    GeoEmbeddingEncoder.validate_coordinates(coordinates)

        with self.assertRaises(TypeError):
            GeoEmbeddingEncoder.validate_coordinates([[0.0, 0.0]])

    def test_coordinate_helpers_validate_before_encoding(self):
        encoder = DummyStaticEncoder()
        np.testing.assert_allclose(
            encoder.encode_single(1.0, 2.0).numpy(),
            np.array([[1.0, 2.0, 0.0]], dtype=np.float32),
        )
        with self.assertRaises(ValueError):
            encoder.encode_from_list([(91.0, 0.0)])

    def test_torchspatial_baseline_coordinate_order(self):
        from wrappers.location_model_encoders import (
            TorchSpatialCartesian3DEncoder,
            TorchSpatialDirectEncoder,
            TorchSpatialWrapEncoder,
        )

        coordinates = torch.tensor([[0.0, 90.0]], dtype=torch.float32)

        direct = TorchSpatialDirectEncoder(device="cpu")
        np.testing.assert_allclose(
            direct.encode(coordinates).numpy(),
            np.deg2rad(np.array([[90.0, 0.0]], dtype=np.float32)),
            rtol=1e-6,
            atol=1e-6,
        )

        cartesian = TorchSpatialCartesian3DEncoder(device="cpu")
        np.testing.assert_allclose(
            cartesian.encode(coordinates).numpy(),
            np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

        wrap = TorchSpatialWrapEncoder(device="cpu")
        np.testing.assert_allclose(
            wrap.encode(coordinates).numpy(),
            np.array([[0.0, 1.0, 1.0, 0.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_resolve_years_uses_intersection_for_temporal_encoders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = GeospatialDatasetGenerator(cache_dir=tmpdir)
            generator.encoders = {
                "temporal_a": DummyTemporalEncoder([2020, 2021, 2022]),
                "temporal_b": DummyTemporalEncoder([2021, 2022, 2023]),
                "static": DummyStaticEncoder(),
            }

            self.assertEqual(generator.resolve_years(), [2021, 2022])

    def test_generate_dataset_writes_yearly_pt_files_with_explicit_coordinates(self):
        batches = [
            (np.array([10.0, 20.0]), np.array([1.0, 2.0])),
            (np.array([10.0, 20.0]), np.array([1.0, 2.0])),
        ]
        generator = SequenceGenerator(batches=batches)
        generator.encoders = {
            "static": DummyStaticEncoder(),
            "temporal": DummyTemporalEncoder([2020, 2021]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_prefix = str(Path(tmpdir) / "dataset")
            outputs = generator.generate_dataset(
                n_points=2,
                output_path=output_prefix,
                output_format="pt",
                plot_results=False,
            )

            self.assertEqual(
                outputs,
                [f"{output_prefix}_2020.pt", f"{output_prefix}_2021.pt"],
            )

            dataset_2020 = torch.load(outputs[0])
            self.assertEqual(dataset_2020["metadata"]["year"], 2020)
            self.assertEqual(
                dataset_2020["metadata"]["coordinate_order"]["coordinates"], "lat_lon"
            )
            np.testing.assert_allclose(
                dataset_2020["coordinates_latlon"].numpy(),
                np.array([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32),
            )
            np.testing.assert_allclose(
                dataset_2020["coordinates_lonlat"].numpy(),
                np.array([[10.0, 1.0], [20.0, 2.0]], dtype=np.float32),
            )
            self.assertEqual(tuple(dataset_2020["static_embeddings"].shape), (2, 3))
            self.assertEqual(tuple(dataset_2020["temporal_embeddings"].shape), (2, 3))

    def test_sample_valid_dataset_resamples_until_target_count(self):
        batches = [
            (np.array([10.0, 20.0]), np.array([1.0, 2.0])),
            (np.array([30.0]), np.array([3.0])),
        ]
        generator = SequenceGenerator(batches=batches)
        invalid_encoder = DummyInvalidEncoder()
        generator.encoders = {"invalid": invalid_encoder}

        longitude, latitude, embeddings = generator.sample_valid_dataset(n_points=2)

        self.assertEqual(len(longitude), 2)
        self.assertEqual(len(latitude), 2)
        self.assertEqual(embeddings["invalid"].shape, (2, 3))
        self.assertGreaterEqual(invalid_encoder.encode_calls, 2)

    def test_sample_from_wgs84_file_index_samples_expected_pixels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raster_path = Path(tmpdir) / "sample.tif"
            transform = from_bounds(10.0, 20.0, 12.0, 22.0, 2, 2)
            data = np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[10.0, 20.0], [30.0, 40.0]],
                ],
                dtype=np.float32,
            )
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                height=2,
                width=2,
                count=2,
                dtype="float32",
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(data)

            index_df = gpd.GeoDataFrame(
                {"filepath": [str(raster_path)]},
                geometry=[box(10.0, 20.0, 12.0, 22.0)],
                crs="EPSG:4326",
            )
            coordinates = torch.tensor(
                [
                    [21.5, 10.5],
                    [20.5, 11.5],
                    [30.0, 30.0],
                ],
                dtype=torch.float32,
            )

            sampled = _sample_from_wgs84_file_index(
                coordinates,
                index_df,
                embedding_dim=2,
            ).numpy()

            np.testing.assert_allclose(sampled[0], np.array([1.0, 10.0], dtype=np.float32))
            np.testing.assert_allclose(sampled[1], np.array([4.0, 40.0], dtype=np.float32))
            self.assertTrue(np.isnan(sampled[2]).all())

    def test_sample_valid_dataset_uses_direct_encoder_sampling_when_available(self):
        generator = DirectSamplingGenerator()
        encoder = DummyCoverageEncoder()
        generator.encoders = {"coverage": encoder}

        longitude, latitude, embeddings = generator.sample_valid_dataset(n_points=3)

        self.assertEqual(encoder.sample_calls, 1)
        np.testing.assert_allclose(longitude, np.array([10.0, 11.0, 12.0]))
        np.testing.assert_allclose(latitude, np.array([1.0, 2.0, 3.0]))
        self.assertEqual(embeddings["coverage"].shape, (3, 3))

    def test_streamed_zarr_writer_uses_chunked_compatible_schema(self):
        class FakeArray:
            def __init__(self, shape):
                self.values = np.zeros(shape, dtype=np.float32)

            def __setitem__(self, item, values):
                self.values[item] = values

        class FakeGroup(dict):
            def __init__(self):
                super().__init__()
                self.attrs = {}

            def create_array(self, name, *, shape, dtype, chunks):
                self[name] = FakeArray(shape)
                return self[name]

        class FakeZarr:
            groups = {}

            @classmethod
            def open_group(cls, path, mode):
                if mode == "w":
                    cls.groups[path] = FakeGroup()
                return cls.groups[path]

        generator = DirectSamplingGenerator()
        generator.encoders = {"coverage": DummyCoverageEncoder()}
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            "sys.modules", {"zarr": FakeZarr}
        ):
            output_file = generator.save_zarr_dataset(3, str(Path(tmpdir) / "data"), None)

        group = FakeZarr.groups[output_file]
        self.assertEqual(group["coordinates"].values.shape, (3, 2))
        np.testing.assert_allclose(
            group["coordinates_latlon"].values,
            np.array([[1.0, 10.0], [2.0, 11.0], [3.0, 12.0]], dtype=np.float32),
        )
        self.assertIn("metadata", group.attrs)

    def test_seed_reproduces_fibonacci_coordinates_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = GeospatialDatasetGenerator(cache_dir=tmpdir, seed=123)
            second = GeospatialDatasetGenerator(cache_dir=tmpdir, seed=123)
            first_lon, first_lat = first.fibonacci_sphere_sampling(10)
            second_lon, second_lat = second.fibonacci_sphere_sampling(10)

            np.testing.assert_allclose(first_lon, second_lon)
            np.testing.assert_allclose(first_lat, second_lat)
            self.assertEqual(
                first._dataset_metadata(10, [], None)["sampling"]["seed"], 123
            )

    def test_coordinate_export_round_trips_documented_npz_and_csv_forms(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = GeospatialDatasetGenerator(cache_dir=tmpdir)
            latitude = np.array([1.0, -2.0], dtype=np.float32)
            longitude = np.array([3.0, -4.0], dtype=np.float32)

            npz_path = generator.save_coordinates(latitude, longitude, f"{tmpdir}/coords")
            csv_path = generator.save_coordinates(latitude, longitude, f"{tmpdir}/coords.csv")

            for path in (npz_path, csv_path):
                loaded_latitude, loaded_longitude = generator.load_coordinates(path)
                np.testing.assert_allclose(loaded_latitude, latitude)
                np.testing.assert_allclose(loaded_longitude, longitude)

    def test_supplied_coordinates_are_never_silently_resampled_when_invalid(self):
        generator = SequenceGenerator(batches=[])
        generator.encoders = {"invalid": DummyInvalidEncoder()}
        with tempfile.TemporaryDirectory() as tmpdir:
            coordinate_path = generator.save_coordinates(
                np.array([1.0], dtype=np.float32),
                np.array([2.0], dtype=np.float32),
                f"{tmpdir}/input.npz",
            )
            with self.assertRaisesRegex(RuntimeError, "never silently resampled"):
                generator.generate_dataset(
                    encoders=["invalid"],
                    coordinates_in=coordinate_path,
                    output_path=f"{tmpdir}/nested/output",
                    plot_results=False,
                )

    def test_sampling_stops_after_configured_attempt_limit(self):
        class NoLandGenerator(GeospatialDatasetGenerator):
            def fibonacci_sphere_sampling(self, n_points):
                return np.zeros(n_points), np.zeros(n_points)

            def vectorized_land_filter(self, longitude, latitude, batch_size=10000):
                return np.array([]), np.array([])

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = NoLandGenerator(cache_dir=tmpdir, max_sampling_attempts=2)
            generator.encoders = {"static": DummyStaticEncoder()}
            with self.assertRaisesRegex(RuntimeError, "after 2 attempts"):
                generator.sample_valid_dataset(1)

    def test_vectorized_land_filter_excludes_antarctica_before_land_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = GeospatialDatasetGenerator(cache_dir=tmpdir)
            generator.land_mask = gpd.GeoDataFrame(
                geometry=[box(-180.0, -90.0, 180.0, 90.0)],
                crs="EPSG:4326",
            )
            longitude = np.array([0.0, 10.0, 20.0], dtype=np.float64)
            latitude = np.array(
                [ANTARCTICA_LATITUDE_CUTOFF - 5.0, ANTARCTICA_LATITUDE_CUTOFF + 1.0, 0.0],
                dtype=np.float64,
            )

            land_longitude, land_latitude = generator.vectorized_land_filter(
                longitude, latitude, batch_size=10
            )

            np.testing.assert_allclose(land_longitude, np.array([10.0, 20.0]))
            np.testing.assert_allclose(
                land_latitude,
                np.array([ANTARCTICA_LATITUDE_CUTOFF + 1.0, 0.0]),
            )


if __name__ == "__main__":
    unittest.main()
