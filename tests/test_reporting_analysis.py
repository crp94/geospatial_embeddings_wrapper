"""Offline contract tests for the dependency-light reporting analysis layer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

from reporting.analysis import (
    CoordinateMatchError,
    comparison_metrics,
    equal_area_stratified_sample,
    group_temporal_artifacts,
    haversine_km,
    linear_cka,
    match_coordinates,
    project_embeddings,
    quality_statistics,
    temporal_displacement,
)
from reporting.artifacts import ArtifactValidationError, DatasetArtifact, load_dataset_artifact


def _artifact(name: str, year: int | None = None, coordinates=None, values=None) -> DatasetArtifact:
    coordinates = np.asarray(
        coordinates if coordinates is not None else [[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float
    )
    values = np.asarray(values if values is not None else np.eye(len(coordinates), 3), dtype=float)
    metadata = {"n_points": len(coordinates), "year": year, "encoders": ["demo"]}
    return DatasetArtifact(Path(name), coordinates[:, 0], coordinates[:, 1], {"demo": values}, metadata)


class ReportingArtifactTests(unittest.TestCase):
    def test_pt_and_csv_loaders_normalize_named_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            coordinates = torch.tensor([[40.0, -3.0], [34.0, -118.0]])
            pt_path = root / "data.pt"
            torch.save({
                "coordinates_latlon": coordinates,
                "demo_embeddings": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "metadata": {"n_points": 2, "year": 2020, "coordinate_order": {"coordinates_latlon": "lat_lon"}},
            }, pt_path)
            pt = load_dataset_artifact(pt_path)
            self.assertEqual(pt.encoder_names, ("demo",))
            np.testing.assert_allclose(pt.coordinates, coordinates.numpy())
            self.assertEqual(pt.year, 2020)

            csv_path = root / "data.csv"
            pd.DataFrame({
                "latitude": [40, 34], "longitude": [-3, -118], "demo_emb_0000": [1, 3], "demo_emb_0001": [2, 4], "year": [2021, 2021],
            }).to_csv(csv_path, index=False)
            csv = load_dataset_artifact(csv_path)
            self.assertEqual(csv.year, 2021)
            np.testing.assert_allclose(csv.embedding("demo"), [[1, 2], [3, 4]])

    def test_loader_rejects_invalid_coordinate_schema_and_embedding_shape(self):
        with self.assertRaises(ArtifactValidationError):
            DatasetArtifact(Path("bad"), np.array([91]), np.array([0]), {"demo": np.ones((1, 1))})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.csv"
            pd.DataFrame({"latitude": [0], "longitude": [0], "demo_emb_0001": [1]}).to_csv(path, index=False)
            with self.assertRaises(ArtifactValidationError):
                load_dataset_artifact(path)

    def test_chunk_iteration_does_not_require_full_array_read(self):
        class LazyArray:
            shape = (3, 2)
            def __getitem__(self, index):
                return np.array([[1, 2], [3, 4], [5, 6]])[index]
        artifact = DatasetArtifact(Path("lazy.zarr"), np.array([0, 1, 2]), np.array([0, 1, 2]), {"demo": LazyArray()})
        batches = list(artifact.iter_embedding_batches("demo", batch_size=2))
        self.assertEqual([(start, end) for start, end, _ in batches], [(0, 2), (2, 3)])
        np.testing.assert_allclose(batches[1][2], [[5, 6]])

    def test_zarr_loader_keeps_embedding_array_lazy(self):
        test_case = self
        class Group:
            attrs = {"metadata": '{"n_points": 2, "encoders": ["demo"]}'}
            arrays = {
                "coordinates_latlon": np.array([[0.0, 1.0], [2.0, 3.0]]),
                "demo_embeddings": np.array([[1.0, 2.0], [3.0, 4.0]]),
            }
            def array_keys(self): return self.arrays.keys()
            def __getitem__(self, key): return self.arrays[key]
        class FakeZarr:
            @staticmethod
            def open_group(path, mode="r"):
                test_case.assertEqual(mode, "r")
                return Group()
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "data.zarr"
            source.mkdir()
            with mock.patch.dict("sys.modules", {"zarr": FakeZarr}):
                artifact = load_dataset_artifact(source)
        self.assertEqual(artifact.n_points, 2)
        np.testing.assert_allclose(artifact.embedding("demo"), [[1, 2], [3, 4]])


class ReportingAnalysisTests(unittest.TestCase):
    def test_stratified_sampling_is_repeatable_bounded_and_unique(self):
        lat = np.linspace(-80, 80, 100)
        lon = np.linspace(-179, 179, 100)
        first = equal_area_stratified_sample(lat, lon, 20, seed=7)
        second = equal_area_stratified_sample(lat, lon, 20, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual(len(np.unique(first)), 20)
        np.testing.assert_array_equal(equal_area_stratified_sample(lat, lon, 100), np.arange(100))

    def test_strict_and_nearest_matching_are_order_independent_and_bounded(self):
        left = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
        right = left[[2, 0, 1]]
        strict = match_coordinates(left, right)
        np.testing.assert_array_equal(strict.right_indices, [1, 2, 0])
        self.assertEqual(strict.n_matches, 3)
        with self.assertRaises(CoordinateMatchError):
            match_coordinates(left, right[:-1])
        with self.assertRaises(CoordinateMatchError):
            match_coordinates(np.array([[0, 0], [0, 0]]), np.array([[0, 0], [1, 1]]))
        nearest = match_coordinates(left, right + 0.001, mode="nearest", tolerance_km=1.0)
        self.assertEqual(nearest.n_matches, 3)
        self.assertTrue(np.all(nearest.distances_km < 1.0))
        self.assertEqual(match_coordinates(left, right + 10, mode="nearest", tolerance_km=1.0).n_matches, 0)

    def test_quality_stats_and_projection_handle_nonfinite_values(self):
        artifact = _artifact("quality", values=[[3, 4], [0, 0], [np.nan, 1], [1, 0]])
        stats = quality_statistics(artifact, "demo", batch_size=2)
        self.assertEqual(stats["dimensions"], 2)
        self.assertEqual(stats["finite_row_rate"], 0.75)
        self.assertAlmostEqual(stats["zero_vector_rate"], 1 / 3)
        projections = project_embeddings(np.array([[1, 2], [2, 3], [3, 5], [np.nan, 0]]), methods=("pca", "ica"), seed=4)
        self.assertEqual(set(projections), {"pca", "ica"})
        self.assertEqual(projections["pca"].shape, (4, 3))
        self.assertTrue(np.isnan(projections["ica"][-1]).all())

    def test_pairwise_metrics_and_geodesic_distance_are_known_values(self):
        values = np.eye(4)
        coordinates = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        self.assertAlmostEqual(linear_cka(values, values), 1.0)
        metrics = comparison_metrics(values, values, coordinates, seed=2, n_pairs=6)
        self.assertAlmostEqual(metrics["mean_row_cosine_similarity"], 1.0)
        self.assertAlmostEqual(metrics["linear_cka"], 1.0)
        self.assertAlmostEqual(metrics["neighborhood_agreement"]["k10"], 1.0)
        self.assertTrue(metrics["geodesic_similarity_curve"])
        self.assertAlmostEqual(float(haversine_km(np.array([[0, 0]]), np.array([[0, 1]]))[0]), 111.195, places=2)

    def test_temporal_grouping_and_displacement(self):
        first = _artifact("a", 2020, values=[[1, 0], [0, 1], [1, 1], [2, 0]])
        second = _artifact("b", 2021, coordinates=first.coordinates[[3, 1, 0, 2]], values=[[2, 0], [0, 1], [0, 1], [1, 1]])
        groups = group_temporal_artifacts([first, second, _artifact("untimed")])
        self.assertEqual([item.year for item in groups["demo"]], [2020, 2021])
        change = temporal_displacement(first, second, "demo")
        self.assertEqual(change["summary"]["n_matched_rows"], 4)
        self.assertEqual(change["coordinates"].shape, (4, 2))
        self.assertEqual(change["latitude"].shape, (4,))
        np.testing.assert_allclose(change["displacement"], change["normalized_displacement"])
        self.assertGreater(change["summary"]["mean_normalized_displacement"], 0)
