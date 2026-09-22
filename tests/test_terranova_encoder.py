import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch

from wrappers.registry import get_encoder_class, normalize_encoder_name


class FakeTerraNova:
    """Offline stand-in for ``terranova.TerraNova`` that records every call."""

    from_pretrained_calls: list[dict] = []

    def __init__(self):
        self.embed_calls: list[dict] = []
        self.backbone_hash = "abc123def456"
        self.device = torch.device("cpu")

    @classmethod
    def from_pretrained(cls, repo_or_dir="crp94/terranova", device=None, revision=None, cache_dir=None):
        cls.from_pretrained_calls.append(
            {"repo_or_dir": repo_or_dir, "device": device, "revision": revision, "cache_dir": cache_dir}
        )
        return cls()

    def embed(self, coords=None, countries=None, year=2015, space="spatiotemporal", batch_size=8192, task=None):
        coords = np.asarray(coords, dtype=np.float64)
        self.embed_calls.append(
            {"coords": coords.copy(), "year": year, "space": space, "batch_size": batch_size}
        )
        n = coords.shape[0]
        out = np.zeros((n, 256), dtype=np.float32)
        out[:, 0] = coords[:, 0]  # lon
        out[:, 1] = coords[:, 1]  # lat
        out[:, 2] = np.asarray(year, dtype=np.float64)  # scalar or [N], as upstream
        return out


def _install_fake_terranova():
    module = types.ModuleType("terranova")
    module.TerraNova = FakeTerraNova
    FakeTerraNova.from_pretrained_calls = []
    return mock.patch.dict(sys.modules, {"terranova": module})


class TerraNovaEncoderTests(unittest.TestCase):
    def setUp(self):
        patcher = _install_fake_terranova()
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_registry_names_and_aliases_resolve(self):
        self.assertEqual(normalize_encoder_name("terranova"), "terranova")
        self.assertEqual(normalize_encoder_name("terra_nova"), "terranova")
        self.assertEqual(normalize_encoder_name("terranova_st"), "terranova")
        self.assertEqual(normalize_encoder_name("terranova_spatial"), "terranova_spatial")
        self.assertEqual(get_encoder_class("terranova").__name__, "TerraNovaEncoder")
        self.assertEqual(get_encoder_class("terranova_spatial").__name__, "TerraNovaSpatialEncoder")

    def test_encode_flips_to_lonlat_and_passes_year_through(self):
        encoder = get_encoder_class("terranova")(device="cpu")
        coords = torch.tensor([[41.9, 12.5], [-33.9, 151.2]], dtype=torch.float32)

        embeddings = encoder.encode(coords, year=1999)

        self.assertEqual(embeddings.shape, (2, 256))
        self.assertEqual(embeddings.dtype, torch.float32)
        call = encoder.model.embed_calls[-1]
        np.testing.assert_allclose(call["coords"], [[12.5, 41.9], [151.2, -33.9]], atol=1e-5)
        self.assertEqual(call["year"], 1999)
        self.assertEqual(call["space"], "spatiotemporal")
        self.assertAlmostEqual(float(embeddings[0, 0]), 12.5, places=4)
        self.assertAlmostEqual(float(embeddings[0, 1]), 41.9, places=4)

    def test_encode_defaults_to_reference_year_when_none_given(self):
        encoder = get_encoder_class("terranova")(device="cpu")
        encoder.encode(torch.tensor([[0.0, 0.0]]))
        self.assertEqual(encoder.model.embed_calls[-1]["year"], 2015)

    def test_spatiotemporal_variant_is_temporal_over_documented_year_range(self):
        encoder = get_encoder_class("terranova")(device="cpu")
        self.assertTrue(encoder.is_temporal())
        years = encoder.get_available_years()
        self.assertEqual(years[0], 1900)
        self.assertEqual(years[-1], 2035)
        self.assertEqual(len(years), 136)
        self.assertEqual(encoder.get_embedding_dim(), 256)
        self.assertEqual(encoder.name, "TerraNova")

    def test_spatial_variant_is_static_and_uses_spatial_space(self):
        encoder = get_encoder_class("terranova_spatial")(device="cpu")
        self.assertFalse(encoder.is_temporal())
        self.assertIsNone(encoder.get_available_years())
        encoder.encode(torch.tensor([[10.0, 20.0]]))
        self.assertEqual(encoder.model.embed_calls[-1]["space"], "spatial")
        self.assertEqual(encoder.get_embedding_dim(), 256)
        self.assertEqual(encoder.name, "TerraNova-spatial")

    def test_default_load_uses_public_hub_repo_and_device(self):
        get_encoder_class("terranova")(device="cpu")
        call = FakeTerraNova.from_pretrained_calls[-1]
        self.assertEqual(call["repo_or_dir"], "crp94/terranova")
        self.assertEqual(call["device"], "cpu")
        self.assertIsNone(call["revision"])
        self.assertIsNone(call["cache_dir"])

    def test_data_root_spec_overrides_repo_revision_cache_space_and_batch_size(self):
        encoder = get_encoder_class("terranova")(
            device="cpu",
            data_root="repo=/local/snapshot;revision=v1;cache=/tmp/hf;space=time;batch_size=16",
        )
        call = FakeTerraNova.from_pretrained_calls[-1]
        self.assertEqual(call["repo_or_dir"], "/local/snapshot")
        self.assertEqual(call["revision"], "v1")
        self.assertEqual(call["cache_dir"], "/tmp/hf")
        encoder.encode(torch.tensor([[1.0, 2.0]]), year=2000)
        self.assertEqual(encoder.model.embed_calls[-1]["space"], "time")
        self.assertEqual(encoder.model.embed_calls[-1]["batch_size"], 16)

    def test_plain_path_data_root_is_treated_as_local_snapshot(self):
        get_encoder_class("terranova")(device="cpu", data_root="/models/terranova")
        self.assertEqual(FakeTerraNova.from_pretrained_calls[-1]["repo_or_dir"], "/models/terranova")

    def test_metadata_records_provenance_and_licence(self):
        encoder = get_encoder_class("terranova")(device="cpu")
        metadata = encoder.get_metadata()
        self.assertEqual(metadata["source_type"], "terranova")
        self.assertEqual(metadata["embedding_space"], "spatiotemporal")
        self.assertEqual(metadata["hf_repo"], "crp94/terranova")
        self.assertEqual(metadata["year_range"], [1900, 2035])
        self.assertEqual(metadata["default_year"], 2015)
        self.assertEqual(metadata["backbone_hash"], "abc123def456")
        self.assertEqual(metadata["native_coordinate_order"], "lon_lat")
        self.assertEqual(metadata["license"], "CC-BY-4.0")
        self.assertIn("2607.29527", metadata["citation"])

    def test_year_outside_documented_range_is_rejected_before_calling_model(self):
        encoder = get_encoder_class("terranova")(device="cpu")
        with self.assertRaises(ValueError):
            encoder.encode(torch.tensor([[0.0, 0.0]]), year=1850)
        self.assertEqual(encoder.model.embed_calls, [])


    def test_encode_with_years_makes_one_vectorised_call_with_per_row_years(self):
        encoder = get_encoder_class("terranova")(device="cpu")
        coords = torch.tensor([[41.9, 12.5], [-33.9, 151.2], [40.7, -74.0]])
        years = np.array([1950, 2015, 2030])

        embeddings = encoder.encode_with_years(coords, years)

        self.assertEqual(len(encoder.model.embed_calls), 1)
        call = encoder.model.embed_calls[0]
        np.testing.assert_array_equal(np.asarray(call["year"]), years)
        np.testing.assert_allclose(call["coords"][:, 0], [12.5, 151.2, -74.0], atol=1e-5)
        np.testing.assert_allclose(embeddings[:, 2].numpy(), years.astype(np.float32))

    def test_encode_with_years_rejects_out_of_range_rows_before_calling_model(self):
        encoder = get_encoder_class("terranova")(device="cpu")
        with self.assertRaises(ValueError):
            encoder.encode_with_years(torch.zeros((2, 2)), np.array([2015, 2040]))
        self.assertEqual(encoder.model.embed_calls, [])


class TerraNovaMissingPackageTests(unittest.TestCase):
    def test_missing_package_raises_import_error_with_install_hint(self):
        with mock.patch.dict(sys.modules, {"terranova": None}):
            with self.assertRaises(ImportError) as ctx:
                get_encoder_class("terranova")(device="cpu")
        self.assertIn("terranova-model", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
