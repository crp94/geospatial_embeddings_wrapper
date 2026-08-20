"""Offline end-to-end checks for the public report orchestration API."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from reporting import ReportConfig, generate_report


class ReportingIntegrationTests(unittest.TestCase):
    def _write_artifact(self, path: Path) -> None:
        coordinates = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        torch.save(
            {
                "coordinates_latlon": coordinates,
                "demo_embeddings": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]),
                "metadata": {
                    "n_points": 5,
                    "year": 2020,
                    "encoders": ["demo"],
                    "coordinate_order": {"coordinates_latlon": "lat_lon"},
                },
            },
            path,
        )

    def test_report_writes_static_outputs_tables_and_reproducibility_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "fixture.pt"
            self._write_artifact(source)
            output = generate_report(
                [source],
                ReportConfig(output_dir=root / "report", no_umap=True, formats=("png", "pdf")),
            )

            for relative_path in (
                "manifest.json",
                "report.pdf",
                "figures/coverage_fixture_2020.png",
                "tables/quality_metrics.csv",
                "tables/comparison_metrics.csv",
                "tables/temporal_metrics.csv",
                "tables/probe_measurements.csv",
            ):
                self.assertTrue((output / relative_path).exists(), relative_path)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["configuration"]["seed"], 42)
            self.assertEqual(manifest["sampling"]["strategy"], "equal_area_stratified")
            self.assertEqual(manifest["inputs"][0]["kind"], "file")
            self.assertTrue(manifest["inputs"][0]["sha256"])

            with self.assertRaisesRegex(ValueError, "already exists"):
                generate_report([source], ReportConfig(output_dir=output, no_umap=True, formats=("png",)))
