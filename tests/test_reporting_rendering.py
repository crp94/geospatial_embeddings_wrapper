import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from reporting.probes import (
    ProbeConfigurationError,
    haversine_km,
    interpolate_polyline,
    load_probe_definitions,
    measure_probe_embeddings,
    sample_probe_rows,
)
from reporting.render import (
    build_html_report,
    render_atlas,
    render_comparison,
    render_coverage,
    render_probes,
    render_quality,
    render_temporal,
    write_pdf_report,
)


class ProbeTests(unittest.TestCase):
    def test_builtin_probes_and_interpolation_are_valid(self):
        probes = load_probe_definitions()
        self.assertGreaterEqual(len(probes), 4)
        samples = interpolate_polyline([(0, 179), (0, -179)], 3)
        # Both signed representations identify the antimeridian.
        self.assertAlmostEqual(abs(samples[1, 1]), 180.0, places=5)
        self.assertAlmostEqual(float(haversine_km(0, 0, 0, 1)), 111.195, places=2)

    def test_custom_probe_validation_and_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "probes.json"
            config.write_text(json.dumps({"probes": [{"name": "test", "coordinates": [[0, 0], [0, 2]], "sample_count": 3}]}))
            probe = load_probe_definitions(config)[0]
            sampled = sample_probe_rows([0, 0, 0], [0, 1, 2], probe, max_gap_km=2)
            self.assertListEqual(sampled["row_index"].tolist(), [0, 1, 2])
            measured = measure_probe_embeddings(sampled, {"encoder": np.array([[0, 0], [3, 4], [6, 8]])})
            self.assertEqual(len(measured), 3)
            self.assertAlmostEqual(measured.loc[1, "step_embedding_distance"], 5.0)
            config.write_text(json.dumps({"probes": [{"name": "bad", "coordinates": [[91, 0], [0, 0]]}]}))
            with self.assertRaises(ProbeConfigurationError):
                load_probe_definitions(config)


class RenderTests(unittest.TestCase):
    def test_all_renderers_and_report_are_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coverage = {"latitude": np.array([0, 10, -10]), "longitude": np.array([0, 20, -20]), "status": ["valid", "invalid", "valid"]}
            atlas = {"latitude": coverage["latitude"], "longitude": coverage["longitude"], "projection": np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])}
            paths = [
                render_coverage(coverage, root / "coverage.png"),
                render_atlas(atlas, root / "atlas.png"),
                render_quality({"dimension": 3, "finite_rate": 1.0, "norms": [1, 2, 3]}, root / "quality.png"),
                render_comparison(pd.DataFrame({"encoder_a": ["a"], "encoder_b": ["b"], "cka": [0.8]}), root / "comparison.png"),
                render_temporal({"latitude": [0, 1], "longitude": [0, 1], "displacement": [0.1, 0.5]}, root / "temporal.png"),
            ]
            probe_rows = pd.DataFrame({"probe": ["p", "p"], "sample_index": [0, 1], "along_fraction": [0, 1], "latitude": [0, 1], "longitude": [0, 1], "embedding_norm": [1, 2], "step_embedding_distance": [np.nan, 0.5]})
            paths.append(render_probes(probe_rows, root / "probes.png"))
            self.assertTrue(all(path.exists() and path.stat().st_size > 100 for path in paths))
            html = build_html_report(root / "index.html", [{"title": "Coverage", "images": [paths[0]], "table": {"rows": 3}}])
            pdf = write_pdf_report(paths, root / "report.pdf")
            self.assertIn("data:image/png;base64", html.read_text())
            self.assertTrue(pdf.exists() and pdf.stat().st_size > 100)
