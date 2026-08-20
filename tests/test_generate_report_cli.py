"""Dependency-light contract tests for the reporting command line interface."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.generate_report import build_parser


class GenerateReportCliTests(unittest.TestCase):
    def test_documented_options_parse_without_loading_reporting_extras(self) -> None:
        args = build_parser().parse_args(
            [
                "first.zarr",
                "second.pt",
                "--output-dir",
                "reports/comparison",
                "--match-mode",
                "nearest",
                "--match-tolerance-km",
                "2.5",
                "--analysis-sample-size",
                "500",
                "--seed",
                "7",
                "--no-umap",
                "--formats",
                "html",
                "png",
                "--overwrite",
            ]
        )

        self.assertEqual(args.inputs, [Path("first.zarr"), Path("second.pt")])
        self.assertEqual(args.output_dir, Path("reports/comparison"))
        self.assertEqual(args.match_mode, "nearest")
        self.assertEqual(args.match_tolerance_km, 2.5)
        self.assertEqual(args.analysis_sample_size, 500)
        self.assertEqual(args.seed, 7)
        self.assertTrue(args.no_umap)
        self.assertEqual(args.formats, ["html", "png"])
        self.assertTrue(args.overwrite)

    def test_defaults_preserve_safe_exact_matching(self) -> None:
        args = build_parser().parse_args(["dataset.pt", "--output-dir", "report"])

        self.assertEqual(args.match_mode, "strict")
        self.assertEqual(args.match_tolerance_km, 1.0)
        self.assertEqual(args.analysis_sample_size, 100_000)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.formats, ["html", "png", "pdf"])
        self.assertFalse(args.overwrite)
