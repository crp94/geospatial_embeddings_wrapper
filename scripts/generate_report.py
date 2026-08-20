#!/usr/bin/env python3
"""Generate an offline visual audit for geospatial embedding datasets.

The command intentionally loads no encoders: it only reads artifacts produced
by :mod:`scripts.generate_dataset` (or compatible coordinate/embedding files).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


# Keep direct ``python scripts/generate_report.py`` execution equivalent to the
# installed console entry point.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


OUTPUT_FORMATS = ("html", "png", "pdf")


def build_parser() -> argparse.ArgumentParser:
    """Build the public report command parser without importing heavy extras."""
    parser = argparse.ArgumentParser(
        description=(
            "Create an offline coverage, quality, embedding-atlas, comparison, "
            "temporal, and geographic-probe report from saved embedding datasets."
        )
    )
    parser.add_argument(
        "inputs",
        metavar="INPUT",
        nargs="+",
        type=Path,
        help="Input .pt, .zarr, or wide CSV dataset artifact(s).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory in which report files will be written.",
    )
    parser.add_argument(
        "--match-mode",
        choices=("strict", "nearest"),
        default="strict",
        help=(
            "How to align coordinates across inputs (default: strict exact pairs, "
            "irrespective of row order)."
        ),
    )
    parser.add_argument(
        "--match-tolerance-km",
        type=float,
        default=1.0,
        metavar="KM",
        help="Maximum nearest-coordinate distance in km (default: 1.0; nearest mode only).",
    )
    parser.add_argument(
        "--analysis-sample-size",
        type=int,
        default=100_000,
        metavar="N",
        help="Maximum deterministic equal-area sample used for expensive analyses (default: 100000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for sampling, reducers, and stochastic comparison analyses (default: 42).",
    )
    parser.add_argument(
        "--probes",
        type=Path,
        help="Optional JSON file defining additional geographic probe transects.",
    )
    parser.add_argument(
        "--no-umap",
        action="store_true",
        help="Skip the optional UMAP atlas for a faster report.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=OUTPUT_FORMATS,
        default=list(OUTPUT_FORMATS),
        metavar="FORMAT",
        help="Output formats to create: html png pdf (default: all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing report output directory.",
    )
    return parser


def _config_from_args(args: argparse.Namespace):
    """Translate parsed CLI options into the stable reporting configuration."""
    # Import lazily so ``--help`` is available in minimal installations and so
    # import failures state exactly which report dependencies are missing.
    from reporting import ReportConfig

    return ReportConfig(
        output_dir=args.output_dir,
        match_mode=args.match_mode,
        match_tolerance_km=args.match_tolerance_km,
        analysis_sample_size=args.analysis_sample_size,
        seed=args.seed,
        probes_path=args.probes,
        no_umap=args.no_umap,
        formats=tuple(dict.fromkeys(args.formats)),
        overwrite=args.overwrite,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the report CLI and return a conventional process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
        from reporting import generate_report

        output_dir = generate_report(args.inputs, config)
    except (ImportError, ModuleNotFoundError) as exc:
        parser.error(
            f"Reporting dependencies are unavailable ({exc}). Install with: "
            'pip install ".[reports]"'
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    print(f"Report written to {output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
