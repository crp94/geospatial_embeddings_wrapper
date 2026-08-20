"""Public orchestration API for offline geospatial embedding audit reports.

Importing this module is intentionally lightweight.  Scientific and rendering
dependencies are imported only by :func:`generate_report`, so applications can
inspect :class:`ReportConfig` without importing Plotly, Matplotlib, or UMAP.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Sequence


_FORMATS = frozenset({"html", "png", "pdf"})


@dataclass(frozen=True)
class ReportConfig:
    """Configuration shared by the CLI and programmatic report generation."""

    output_dir: Path | str
    match_mode: str = "strict"
    match_tolerance_km: float = 1.0
    analysis_sample_size: int = 100_000
    seed: int = 42
    probes_path: Path | str | None = None
    no_umap: bool = False
    formats: tuple[str, ...] = ("html", "png", "pdf")
    overwrite: bool = False

    def __post_init__(self) -> None:
        output_dir = Path(self.output_dir).expanduser()
        probes_path = Path(self.probes_path) if self.probes_path is not None else None
        formats = tuple(dict.fromkeys(str(value).lower() for value in self.formats))
        if not formats or any(value not in _FORMATS for value in formats):
            raise ValueError("formats must be a non-empty subset of: html, png, pdf")
        if self.match_mode not in {"strict", "nearest"}:
            raise ValueError("match_mode must be either 'strict' or 'nearest'")
        if self.match_tolerance_km <= 0:
            raise ValueError("match_tolerance_km must be positive")
        if self.analysis_sample_size <= 0:
            raise ValueError("analysis_sample_size must be positive")
        resolved_output = output_dir.resolve()
        if (
            not output_dir.name
            or resolved_output in {Path.cwd().resolve(), Path.cwd().resolve().parent, Path("/")}
        ):
            raise ValueError("output_dir must name a dedicated report directory, not a workspace or filesystem root")
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "probes_path", probes_path)
        object.__setattr__(self, "formats", formats)


def _source_id(artifact: Any, used: set[str]) -> str:
    """Return a readable unique label which remains stable across report runs."""
    source = Path(artifact.source)
    year = getattr(artifact, "metadata", {}).get("year")
    base = source.stem or source.name
    if year is not None and str(year) not in base:
        base = f"{base}_{year}"
    candidate, number = base, 2
    while candidate in used:
        candidate = f"{base}_{number}"
        number += 1
    used.add(candidate)
    return candidate


def _scalar_metrics(values: dict[str, Any]) -> dict[str, Any]:
    """Keep table/JSON metrics compact while retaining all scalar diagnostics."""
    import numpy as np

    result: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, dict):
            for nested_key, nested_value in _scalar_metrics(value).items():
                result[f"{key}_{nested_key}"] = nested_value
            continue
        if isinstance(value, (str, bool, int, float)) or np.isscalar(value):
            if isinstance(value, np.generic):
                value = value.item()
            result[str(key)] = value
    return result


def _json_default(value: Any) -> Any:
    """Serialize NumPy and pathlib values used in report provenance."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _source_fingerprint(path: Path) -> dict[str, Any]:
    """Hash files and create a deterministic content inventory for Zarr trees."""
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return {"path": str(path), "kind": "file", "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}
    if path.is_dir():
        file_count = 0
        total_size = 0
        for member in sorted(item for item in path.rglob("*") if item.is_file()):
            relative = member.relative_to(path).as_posix().encode("utf-8")
            size = member.stat().st_size
            digest.update(relative); digest.update(b"\0"); digest.update(str(size).encode("ascii")); digest.update(b"\0")
            with member.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            file_count += 1; total_size += size
        return {
            "path": str(path), "kind": "directory", "sha256": digest.hexdigest(),
            "file_count": file_count, "size_bytes": total_size,
        }
    raise ValueError(f"Input artifact does not exist: {path}")


def _prepare_output(config: ReportConfig) -> tuple[Path, Path, Path, Path]:
    output = Path(config.output_dir)
    if output.exists():
        if not config.overwrite:
            raise ValueError(f"Report output directory already exists: {output}. Use --overwrite to replace it.")
        if not output.is_dir():
            raise ValueError(f"Report output path is not a directory: {output}")
        # The directory is an explicit user target guarded by --overwrite.
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)
    figures, tables, assets = output / "figures", output / "tables", output / "assets"
    for directory in (figures, tables, assets):
        directory.mkdir()
    return output, figures, tables, assets


def _embedding_rows(artifact: Any, encoder_name: str, indexes: Any) -> Any:
    """Read selected embedding rows in a Zarr-compatible deterministic order."""
    import numpy as np

    indexes = np.asarray(indexes, dtype=np.int64)
    if len(indexes) == 0:
        return artifact.embedding(encoder_name, indexes)
    # Some Zarr backends require monotonically increasing fancy indexes. Restore
    # the coordinate-match order after the read so rowwise metrics stay valid.
    order = np.argsort(indexes)
    sorted_values = artifact.embedding(encoder_name, indexes[order])
    return sorted_values[np.argsort(order)]


def _interactive_coverage(latitude: Any, longitude: Any, name: str) -> Any:
    """Create a small offline Plotly map only when HTML output was requested."""
    import plotly.graph_objects as go

    return go.Figure(
        go.Scattergeo(
            lon=longitude,
            lat=latitude,
            mode="markers",
            name=name,
            marker={"size": 3, "opacity": 0.65, "color": "#1677b8"},
            hovertemplate="lat=%{lat:.4f}<br>lon=%{lon:.4f}<extra></extra>",
        )
    ).update_layout(title=f"Coverage: {name}", geo={"projection_type": "equirectangular"}, margin={"l": 0, "r": 0, "t": 45, "b": 0})


def generate_report(input_paths: Sequence[str | Path], config: ReportConfig) -> Path:
    """Generate an audit report and return its output directory.

    The function is additive: inputs are opened read-only and all generated
    figures, tables, and provenance remain underneath ``config.output_dir``.
    """
    if not input_paths:
        raise ValueError("At least one input artifact is required")
    if not isinstance(config, ReportConfig):
        raise TypeError("config must be a ReportConfig")

    # Heavy optional modules live behind this boundary so the base package does
    # not acquire a plotting import-time dependency.
    import numpy as np
    import pandas as pd
    from reporting.analysis import (
        comparison_metrics,
        equal_area_stratified_sample,
        group_temporal_artifacts,
        match_coordinates,
        project_embeddings,
        quality_statistics,
        temporal_displacement,
    )
    from reporting.artifacts import load_dataset_artifact
    from reporting.probes import load_probe_definitions, measure_probe_embeddings, sample_probe_rows
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

    paths = [Path(value).expanduser().resolve() for value in input_paths]
    if len(set(paths)) != len(paths):
        raise ValueError("Each input artifact may be specified only once")
    for path in paths:
        if not path.exists():
            raise ValueError(f"Input artifact does not exist: {path}")
    output, figures_dir, tables_dir, _assets_dir = _prepare_output(config)
    warnings: list[str] = []
    figure_paths: list[Path] = []
    sections: list[dict[str, Any]] = []
    try:
        artifacts = [load_dataset_artifact(path) for path in paths]
        used_ids: set[str] = set()
        labels = {id(artifact): _source_id(artifact, used_ids) for artifact in artifacts}
        samples: dict[int, np.ndarray] = {}
        for artifact in artifacts:
            samples[id(artifact)] = np.asarray(
                equal_area_stratified_sample(
                    artifact.latitude, artifact.longitude, config.analysis_sample_size, seed=config.seed
                ), dtype=np.int64,
            )

        # Coverage, provenance, and full-data streamed quality statistics.
        quality_rows: list[dict[str, Any]] = []
        for artifact in artifacts:
            label = labels[id(artifact)]
            latitude, longitude = np.asarray(artifact.latitude), np.asarray(artifact.longitude)
            coverage_path = figures_dir / f"coverage_{label}.png"
            figure_paths.append(render_coverage({"latitude": latitude, "longitude": longitude, "status": np.full(len(latitude), "valid")}, coverage_path, title=f"Coverage — {label}"))
            interactive = _interactive_coverage(latitude[samples[id(artifact)]], longitude[samples[id(artifact)]], label) if "html" in config.formats else None
            sections.append({"title": f"Coverage — {label}", "images": [coverage_path], "text": f"{artifact.n_points:,} input rows; map shows the deterministic analysis sample interactively.", "plotly_figure": interactive})
            for encoder_name in artifact.encoder_names:
                metrics = quality_statistics(artifact, encoder_name)
                quality_rows.append({"dataset": label, "encoder": encoder_name, **_scalar_metrics(metrics)})

        quality_frame = pd.DataFrame(quality_rows)
        quality_frame.to_csv(tables_dir / "quality_metrics.csv", index=False)
        if not quality_frame.empty:
            quality_path = figures_dir / "quality_metrics.png"
            figure_paths.append(render_quality(quality_frame, quality_path, title="Embedding quality summary"))
            sections.append({"title": "Quality and provenance", "images": [quality_path], "table": quality_frame, "text": "Quality metrics are computed over each full artifact; source-specific arrays are read in batches when supported."})

        # Per-encoder geographic RGB atlases fitted only on the stable sample.
        projections: dict[tuple[int, str], dict[str, Any]] = {}
        reducer_methods = ("pca", "ica") if config.no_umap else ("pca", "ica", "umap")
        for artifact in artifacts:
            label, indexes = labels[id(artifact)], samples[id(artifact)]
            latitude, longitude = np.asarray(artifact.latitude)[indexes], np.asarray(artifact.longitude)[indexes]
            for encoder_name in artifact.encoder_names:
                values = _embedding_rows(artifact, encoder_name, indexes)
                try:
                    projection_set = project_embeddings(values, methods=reducer_methods, seed=config.seed)
                except (ImportError, RuntimeError) as exc:
                    if config.no_umap:
                        raise
                    warnings.append(f"UMAP atlas skipped for {label}/{encoder_name}: {exc}")
                    projection_set = project_embeddings(values, methods=("pca", "ica"), seed=config.seed)
                projections[(id(artifact), encoder_name)] = projection_set
                for method, components in projection_set.items():
                    atlas_path = figures_dir / f"atlas_{label}_{encoder_name}_{method}.png"
                    figure_paths.append(render_atlas({"latitude": latitude, "longitude": longitude, "projection": components}, atlas_path, title=f"{method.upper()} atlas — {label} / {encoder_name}"))
                    sections.append({"title": f"{method.upper()} atlas — {label} / {encoder_name}", "images": [atlas_path], "text": "Projection colors are comparable only within a shared reducer fit; independent encoders use separate feature spaces."})

        # Every encoder pair is comparable when the feature metrics permit it;
        # coordinate matching applies whenever sources differ.
        series = [(artifact, encoder_name, f"{labels[id(artifact)]}/{encoder_name}") for artifact in artifacts for encoder_name in artifact.encoder_names]
        comparison_rows: list[dict[str, Any]] = []
        for index, (left_artifact, left_name, left_label) in enumerate(series):
            for right_artifact, right_name, right_label in series[index + 1:]:
                try:
                    if left_artifact is right_artifact:
                        left_indexes = right_indexes = samples[id(left_artifact)]
                        distances = np.zeros(len(left_indexes), dtype=float)
                    else:
                        match = match_coordinates(left_artifact, right_artifact, mode=config.match_mode, tolerance_km=config.match_tolerance_km)
                        left_indexes, right_indexes = match.left_indices, match.right_indices
                        distances = match.distances_km
                        if len(left_indexes) == 0:
                            raise ValueError("no matched coordinate rows")
                        if len(left_indexes) > config.analysis_sample_size:
                            rng = np.random.default_rng(config.seed)
                            picked = np.sort(rng.choice(len(left_indexes), config.analysis_sample_size, replace=False))
                            left_indexes, right_indexes, distances = left_indexes[picked], right_indexes[picked], distances[picked]
                    metrics = comparison_metrics(_embedding_rows(left_artifact, left_name, left_indexes), _embedding_rows(right_artifact, right_name, right_indexes), coordinates=np.column_stack((np.asarray(left_artifact.latitude)[left_indexes], np.asarray(left_artifact.longitude)[left_indexes])), seed=config.seed)
                    flattened = _scalar_metrics(metrics)
                    if "linear_cka" in flattened:
                        flattened["cka"] = flattened["linear_cka"]
                    comparison_rows.append({"encoder_a": left_label, "encoder_b": right_label, "matched_rows": int(len(left_indexes)), "max_match_distance_km": float(np.max(distances)) if len(distances) else 0.0, **flattened})
                except (ValueError, RuntimeError) as exc:
                    warnings.append(f"Skipped comparison {left_label} ↔ {right_label}: {exc}")
        comparison_frame = pd.DataFrame(comparison_rows)
        comparison_frame.to_csv(tables_dir / "comparison_metrics.csv", index=False)
        comparison_path = figures_dir / "comparison_metrics.png"
        figure_paths.append(render_comparison(comparison_frame, comparison_path, title="Cross-encoder comparison"))
        sections.append({"title": "Cross-encoder comparison", "images": [comparison_path], "table": comparison_frame, "text": "Metrics use the selected coordinate-match policy and deterministic analysis sample. " + (" ".join(warnings) if warnings else "")})

        # Consecutive compatible temporal artifacts are handled separately.
        temporal_rows: list[dict[str, Any]] = []
        for encoder_name, temporal_artifacts in group_temporal_artifacts(artifacts).items():
            for left_artifact, right_artifact in zip(temporal_artifacts, temporal_artifacts[1:]):
                try:
                    match = match_coordinates(left_artifact, right_artifact, mode=config.match_mode, tolerance_km=config.match_tolerance_km)
                    change = temporal_displacement(left_artifact, right_artifact, encoder_name, match=match)
                    temporal_rows.append({"encoder": encoder_name, "from": labels[id(left_artifact)], "to": labels[id(right_artifact)], **_scalar_metrics(change)})
                    temporal_path = figures_dir / f"temporal_{encoder_name}_{labels[id(left_artifact)]}_to_{labels[id(right_artifact)]}.png"
                    coordinates = np.asarray(change["coordinates"])
                    figure_paths.append(render_temporal({"latitude": coordinates[:, 0], "longitude": coordinates[:, 1], "displacement": change["normalized_displacement"]}, temporal_path, title=f"Temporal change — {encoder_name}"))
                    sections.append({"title": f"Temporal change — {encoder_name}", "images": [temporal_path], "text": f"{labels[id(left_artifact)]} to {labels[id(right_artifact)]}."})
                except (ValueError, RuntimeError) as exc:
                    warnings.append(f"Skipped temporal comparison for {encoder_name}: {exc}")
        temporal_frame = pd.DataFrame(temporal_rows)
        temporal_frame.to_csv(tables_dir / "temporal_metrics.csv", index=False)
        if temporal_frame.empty:
            sections.append({"title": "Temporal change", "text": "No compatible temporal series was detected."})

        probes = load_probe_definitions()
        if config.probes_path is not None:
            custom = load_probe_definitions(config.probes_path)
            overrides = {probe["name"]: probe for probe in custom}
            probes = [overrides.pop(probe["name"], probe) for probe in probes] + list(overrides.values())
        probe_frames: list[Any] = []
        for artifact in artifacts:
            sampled = [sample_probe_rows(artifact.latitude, artifact.longitude, probe) for probe in probes]
            rows = pd.concat(sampled, ignore_index=True)
            measurements = measure_probe_embeddings(rows, {name: artifact.embedding(name) for name in artifact.encoder_names})
            measurements.insert(0, "dataset", labels[id(artifact)])
            probe_frames.append(measurements)
        probe_frame = pd.concat(probe_frames, ignore_index=True) if probe_frames else pd.DataFrame()
        probe_frame.to_csv(tables_dir / "probe_measurements.csv", index=False)
        if not probe_frame.empty:
            probe_path = figures_dir / "geographic_probes.png"
            figure_paths.append(render_probes(probe_frame, probe_path))
            sections.append({"title": "Geographic probes", "images": [probe_path], "text": "Nearest-row distances and gap warnings are included in the exported probe table."})

        if "png" not in config.formats:
            # PNG figures feed HTML/PDF and are intentionally retained for a
            # self-contained report even when they were not explicitly listed.
            warnings.append("Static PNG assets were retained because HTML/PDF reports embed them.")
        manifest = {
            "report_schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": asdict(config),
            "inputs": [_source_fingerprint(path) for path in paths],
            "artifacts": [{"label": labels[id(item)], "metadata": item.metadata, "n_points": item.n_points, "encoders": list(item.encoder_names)} for item in artifacts],
            "sampling": {"strategy": "equal_area_stratified", "sample_size_cap": config.analysis_sample_size, "seed": config.seed},
            "warnings": warnings,
            "files": sorted(
                str(path.relative_to(output))
                for path in figure_paths + list(tables_dir.glob("*.csv"))
                + [output / "manifest.json"]
                + ([output / "index.html"] if "html" in config.formats else [])
                + ([output / "report.pdf"] if "pdf" in config.formats else [])
            ),
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default, sort_keys=True) + "\n", encoding="utf-8")
        if "html" in config.formats:
            build_html_report(output / "index.html", sections, title="Geospatial embedding audit report")
        if "pdf" in config.formats:
            write_pdf_report(figure_paths, output / "report.pdf", title="Geospatial embedding audit report")
        return output
    except Exception:
        # Do not leave a misleading partial report when load/analysis/rendering
        # fails. The user can inspect the original input artifacts unchanged.
        shutil.rmtree(output, ignore_errors=True)
        raise


__all__ = ["ReportConfig", "generate_report"]
