"""Offline static and interactive rendering helpers for embedding audit reports.

All public functions accept plain dictionaries, pandas DataFrames, and NumPy
arrays so the report orchestration layer does not need to expose internal types.
They return their written :class:`pathlib.Path` and never require network access.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _array(data: Mapping[str, Any] | pd.DataFrame, *names: str) -> np.ndarray:
    for name in names:
        if isinstance(data, pd.DataFrame) and name in data:
            return data[name].to_numpy()
        if isinstance(data, Mapping) and name in data:
            return np.asarray(data[name])
    raise ValueError(f"Expected one of {names!r}")


def _coordinates(data: Mapping[str, Any] | pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return _array(data, "latitude", "lat"), _array(data, "longitude", "lon")


def _save(fig: plt.Figure, output_path: str | Path) -> Path:
    path = _path(output_path)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _world_axis(fig: plt.Figure) -> Any:
    """Use Cartopy when installed, otherwise a standards-only lon/lat axis."""
    try:
        import cartopy.crs as ccrs  # type: ignore
        axis = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        axis.set_global()
        # Deliberately do not request Cartopy's Natural Earth features here.
        # Cartopy may download those shapefiles on first use, which violates the
        # report command's promise that it can render offline.
        axis.gridlines(draw_labels=False, linewidth=0.35, color="#8aa0b0", alpha=0.35)
        axis.set_facecolor("#eef6fb")
        axis._report_transform = ccrs.PlateCarree()  # type: ignore[attr-defined]
        return axis
    except (ImportError, ModuleNotFoundError):
        axis = fig.add_subplot(1, 1, 1)
        axis.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude", ylabel="Latitude", facecolor="#eef6fb")
        axis.grid(alpha=0.25, linewidth=0.4)
        axis._report_transform = None  # type: ignore[attr-defined]
        return axis


def _scatter_map(axis: Any, longitude: np.ndarray, latitude: np.ndarray, **kwargs: Any) -> Any:
    transform = getattr(axis, "_report_transform", None)
    if transform is not None:
        kwargs["transform"] = transform
    return axis.scatter(longitude, latitude, **kwargs)


def render_coverage(coverage: Mapping[str, Any] | pd.DataFrame, output_path: str | Path, *, title: str = "Coverage") -> Path:
    """Render a world map for coordinates with optional categorical status."""
    latitude, longitude = _coordinates(coverage)
    status = None
    try:
        status = _array(coverage, "status", "coverage_status")
    except ValueError:
        pass
    fig = plt.figure(figsize=(12, 5.6))
    axis = _world_axis(fig)
    if status is None:
        _scatter_map(axis, longitude, latitude, s=3, alpha=0.55, c="#1677b8", linewidths=0)
    else:
        statuses = pd.Series(status.astype(str))
        palette = {"valid": "#1b9e77", "invalid": "#d95f02", "unmatched": "#7570b3", "dropped": "#e7298a"}
        for item in sorted(statuses.unique()):
            selected = statuses.to_numpy() == item
            _scatter_map(axis, longitude[selected], latitude[selected], s=4, alpha=0.7,
                         c=palette.get(item.lower(), "#555555"), label=item, linewidths=0)
        axis.legend(loc="lower left", frameon=True, fontsize=8)
    axis.set_title(title)
    return _save(fig, output_path)


def render_atlas(atlas: Mapping[str, Any] | pd.DataFrame, output_path: str | Path, *, title: str = "Embedding atlas") -> Path:
    """Render an RGB geographic embedding projection.

    ``atlas`` needs latitude/longitude and either ``rgb`` (N×3) or
    ``projection``/``components`` (N×3); component values are robustly scaled.
    """
    latitude, longitude = _coordinates(atlas)
    raw = None
    for name in ("rgb", "projection", "components"):
        if isinstance(atlas, Mapping) and name in atlas:
            raw = np.asarray(atlas[name], dtype=float)
            break
    if raw is None or raw.ndim != 2 or raw.shape[0] != len(latitude) or raw.shape[1] < 3:
        raise ValueError("atlas requires an N×3 rgb, projection, or components array")
    rgb = raw[:, :3].copy()
    if np.nanmin(rgb) < 0 or np.nanmax(rgb) > 1:
        low, high = np.nanpercentile(rgb, [2, 98], axis=0)
        rgb = (rgb - low) / np.maximum(high - low, 1e-12)
    rgb = np.nan_to_num(np.clip(rgb, 0, 1), nan=0.5)
    fig = plt.figure(figsize=(12, 5.6))
    axis = _world_axis(fig)
    _scatter_map(axis, longitude, latitude, s=4, c=rgb, alpha=0.85, linewidths=0, rasterized=True)
    axis.set_title(title)
    return _save(fig, output_path)


def _as_metric_frame(data: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, Mapping):
        scalar = {key: value for key, value in data.items() if np.isscalar(value) or isinstance(value, str)}
        if scalar:
            return pd.DataFrame([scalar])
        return pd.DataFrame(data)
    raise TypeError("Expected a mapping or pandas DataFrame")


def render_quality(quality: Mapping[str, Any] | pd.DataFrame, output_path: str | Path, *, title: str = "Embedding quality") -> Path:
    """Render a readable quality metrics table and optional norm distribution."""
    frame = _as_metric_frame(quality)
    norm_values = None
    if isinstance(quality, Mapping) and "norms" in quality:
        norm_values = np.asarray(quality["norms"], dtype=float).reshape(-1)
    fig, axes = plt.subplots(1, 2 if norm_values is not None else 1, figsize=(12, 4.8))
    axes = np.atleast_1d(axes)
    axes[0].axis("off")
    display = frame.drop(columns=[column for column in frame if column == "norms"], errors="ignore").round(5)
    table = axes[0].table(cellText=display.astype(str).values, colLabels=display.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.35)
    axes[0].set_title(title)
    if norm_values is not None:
        values = norm_values[np.isfinite(norm_values)]
        axes[1].hist(values, bins=min(60, max(10, len(values) // 5)), color="#1677b8", alpha=0.85)
        axes[1].set(title="Embedding norm distribution", xlabel="L2 norm", ylabel="Rows")
        axes[1].grid(alpha=0.2)
    return _save(fig, output_path)


def render_comparison(comparison: Mapping[str, Any] | pd.DataFrame, output_path: str | Path, *, title: str = "Encoder comparison") -> Path:
    """Render a pairwise metric heatmap from long-form comparison results."""
    frame = _as_metric_frame(comparison)
    left = next((name for name in ("encoder_a", "left", "source") if name in frame), None)
    right = next((name for name in ("encoder_b", "right", "target") if name in frame), None)
    metric = next((name for name in ("cka", "mean_cosine_similarity", "neighborhood_agreement", "value") if name in frame), None)
    fig, axis = plt.subplots(figsize=(7, 6))
    if left is None or right is None or metric is None:
        axis.axis("off")
        axis.text(0.5, 0.5, "No pairwise comparison metrics available", ha="center", va="center")
    else:
        names = sorted(set(frame[left].astype(str)) | set(frame[right].astype(str)))
        matrix = pd.DataFrame(np.nan, index=names, columns=names)
        for row in frame[[left, right, metric]].dropna().itertuples(index=False):
            a, b, value = row
            matrix.loc[str(a), str(b)] = float(value); matrix.loc[str(b), str(a)] = float(value)
        values = matrix.to_numpy(dtype=float, copy=True)
        np.fill_diagonal(values, 1.0)
        matrix.iloc[:, :] = values
        image = axis.imshow(values, vmin=-1, vmax=1, cmap="viridis")
        axis.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names, title=title)
        plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
        for i in range(len(names)):
            for j in range(len(names)):
                value = matrix.iat[i, j]
                if np.isfinite(value): axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                                                  color="white" if value < 0.5 else "black")
        fig.colorbar(image, ax=axis, label=metric)
    return _save(fig, output_path)


def render_temporal(temporal: Mapping[str, Any] | pd.DataFrame, output_path: str | Path, *, title: str = "Temporal embedding change") -> Path:
    """Render a displacement world map or a temporal summary line chart."""
    fig = plt.figure(figsize=(12, 5.6))
    try:
        latitude, longitude = _coordinates(temporal)
        change = _array(temporal, "displacement", "cosine_change", "value").astype(float)
        axis = _world_axis(fig)
        scatter = _scatter_map(axis, longitude, latitude, s=5, c=change, cmap="magma", linewidths=0, alpha=0.85)
        fig.colorbar(scatter, ax=axis, label="Embedding change")
        axis.set_title(title)
    except ValueError:
        frame = _as_metric_frame(temporal)
        axis = fig.add_subplot(1, 1, 1)
        x = next((column for column in ("year", "year_to", "time") if column in frame), frame.columns[0])
        y = next((column for column in ("mean_displacement", "displacement", "mean_cosine_change", "value") if column in frame), frame.columns[-1])
        axis.plot(frame[x], frame[y], marker="o", color="#c23b22")
        axis.set(title=title, xlabel=x.replace("_", " "), ylabel=y.replace("_", " "))
        axis.grid(alpha=0.25)
    return _save(fig, output_path)


def render_probes(measurements: pd.DataFrame, output_path: str | Path, *, title: str = "Geographic probes") -> Path:
    """Render probe paths, embedding norms, and consecutive embedding distance."""
    if measurements.empty:
        raise ValueError("No probe measurements to render")
    required = {"probe", "latitude", "longitude"}
    if not required.issubset(measurements.columns):
        raise ValueError(f"measurements must contain {sorted(required)}")
    fig = plt.figure(figsize=(13, 8))
    axis_map = fig.add_subplot(2, 2, (1, 2))
    # A plain axis makes multiple paths legible and avoids a hard Cartopy dependency here.
    axis_map.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude", ylabel="Latitude", facecolor="#eef6fb", title=title)
    axis_map.grid(alpha=0.25)
    for name, rows in measurements.groupby("probe", sort=False):
        axis_map.plot(rows["longitude"], rows["latitude"], marker="o", markersize=2.5, linewidth=1.1, label=name)
    axis_map.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7)
    for axis, column, label in ((fig.add_subplot(2, 2, 3), "embedding_norm", "Embedding norm"),
                                (fig.add_subplot(2, 2, 4), "step_embedding_distance", "Consecutive distance")):
        if column in measurements:
            group_columns = ["probe"] + (["encoder"] if "encoder" in measurements else [])
            for names, rows in measurements.groupby(group_columns, sort=False):
                name = " / ".join(names) if isinstance(names, tuple) else str(names)
                axis.plot(rows.get("along_fraction", rows["sample_index"]), rows[column], label=name)
            axis.set(xlabel="Along probe fraction", ylabel=label)
            axis.grid(alpha=0.25)
    return _save(fig, output_path)


def _image_data_uri(path: str | Path) -> str:
    suffix = Path(path).suffix.lower().lstrip(".") or "png"
    mime = "image/svg+xml" if suffix == "svg" else f"image/{suffix}"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_html_report(output_path: str | Path, sections: Sequence[Mapping[str, Any]], *, title: str = "Geospatial embedding report") -> Path:
    """Build a self-contained HTML report from static images, tables, and Plotly figures.

    Plotly is optional. If absent, supplied Plotly figures are represented by a
    clear note while all static visualisations and tables stay available.
    """
    content: list[str] = []
    plotly_available = False
    try:
        import plotly.io as pio  # type: ignore
        plotly_available = True
    except ImportError:
        pio = None  # type: ignore
    for section in sections:
        heading = html.escape(str(section.get("title", "Untitled section")))
        body = [f"<section><h2>{heading}</h2>"]
        if section.get("text"):
            body.append(f"<p>{html.escape(str(section['text']))}</p>")
        table = section.get("table")
        if table is not None:
            frame = _as_metric_frame(table) if not isinstance(table, pd.DataFrame) else table
            body.append(frame.to_html(index=False, escape=True, classes="metrics"))
        for image in section.get("images", []):
            body.append(f'<img loading="lazy" src="{_image_data_uri(image)}" alt="{heading}">')
        figure = section.get("plotly_figure")
        if figure is not None:
            if plotly_available:
                body.append(pio.to_html(figure, include_plotlyjs="inline", full_html=False))
            else:
                body.append("<p class=warning>Interactive chart omitted because Plotly is not installed.</p>")
        body.append("</section>")
        content.append("\n".join(body))
    document = f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{html.escape(title)}</title><style>
body{{font-family:system-ui,sans-serif;max-width:1280px;margin:auto;padding:1.4rem;color:#18212b;background:#fafafa}} section{{background:white;padding:1.2rem;margin:1rem 0;border-radius:8px;box-shadow:0 1px 3px #0002}} img{{max-width:100%;height:auto;margin:.5rem 0}} table.metrics{{border-collapse:collapse;max-width:100%;overflow:auto;font-size:.9rem}} .metrics th,.metrics td{{border:1px solid #d8dee4;padding:.35rem .5rem;text-align:right}} .metrics th{{background:#edf3f8}} .warning{{color:#8a4b00}}</style></head><body><h1>{html.escape(title)}</h1>{''.join(content)}</body></html>"""
    path = _path(output_path)
    path.write_text(document, encoding="utf-8")
    return path


def write_pdf_report(figure_paths: Iterable[str | Path], output_path: str | Path, *, title: str = "Geospatial embedding report") -> Path:
    """Combine PNG/JPEG figures into a portable PDF without external converters."""
    path = _path(output_path)
    figures = [Path(figure) for figure in figure_paths]
    with PdfPages(path) as pdf:
        if not figures:
            figure = plt.figure(figsize=(8.27, 11.69))
            figure.text(0.5, 0.5, f"{title}\n\nNo static figures were generated.", ha="center", va="center")
            pdf.savefig(figure, bbox_inches="tight"); plt.close(figure)
        for image_path in figures:
            image = plt.imread(image_path)
            figure, axis = plt.subplots(figsize=(11.69, 8.27))
            axis.imshow(image); axis.axis("off"); axis.set_title(title, fontsize=10)
            pdf.savefig(figure, bbox_inches="tight"); plt.close(figure)
    return path
