"""Reproducible geographic probe definitions and nearest-row measurements.

The probe format deliberately uses the repository-wide ``(latitude, longitude)``
convention.  It does not depend on raster, model, or network packages.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


class ProbeConfigurationError(ValueError):
    """Raised when a probe JSON document does not satisfy the public schema."""


def _as_coordinates(value: Any, name: str) -> list[tuple[float, float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        raise ProbeConfigurationError(f"Probe {name!r} must contain at least two coordinates")
    coordinates: list[tuple[float, float]] = []
    for index, coordinate in enumerate(value):
        if not isinstance(coordinate, Sequence) or isinstance(coordinate, (str, bytes)) or len(coordinate) != 2:
            raise ProbeConfigurationError(f"Probe {name!r} coordinate {index} must be [latitude, longitude]")
        latitude, longitude = float(coordinate[0]), float(coordinate[1])
        if not np.isfinite(latitude) or not np.isfinite(longitude) or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ProbeConfigurationError(f"Probe {name!r} coordinate {index} is outside valid latitude/longitude ranges")
        coordinates.append((latitude, longitude))
    return coordinates


def _normalise_probe(probe: Mapping[str, Any]) -> dict[str, Any]:
    name = probe.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ProbeConfigurationError("Every probe requires a non-empty name")
    coordinates = _as_coordinates(probe.get("coordinates"), name)
    sample_count = int(probe.get("sample_count", 50))
    if sample_count < 2:
        raise ProbeConfigurationError(f"Probe {name!r} sample_count must be at least 2")
    return {
        "name": name,
        "description": str(probe.get("description", "")),
        "coordinates": coordinates,
        "sample_count": sample_count,
    }


def load_probe_definitions(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load validated built-in probes or a JSON file in the same schema.

    Custom files may contain either a top-level ``{"probes": [...]}`` object or
    a list of probe objects.  Names must be unique so output tables are stable.
    """
    if path is None:
        text = resources.files(__package__).joinpath("default_probes.json").read_text(encoding="utf-8")
        source = "built-in probes"
    else:
        source_path = Path(path)
        try:
            text = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ProbeConfigurationError(f"Cannot read probe configuration {source_path}: {exc}") from exc
        source = str(source_path)
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProbeConfigurationError(f"Invalid JSON in {source}: {exc}") from exc
    entries = document.get("probes") if isinstance(document, Mapping) else document
    if not isinstance(entries, list):
        raise ProbeConfigurationError(f"{source} must be a list or an object with a 'probes' list")
    probes = [_normalise_probe(entry) for entry in entries if isinstance(entry, Mapping)]
    if len(probes) != len(entries):
        raise ProbeConfigurationError(f"Every probe in {source} must be an object")
    names = [probe["name"] for probe in probes]
    if len(set(names)) != len(names):
        raise ProbeConfigurationError(f"Probe names in {source} must be unique")
    return probes


def haversine_km(latitude_a: Any, longitude_a: Any, latitude_b: Any, longitude_b: Any) -> np.ndarray:
    """Return great-circle distances in kilometres with NumPy broadcasting."""
    lat_a, lon_a, lat_b, lon_b = np.broadcast_arrays(
        np.asarray(latitude_a, dtype=float), np.asarray(longitude_a, dtype=float),
        np.asarray(latitude_b, dtype=float), np.asarray(longitude_b, dtype=float),
    )
    lat_a, lon_a, lat_b, lon_b = map(np.deg2rad, (lat_a, lon_a, lat_b, lon_b))
    dlat = lat_b - lat_a
    dlon = (lon_b - lon_a + np.pi) % (2 * np.pi) - np.pi
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def interpolate_polyline(coordinates: Sequence[Sequence[float]], sample_count: int) -> np.ndarray:
    """Linearly interpolate a lat/lon polyline by great-circle segment length.

    Longitude interpolation takes the short way around the antimeridian.  This
    is intentionally a lightweight visual/probe sampler, not a geodesic solver.
    """
    coords = np.asarray(_as_coordinates(coordinates, "probe"), dtype=float)
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    segment_lengths = haversine_km(coords[:-1, 0], coords[:-1, 1], coords[1:, 0], coords[1:, 1])
    total = float(segment_lengths.sum())
    if total == 0:
        return np.repeat(coords[:1], sample_count, axis=0)
    targets = np.linspace(0.0, total, sample_count)
    starts = np.r_[0.0, np.cumsum(segment_lengths)[:-1]]
    segment_indexes = np.minimum(np.searchsorted(starts + segment_lengths, targets, side="right"), len(segment_lengths) - 1)
    fractions = (targets - starts[segment_indexes]) / segment_lengths[segment_indexes]
    begins, ends = coords[segment_indexes], coords[segment_indexes + 1]
    latitude = begins[:, 0] + fractions * (ends[:, 0] - begins[:, 0])
    delta_lon = (ends[:, 1] - begins[:, 1] + 180.0) % 360.0 - 180.0
    longitude = (begins[:, 1] + fractions * delta_lon + 180.0) % 360.0 - 180.0
    return np.column_stack((latitude, longitude))


def sample_probe_rows(
    latitude: Any,
    longitude: Any,
    probe: Mapping[str, Any],
    *,
    sample_count: int | None = None,
    max_gap_km: float = 100.0,
) -> pd.DataFrame:
    """Select the closest dataset row to each evenly-spaced probe location.

    Returns a table that retains requested and selected coordinates, source row
    indexes, and a boolean gap warning. Repeated selected rows are permitted:
    that accurately reflects a sparse source dataset.
    """
    lat = np.asarray(latitude, dtype=float).reshape(-1)
    lon = np.asarray(longitude, dtype=float).reshape(-1)
    if lat.shape != lon.shape or len(lat) == 0:
        raise ValueError("latitude and longitude must be non-empty arrays of equal length")
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError("dataset coordinates must be finite")
    normalised = _normalise_probe(probe)
    requested = interpolate_polyline(normalised["coordinates"], sample_count or normalised["sample_count"])
    nearest_indexes = np.empty(len(requested), dtype=int)
    nearest_distances = np.empty(len(requested), dtype=float)
    # Iteration prevents accidentally materialising N_probe * N_dataset matrices.
    for i, (requested_lat, requested_lon) in enumerate(requested):
        distances = haversine_km(requested_lat, requested_lon, lat, lon)
        nearest_indexes[i] = int(np.argmin(distances))
        nearest_distances[i] = float(distances[nearest_indexes[i]])
    fractions = np.linspace(0.0, 1.0, len(requested))
    return pd.DataFrame({
        "probe": normalised["name"],
        "probe_description": normalised["description"],
        "sample_index": np.arange(len(requested), dtype=int),
        "along_fraction": fractions,
        "requested_latitude": requested[:, 0],
        "requested_longitude": requested[:, 1],
        "row_index": nearest_indexes,
        "latitude": lat[nearest_indexes],
        "longitude": lon[nearest_indexes],
        "nearest_distance_km": nearest_distances,
        "gap_warning": nearest_distances > max_gap_km,
    })


def measure_probe_embeddings(
    sampled_rows: pd.DataFrame,
    embeddings: Mapping[str, Any],
    *,
    projections: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Add norms, consecutive distances, and optional projection values to probes."""
    required = {"probe", "sample_index", "row_index"}
    if not required.issubset(sampled_rows.columns):
        raise ValueError(f"sampled_rows must contain {sorted(required)}")
    tables: list[pd.DataFrame] = []
    for encoder_name, values in embeddings.items():
        array = np.asarray(values, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"Embeddings for {encoder_name!r} must be two-dimensional")
        indices = sampled_rows["row_index"].to_numpy(dtype=int)
        if np.any(indices < 0) or np.any(indices >= len(array)):
            raise ValueError(f"Probe row indexes are out of bounds for {encoder_name!r}")
        selected = array[indices]
        table = sampled_rows.copy()
        table.insert(1, "encoder", encoder_name)
        table["embedding_norm"] = np.linalg.norm(selected, axis=1)
        table["step_embedding_distance"] = np.nan
        for _, index in table.groupby("probe", sort=False).groups.items():
            rows = np.asarray(list(index), dtype=int)
            table.loc[rows[1:], "step_embedding_distance"] = np.linalg.norm(np.diff(selected[rows], axis=0), axis=1)
        if projections is not None and encoder_name in projections:
            projected = np.asarray(projections[encoder_name], dtype=float)
            if projected.ndim != 2 or projected.shape[0] < len(array):
                raise ValueError(f"Projection for {encoder_name!r} must have one row per embedding")
            projected = projected[indices]
            for component in range(min(3, projected.shape[1])):
                table[f"projection_{component + 1}"] = projected[:, component]
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
