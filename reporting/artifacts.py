"""Portable, validated access to generated geospatial embedding datasets.

The report subsystem deliberately uses this small abstraction instead of tying
analysis to Torch or Zarr.  A :class:`DatasetArtifact` always exposes public
coordinates in the repository-wide ``(latitude, longitude)`` order and named
two-dimensional embedding arrays.  Zarr arrays remain lazy until read through
``embedding`` or ``iter_embedding_batches``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd


class ArtifactValidationError(ValueError):
    """Raised when an input artifact cannot safely be interpreted."""


_COORDINATE_KEYS = {
    "coordinates",
    "coordinates_latlon",
    "coordinates_lonlat",
    "latitude",
    "longitude",
    "metadata",
    "metadata_json",
    "year",
}
_CSV_EMBEDDING_PATTERN = re.compile(r"^(?P<name>.+)_emb_(?P<dimension>\d+)$")


def _as_numpy(value: Any) -> np.ndarray:
    """Convert torch-like values to NumPy without requiring torch at import time."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _decode_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ArtifactValidationError("metadata is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ArtifactValidationError("metadata must be a mapping or JSON object")
    return dict(value)


def _validate_coordinates(latitude: np.ndarray, longitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    latitude = np.asarray(latitude, dtype=np.float64).reshape(-1)
    longitude = np.asarray(longitude, dtype=np.float64).reshape(-1)
    if latitude.size == 0:
        raise ArtifactValidationError("an artifact must contain at least one coordinate")
    if latitude.shape != longitude.shape:
        raise ArtifactValidationError("latitude and longitude must have equal length")
    if not np.isfinite(latitude).all() or not np.isfinite(longitude).all():
        raise ArtifactValidationError("coordinates must be finite")
    if np.any((latitude < -90) | (latitude > 90)):
        raise ArtifactValidationError("latitude values must be in [-90, 90]")
    if np.any((longitude < -180) | (longitude > 180)):
        raise ArtifactValidationError("longitude values must be in [-180, 180]")
    return latitude, longitude


def _coordinates_from_mapping(values: Mapping[str, Any], metadata: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Read explicit coordinate arrays while rejecting ambiguous ordering."""
    if "coordinates_latlon" in values:
        coordinates = _as_numpy(values["coordinates_latlon"])
        expected_order = "lat_lon"
    elif "coordinates" in values:
        coordinates = _as_numpy(values["coordinates"])
        expected_order = (metadata.get("coordinate_order") or {}).get("coordinates", "lat_lon")
    elif "latitude" in values and "longitude" in values:
        return _validate_coordinates(_as_numpy(values["latitude"]), _as_numpy(values["longitude"]))
    else:
        raise ArtifactValidationError(
            "artifact must contain coordinates_latlon, coordinates, or latitude/longitude"
        )
    if expected_order != "lat_lon":
        raise ArtifactValidationError(
            "reporting only accepts public coordinates in lat_lon order; "
            f"received {expected_order!r}"
        )
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ArtifactValidationError("coordinate array must have shape (n, 2)")
    return _validate_coordinates(coordinates[:, 0], coordinates[:, 1])


@dataclass
class DatasetArtifact:
    """A validated embedding dataset, with optional lazy backing arrays.

    ``embeddings`` values only need support NumPy-style slicing.  This permits
    Zarr datasets to be analysed in chunks without materializing whole arrays.
    """

    source: Path
    latitude: np.ndarray
    longitude: np.ndarray
    embeddings: Mapping[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source = Path(self.source)
        self.latitude, self.longitude = _validate_coordinates(self.latitude, self.longitude)
        self.metadata = dict(self.metadata)
        if not self.embeddings:
            raise ArtifactValidationError("artifact contains no named embeddings")
        checked: dict[str, Any] = {}
        for name, values in self.embeddings.items():
            if not isinstance(name, str) or not name:
                raise ArtifactValidationError("embedding names must be non-empty strings")
            shape = getattr(values, "shape", None)
            if shape is None or len(shape) != 2:
                shape = np.asarray(values).shape
            if len(shape) != 2 or shape[0] != self.latitude.size or shape[1] < 1:
                raise ArtifactValidationError(
                    f"embedding {name!r} must have shape (n, dimensions) matching coordinates"
                )
            checked[name] = values
        self.embeddings = checked
        declared_n = self.metadata.get("n_points")
        if declared_n is not None and int(declared_n) != self.n_points:
            raise ArtifactValidationError(
                f"metadata n_points={declared_n} does not match {self.n_points} coordinates"
            )

    @property
    def n_points(self) -> int:
        return int(self.latitude.size)

    @property
    def encoder_names(self) -> tuple[str, ...]:
        return tuple(self.embeddings)

    @property
    def year(self) -> int | None:
        value = self.metadata.get("year")
        return None if value is None else int(value)

    @property
    def coordinates(self) -> np.ndarray:
        """Coordinates as a newly allocated ``(n, 2)`` latitude/longitude array."""
        return np.column_stack((self.latitude, self.longitude))

    def embedding(self, name: str, indices: np.ndarray | None = None) -> np.ndarray:
        """Materialize one named embedding, optionally only selected row indices."""
        try:
            values = self.embeddings[name]
        except KeyError as exc:
            raise KeyError(f"unknown encoder {name!r}; available: {', '.join(self.encoder_names)}") from exc
        values = values[:] if indices is None else values[indices]
        return _as_numpy(values).astype(np.float64, copy=False)

    def iter_embedding_batches(
        self, name: str, batch_size: int = 100_000
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield ``(start, end, values)`` batches without forcing Zarr into memory."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        values = self.embeddings[name]
        for start in range(0, self.n_points, batch_size):
            end = min(start + batch_size, self.n_points)
            yield start, end, _as_numpy(values[start:end]).astype(np.float64, copy=False)


def _embedding_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    declared = values.get("metadata", {})
    if isinstance(declared, Mapping):
        expected = set(declared.get("encoders") or ())
    else:
        expected = set()
    for key, value in values.items():
        if key in _COORDINATE_KEYS:
            continue
        name = key[:-11] if key.endswith("_embeddings") else key
        shape = getattr(value, "shape", None)
        if shape is None:
            shape = np.asarray(value).shape
        if len(shape) == 2 and (key.endswith("_embeddings") or key in expected):
            result[name] = value
    return result


def _load_pt(path: Path) -> DatasetArtifact:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - core dependency in this project
        raise ImportError("loading .pt artifacts requires torch") from exc
    values = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(values, Mapping):
        raise ArtifactValidationError(".pt artifact must be a mapping")
    metadata = _decode_metadata(values.get("metadata"))
    latitude, longitude = _coordinates_from_mapping(values, metadata)
    return DatasetArtifact(path, latitude, longitude, _embedding_mapping(values), metadata)


def _load_csv(path: Path) -> DatasetArtifact:
    frame = pd.read_csv(path)
    if not {"latitude", "longitude"}.issubset(frame.columns):
        raise ArtifactValidationError("CSV artifact must contain latitude and longitude columns")
    grouped: dict[str, list[tuple[int, str]]] = {}
    for column in frame.columns:
        match = _CSV_EMBEDDING_PATTERN.match(column)
        if match:
            grouped.setdefault(match.group("name"), []).append((int(match.group("dimension")), column))
    embeddings: dict[str, np.ndarray] = {}
    for name, columns in grouped.items():
        columns.sort()
        dimensions = [dimension for dimension, _ in columns]
        if dimensions != list(range(dimensions[-1] + 1)):
            raise ArtifactValidationError(f"CSV embedding {name!r} has non-contiguous dimensions")
        embeddings[name] = frame[[column for _, column in columns]].to_numpy(dtype=np.float64)
    metadata: dict[str, Any] = {"n_points": len(frame), "encoders": list(embeddings)}
    if "year" in frame.columns:
        years = frame["year"].dropna().unique()
        if len(years) == 1:
            metadata["year"] = int(years[0])
        elif len(years) > 1:
            raise ArtifactValidationError("a CSV artifact must contain a single year")
    return DatasetArtifact(path, frame["latitude"].to_numpy(), frame["longitude"].to_numpy(), embeddings, metadata)


def _load_zarr(path: Path) -> DatasetArtifact:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("loading .zarr artifacts requires the optional zarr dependency") from exc
    group = zarr.open_group(str(path), mode="r")
    metadata = _decode_metadata(group.attrs.get("metadata"))
    keys = list(group.array_keys()) if hasattr(group, "array_keys") else list(group.keys())
    values = {key: group[key] for key in keys}
    latitude, longitude = _coordinates_from_mapping(values, metadata)
    return DatasetArtifact(path, latitude, longitude, _embedding_mapping(values), metadata)


def load_dataset_artifact(path: str | Path) -> DatasetArtifact:
    """Load a generated ``.pt``, wide ``.csv``, or ``.zarr`` dataset safely.

    The loader never guesses a lon/lat ordering.  Legacy generic ``coordinates``
    are accepted only as the established repository default, ``lat_lon``.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"dataset artifact does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".pt":
        return _load_pt(source)
    if suffix == ".csv":
        return _load_csv(source)
    if suffix == ".zarr" or source.is_dir() and source.name.endswith(".zarr"):
        return _load_zarr(source)
    raise ValueError("supported artifact formats are .pt, .csv, and .zarr")
