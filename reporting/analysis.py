"""Deterministic numerical analysis for geospatial embedding reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .artifacts import DatasetArtifact

EARTH_RADIUS_KM = 6_371.0088


class CoordinateMatchError(ValueError):
    """Raised when artifacts cannot be matched under a requested policy."""


@dataclass(frozen=True)
class CoordinateMatch:
    """One-to-one row correspondences between two coordinate arrays."""

    left_indices: np.ndarray
    right_indices: np.ndarray
    distances_km: np.ndarray
    mode: str

    @property
    def n_matches(self) -> int:
        return int(self.left_indices.size)


def _coordinates(value: DatasetArtifact | np.ndarray) -> np.ndarray:
    coordinates = value.coordinates if isinstance(value, DatasetArtifact) else np.asarray(value)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (n, 2) in latitude/longitude order")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite")
    return coordinates.astype(np.float64, copy=False)


def haversine_km(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Great-circle distance between broadcast-compatible lat/lon arrays."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    lat1, lon1 = np.deg2rad(left[..., 0]), np.deg2rad(left[..., 1])
    lat2, lon2 = np.deg2rad(right[..., 0]), np.deg2rad(right[..., 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    term = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(np.clip(term, 0.0, 1.0)))


def equal_area_stratified_sample(
    latitude: np.ndarray,
    longitude: np.ndarray,
    sample_size: int,
    seed: int = 42,
    latitude_bins: int = 18,
    longitude_bins: int = 36,
) -> np.ndarray:
    """Return deterministic row indices balanced across equal-area world cells.

    Latitude cells are uniform in ``sin(latitude)``, giving them equal surface
    area.  The allocation gives every occupied cell an initial equal quota and
    fills leftover slots uniformly from rows not already selected.
    """
    latitude = np.asarray(latitude, dtype=np.float64).reshape(-1)
    longitude = np.asarray(longitude, dtype=np.float64).reshape(-1)
    if latitude.shape != longitude.shape:
        raise ValueError("latitude and longitude must have equal length")
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if latitude_bins < 1 or longitude_bins < 1:
        raise ValueError("bin counts must be positive")
    count = latitude.size
    if count <= sample_size:
        return np.arange(count, dtype=np.int64)
    sin_latitude = (np.sin(np.deg2rad(latitude)) + 1.0) / 2.0
    lat_cell = np.minimum((sin_latitude * latitude_bins).astype(int), latitude_bins - 1)
    lon_normalized = ((longitude + 180.0) % 360.0) / 360.0
    lon_cell = np.minimum((lon_normalized * longitude_bins).astype(int), longitude_bins - 1)
    cell_id = lat_cell * longitude_bins + lon_cell
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    occupied = np.unique(cell_id)
    quota = max(1, sample_size // occupied.size)
    for cell in occupied:
        candidates = np.flatnonzero(cell_id == cell)
        take = min(quota, candidates.size)
        selected.append(rng.choice(candidates, size=take, replace=False))
    result = np.concatenate(selected)
    if result.size > sample_size:
        result = rng.choice(result, size=sample_size, replace=False)
    elif result.size < sample_size:
        available = np.setdiff1d(np.arange(count), result, assume_unique=False)
        fill = rng.choice(available, size=sample_size - result.size, replace=False)
        result = np.concatenate((result, fill))
    return np.sort(result.astype(np.int64, copy=False))


def _duplicate_coordinate_indices(coordinates: np.ndarray) -> np.ndarray:
    _, index, counts = np.unique(coordinates, axis=0, return_index=True, return_counts=True)
    return np.sort(index[counts > 1])


def match_coordinates(
    left: DatasetArtifact | np.ndarray,
    right: DatasetArtifact | np.ndarray,
    mode: str = "strict",
    tolerance_km: float = 1.0,
) -> CoordinateMatch:
    """Match coordinate rows despite ordering, using exact or bounded-nearest policy.

    ``strict`` requires both sides to have the same unique coordinate set.  The
    nearest policy uses all radius candidates and deterministic distance-ordered
    greedy assignment to produce a one-to-one set of matches.
    """
    left_coordinates, right_coordinates = _coordinates(left), _coordinates(right)
    if mode not in {"strict", "nearest"}:
        raise ValueError("mode must be 'strict' or 'nearest'")
    if tolerance_km < 0:
        raise ValueError("tolerance_km must be non-negative")
    if mode == "strict":
        if _duplicate_coordinate_indices(left_coordinates).size or _duplicate_coordinate_indices(right_coordinates).size:
            raise CoordinateMatchError("strict matching does not permit duplicate coordinate rows")
        right_lookup = {tuple(row): index for index, row in enumerate(right_coordinates)}
        try:
            right_indices = np.asarray([right_lookup[tuple(row)] for row in left_coordinates], dtype=np.int64)
        except KeyError as exc:
            raise CoordinateMatchError("strict matching requires identical coordinate sets") from exc
        if left_coordinates.shape[0] != right_coordinates.shape[0] or len(set(right_indices)) != right_coordinates.shape[0]:
            raise CoordinateMatchError("strict matching requires identical coordinate sets")
        return CoordinateMatch(
            np.arange(left_coordinates.shape[0], dtype=np.int64), right_indices,
            np.zeros(left_coordinates.shape[0], dtype=np.float64), mode,
        )

    if left_coordinates.size == 0 or right_coordinates.size == 0:
        return CoordinateMatch(np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=float), mode)
    try:
        from sklearn.neighbors import BallTree
    except ImportError as exc:  # pragma: no cover - reporting extra declares sklearn
        raise ImportError("nearest coordinate matching requires scikit-learn") from exc
    tree = BallTree(np.deg2rad(right_coordinates), metric="haversine")
    candidate_rows, candidate_distances = tree.query_radius(
        np.deg2rad(left_coordinates), r=tolerance_km / EARTH_RADIUS_KM,
        return_distance=True, sort_results=True,
    )
    edges: list[tuple[float, int, int]] = []
    for left_index, (rows, distances) in enumerate(zip(candidate_rows, candidate_distances)):
        edges.extend((float(distance * EARTH_RADIUS_KM), left_index, int(right_index)) for right_index, distance in zip(rows, distances))
    edges.sort()
    used_left: set[int] = set()
    used_right: set[int] = set()
    accepted: list[tuple[int, int, float]] = []
    for distance, left_index, right_index in edges:
        if left_index not in used_left and right_index not in used_right:
            used_left.add(left_index)
            used_right.add(right_index)
            accepted.append((left_index, right_index, distance))
    accepted.sort(key=lambda item: item[0])
    return CoordinateMatch(
        np.asarray([item[0] for item in accepted], dtype=np.int64),
        np.asarray([item[1] for item in accepted], dtype=np.int64),
        np.asarray([item[2] for item in accepted], dtype=np.float64), mode,
    )


def _json_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def quality_statistics(
    artifact: DatasetArtifact,
    encoder_name: str,
    batch_size: int = 100_000,
) -> dict[str, Any]:
    """Compute streaming, JSON-safe quality statistics for one embedding array."""
    dimensions = int(artifact.embeddings[encoder_name].shape[1])
    finite_values = np.zeros(dimensions, dtype=np.int64)
    sums = np.zeros(dimensions, dtype=np.float64)
    sums_squared = np.zeros(dimensions, dtype=np.float64)
    finite_rows = 0
    zero_rows = 0
    norms: list[np.ndarray] = []
    for _, _, values in artifact.iter_embedding_batches(encoder_name, batch_size):
        finite = np.isfinite(values)
        finite_values += finite.sum(axis=0)
        cleaned = np.where(finite, values, 0.0)
        sums += cleaned.sum(axis=0)
        sums_squared += (cleaned * cleaned).sum(axis=0)
        row_finite = finite.all(axis=1)
        valid = values[row_finite]
        finite_rows += int(row_finite.sum())
        if valid.size:
            batch_norms = np.linalg.norm(valid, axis=1)
            zero_rows += int(np.count_nonzero(np.isclose(batch_norms, 0.0)))
            norms.append(batch_norms)
    means = np.divide(sums, finite_values, out=np.full(dimensions, np.nan), where=finite_values > 0)
    variance = np.divide(sums_squared, finite_values, out=np.full(dimensions, np.nan), where=finite_values > 0) - means**2
    all_norms = np.concatenate(norms) if norms else np.array([], dtype=float)
    return {
        "encoder": encoder_name,
        "n_rows": artifact.n_points,
        "dimensions": dimensions,
        "finite_value_rate": _json_float(float(finite_values.sum() / (artifact.n_points * dimensions))),
        "finite_row_rate": _json_float(float(finite_rows / artifact.n_points)),
        "zero_vector_rate": _json_float(float(zero_rows / finite_rows)) if finite_rows else None,
        "duplicate_coordinate_count": int(artifact.n_points - np.unique(artifact.coordinates, axis=0).shape[0]),
        "norm": {
            "mean": _json_float(float(np.mean(all_norms))) if all_norms.size else None,
            "std": _json_float(float(np.std(all_norms))) if all_norms.size else None,
            "min": _json_float(float(np.min(all_norms))) if all_norms.size else None,
            "max": _json_float(float(np.max(all_norms))) if all_norms.size else None,
        },
        "dimension_mean": [_json_float(value) for value in means],
        "dimension_std": [_json_float(value) for value in np.sqrt(np.maximum(variance, 0.0))],
    }


def _standardized(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values).all(axis=1)
    if valid.sum() < 2:
        raise ValueError("at least two finite embedding rows are required")
    selected = values[valid]
    scale = selected.std(axis=0)
    scale[scale == 0] = 1.0
    return (selected - selected.mean(axis=0)) / scale, valid


def project_embeddings(
    values: np.ndarray,
    methods: Sequence[str] = ("pca", "ica"),
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Create three-channel PCA/ICA/optional-UMAP projections.

    Output arrays retain input row count; rows containing non-finite source
    values are filled with NaN.  Fewer than three possible components are
    zero-padded, which keeps renderer interfaces uniform for tiny fixtures.
    """
    from sklearn.decomposition import FastICA, PCA

    standardized, valid = _standardized(values)
    components = min(3, standardized.shape[0], standardized.shape[1])
    output: dict[str, np.ndarray] = {}
    for method in methods:
        if method not in {"pca", "ica", "umap"}:
            raise ValueError(f"unknown projection method {method!r}")
        if method == "pca":
            projected = PCA(n_components=components, random_state=seed).fit_transform(standardized)
        elif method == "ica":
            projected = FastICA(n_components=components, random_state=seed, whiten="unit-variance", max_iter=1000).fit_transform(standardized)
        else:
            try:
                import umap
            except ImportError as exc:
                raise ImportError("UMAP projection requires optional umap-learn") from exc
            projected = umap.UMAP(n_components=components, random_state=seed).fit_transform(standardized)
        padded = np.zeros((projected.shape[0], 3), dtype=np.float64)
        padded[:, :components] = projected
        full = np.full((np.asarray(values).shape[0], 3), np.nan, dtype=np.float64)
        full[valid] = padded
        output[method] = full
    return output


def _row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = left_norm * right_norm
    return np.divide(np.sum(left * right, axis=1), denominator, out=np.full(left.shape[0], np.nan), where=denominator > 0)


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Return centered linear CKA, invariant to isotropic feature scaling."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape[0] != right.shape[0] or left.shape[0] < 2:
        raise ValueError("CKA requires equal row counts of at least two")
    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    left, right = left[valid], right[valid]
    if left.shape[0] < 2:
        return float("nan")
    left = left - left.mean(axis=0)
    right = right - right.mean(axis=0)
    cross = np.linalg.norm(left.T @ right, ord="fro") ** 2
    denominator = np.linalg.norm(left.T @ left, ord="fro") * np.linalg.norm(right.T @ right, ord="fro")
    return float(cross / denominator) if denominator else float("nan")


def neighborhood_agreement(left: np.ndarray, right: np.ndarray, k: int = 10) -> float:
    """Mean overlap of cosine-nearest-neighbor sets in two embedding spaces."""
    if k < 1:
        raise ValueError("k must be positive")
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape[0] != right.shape[0]:
        raise ValueError("neighbor agreement requires equal row counts")
    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    left, right = left[valid], right[valid]
    if left.shape[0] < 2:
        return float("nan")
    from sklearn.neighbors import NearestNeighbors

    use_k = min(k, left.shape[0] - 1)
    # Query the fitted rows explicitly.  ``kneighbors(X=None)`` has a special
    # self-exclusion path in scikit-learn and rejects n_neighbors == n_samples.
    left_neighbors = NearestNeighbors(n_neighbors=use_k + 1, metric="cosine").fit(left).kneighbors(left, return_distance=False)[:, 1:]
    right_neighbors = NearestNeighbors(n_neighbors=use_k + 1, metric="cosine").fit(right).kneighbors(right, return_distance=False)[:, 1:]
    return float(np.mean([len(set(a).intersection(b)) / use_k for a, b in zip(left_neighbors, right_neighbors)]))


def geodesic_similarity_curve(
    values: np.ndarray,
    coordinates: np.ndarray,
    seed: int = 42,
    n_pairs: int = 10_000,
    distance_bins_km: Sequence[float] = (0, 10, 50, 100, 500, 1_000, 5_000, 20_100),
) -> list[dict[str, Any]]:
    """Summarize cosine similarity by randomly sampled geodesic distance bands."""
    values, coordinates = np.asarray(values, dtype=np.float64), _coordinates(coordinates)
    valid = np.isfinite(values).all(axis=1)
    values, coordinates = values[valid], coordinates[valid]
    if values.shape[0] < 2:
        return []
    rng = np.random.default_rng(seed)
    pairs = min(n_pairs, values.shape[0] * (values.shape[0] - 1) // 2)
    left_indices = rng.integers(0, values.shape[0], size=pairs)
    right_indices = rng.integers(0, values.shape[0], size=pairs)
    same = left_indices == right_indices
    while np.any(same):
        right_indices[same] = rng.integers(0, values.shape[0], size=int(same.sum()))
        same = left_indices == right_indices
    distances = haversine_km(coordinates[left_indices], coordinates[right_indices])
    similarity = _row_cosine(values[left_indices], values[right_indices])
    bins = np.asarray(distance_bins_km, dtype=float)
    if bins.ndim != 1 or bins.size < 2 or np.any(np.diff(bins) <= 0):
        raise ValueError("distance_bins_km must be increasing with at least two boundaries")
    records: list[dict[str, Any]] = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (distances >= lower) & ((distances < upper) if upper != bins[-1] else (distances <= upper))
        selected = similarity[mask]
        records.append({
            "distance_min_km": float(lower), "distance_max_km": float(upper), "n_pairs": int(mask.sum()),
            "mean_cosine_similarity": _json_float(float(np.nanmean(selected))) if selected.size else None,
            "p05_cosine_similarity": _json_float(float(np.nanquantile(selected, 0.05))) if selected.size else None,
            "p95_cosine_similarity": _json_float(float(np.nanquantile(selected, 0.95))) if selected.size else None,
        })
    return records


def comparison_metrics(
    left: np.ndarray,
    right: np.ndarray,
    coordinates: np.ndarray | None = None,
    seed: int = 42,
    n_pairs: int = 10_000,
) -> dict[str, Any]:
    """Calculate complementary pairwise embedding comparison metrics."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("comparison requires two-dimensional embeddings with equal row counts")
    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    left, right = left[valid], right[valid]
    comparable_dimensions = left.shape[1] == right.shape[1]
    if comparable_dimensions:
        cosine = _row_cosine(left, right)
        norms_left = np.linalg.norm(left, axis=1, keepdims=True)
        norms_right = np.linalg.norm(right, axis=1, keepdims=True)
        normalized_distance = np.linalg.norm(
            np.divide(left, norms_left, out=np.zeros_like(left), where=norms_left > 0)
            - np.divide(right, norms_right, out=np.zeros_like(right), where=norms_right > 0), axis=1
        )
    else:
        cosine = np.array([], dtype=float)
        normalized_distance = np.array([], dtype=float)
    result: dict[str, Any] = {
        "n_valid_rows": int(valid.sum()),
        "same_embedding_dimension": comparable_dimensions,
        "mean_row_cosine_similarity": _json_float(float(np.nanmean(cosine))) if cosine.size else None,
        "mean_normalized_distance": _json_float(float(np.nanmean(normalized_distance))) if normalized_distance.size else None,
        "linear_cka": _json_float(linear_cka(left, right)) if left.shape[0] >= 2 else None,
        "neighborhood_agreement": {
            "k10": _json_float(neighborhood_agreement(left, right, 10)),
            "k50": _json_float(neighborhood_agreement(left, right, 50)),
        } if left.shape[0] >= 2 else {"k10": None, "k50": None},
    }
    if coordinates is not None and comparable_dimensions:
        coordinates = _coordinates(coordinates)
        if coordinates.shape[0] != valid.shape[0]:
            raise ValueError("coordinates must have one row per original embedding row")
        result["geodesic_similarity_curve"] = geodesic_similarity_curve(left, coordinates[valid], seed, n_pairs)
    return result


def group_temporal_artifacts(artifacts: Iterable[DatasetArtifact]) -> dict[str, list[DatasetArtifact]]:
    """Group artifacts by shared encoder name, ordered by their metadata year."""
    groups: dict[str, list[DatasetArtifact]] = {}
    for artifact in artifacts:
        if artifact.year is None:
            continue
        for encoder in artifact.encoder_names:
            groups.setdefault(encoder, []).append(artifact)
    for name in tuple(groups):
        groups[name].sort(key=lambda artifact: artifact.year if artifact.year is not None else -1)
        if len(groups[name]) < 2:
            del groups[name]
    return groups


def temporal_displacement(
    left: DatasetArtifact,
    right: DatasetArtifact,
    encoder_name: str,
    match: CoordinateMatch | None = None,
    mode: str = "strict",
    tolerance_km: float = 1.0,
) -> dict[str, Any]:
    """Calculate per-location normalized embedding change between two years."""
    if match is None:
        match = match_coordinates(left, right, mode=mode, tolerance_km=tolerance_km)
    left_values = left.embedding(encoder_name, match.left_indices)
    right_values = right.embedding(encoder_name, match.right_indices)
    valid = np.isfinite(left_values).all(axis=1) & np.isfinite(right_values).all(axis=1)
    left_values, right_values = left_values[valid], right_values[valid]
    left_norm = np.linalg.norm(left_values, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right_values, axis=1, keepdims=True)
    normalized_left = np.divide(left_values, left_norm, out=np.zeros_like(left_values), where=left_norm > 0)
    normalized_right = np.divide(right_values, right_norm, out=np.zeros_like(right_values), where=right_norm > 0)
    displacement = np.linalg.norm(normalized_left - normalized_right, axis=1)
    cosine_change = 1.0 - _row_cosine(left_values, right_values)
    coordinates = left.coordinates[match.left_indices][valid]
    return {
        "encoder": encoder_name,
        "left_year": left.year,
        "right_year": right.year,
        "coordinates": coordinates,
        # Named aliases make this payload directly consumable by the static and
        # interactive map renderers without forcing callers to unpack the
        # coordinate matrix themselves.
        "latitude": coordinates[:, 0],
        "longitude": coordinates[:, 1],
        "left_indices": match.left_indices[valid],
        "right_indices": match.right_indices[valid],
        "match_distances_km": match.distances_km[valid],
        "normalized_displacement": displacement,
        "displacement": displacement,
        "cosine_change": cosine_change,
        "summary": {
            "n_matched_rows": match.n_matches,
            "n_valid_rows": int(valid.sum()),
            "mean_normalized_displacement": _json_float(float(np.nanmean(displacement))) if displacement.size else None,
            "mean_cosine_change": _json_float(float(np.nanmean(cosine_change))) if cosine_change.size else None,
        },
    }
