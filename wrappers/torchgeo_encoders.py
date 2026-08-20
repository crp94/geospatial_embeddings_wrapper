"""
TorchGeo-backed embedding encoders.

These adapters expose precomputed Earth embedding products through the same
coordinate-based interface as the model-backed encoders in this repository.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path
import logging
import os
import tempfile
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import torch
from pyproj import Transformer
from shapely.geometry import box
from sklearn.neighbors import BallTree
from torchgeo.datasets import (
    ClayEmbeddings,
    CopernicusEmbed,
    GoogleSatelliteEmbedding,
    TesseraEmbeddings,
)

from .embedding_encoder import GeoEmbeddingEncoder

LOGGER = logging.getLogger(__name__)


def _year_overlap_mask(index: pd.IntervalIndex, year: int) -> np.ndarray:
    """Return a boolean mask for rows whose interval overlaps the requested year."""
    year_start = pd.Timestamp(year=year, month=1, day=1)
    year_end = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59)
    year_interval = pd.Interval(year_start, year_end, closed="both")
    return index.overlaps(year_interval)


def _sample_latitudes_on_sphere(
    south: np.ndarray,
    north: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample latitudes uniformly on the sphere between south/north bounds."""
    south_sin = np.sin(np.deg2rad(south))
    north_sin = np.sin(np.deg2rad(north))
    return np.rad2deg(np.arcsin(rng.uniform(south_sin, north_sin)))


def _sample_from_wgs84_file_index(
    coordinates: torch.Tensor,
    index_df: gpd.GeoDataFrame,
    embedding_dim: int,
    filepath_column: str = "filepath",
    rasterio_env_options: dict[str, str] | None = None,
    max_file_retries: int = 1,
    retry_wait_seconds: float = 0.0,
) -> torch.Tensor:
    """
    Sample raster embeddings using a WGS84 polygon index and per-file reprojection.

    The input coordinates must follow the repository convention of `(lat, lon)`.
    The index geometries are assumed to be in `EPSG:4326`.
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            f"Expected coordinates with shape (N, 2), received {tuple(coordinates.shape)}"
        )

    lon = coordinates[:, 1].detach().cpu().numpy()
    lat = coordinates[:, 0].detach().cpu().numpy()

    points = gpd.GeoDataFrame(
        {
            "row_idx": np.arange(len(coordinates), dtype=np.int64),
            "longitude": lon,
            "latitude": lat,
        },
        geometry=gpd.points_from_xy(x=lon, y=lat),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(
        points,
        index_df[[filepath_column, "geometry"]],
        how="left",
        predicate="intersects",
    )
    joined = joined.sort_values("row_idx").drop_duplicates(subset="row_idx", keep="first")

    embeddings = np.full((len(coordinates), embedding_dim), np.nan, dtype=np.float32)

    valid_rows = joined.dropna(subset=[filepath_column])
    for filepath, group in valid_rows.groupby(filepath_column, sort=False):
        row_indices = group["row_idx"].to_numpy(dtype=np.int64)
        sample_lon = group["longitude"].to_numpy(dtype=np.float64)
        sample_lat = group["latitude"].to_numpy(dtype=np.float64)

        env_options = rasterio_env_options or {}
        samples = None
        last_error: Exception | None = None
        for attempt in range(1, max_file_retries + 1):
            try:
                with rasterio.Env(**env_options):
                    with rasterio.open(filepath) as src:
                        transformer = Transformer.from_crs(
                            "EPSG:4326", src.crs, always_xy=True
                        )
                        sample_x, sample_y = transformer.transform(sample_lon, sample_lat)
                        samples = np.asarray(list(src.sample(list(zip(sample_x, sample_y)))))
                break
            except Exception as exc:
                last_error = exc
                if attempt == max_file_retries:
                    LOGGER.warning(
                        "Failed to sample remote raster after %s attempts: %s (%s points)",
                        attempt,
                        filepath,
                        len(row_indices),
                    )
                elif retry_wait_seconds > 0:
                    time.sleep(retry_wait_seconds)

        if samples is None:
            if last_error is not None:
                LOGGER.warning("Last sampling error for %s: %s", filepath, last_error)
            continue

        if samples.ndim == 1:
            samples = samples[:, None]

        embeddings[row_indices] = samples.astype(np.float32, copy=False)

    return torch.from_numpy(embeddings)


class TorchGeoRasterPointEncoder(GeoEmbeddingEncoder):
    """Base adapter for TorchGeo raster embedding products."""

    dataset_cls = None
    encoder_id = "torchgeo_raster"
    default_data_root: str | None = None
    supports_auto_download = False
    temporal = False
    all_zero_invalid = False
    reference_year: int | None = None
    batch_size = 50000

    def __init__(
        self,
        device: str | None = None,
        data_root: str | None = None,
        cache: bool = True,
    ) -> None:
        super().__init__(device)

        if self.dataset_cls is None:
            raise ValueError("dataset_cls must be defined on raster encoder subclasses")

        root = data_root or self.default_data_root or f"./data_cache/{self.encoder_id}"
        dataset_kwargs: dict[str, Any] = {
            "paths": root,
            "cache": cache,
            "time_series": False,
        }
        if self.supports_auto_download:
            dataset_kwargs["download"] = data_root is None

        self.dataset = self.dataset_cls(**dataset_kwargs)
        self._embedding_dim = self._infer_embedding_dim()
        self._band_indexes = getattr(self.dataset, "band_indexes", None)
        self._transformer = Transformer.from_crs(
            "EPSG:4326", self.dataset.crs, always_xy=True
        )
        self._available_years = (
            self._infer_available_years() if self.temporal else None
        )

    def _infer_embedding_dim(self) -> int:
        with rasterio.open(self.dataset.files[0]) as src:
            if self._uses_dataset_band_indexes():
                band_indexes = getattr(self.dataset, "band_indexes", None)
                assert band_indexes is not None
                return len(band_indexes)
            return src.count

    def _uses_dataset_band_indexes(self) -> bool:
        return bool(getattr(self.dataset, "all_bands", ())) and bool(
            getattr(self.dataset, "bands", ())
        )

    def _infer_available_years(self) -> list[int]:
        years: set[int] = set()
        for interval in self.dataset.index.index:
            start = pd.Timestamp(interval.left)
            stop = pd.Timestamp(interval.right)
            for year in range(start.year, stop.year + 1):
                years.add(year)
        return sorted(years)

    def _select_index(self, year: int | None) -> gpd.GeoDataFrame:
        if self.temporal:
            if year is None:
                raise ValueError(f"{self.name} requires an explicit year")
            if self._available_years is not None and year not in self._available_years:
                raise ValueError(
                    f"Year {year} is not available for {self.name}. "
                    f"Available years: {self._available_years}"
                )
            mask = _year_overlap_mask(self.dataset.index.index, year)
            return self.dataset.index.loc[mask]

        return self.dataset.index

    def encode(
        self, coordinates: torch.Tensor, year: int | None = None
    ) -> torch.Tensor:
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError(
                f"Expected coordinates with shape (N, 2), received {tuple(coordinates.shape)}"
            )

        lon = coordinates[:, 1].detach().cpu().numpy()
        lat = coordinates[:, 0].detach().cpu().numpy()
        x, y = self._transformer.transform(lon, lat)

        points = gpd.GeoDataFrame(
            {
                "row_idx": np.arange(len(coordinates), dtype=np.int64),
                "x": x,
                "y": y,
            },
            geometry=gpd.points_from_xy(x=x, y=y),
            crs=self.dataset.crs,
        )

        index_df = self._select_index(year)
        joined = gpd.sjoin(
            points,
            index_df[["filepath", "geometry"]],
            how="left",
            predicate="intersects",
        )
        joined = joined.sort_values("row_idx").drop_duplicates(subset="row_idx", keep="first")

        embeddings = np.full(
            (len(coordinates), self._embedding_dim), np.nan, dtype=np.float32
        )

        valid_rows = joined.dropna(subset=["filepath"])
        for filepath, group in valid_rows.groupby("filepath", sort=False):
            row_indices = group["row_idx"].to_numpy(dtype=np.int64)
            sample_points = list(zip(group["x"].to_numpy(), group["y"].to_numpy()))

            with rasterio.open(filepath) as src:
                samples = np.asarray(
                    list(src.sample(sample_points, indexes=self._band_indexes))
                )

            if samples.ndim == 1:
                samples = samples[:, None]

            embeddings[row_indices] = samples.astype(np.float32, copy=False)

        return torch.from_numpy(embeddings)

    def get_embedding_dim(self) -> int:
        return self._embedding_dim

    def is_temporal(self) -> bool:
        return self.temporal

    def get_available_years(self) -> list[int] | None:
        return self._available_years

    def validate_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        valid = super().validate_embeddings(embeddings)
        if self.all_zero_invalid:
            valid = valid & (embeddings.abs().sum(dim=1) > 0)
        return valid

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update(
            {
                "source_type": "torchgeo_raster",
                "reference_year": self.reference_year,
            }
        )
        return metadata


class CopernicusEmbedEncoder(TorchGeoRasterPointEncoder):
    """Adapter for the Copernicus-Embed annual raster product."""

    dataset_cls = CopernicusEmbed
    encoder_id = "copernicus_embed"
    supports_auto_download = True
    all_zero_invalid = True
    reference_year = 2021

    @property
    def name(self) -> str:
        return "Copernicus-Embed"


class TesseraEmbeddingsEncoder(TorchGeoRasterPointEncoder):
    """Adapter for the Tessera annual embedding raster product."""

    dataset_cls = TesseraEmbeddings
    encoder_id = "tessera"
    temporal = True
    batch_size = 250000

    def __init__(
        self,
        device: str | None = None,
        data_root: str | None = None,
        cache: bool = True,
    ) -> None:
        if data_root is not None:
            self._mode = "local"
            super().__init__(device=device, data_root=data_root, cache=cache)
            return

        GeoEmbeddingEncoder.__init__(self, device)
        self._mode = "remote"
        self._embedding_dim = 128

        from geotessera import GeoTessera

        self._cache_root = Path("./data_cache/tessera")
        self._cache_root.mkdir(parents=True, exist_ok=True)
        self._registry_cache_dir = self._cache_root / "registry_cache"
        self._embeddings_dir = self._cache_root / "tiles"
        self._registry_cache_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_partial_tile_cache()

        self._geotessera = GeoTessera(
            cache_dir=self._registry_cache_dir,
            embeddings_dir=self._embeddings_dir,
        )
        self._available_years = list(
            map(int, self._geotessera.registry.get_available_years())
        )
        self._tile_centers_by_year: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _cleanup_partial_tile_cache(self) -> None:
        representation_root = self._embeddings_dir / "global_0.1_degree_representation"
        landmask_root = self._embeddings_dir / "global_0.1_degree_tiff_all"
        if not representation_root.exists():
            return

        for npy_path in representation_root.rglob("*.npy"):
            if npy_path.name.startswith("."):
                npy_path.unlink(missing_ok=True)
                continue

            if npy_path.name.endswith("_scales.npy"):
                embedding_path = npy_path.with_name(npy_path.name.replace("_scales.npy", ".npy"))
                if not embedding_path.exists():
                    npy_path.unlink(missing_ok=True)
                continue

            scales_path = npy_path.with_name(f"{npy_path.stem}_scales.npy")
            landmask_path = landmask_root / f"{npy_path.stem}.tiff"
            if not scales_path.exists() or not landmask_path.exists():
                npy_path.unlink(missing_ok=True)
                scales_path.unlink(missing_ok=True)

    def _get_tile_centers(self, year: int) -> tuple[np.ndarray, np.ndarray]:
        if year in self._tile_centers_by_year:
            return self._tile_centers_by_year[year]
        registry_gdf = self._geotessera.registry._registry_gdf
        year_index = registry_gdf.loc[year].index
        lon = year_index.get_level_values("lon_i").to_numpy(dtype=np.float64) / 100.0
        lat = year_index.get_level_values("lat_i").to_numpy(dtype=np.float64) / 100.0
        self._tile_centers_by_year[year] = (lon, lat)
        return self._tile_centers_by_year[year]

    def encode(
        self, coordinates: torch.Tensor, year: int | None = None
    ) -> torch.Tensor:
        if self._mode == "local":
            return super().encode(coordinates, year=year)

        if year is None:
            raise ValueError(f"{self.name} requires an explicit year")
        if year not in self._available_years:
            raise ValueError(
                f"Year {year} is not available for {self.name}. "
                f"Available years: {self._available_years}"
            )

        coords_np = coordinates.detach().cpu().numpy()
        points = [(float(lon), float(lat)) for lat, lon in coords_np]
        embeddings = self._geotessera.sample_embeddings_at_points(
            points,
            year=year,
            auto_download=True,
        )
        return torch.from_numpy(np.asarray(embeddings, dtype=np.float32))

    def get_embedding_dim(self) -> int:
        if self._mode == "local":
            return super().get_embedding_dim()
        return self._embedding_dim

    def get_available_years(self) -> list[int] | None:
        if self._mode == "local":
            return super().get_available_years()
        return self._available_years

    def supports_coverage_sampling(self) -> bool:
        return self._mode == "remote"

    def get_sampling_oversample_factor(self) -> float:
        return 1.05 if self._mode == "remote" else super().get_sampling_oversample_factor()

    def sample_candidate_coordinates(
        self,
        n_points: int,
        year: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._mode != "remote":
            return super().sample_candidate_coordinates(n_points, year=year, rng=rng)
        if year is None:
            raise ValueError(f"{self.name} requires an explicit year")
        rng = rng or np.random.default_rng()
        lon_centers, lat_centers = self._get_tile_centers(year)
        weights = np.clip(np.cos(np.deg2rad(lat_centers)), a_min=1e-6, a_max=None)
        weights = weights / weights.sum()
        chosen = rng.choice(len(lon_centers), size=n_points, replace=True, p=weights)
        longitude = lon_centers[chosen] + rng.uniform(-0.05, 0.05, size=n_points)
        latitude = lat_centers[chosen] + rng.uniform(-0.05, 0.05, size=n_points)
        return longitude.astype(np.float64), latitude.astype(np.float64)

    def get_metadata(self) -> dict[str, Any]:
        metadata = (
            super().get_metadata()
            if self._mode == "local"
            else GeoEmbeddingEncoder.get_metadata(self)
        )
        metadata.update(
            {
                "source_type": f"tessera_{self._mode}",
            }
        )
        if self._mode == "remote":
            metadata["cache_root"] = str(self._cache_root)
        return metadata

    @property
    def name(self) -> str:
        return "TESSERA"


class GoogleSatelliteEmbeddingEncoder(TorchGeoRasterPointEncoder):
    """Adapter for the Google Satellite Embedding annual raster product."""

    dataset_cls = GoogleSatelliteEmbedding
    encoder_id = "google_satellite_embedding"
    temporal = True
    batch_size = 250000
    index_url = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
    remote_index_download_attempts = 3
    remote_index_columns = (
        "year",
        "path",
        "wgs84_west",
        "wgs84_south",
        "wgs84_east",
        "wgs84_north",
    )
    remote_rasterio_env = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
        "CPL_VSIL_CURL_USE_HEAD": "NO",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MAX_RETRY": "4",
        "GDAL_HTTP_RETRY_DELAY": "2",
    }

    def __init__(
        self,
        device: str | None = None,
        data_root: str | None = None,
        cache: bool = True,
    ) -> None:
        if data_root is not None:
            self._mode = "local"
            super().__init__(device=device, data_root=data_root, cache=cache)
            return

        GeoEmbeddingEncoder.__init__(self, device)
        self._mode = "remote"
        self._embedding_dim = 64
        self._cache_root = Path("./data_cache/google_satellite_embedding")
        self._cache_root.mkdir(parents=True, exist_ok=True)
        self._index_path = self._cache_root / "aef_index.parquet"
        self._ensure_remote_index()
        self._year_indexes = self._load_remote_index(self._index_path)
        self._available_years = sorted(self._year_indexes.keys())
        self._sampling_bounds_by_year = self._build_sampling_bounds()

    def _ensure_remote_index(self) -> None:
        """Ensure the cached annual index is a complete, readable Parquet file.

        Never write directly to the cache path: an interrupted process must not
        make a partial index look like a valid cached download on the next run.
        """
        if self._index_path.exists():
            try:
                self._validate_remote_index(self._index_path)
                return
            except Exception as exc:
                LOGGER.warning(
                    "Discarding invalid Google Satellite Embedding index cache %s: %s",
                    self._index_path,
                    exc,
                )
                self._index_path.unlink(missing_ok=True)

        last_error: Exception | None = None
        for attempt in range(1, self.remote_index_download_attempts + 1):
            temporary_path: Path | None = None
            response = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{self._index_path.name}.",
                    suffix=".part",
                    dir=self._index_path.parent,
                    delete=False,
                ) as file_handle:
                    temporary_path = Path(file_handle.name)
                    response = requests.get(self.index_url, stream=True, timeout=120)
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file_handle.write(chunk)
                    file_handle.flush()
                    os.fsync(file_handle.fileno())

                self._validate_remote_index(temporary_path)
                temporary_path.replace(self._index_path)
                LOGGER.info("Cached Google Satellite Embedding index at %s", self._index_path)
                return
            except Exception as exc:
                last_error = exc
                if attempt < self.remote_index_download_attempts:
                    LOGGER.warning(
                        "Google Satellite Embedding index download attempt %d/%d failed: %s",
                        attempt,
                        self.remote_index_download_attempts,
                        exc,
                    )
            finally:
                if response is not None:
                    response.close()
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

        raise RuntimeError(
            "Could not download a valid Google Satellite Embedding index after "
            f"{self.remote_index_download_attempts} attempts: {self.index_url}"
        ) from last_error

    @classmethod
    def _validate_remote_index(cls, path: Path) -> None:
        """Check that a downloaded index has the columns required by this adapter."""
        index_df = pd.read_parquet(path, columns=list(cls.remote_index_columns))
        if index_df.empty:
            raise ValueError("the Parquet index has no rows")
        bounds_columns = [
            "wgs84_west",
            "wgs84_south",
            "wgs84_east",
            "wgs84_north",
        ]
        bounds = index_df[bounds_columns].apply(pd.to_numeric, errors="coerce")
        if (
            index_df["year"].isna().any()
            or index_df["path"].isna().any()
            or bounds.isna().any().any()
        ):
            raise ValueError("the Parquet index has missing required values")
        if not np.isfinite(bounds.to_numpy()).all():
            raise ValueError("the Parquet index has non-finite WGS84 bounds")
        if (
            (bounds["wgs84_west"] > bounds["wgs84_east"]).any()
            or (bounds["wgs84_south"] > bounds["wgs84_north"]).any()
            or (bounds["wgs84_west"] < -180).any()
            or (bounds["wgs84_east"] > 180).any()
            or (bounds["wgs84_south"] < -90).any()
            or (bounds["wgs84_north"] > 90).any()
        ):
            raise ValueError("the Parquet index has invalid WGS84 bounds")

    def _load_remote_index(self, path: Path) -> dict[int, gpd.GeoDataFrame]:
        columns = list(self.remote_index_columns)
        index_df = pd.read_parquet(path, columns=columns)
        index_df["filepath"] = index_df["path"].str.replace(
            "s3://us-west-2.opendata.source.coop/",
            "/vsicurl/https://data.source.coop/",
            regex=False,
        )
        geometry = [
            box(west, south, east, north)
            for west, south, east, north in zip(
                index_df["wgs84_west"],
                index_df["wgs84_south"],
                index_df["wgs84_east"],
                index_df["wgs84_north"],
            )
        ]
        gdf = gpd.GeoDataFrame(
            index_df[["year", "filepath"]].copy(),
            geometry=geometry,
            crs="EPSG:4326",
        )
        return {
            int(year): subset.reset_index(drop=True)
            for year, subset in gdf.groupby("year", sort=True)
        }

    def _build_sampling_bounds(self) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        sampling: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for year, year_index in self._year_indexes.items():
            bounds = np.asarray([geom.bounds for geom in year_index.geometry], dtype=np.float64)
            west = bounds[:, 0]
            south = bounds[:, 1]
            east = bounds[:, 2]
            north = bounds[:, 3]
            mid_lat = 0.5 * (south + north)
            weights = np.clip((east - west) * (north - south) * np.cos(np.deg2rad(mid_lat)), a_min=1e-6, a_max=None)
            weights = weights / weights.sum()
            sampling[year] = (west, south, east, north, weights)
        return sampling

    def encode(
        self, coordinates: torch.Tensor, year: int | None = None
    ) -> torch.Tensor:
        if self._mode == "local":
            return super().encode(coordinates, year=year)

        if year is None:
            raise ValueError(f"{self.name} requires an explicit year")
        if year not in self._year_indexes:
            raise ValueError(
                f"Year {year} is not available for {self.name}. "
                f"Available years: {self._available_years}"
            )
        return _sample_from_wgs84_file_index(
            coordinates,
            self._year_indexes[year],
            embedding_dim=self._embedding_dim,
            rasterio_env_options=self.remote_rasterio_env,
            max_file_retries=4,
            retry_wait_seconds=2.0,
        )

    def get_embedding_dim(self) -> int:
        if self._mode == "local":
            return super().get_embedding_dim()
        return self._embedding_dim

    def get_available_years(self) -> list[int] | None:
        if self._mode == "local":
            return super().get_available_years()
        return self._available_years

    def supports_coverage_sampling(self) -> bool:
        return self._mode == "remote"

    def get_sampling_oversample_factor(self) -> float:
        return 1.1 if self._mode == "remote" else super().get_sampling_oversample_factor()

    def sample_candidate_coordinates(
        self,
        n_points: int,
        year: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._mode != "remote":
            return super().sample_candidate_coordinates(n_points, year=year, rng=rng)
        if year is None:
            raise ValueError(f"{self.name} requires an explicit year")
        rng = rng or np.random.default_rng()
        west, south, east, north, weights = self._sampling_bounds_by_year[year]
        chosen = rng.choice(len(weights), size=n_points, replace=True, p=weights)
        longitude = rng.uniform(west[chosen], east[chosen])
        latitude = _sample_latitudes_on_sphere(south[chosen], north[chosen], rng)
        return longitude.astype(np.float64), latitude.astype(np.float64)

    def get_metadata(self) -> dict[str, Any]:
        metadata = (
            super().get_metadata()
            if self._mode == "local"
            else GeoEmbeddingEncoder.get_metadata(self)
        )
        metadata.update(
            {
                "source_type": f"google_satellite_embedding_{self._mode}",
            }
        )
        if self._mode == "remote":
            metadata["index_url"] = self.index_url
            metadata["index_path"] = str(self._index_path)
        return metadata

    @property
    def name(self) -> str:
        return "Google Satellite Embedding"


class LGNDClayEncoder(GeoEmbeddingEncoder):
    """
    Adapter for LGND Clay parquet embeddings.

    This encoder performs nearest-neighbor lookup over the patch centroids in the
    provided parquet product. It therefore approximates arbitrary coordinates using
    the nearest available native patch embedding.
    """

    def __init__(self, device: str | None = None, data_root: str | None = None) -> None:
        super().__init__(device)
        if data_root is None:
            raise ValueError(
                "LGND Clay requires a local parquet path via --encoder_root "
                "lgnd_clay=/path/to/data.parquet"
            )

        self.dataset = ClayEmbeddings(root=data_root)
        self.data = self.dataset.data.copy()
        self._embedding_key = (
            "embedding" if "embedding" in self.data.columns else "embeddings"
        )
        self._time_key = None
        for candidate in ("date", "datetime"):
            if candidate in self.data.columns:
                self._time_key = candidate
                break

        centroids = self.data.geometry.centroid
        self._latitudes = centroids.y.to_numpy(dtype=np.float64)
        self._longitudes = centroids.x.to_numpy(dtype=np.float64)
        self._embeddings = np.stack(self.data[self._embedding_key].to_list()).astype(
            np.float32
        )
        self._embedding_dim = int(self._embeddings.shape[1])
        self._tree = BallTree(
            np.deg2rad(np.column_stack([self._latitudes, self._longitudes])),
            metric="haversine",
        )
        self._reference_years = self._infer_reference_years()

    def _infer_reference_years(self) -> list[int] | None:
        if self._time_key is None:
            return None
        timestamps = pd.to_datetime(self.data[self._time_key])
        years = sorted(set(int(year) for year in timestamps.dt.year))
        return years or None

    def encode(
        self, coordinates: torch.Tensor, year: int | None = None
    ) -> torch.Tensor:
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError(
                f"Expected coordinates with shape (N, 2), received {tuple(coordinates.shape)}"
            )

        query = np.deg2rad(coordinates.detach().cpu().numpy())
        _, indices = self._tree.query(query, k=1)
        embeddings = self._embeddings[indices[:, 0]]
        return torch.from_numpy(embeddings)

    def get_embedding_dim(self) -> int:
        return self._embedding_dim

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update(
            {
                "source_type": "torchgeo_tabular_nearest",
                "reference_years": self._reference_years,
                "native_coordinate_order": "lon_lat_geometry",
            }
        )
        return metadata

    @property
    def name(self) -> str:
        return "LGND Clay"
