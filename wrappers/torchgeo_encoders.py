"""
TorchGeo-backed embedding encoders.

These adapters expose precomputed Earth embedding products through the same
coordinate-based interface as the model-backed encoders in this repository.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

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


def _year_overlap_mask(index: pd.IntervalIndex, year: int) -> np.ndarray:
    """Return a boolean mask for rows whose interval overlaps the requested year."""
    year_start = pd.Timestamp(year=year, month=1, day=1)
    year_end = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59)
    year_interval = pd.Interval(year_start, year_end, closed="both")
    return index.overlaps(year_interval)


def _sample_from_wgs84_file_index(
    coordinates: torch.Tensor,
    index_df: gpd.GeoDataFrame,
    embedding_dim: int,
    filepath_column: str = "filepath",
    rasterio_env_options: dict[str, str] | None = None,
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
        with rasterio.Env(**env_options):
            with rasterio.open(filepath) as src:
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                sample_x, sample_y = transformer.transform(sample_lon, sample_lat)
                samples = np.asarray(list(src.sample(list(zip(sample_x, sample_y)))))

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
    batch_size = 2000

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

        self._geotessera = GeoTessera(
            cache_dir=self._registry_cache_dir,
            embeddings_dir=self._embeddings_dir,
        )
        self._available_years = list(
            map(int, self._geotessera.registry.get_available_years())
        )

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
    batch_size = 5000
    index_url = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
    remote_rasterio_env = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
        "CPL_VSIL_CURL_USE_HEAD": "NO",
        "GDAL_HTTP_MULTIRANGE": "YES",
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

    def _ensure_remote_index(self) -> None:
        if self._index_path.exists():
            return

        response = requests.get(self.index_url, stream=True, timeout=120)
        response.raise_for_status()
        with open(self._index_path, "wb") as file_handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_handle.write(chunk)

    def _load_remote_index(self, path: Path) -> dict[int, gpd.GeoDataFrame]:
        columns = [
            "year",
            "path",
            "wgs84_west",
            "wgs84_south",
            "wgs84_east",
            "wgs84_north",
        ]
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
        )

    def get_embedding_dim(self) -> int:
        if self._mode == "local":
            return super().get_embedding_dim()
        return self._embedding_dim

    def get_available_years(self) -> list[int] | None:
        if self._mode == "local":
            return super().get_available_years()
        return self._available_years

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
