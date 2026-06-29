#!/usr/bin/env python3
"""
Geospatial Embedding Dataset Generator.

Generates land-only coordinate datasets with embeddings from model-backed encoders
(GeoCLIP, SatCLIP) and TorchGeo-backed Earth embedding products.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import zipfile
from shapely.vectorized import contains
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeatures

    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    print("Warning: Cartopy not available. Maps will use basic matplotlib plots.")

# Add parent directory to path to import wrappers
sys.path.insert(0, str(Path(__file__).parent.parent))

from wrappers.registry import get_encoder_class, list_encoder_names, normalize_encoder_name

warnings.filterwarnings("ignore")

DEFAULT_ENCODERS = ["geoclip", "satclip"]
ANTARCTICA_LATITUDE_CUTOFF = -60.0


def parse_encoder_roots(values: list[str] | None) -> dict[str, str]:
    """Parse repeated encoder=path arguments."""
    roots: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                f"Invalid --encoder_root value '{value}'. Expected encoder=/path/to/data"
            )
        raw_name, raw_path = value.split("=", 1)
        roots[normalize_encoder_name(raw_name)] = raw_path
    return roots


class GeospatialDatasetGenerator:
    """Generate land coordinate datasets with embeddings from multiple backends."""

    def __init__(
        self,
        cache_dir: str = "./data_cache",
        encoder_roots: dict[str, str] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.land_mask = None
        self.encoders: dict[str, object] = {}
        self.encoder_roots = encoder_roots or {}
        self.rng = np.random.default_rng()

    def initialize_encoders(
        self, encoder_names: list[str] | None = None, device: str | None = None
    ) -> dict[str, object]:
        """Initialize selected encoders from the shared registry."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        encoders_to_use = encoder_names or list(DEFAULT_ENCODERS)
        resolved_names = [normalize_encoder_name(name) for name in encoders_to_use]

        print("Initializing encoders...")
        for encoder_name in resolved_names:
            encoder_class = get_encoder_class(encoder_name)
            encoder_root = self.encoder_roots.get(encoder_name)

            try:
                print(f"  Loading {encoder_name}...")
                self.encoders[encoder_name] = encoder_class(
                    device=device,
                    data_root=encoder_root,
                )
                dim = self.encoders[encoder_name].get_embedding_dim()
                print(f"  [OK] {encoder_name} ready (dim={dim})")
            except Exception as exc:
                print(f"  [ERROR] Could not initialize {encoder_name}: {exc}")

        if not self.encoders:
            raise RuntimeError("No encoders were successfully initialized.")

        print(f"\nActive encoders: {', '.join(self.encoders.keys())}")
        return self.encoders

    def resolve_years(self, requested_years: list[int] | None = None) -> list[int | None]:
        """Resolve which yearly datasets to generate."""
        if requested_years:
            requested = sorted(set(requested_years))
        else:
            requested = []

        temporal_encoders = {
            name: encoder
            for name, encoder in self.encoders.items()
            if encoder.is_temporal()
        }

        if not temporal_encoders:
            return requested or [None]

        if requested:
            for year in requested:
                unsupported = [
                    name
                    for name, encoder in temporal_encoders.items()
                    if year not in (encoder.get_available_years() or [])
                ]
                if unsupported:
                    raise ValueError(
                        f"Year {year} is not available for: {', '.join(sorted(unsupported))}"
                    )
            return requested

        year_sets = [
            set(encoder.get_available_years() or []) for encoder in temporal_encoders.values()
        ]
        common_years = sorted(set.intersection(*year_sets))
        if not common_years:
            raise RuntimeError(
                "Selected temporal encoders do not share any common years. "
                "Pass --years explicitly to constrain the run."
            )
        return common_years

    def download_land_mask(self) -> Path:
        """Download Natural Earth land shapefile for masking."""
        land_file = self.cache_dir / "ne_110m_land.shp"

        if land_file.exists():
            print("Land mask already downloaded")
            return land_file

        print("Downloading Natural Earth land data...")
        url = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"
        zip_path = self.cache_dir / "ne_110m_land.zip"

        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        }

        try:
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            with open(zip_path, "wb") as file_handle:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file_handle.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rDownloading: {percent:.1f}%", end="", flush=True)
            print()

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.cache_dir)
            print("Successfully downloaded and extracted Natural Earth data")
        except Exception as exc:
            raise RuntimeError(f"Failed to download Natural Earth data: {exc}") from exc
        finally:
            if zip_path.exists():
                zip_path.unlink()

        return land_file

    def load_land_mask(self) -> None:
        """Load land shapefile as mask."""
        if self.land_mask is not None:
            return

        land_file = self.download_land_mask()
        print("Loading land mask...")
        self.land_mask = gpd.read_file(land_file)
        print(f"Land mask loaded with {len(self.land_mask)} land polygons")

    def fibonacci_sphere_sampling(self, n_points: int) -> tuple[np.ndarray, np.ndarray]:
        """Fibonacci sphere sampling for nearly uniform distribution."""
        print(f"Generating {n_points:,} Fibonacci sphere points...")
        index_offset = self.rng.random()
        azimuth_offset = self.rng.uniform(0.0, 2.0 * np.pi)
        indices = np.arange(0, n_points, dtype=float) + index_offset
        phi = np.arccos(1 - 2 * indices / n_points)
        theta = np.pi * (1 + 5**0.5) * indices + azimuth_offset

        latitude_deg = 90 - np.degrees(phi)
        longitude_deg = np.degrees(theta) % 360 - 180
        return longitude_deg, latitude_deg

    def vectorized_land_filter(
        self, longitude: np.ndarray, latitude: np.ndarray, batch_size: int = 10000
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized land filtering against Natural Earth polygons."""
        self.load_land_mask()

        print("Filtering for land points (vectorized)...")
        non_antarctic_mask = latitude > ANTARCTICA_LATITUDE_CUTOFF
        if not non_antarctic_mask.any():
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        candidate_longitude = longitude[non_antarctic_mask]
        candidate_latitude = latitude[non_antarctic_mask]
        land_union = self.land_mask.geometry.unary_union
        land_mask = np.zeros(len(candidate_longitude), dtype=bool)

        for start_idx in tqdm(
            range(0, len(candidate_longitude), batch_size), desc="Land filtering"
        ):
            end_idx = min(start_idx + batch_size, len(candidate_longitude))
            batch_mask = contains(
                land_union,
                candidate_longitude[start_idx:end_idx],
                candidate_latitude[start_idx:end_idx],
            )
            land_mask[start_idx:end_idx] = batch_mask

        land_longitude = candidate_longitude[land_mask]
        land_latitude = candidate_latitude[land_mask]
        print(
            f"Found {len(land_longitude):,} land points out of {len(longitude):,} total"
        )
        return land_longitude, land_latitude

    def _get_direct_sampling_encoder(self) -> object | None:
        """Return a single encoder that can sample from its own coverage, if available."""
        if len(self.encoders) != 1:
            return None
        encoder = next(iter(self.encoders.values()))
        if getattr(encoder, "supports_coverage_sampling", lambda: False)():
            return encoder
        return None

    def get_embeddings(
        self, latitude: np.ndarray, longitude: np.ndarray, year: int | None = None
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Get embeddings from all initialized encoders and a combined validity mask."""
        if not self.encoders:
            raise RuntimeError("No encoders initialized. Call initialize_encoders() first.")

        print(
            f"\nGenerating embeddings for {len(latitude):,} coordinates"
            + (f" for year {year}" if year is not None else "")
            + "..."
        )

        coordinates = torch.tensor(
            np.column_stack([latitude, longitude]), dtype=torch.float32
        )

        all_embeddings: dict[str, np.ndarray] = {}
        combined_valid = np.ones(len(latitude), dtype=bool)

        for encoder_name, encoder in self.encoders.items():
            print(f"  Processing with {encoder_name}...")
            batch_size = getattr(encoder, "batch_size", 5000)
            embeddings_list: list[np.ndarray] = []
            validity_list: list[np.ndarray] = []

            for start_idx in tqdm(
                range(0, len(coordinates), batch_size),
                desc=f"  {encoder_name}",
                leave=False,
            ):
                end_idx = min(start_idx + batch_size, len(coordinates))
                batch_coords = coordinates[start_idx:end_idx]
                batch_embeddings = encoder.encode(batch_coords, year=year).detach().cpu().float()
                batch_valid = encoder.validate_embeddings(batch_embeddings).detach().cpu().numpy()
                embeddings_list.append(batch_embeddings.numpy())
                validity_list.append(batch_valid.astype(bool))

            embeddings = np.concatenate(embeddings_list, axis=0)
            valid_mask = np.concatenate(validity_list, axis=0)
            combined_valid &= valid_mask
            all_embeddings[encoder_name] = embeddings

            valid_embeddings = embeddings[valid_mask]
            if len(valid_embeddings) > 0:
                print(
                    f"    Shape: {embeddings.shape}, "
                    f"Valid: {int(valid_mask.sum()):,}/{len(valid_mask):,}, "
                    f"Mean: {valid_embeddings.mean():.4f}, "
                    f"Std: {valid_embeddings.std():.4f}"
                )
            else:
                print(f"    Shape: {embeddings.shape}, Valid: 0/{len(valid_mask):,}")

        return all_embeddings, combined_valid

    def sample_valid_dataset(
        self, n_points: int, year: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Sample land coordinates until enough valid embeddings are collected."""
        direct_sampling_encoder = self._get_direct_sampling_encoder()
        oversample_factor = (
            direct_sampling_encoder.get_sampling_oversample_factor()
            if direct_sampling_encoder is not None
            else 3.5
        )
        collected_longitude: list[np.ndarray] = []
        collected_latitude: list[np.ndarray] = []
        collected_embeddings = {name: [] for name in self.encoders.keys()}

        while sum(len(chunk) for chunk in collected_longitude) < n_points:
            remaining = n_points - sum(len(chunk) for chunk in collected_longitude)
            initial_points = max(int(np.ceil(remaining * oversample_factor)), remaining)

            if direct_sampling_encoder is not None:
                longitude, latitude = direct_sampling_encoder.sample_candidate_coordinates(
                    initial_points,
                    year=year,
                    rng=self.rng,
                )
            else:
                longitude, latitude = self.fibonacci_sphere_sampling(initial_points)
            land_longitude, land_latitude = self.vectorized_land_filter(longitude, latitude)
            if len(land_longitude) == 0:
                continue

            batch_embeddings, valid_mask = self.get_embeddings(
                land_latitude, land_longitude, year=year
            )
            if not valid_mask.any():
                print("No valid embeddings found in this batch. Resampling...")
                continue

            valid_longitude = land_longitude[valid_mask]
            valid_latitude = land_latitude[valid_mask]
            take = min(len(valid_longitude), remaining)
            if len(valid_longitude) > take:
                chosen = self.rng.choice(len(valid_longitude), take, replace=False)
            else:
                chosen = np.arange(take)

            collected_longitude.append(valid_longitude[chosen])
            collected_latitude.append(valid_latitude[chosen])
            for encoder_name, embeddings in batch_embeddings.items():
                collected_embeddings[encoder_name].append(embeddings[valid_mask][chosen])

            current = sum(len(chunk) for chunk in collected_longitude)
            print(f"Collected {current:,}/{n_points:,} valid points")

        longitude = np.concatenate(collected_longitude, axis=0)[:n_points]
        latitude = np.concatenate(collected_latitude, axis=0)[:n_points]
        embeddings = {
            name: np.concatenate(chunks, axis=0)[:n_points]
            for name, chunks in collected_embeddings.items()
        }
        return longitude, latitude, embeddings

    def build_dataset(
        self,
        latitude: np.ndarray,
        longitude: np.ndarray,
        embeddings_dict: dict[str, np.ndarray],
        year: int | None = None,
    ) -> dict[str, object]:
        """Build the in-memory dataset dictionary with explicit coordinate conventions."""
        coordinates_latlon = np.column_stack([latitude, longitude]).astype(np.float32)
        coordinates_lonlat = np.column_stack([longitude, latitude]).astype(np.float32)

        dataset: dict[str, object] = {
            "metadata": {
                "coordinate_order": {
                    "coordinates": "lat_lon",
                    "coordinates_latlon": "lat_lon",
                    "coordinates_lonlat": "lon_lat",
                },
                "year": year,
                "n_points": int(len(latitude)),
                "encoders": list(embeddings_dict.keys()),
                "encoder_metadata": {
                    name: encoder.get_metadata()
                    for name, encoder in self.encoders.items()
                },
            },
            "longitude": torch.from_numpy(longitude.astype(np.float32)),
            "latitude": torch.from_numpy(latitude.astype(np.float32)),
            "coordinates": torch.from_numpy(coordinates_latlon),
            "coordinates_latlon": torch.from_numpy(coordinates_latlon),
            "coordinates_lonlat": torch.from_numpy(coordinates_lonlat),
        }

        for encoder_name, embeddings in embeddings_dict.items():
            dataset[f"{encoder_name}_embeddings"] = torch.from_numpy(
                embeddings.astype(np.float32)
            )

        return dataset

    def save_dataset(
        self,
        latitude: np.ndarray,
        longitude: np.ndarray,
        embeddings_dict: dict[str, np.ndarray],
        output_path: str,
        output_format: str,
        year: int | None = None,
    ) -> str:
        """Persist the dataset to disk."""
        suffix = f"_{year}" if year is not None else ""
        if output_format == "pt":
            dataset = self.build_dataset(latitude, longitude, embeddings_dict, year=year)
            output_file = f"{output_path}{suffix}.pt"
            torch.save(dataset, output_file)
            return output_file

        df_data: dict[str, np.ndarray] = {
            "longitude": longitude.astype(np.float32),
            "latitude": latitude.astype(np.float32),
        }
        for encoder_name, embeddings in embeddings_dict.items():
            for dim_idx in range(embeddings.shape[1]):
                df_data[f"{encoder_name}_emb_{dim_idx:04d}"] = embeddings[:, dim_idx]
        if year is not None:
            df_data["year"] = np.full(len(latitude), year, dtype=np.int32)

        output_file = f"{output_path}{suffix}.csv"
        pd.DataFrame(df_data).to_csv(output_file, index=False)
        return output_file

    def generate_dataset(
        self,
        n_points: int = 100000,
        encoders: list[str] | None = None,
        output_format: str = "pt",
        output_path: str = "geospatial_dataset",
        device: str | None = None,
        plot_results: bool = True,
        years: list[int] | None = None,
    ) -> list[str]:
        """Generate one or more datasets, splitting by year when requested."""
        self.initialize_encoders(encoders, device)
        resolved_years = self.resolve_years(years)

        output_files: list[str] = []
        for year in resolved_years:
            print("\n" + "=" * 60)
            if year is None:
                print("Generating static dataset")
            else:
                print(f"Generating dataset for year {year}")
            print("=" * 60)

            longitude, latitude, embeddings_dict = self.sample_valid_dataset(
                n_points=n_points, year=year
            )
            output_file = self.save_dataset(
                latitude=latitude,
                longitude=longitude,
                embeddings_dict=embeddings_dict,
                output_path=output_path,
                output_format=output_format,
                year=year,
            )
            output_files.append(output_file)
            print(f"[OK] Saved dataset to {output_file}")

            if plot_results:
                self.plot_sampled_locations(
                    longitude,
                    latitude,
                    f"{Path(output_file).with_suffix('')}_locations.png",
                    year=year,
                )
                for encoder_name, embeddings in embeddings_dict.items():
                    self.plot_ica_embeddings(
                        longitude,
                        latitude,
                        embeddings,
                        encoder_name,
                        f"{Path(output_file).with_suffix('')}_{encoder_name}_ica.png",
                        year=year,
                    )

        return output_files

    def plot_sampled_locations(
        self,
        longitude: np.ndarray,
        latitude: np.ndarray,
        save_path: str,
        year: int | None = None,
    ) -> None:
        """Create a plot showing the sampled land locations."""
        print(f"\nCreating location plot for {len(longitude):,} points...")
        fig = plt.figure(figsize=(15, 8))

        title_suffix = f"\nYear {year}" if year is not None else ""
        if CARTOPY_AVAILABLE:
            ax = plt.axes(projection=ccrs.Robinson())
            ax.add_feature(cfeatures.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeatures.LAND, color="lightgray", alpha=0.5)
            ax.add_feature(cfeatures.OCEAN, color="lightblue", alpha=0.3)
            ax.add_feature(cfeatures.BORDERS, linewidth=0.3, alpha=0.7)

            ax.scatter(
                longitude,
                latitude,
                c="red",
                s=0.5,
                alpha=0.6,
                transform=ccrs.PlateCarree(),
                label=f"{len(longitude):,} sampled points",
            )
            ax.set_title(
                f"Geospatial Dataset: {len(longitude):,} Land Points{title_suffix}",
                fontsize=14,
                pad=20,
            )
            ax.legend(loc="lower left")
            ax.gridlines(draw_labels=False, alpha=0.3)
        else:
            ax = plt.subplot(111)
            ax.scatter(longitude, latitude, c="red", s=0.5, alpha=0.6)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(f"Geospatial Dataset: {len(longitude):,} Land Points{title_suffix}")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] Location plot saved to {save_path}")

    def project_embeddings_to_rgb(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings to RGB using ICA, with a robust fallback."""
        if embeddings.shape[0] < 3:
            base = np.zeros((embeddings.shape[0], 3), dtype=np.float32)
            base[:, : min(3, embeddings.shape[1])] = embeddings[:, : min(3, embeddings.shape[1])]
            return base

        fit_sample_size = min(len(embeddings), 100000)
        if len(embeddings) > fit_sample_size:
            sample_idx = np.random.default_rng(42).choice(
                len(embeddings), size=fit_sample_size, replace=False
            )
            fit_embeddings = embeddings[sample_idx]
        else:
            fit_embeddings = embeddings

        scaler = StandardScaler()
        fit_scaled = scaler.fit_transform(fit_embeddings)

        try:
            ica = FastICA(
                n_components=3,
                whiten="unit-variance",
                random_state=42,
                max_iter=1000,
            )
            ica.fit(fit_scaled)

            projected_chunks: list[np.ndarray] = []
            batch_size = 100000
            for start_idx in range(0, len(embeddings), batch_size):
                end_idx = min(start_idx + batch_size, len(embeddings))
                batch_scaled = scaler.transform(embeddings[start_idx:end_idx])
                projected_chunks.append(ica.transform(batch_scaled))
            projected = np.concatenate(projected_chunks, axis=0)
        except Exception:
            fallback_scaled = scaler.fit_transform(embeddings)
            projected = fallback_scaled[:, :3]

        if projected.shape[1] < 3:
            projected = np.pad(
                projected,
                ((0, 0), (0, 3 - projected.shape[1])),
                mode="constant",
            )

        projected = np.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
        colors = np.zeros_like(projected, dtype=np.float32)
        for component_idx in range(3):
            component = projected[:, component_idx]
            comp_min = component.min()
            comp_max = component.max()
            if np.isclose(comp_min, comp_max):
                colors[:, component_idx] = 0.5
            else:
                colors[:, component_idx] = (component - comp_min) / (comp_max - comp_min)
        return colors

    def plot_ica_embeddings(
        self,
        longitude: np.ndarray,
        latitude: np.ndarray,
        embeddings: np.ndarray,
        encoder_name: str,
        save_path: str,
        year: int | None = None,
    ) -> None:
        """Create a world map with points colored by ICA-projected embeddings."""
        print(f"\nCreating ICA visualization for {encoder_name}...")
        colors = self.project_embeddings_to_rgb(embeddings)

        fig = plt.figure(figsize=(20, 12))
        title_suffix = f" - Year {year}" if year is not None else ""

        if CARTOPY_AVAILABLE:
            ax = plt.axes(projection=ccrs.Robinson())
            ax.add_feature(cfeatures.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeatures.LAND, color="lightgray", alpha=0.3)
            ax.add_feature(cfeatures.OCEAN, color="lightblue", alpha=0.2)
            ax.add_feature(cfeatures.BORDERS, linewidth=0.3, alpha=0.5)
            ax.scatter(
                longitude,
                latitude,
                c=colors,
                s=1.0,
                alpha=0.8,
                transform=ccrs.PlateCarree(),
            )
            ax.set_title(
                f"{encoder_name} Embeddings (ICA-projected to RGB){title_suffix}\n"
                f"{len(longitude):,} points",
                fontsize=16,
                pad=20,
            )
            ax.gridlines(draw_labels=False, alpha=0.3)
        else:
            ax = plt.subplot(111)
            ax.scatter(longitude, latitude, c=colors, s=1.0, alpha=0.8)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(
                f"{encoder_name} Embeddings (ICA-projected to RGB){title_suffix}\n"
                f"{len(longitude):,} points"
            )
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] ICA plot saved to {save_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate land-only geospatial embedding datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--n_points",
        type=int,
        default=100000,
        help="Number of valid land points to generate per output file",
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=list(DEFAULT_ENCODERS),
        help=(
            "Encoders to use. Canonical names: "
            + ", ".join(list_encoder_names())
            + ". Aliases such as clay, copernicus, and gse are also accepted."
        ),
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        help="Optional explicit years to generate. Temporal encoders will use these years.",
    )
    parser.add_argument(
        "--encoder_root",
        action="append",
        help="Repeatable mapping of encoder=PATH for TorchGeo-backed products.",
    )
    parser.add_argument(
        "--output_format",
        choices=["pt", "csv"],
        default="pt",
        help="Output format (default: pt)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="geospatial_dataset",
        help="Output file path prefix without extension",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Device to run model-backed encoders on (default: auto-detect)",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./data_cache",
        help="Directory for caching land mask data",
    )

    args = parser.parse_args()
    encoder_roots = parse_encoder_roots(args.encoder_root)

    print("=" * 60)
    print("Geospatial Embedding Dataset Generator")
    print("=" * 60)
    print(f"Target points per file: {args.n_points:,}")
    print(f"Encoders: {args.encoders}")
    print(f"Years: {args.years or 'auto/static'}")
    print(f"Output format: {args.output_format}")
    print(f"Device: {args.device or 'auto-detect'}")
    print("=" * 60 + "\n")

    generator = GeospatialDatasetGenerator(
        cache_dir=args.cache_dir,
        encoder_roots=encoder_roots,
    )
    output_files = generator.generate_dataset(
        n_points=args.n_points,
        encoders=args.encoders,
        output_format=args.output_format,
        output_path=args.output_path,
        device=args.device,
        plot_results=not args.no_plot,
        years=args.years,
    )

    print("\n" + "=" * 60)
    print("[OK] Dataset generation complete!")
    for output_file in output_files:
        print(f"Output: {output_file}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
