#!/usr/bin/env python3
"""
Geospatial Embedding Dataset Generator

Generates uniformly sampled land-only coordinates with geospatial embeddings
from multiple encoders (GeoCLIP, SatCLIP, etc.) for training location encoders.

Usage:
    # Generate 100k points with both encoders (default)
    python generate_dataset.py

    # Generate custom number of points with specific encoders
    python generate_dataset.py --n_points 50000 --encoders geoclip satclip

    # Generate CSV output without plots
    python generate_dataset.py --output_format csv --no_plot

    # Use only GeoCLIP
    python generate_dataset.py --encoders geoclip

Requirements:
    pip install geopandas shapely pandas torch tqdm requests matplotlib cartopy
"""

import numpy as np
import pandas as pd
import torch
import geopandas as gpd
from shapely.geometry import Point
from shapely.vectorized import contains
import requests
import zipfile
import os
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeatures
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    print("Warning: Cartopy not available. Maps will use basic matplotlib plots.")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

# Add parent directory to path to import wrappers
sys.path.insert(0, str(Path(__file__).parent.parent))

from wrappers import GeoEmbeddingEncoder, GeoCLIPEncoder, SatCLIPEncoder

warnings.filterwarnings('ignore')


class GeospatialDatasetGenerator:
    """
    Generate datasets of land coordinates with geospatial embeddings.

    Uses the unified embedding interface to support multiple encoders
    (GeoCLIP, SatCLIP, and any future additions).
    """

    def __init__(self, cache_dir="./data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.land_mask = None
        self.encoders = {}

    def initialize_encoders(self, encoder_names=None, device=None):
        """
        Initialize embedding encoders.

        Args:
            encoder_names: List of encoder names ('geoclip', 'satclip').
                          If None, initializes all available encoders.
            device: Device to run models on ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        available_encoders = {
            'geoclip': GeoCLIPEncoder,
            'satclip': SatCLIPEncoder
        }

        encoders_to_use = encoder_names or list(available_encoders.keys())

        print("Initializing encoders...")
        for encoder_name in encoders_to_use:
            if encoder_name.lower() not in available_encoders:
                print(f"Warning: Unknown encoder '{encoder_name}'. Skipping.")
                continue

            try:
                encoder_class = available_encoders[encoder_name.lower()]
                print(f"  Loading {encoder_name.upper()}...")
                self.encoders[encoder_name.lower()] = encoder_class(device=device)
                print(f"  [OK] {encoder_name.upper()} ready "
                      f"(dim={self.encoders[encoder_name.lower()].get_embedding_dim()})")
            except Exception as e:
                print(f"  [ERROR] Could not initialize {encoder_name.upper()}: {e}")

        if not self.encoders:
            raise RuntimeError("No encoders were successfully initialized!")

        print(f"\nActive encoders: {', '.join(self.encoders.keys())}")
        return self.encoders

    def download_land_mask(self):
        """Download Natural Earth land shapefile for masking"""
        land_file = self.cache_dir / "ne_110m_land.shp"

        if land_file.exists():
            print("Land mask already downloaded")
            return land_file

        print("Downloading Natural Earth land data...")
        url = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"
        zip_path = self.cache_dir / "ne_110m_land.zip"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        try:
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rDownloading: {percent:.1f}%", end='', flush=True)
            print()

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.cache_dir)
            print("Successfully downloaded and extracted Natural Earth data")

        except Exception as e:
            raise RuntimeError(f"Failed to download Natural Earth data: {e}")
        finally:
            if zip_path.exists():
                zip_path.unlink()

        return land_file

    def load_land_mask(self):
        """Load land shapefile as mask"""
        if self.land_mask is not None:
            return

        land_file = self.download_land_mask()
        print("Loading land mask...")
        self.land_mask = gpd.read_file(land_file)
        print(f"Land mask loaded with {len(self.land_mask)} land polygons")

    def fibonacci_sphere_sampling(self, n_points):
        """Fibonacci sphere sampling for uniform distribution"""
        print(f"Generating {n_points:,} Fibonacci sphere points...")

        indices = np.arange(0, n_points, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n_points)
        theta = np.pi * (1 + 5 ** 0.5) * indices

        latitude_deg = 90 - np.degrees(phi)
        longitude_deg = np.degrees(theta) % 360 - 180

        return longitude_deg, latitude_deg

    def vectorized_land_filter(self, longitude, latitude, batch_size=10000):
        """Fast vectorized land filtering"""
        self.load_land_mask()

        print("Filtering for land points (vectorized)...")
        land_union = self.land_mask.geometry.unary_union
        land_mask = np.zeros(len(longitude), dtype=bool)

        for i in tqdm(range(0, len(longitude), batch_size), desc="Land filtering"):
            end_idx = min(i + batch_size, len(longitude))
            batch_lon = longitude[i:end_idx]
            batch_lat = latitude[i:end_idx]
            batch_mask = contains(land_union, batch_lon, batch_lat)
            land_mask[i:end_idx] = batch_mask

        land_longitude = longitude[land_mask]
        land_latitude = latitude[land_mask]

        print(f"Found {len(land_longitude):,} land points out of {len(longitude):,} total")
        return land_longitude, land_latitude

    def get_embeddings(self, latitude, longitude):
        """
        Get embeddings from all initialized encoders.

        Args:
            latitude: Array of latitude coordinates
            longitude: Array of longitude coordinates

        Returns:
            Dictionary mapping encoder names to embedding arrays
        """
        if not self.encoders:
            raise RuntimeError("No encoders initialized. Call initialize_encoders() first.")

        print(f"\nGenerating embeddings for {len(latitude):,} coordinates...")

        # Create coordinate tensor in standard (lat, lon) format
        coordinates = torch.Tensor([[lat, lon] for lat, lon in zip(latitude, longitude)])

        all_embeddings = {}

        for encoder_name, encoder in self.encoders.items():
            print(f"  Processing with {encoder_name.upper()}...")

            # Process in batches to avoid memory issues
            batch_size = 5000
            embeddings_list = []

            for i in tqdm(range(0, len(coordinates), batch_size),
                         desc=f"  {encoder_name.upper()}",
                         leave=False):
                end_idx = min(i + batch_size, len(coordinates))
                batch_coords = coordinates[i:end_idx]
                batch_emb = encoder.encode(batch_coords)
                embeddings_list.append(batch_emb)

            # Concatenate all batches
            embeddings = torch.cat(embeddings_list, dim=0).numpy()
            all_embeddings[encoder_name] = embeddings

            # Print statistics
            print(f"    Shape: {embeddings.shape}, "
                  f"Mean: {np.mean(embeddings):.4f}, "
                  f"Std: {np.std(embeddings):.4f}")

        return all_embeddings

    def generate_dataset(self, n_points=100000,
                        encoders=None,
                        output_format="pt",
                        output_path="geospatial_dataset",
                        device=None,
                        plot_results=True):
        """
        Generate the complete dataset.

        Args:
            n_points: Target number of land points
            encoders: List of encoder names to use (None = all available)
            output_format: "pt" or "csv"
            output_path: Output file path (without extension)
            device: Device for models ('cuda', 'cpu', or None)
            plot_results: Whether to create visualization plots
        """
        # Initialize encoders
        self.initialize_encoders(encoders, device)

        # Generate points with oversampling for land filtering
        oversample_factor = 3.5  # ~29% of Earth is land
        initial_points = int(n_points * oversample_factor)

        longitude, latitude = self.fibonacci_sphere_sampling(initial_points)
        land_longitude, land_latitude = self.vectorized_land_filter(longitude, latitude)

        # If we don't have enough, generate more
        while len(land_longitude) < n_points:
            print(f"Need more points. Have {len(land_longitude):,}, need {n_points:,}")
            additional_points = int((n_points - len(land_longitude)) * oversample_factor)
            add_lon, add_lat = self.fibonacci_sphere_sampling(additional_points)
            add_land_lon, add_land_lat = self.vectorized_land_filter(add_lon, add_lat)
            land_longitude = np.concatenate([land_longitude, add_land_lon])
            land_latitude = np.concatenate([land_latitude, add_land_lat])

        # Trim to exact number requested
        if len(land_longitude) > n_points:
            indices = np.random.choice(len(land_longitude), int(n_points), replace=False)
            land_longitude = land_longitude[indices]
            land_latitude = land_latitude[indices]

        print(f"\nFinal dataset has {len(land_longitude):,} points")

        # Get embeddings from all encoders
        embeddings_dict = self.get_embeddings(land_latitude, land_longitude)

        # Save dataset
        if output_format == "pt":
            dataset = {
                'longitude': torch.from_numpy(land_longitude),
                'latitude': torch.from_numpy(land_latitude),
                'coordinates': torch.from_numpy(np.column_stack([land_latitude, land_longitude]))
            }

            # Add embeddings from each encoder
            for encoder_name, embeddings in embeddings_dict.items():
                dataset[f'{encoder_name}_embeddings'] = torch.from_numpy(embeddings)

            output_file = f"{output_path}.pt"
            torch.save(dataset, output_file)
            print(f"\n[OK] Saved PyTorch dataset to {output_file}")

            # Print dataset info
            print("\nDataset Summary:")
            print(f"  Keys: {list(dataset.keys())}")
            print(f"  Number of points: {len(dataset['coordinates'])}")
            print(f"  Coordinates shape: {dataset['coordinates'].shape}")
            for encoder_name in embeddings_dict.keys():
                emb_shape = dataset[f'{encoder_name}_embeddings'].shape
                print(f"  {encoder_name.upper()} embeddings shape: {emb_shape}")

        else:  # CSV format
            df_data = {
                'longitude': land_longitude,
                'latitude': land_latitude,
            }

            # Add embedding columns from each encoder
            for encoder_name, embeddings in embeddings_dict.items():
                for i in range(embeddings.shape[1]):
                    df_data[f'{encoder_name}_emb_{i:03d}'] = embeddings[:, i]

            df = pd.DataFrame(df_data)
            output_file = f"{output_path}.csv"
            df.to_csv(output_file, index=False)
            print(f"\n[OK] Saved CSV dataset to {output_file}")
            print(f"\nDataset shape: {df.shape}")

        # Create visualizations
        if plot_results:
            self.plot_sampled_locations(land_longitude, land_latitude,
                                       f"{output_path}_locations.png")

            # Create PCA visualization for each encoder
            for encoder_name, embeddings in embeddings_dict.items():
                self.plot_pca_embeddings(land_longitude, land_latitude, embeddings,
                                       encoder_name,
                                       f"{output_path}_{encoder_name}_pca.png")

        return output_file

    def plot_sampled_locations(self, longitude, latitude, save_path):
        """Create a plot showing the sampled land locations"""
        print(f"\nCreating location plot for {len(longitude):,} points...")

        fig = plt.figure(figsize=(15, 8))

        if CARTOPY_AVAILABLE:
            ax = plt.axes(projection=ccrs.Robinson())
            ax.add_feature(cfeatures.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeatures.LAND, color='lightgray', alpha=0.5)
            ax.add_feature(cfeatures.OCEAN, color='lightblue', alpha=0.3)
            ax.add_feature(cfeatures.BORDERS, linewidth=0.3, alpha=0.7)

            ax.scatter(longitude, latitude, c='red', s=0.5, alpha=0.6,
                      transform=ccrs.PlateCarree(),
                      label=f'{len(longitude):,} sampled points')

            ax.set_title(f'Geospatial Dataset: {len(longitude):,} Land Points\n'
                        f'(Fibonacci Sphere Sampling)', fontsize=14, pad=20)
            ax.legend(loc='lower left')
            ax.gridlines(draw_labels=False, alpha=0.3)
        else:
            ax = plt.subplot(111)
            ax.scatter(longitude, latitude, c='red', s=0.5, alpha=0.6)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'Geospatial Dataset: {len(longitude):,} Land Points')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Location plot saved to {save_path}")
        plt.close()

        # Create density heatmap
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
        hist, xedges, yedges = np.histogram2d(longitude, latitude, bins=180)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax2.imshow(hist.T, extent=extent, origin='lower',
                       cmap='Reds', aspect='auto', interpolation='gaussian')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title(f'Sampling Density Heatmap ({len(longitude):,} points)')
        plt.colorbar(im, ax=ax2, label='Point Density')

        density_path = save_path.replace('.png', '_density.png')
        plt.tight_layout()
        plt.savefig(density_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Density plot saved to {density_path}")
        plt.close()

    def plot_pca_embeddings(self, longitude, latitude, embeddings, encoder_name, save_path):
        """Create a world map with points colored by PCA-projected embeddings"""
        print(f"\nCreating PCA visualization for {encoder_name.upper()}...")

        # Perform PCA
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        pca = PCA(n_components=3)
        pca_embeddings = pca.fit_transform(embeddings_scaled)

        print(f"  PCA explained variance: {pca.explained_variance_ratio_}")
        print(f"  Total: {pca.explained_variance_ratio_.sum():.3f}")

        # Normalize to RGB
        pca_normalized = np.zeros_like(pca_embeddings)
        for i in range(3):
            comp = pca_embeddings[:, i]
            pca_normalized[:, i] = (comp - comp.min()) / (comp.max() - comp.min())
        colors = pca_normalized

        # Create main plot
        fig = plt.figure(figsize=(20, 12))

        if CARTOPY_AVAILABLE:
            ax = plt.axes(projection=ccrs.Robinson())
            ax.add_feature(cfeatures.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeatures.LAND, color='lightgray', alpha=0.3)
            ax.add_feature(cfeatures.OCEAN, color='lightblue', alpha=0.2)
            ax.add_feature(cfeatures.BORDERS, linewidth=0.3, alpha=0.5)

            ax.scatter(longitude, latitude, c=colors, s=1.0, alpha=0.8,
                      transform=ccrs.PlateCarree())

            ax.set_title(f'{encoder_name.upper()} Embeddings (PCA-projected to RGB)\n'
                        f'{len(longitude):,} points - '
                        f'Explained variance: {pca.explained_variance_ratio_.sum():.1%}',
                        fontsize=16, pad=20)
            ax.gridlines(draw_labels=False, alpha=0.3)
        else:
            ax = plt.subplot(111)
            ax.scatter(longitude, latitude, c=colors, s=1.0, alpha=0.8)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'{encoder_name.upper()} Embeddings (PCA-projected to RGB)\n'
                        f'{len(longitude):,} points - '
                        f'Explained variance: {pca.explained_variance_ratio_.sum():.1%}')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] PCA plot saved to {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate geospatial embedding datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--n_points', type=int, default=100000,
                       help='Number of land points to generate (default: 100000)')
    parser.add_argument('--encoders', nargs='+', choices=['geoclip', 'satclip'],
                       help='Encoders to use (default: all available)')
    parser.add_argument('--output_format', choices=['pt', 'csv'], default='pt',
                       help='Output format (default: pt)')
    parser.add_argument('--output_path', type=str, default='geospatial_dataset',
                       help='Output file path without extension (default: geospatial_dataset)')
    parser.add_argument('--device', choices=['cuda', 'cpu'],
                       help='Device to run models on (default: auto-detect)')
    parser.add_argument('--no_plot', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--cache_dir', type=str, default='./data_cache',
                       help='Directory for caching land mask data (default: ./data_cache)')

    args = parser.parse_args()

    print("="*60)
    print("Geospatial Embedding Dataset Generator")
    print("="*60)
    print(f"Target points: {args.n_points:,}")
    print(f"Encoders: {args.encoders or 'all available'}")
    print(f"Output format: {args.output_format}")
    print(f"Device: {args.device or 'auto-detect'}")
    print("="*60 + "\n")

    generator = GeospatialDatasetGenerator(cache_dir=args.cache_dir)

    output_file = generator.generate_dataset(
        n_points=args.n_points,
        encoders=args.encoders,
        output_format=args.output_format,
        output_path=args.output_path,
        device=args.device,
        plot_results=not args.no_plot
    )

    print("\n" + "="*60)
    print(f"[OK] Dataset generation complete!")
    print(f"Output: {output_file}")
    print("="*60)


if __name__ == "__main__":
    main()
