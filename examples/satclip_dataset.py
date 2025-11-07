#!/usr/bin/env python3
"""
TerraNova SatCLIP Dataset Generator

Generates uniformly sampled land-only coordinates with SatCLIP embeddings
for training TerraNova's location encoder alignment.

Usage:
    # Generate 100k points (default) with fibonacci sampling
    python satclip_dataset_generator.py

    # Generate custom number of points with your SatCLIP checkpoint
    python satclip_dataset_generator.py --n_points 50000 --satclip_checkpoint path/to/satclip-vit16-l10.ckpt

    # Generate CSV output without plots
    python satclip_dataset_generator.py --output_format csv --no_plot

Requirements:
    pip install geopandas shapely pandas torch tqdm requests matplotlib cartopy

    Plus your SatCLIP installation and checkpoint file.
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
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeatures
import warnings
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors

sys.path.append('./satclip')
sys.path.append('./satclip/*')
sys.path.append(r"C:\Users\Carlos Rodriguez\Desktop\aqs-classifier\code\satclip")


warnings.filterwarnings('ignore')


class SatCLIPDatasetGenerator:
    def __init__(self, cache_dir="./data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.land_mask = None

    def download_land_mask(self):
        """Download Natural Earth land shapefile for masking"""
        land_file = self.cache_dir / "ne_110m_land.shp"

        if land_file.exists():
            print("Land mask already downloaded")
            return land_file

        print("Downloading Natural Earth land data...")

        # Use the CDN URL directly as it's more reliable
        url = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"

        zip_path = self.cache_dir / "ne_110m_land.zip"

        # Headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/zip,application/octet-stream,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }

        try:
            print(f"Requesting: {url}")
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()

            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192

            with open(zip_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rDownloading: {percent:.1f}%", end='', flush=True)

                print()  # New line after progress

            # Verify the downloaded file is a valid zip
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.cache_dir)
                print("Successfully downloaded and extracted Natural Earth data")

            except zipfile.BadZipFile:
                print("Downloaded file is not a valid zip. Trying fallback approach...")
                # Remove the corrupted file
                zip_path.unlink(missing_ok=True)

                # Try the original URL with different headers
                fallback_url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/physical/ne_110m_land.zip"
                print(f"Trying fallback URL: {fallback_url}")

                # Simpler headers for fallback
                simple_headers = {'User-Agent': 'curl/7.68.0'}
                response = requests.get(fallback_url, headers=simple_headers, timeout=30)
                response.raise_for_status()

                with open(zip_path, 'wb') as f:
                    f.write(response.content)

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.cache_dir)
                print("Successfully downloaded from fallback URL")

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download Natural Earth data: {e}")
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Downloaded file is not a valid zip file: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error downloading land mask: {e}")
        finally:
            # Clean up zip file
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

    def uniform_sphere_sampling(self, n_points):
        """Generate uniformly distributed points on sphere surface"""
        print(f"Generating {n_points} uniform points on sphere...")

        # Uniform sampling on sphere using inverse transform sampling
        u = np.random.uniform(0, 1, n_points)
        v = np.random.uniform(0, 1, n_points)

        # Convert to lat/lon
        longitude = 2 * np.pi * u - np.pi  # [-π, π]
        latitude = np.arccos(2 * v - 1) - np.pi / 2  # [-π/2, π/2]

        # Convert to degrees
        longitude_deg = np.degrees(longitude)
        latitude_deg = np.degrees(latitude)

        return longitude_deg, latitude_deg

    def fibonacci_sphere_sampling(self, n_points):
        """Alternative: Fibonacci sphere sampling for more even distribution"""
        print(f"Generating {n_points} Fibonacci sphere points...")

        indices = np.arange(0, n_points, dtype=float) + 0.5

        # Golden angle in radians
        phi = np.arccos(1 - 2 * indices / n_points)  # Latitude
        theta = np.pi * (1 + 5 ** 0.5) * indices  # Longitude

        latitude_deg = 90 - np.degrees(phi)  # Convert to standard lat range
        longitude_deg = np.degrees(theta) % 360 - 180  # Convert to standard lon range

        return longitude_deg, latitude_deg

    def filter_land_points(self, longitude, latitude):
        """Filter points to keep only those on land"""
        self.load_land_mask()

        print("Filtering for land points...")
        points = [Point(lon, lat) for lon, lat in zip(longitude, latitude)]

        # Check which points are on land (this can be slow for many points)
        land_mask = np.zeros(len(points), dtype=bool)

        for i, point in enumerate(tqdm(points, desc="Checking land mask")):
            # Check if point is within any land polygon
            land_mask[i] = self.land_mask.contains(point).any()

        land_longitude = longitude[land_mask]
        land_latitude = latitude[land_mask]

        print(f"Found {len(land_longitude)} land points out of {len(longitude)} total")
        return land_longitude, land_latitude

    def vectorized_land_filter(self, longitude, latitude, batch_size=10000):
        """Faster vectorized land filtering"""
        self.load_land_mask()

        print("Filtering for land points (vectorized)...")

        # Combine all land polygons into one geometry for faster checking
        land_union = self.land_mask.geometry.unary_union

        land_mask = np.zeros(len(longitude), dtype=bool)

        # Process in batches to avoid memory issues
        for i in tqdm(range(0, len(longitude), batch_size), desc="Land filtering"):
            end_idx = min(i + batch_size, len(longitude))
            batch_lon = longitude[i:end_idx]
            batch_lat = latitude[i:end_idx]

            # Vectorized point-in-polygon test
            batch_mask = contains(land_union, batch_lon, batch_lat)
            land_mask[i:end_idx] = batch_mask

        land_longitude = longitude[land_mask]
        land_latitude = latitude[land_mask]

        print(f"Found {len(land_longitude)} land points out of {len(longitude)} total")
        return land_longitude, land_latitude

    def get_satclip_embeddings(self, longitude, latitude, checkpoint_path=None):
        """Get SatCLIP embeddings for coordinates using the correct loading pattern

        Args:
            longitude: Array of longitude coordinates
            latitude: Array of latitude coordinates
            checkpoint_path: Path to SatCLIP checkpoint file
        """
        print("Getting SatCLIP embeddings...")

        try:
            # Import SatCLIP following the pattern from provided code
            from satclip.load import get_satclip
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Load the SatCLIP model
            if checkpoint_path is None:
                # You'll need to provide the actual path to your checkpoint
                checkpoint_path = "path/to/satclip-vit16-l10.ckpt"
                print(f"Warning: Using default checkpoint path: {checkpoint_path}")
                print("Please provide the correct path to your SatCLIP checkpoint")
                raise FileNotFoundError("SatCLIP checkpoint not found")

            print(f"Loading SatCLIP model from {checkpoint_path}")

            # Get the full model, not just the location encoder
            full_model = get_satclip(checkpoint_path, device=device, return_all=True)
            full_model.eval()

            # Get just the location encoder for generating embeddings
            location_model = full_model.location
            location_model.eval()

            embeddings = []
            batch_size = 50000  # Reduce batch size to avoid memory issues

            print(f"Processing {len(longitude)} coordinates in batches of {batch_size}")

            for i in tqdm(range(0, len(longitude), batch_size), desc="Getting SatCLIP embeddings"):
                end_idx = min(i + batch_size, len(longitude))
                batch_lon = longitude[i:end_idx]
                batch_lat = latitude[i:end_idx]

                # Create coordinate tensor [longitude, latitude] - this is important!
                # SatCLIP training data uses (lon, lat) format, not (lat, lon)!
                coords = torch.tensor([[float(lon), float(lat)] for lat, lon in zip(batch_lat, batch_lon)],
                                    dtype=torch.float32)

                with torch.no_grad():
                    # Use the location encoder directly
                    # Convert to double precision as the model expects it internally
                    batch_embeddings = location_model(coords.double().to(device))

                    # Ensure we get the right shape
                    if batch_embeddings.dim() == 1:
                        batch_embeddings = batch_embeddings.unsqueeze(0)

                    # Convert to float32 and move to CPU
                    embeddings.append(batch_embeddings.cpu().float())

            # Concatenate all embeddings
            embeddings = torch.cat(embeddings, dim=0).numpy()

            print(f"Generated {len(embeddings)} SatCLIP embeddings of dimension {embeddings.shape[1]}")

            # Basic sanity check and debugging info
            embedding_stats = {
                'mean': np.mean(embeddings),
                'std': np.std(embeddings),
                'min': np.min(embeddings),
                'max': np.max(embeddings),
                'zeros': np.sum(embeddings == 0),
                'nans': np.sum(np.isnan(embeddings))
            }

            print(f"Embedding statistics: {embedding_stats}")

            if np.all(embeddings == 0) or np.all(np.isnan(embeddings)):
                print("WARNING: Generated embeddings are all zeros or NaN!")
                raise ValueError("Invalid embeddings generated")

            # Check if embeddings are identical (another sign of problems)
            if len(embeddings) > 1:
                first_embedding = embeddings[0]
                if np.all([np.allclose(emb, first_embedding, atol=1e-10) for emb in embeddings[:min(10, len(embeddings))]]):
                    print("WARNING: First 10 embeddings are nearly identical - possible model issue!")

            return embeddings

        except (ImportError, FileNotFoundError) as e:
            print(f"SatCLIP not available or checkpoint not found: {e}")
            print("Generating random embeddings as placeholder")
            # Use typical SatCLIP embedding dimension
            embedding_dim = 256  # Adjust based on your model
            embeddings = np.random.randn(len(longitude), embedding_dim).astype(np.float32)
            print(f"Generated {len(embeddings)} random embeddings of dimension {embedding_dim}")
            return embeddings

    def generate_dataset(self, n_points=1e6, sampling_method="fibonacci",
                         output_format="pt", output_path="satclip_dataset",
                         satclip_checkpoint=None, plot_results=True):
        """Generate the complete dataset

        Args:
            n_points: Target number of land points (default 100,000)
            sampling_method: "uniform" or "fibonacci" (default "fibonacci")
            output_format: "pt" or "csv"
            output_path: Output file path (without extension)
            satclip_checkpoint: Path to SatCLIP checkpoint file
            plot_results: Whether to create a plot of sampled locations
        """

        # Sample more points initially since we'll filter for land
        oversample_factor = 10.0  # Assume ~33% of Earth is land
        initial_points = int(n_points * oversample_factor)

        # Generate initial uniform points (default to fibonacci)
        if sampling_method == "fibonacci":
            longitude, latitude = self.fibonacci_sphere_sampling(initial_points)
        else:
            longitude, latitude = self.uniform_sphere_sampling(initial_points)

        # Filter for land points
        land_longitude, land_latitude = self.vectorized_land_filter(longitude, latitude)

        # If we don't have enough land points, sample more
        while len(land_longitude) < n_points:
            print(f"Need more points. Have {len(land_longitude)}, need {n_points}")
            additional_points = int((n_points - len(land_longitude)) * oversample_factor)

            if sampling_method == "fibonacci":
                add_lon, add_lat = self.fibonacci_sphere_sampling(additional_points)
            else:
                add_lon, add_lat = self.uniform_sphere_sampling(additional_points)

            add_land_lon, add_land_lat = self.vectorized_land_filter(add_lon, add_lat)

            land_longitude = np.concatenate([land_longitude, add_land_lon])
            land_latitude = np.concatenate([land_latitude, add_land_lat])

        # Trim to exact number requested
        if len(land_longitude) > n_points:
            indices = np.random.choice(len(land_longitude), int(n_points), replace=False)
            land_longitude = land_longitude[indices]
            land_latitude = land_latitude[indices]

        print(f"Final dataset has {len(land_longitude)} points")

        # Get SatCLIP embeddings
        embeddings = self.get_satclip_embeddings(land_longitude, land_latitude, satclip_checkpoint)

        # Create dataset
        if output_format == "pt":
            # Save as PyTorch tensor
            dataset = {
                'longitude': torch.from_numpy(land_longitude),
                'latitude': torch.from_numpy(land_latitude),
                'satclip_embeddings': torch.from_numpy(embeddings),
                'coordinates': torch.from_numpy(np.column_stack([land_longitude, land_latitude]))
            }

            output_file = f"{output_path}.pt"
            torch.save(dataset, output_file)
            print(f"Saved PyTorch dataset to {output_file}")

        else:  # CSV format
            # Create DataFrame
            df_data = {
                'longitude': land_longitude,
                'latitude': land_latitude,
            }

            # Add embedding columns
            for i in range(embeddings.shape[1]):
                df_data[f'embedding_{i:03d}'] = embeddings[:, i]

            df = pd.DataFrame(df_data)

            output_file = f"{output_path}.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved CSV dataset to {output_file}")

        # Plot results if requested
        if plot_results:
            self.plot_sampled_locations(land_longitude, land_latitude,
                                        f"{output_path}_locations.png")

            # Create PCA embedding visualization
            self.plot_pca_embeddings(land_longitude, land_latitude, embeddings,
                                   f"{output_path}_pca_embeddings.png")

        return output_file

    def plot_sampled_locations(self, longitude, latitude, save_path):
        """Create a plot showing the sampled land locations"""
        print(f"Creating plot of {len(longitude)} sampled locations...")

        try:
            # Create figure with cartopy projection
            fig = plt.figure(figsize=(15, 8))
            ax = plt.axes(projection=ccrs.Robinson())

            # Add map features
            ax.add_feature(cfeatures.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeatures.LAND, color='lightgray', alpha=0.5)
            ax.add_feature(cfeatures.OCEAN, color='lightblue', alpha=0.3)
            ax.add_feature(cfeatures.BORDERS, linewidth=0.3, alpha=0.7)

            # Plot sampled points
            scatter = ax.scatter(longitude, latitude,
                                 c='red', s=0.05, alpha=0.6,
                                 transform=ccrs.PlateCarree(),
                                 label=f'{len(longitude):,} sampled points')

            # Add title and legend
            ax.set_title(f'TerraNova SatCLIP Dataset: {len(longitude):,} Land Points\n'
                         f'(Fibonacci Sphere Sampling)', fontsize=14, pad=20)
            ax.legend(loc='lower left')

            # Add gridlines
            ax.gridlines(draw_labels=False, alpha=0.3)

        except ImportError:
            print("Cartopy not available, using simple matplotlib plot...")
            # Fallback to simple matplotlib plot
            fig, ax = plt.subplots(1, 1, figsize=(15, 8))
            ax.scatter(longitude, latitude, c='red', s=0.05, alpha=0.6)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'TerraNova SatCLIP Dataset: {len(longitude):,} Land Points')
            ax.grid(True, alpha=0.3)

        # Save the plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=1000, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close()

        # Also create a density plot
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))

        # Create 2D histogram for density
        hist, xedges, yedges = np.histogram2d(longitude, latitude, bins=180)

        # Plot heatmap
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax2.imshow(hist.T, extent=extent, origin='lower',
                        cmap='Reds', aspect='auto', interpolation='gaussian')

        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title(f'Sampling Density Heatmap ({len(longitude):,} points)')
        plt.colorbar(im, ax=ax2, label='Point Density')

        # Save density plot
        density_path = save_path.replace('.png', '_density.png')
        plt.tight_layout()
        plt.savefig(density_path, dpi=1000, bbox_inches='tight')
        print(f"Density plot saved to {density_path}")
        plt.close()

    def plot_pca_embeddings(self, longitude, latitude, embeddings, save_path):
        """Create a world map with points colored by PCA-projected SatCLIP embeddings"""
        print(f"Creating PCA embedding visualization for {len(longitude)} points...")

        # Perform PCA on embeddings to reduce to 3D
        print("Performing PCA on embeddings...")
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        pca = PCA(n_components=3)
        pca_embeddings = pca.fit_transform(embeddings_scaled)

        print(f"PCA explained variance: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")

        # Convert PCA components to RGB colors
        # Normalize each PCA component to [0, 1] range
        pca_normalized = np.zeros_like(pca_embeddings)
        for i in range(3):
            comp = pca_embeddings[:, i]
            pca_normalized[:, i] = (comp - comp.min()) / (comp.max() - comp.min())

        # Create RGB colors from normalized PCA components
        colors = pca_normalized

        try:
            # Create figure with cartopy projection
            fig = plt.figure(figsize=(20, 12))
            ax = plt.axes(projection=ccrs.Robinson())

            # Add map features
            ax.add_feature(cfeatures.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeatures.LAND, color='lightgray', alpha=0.3)
            ax.add_feature(cfeatures.OCEAN, color='lightblue', alpha=0.2)
            ax.add_feature(cfeatures.BORDERS, linewidth=0.3, alpha=0.5)

            # Plot points with PCA-based colors
            scatter = ax.scatter(longitude, latitude,
                                c=colors, s=.05, alpha=0.8,
                                transform=ccrs.PlateCarree())

            # Add title
            ax.set_title(f'SatCLIP Embeddings Visualization (PCA-projected to RGB)\n'
                        f'{len(longitude):,} points - '
                        f'Explained variance: {pca.explained_variance_ratio_.sum():.1%}',
                        fontsize=16, pad=20)

            # Add gridlines
            ax.gridlines(draw_labels=False, alpha=0.3)

            # Create a separate legend showing PCA component interpretation
            legend_fig, legend_axes = plt.subplots(1, 3, figsize=(15, 4))

            for i, (comp_name, color_channel) in enumerate([('Red (PC1)', 0), ('Green (PC2)', 1), ('Blue (PC3)', 2)]):
                component_values = pca_embeddings[:, color_channel]

                # Create a scatter plot for this component
                scatter_comp = legend_axes[i].scatter(longitude, latitude,
                                                     c=component_values, cmap='viridis',
                                                     s=0.05, alpha=0.7)

                legend_axes[i].set_title(f'{comp_name}\n'
                                       f'Explained Variance: {pca.explained_variance_ratio_[color_channel]:.1%}')
                legend_axes[i].set_xlabel('Longitude')
                legend_axes[i].set_ylabel('Latitude')

                # Add colorbar
                plt.colorbar(scatter_comp, ax=legend_axes[i])

            plt.tight_layout()
            legend_path = save_path.replace('.png', '_components.png')
            legend_fig.savefig(legend_path, dpi=1000, bbox_inches='tight')
            print(f"PCA components plot saved to {legend_path}")
            plt.close(legend_fig)

        except ImportError:
            print("Cartopy not available, using simple matplotlib plot...")
            # Fallback to simple matplotlib plot
            fig, ax = plt.subplots(1, 1, figsize=(20, 12))

            # Plot with PCA colors
            scatter = ax.scatter(longitude, latitude, c=colors, s=1.0, alpha=0.8)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'SatCLIP Embeddings Visualization (PCA-projected to RGB)\n'
                        f'{len(longitude):,} points - '
                        f'Explained variance: {pca.explained_variance_ratio_.sum():.1%}')
            ax.grid(True, alpha=0.3)

            # Create component plots
            fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
            for i, comp_name in enumerate(['PC1 (Red)', 'PC2 (Green)', 'PC3 (Blue)']):
                component_values = pca_embeddings[:, i]
                scatter_comp = axes[i].scatter(longitude, latitude,
                                             c=component_values, cmap='viridis',
                                             s=0.05, alpha=0.7)
                axes[i].set_title(f'{comp_name}\n'
                                f'Explained Variance: {pca.explained_variance_ratio_[i]:.1%}')
                axes[i].set_xlabel('Longitude')
                axes[i].set_ylabel('Latitude')
                plt.colorbar(scatter_comp, ax=axes[i])

            plt.tight_layout()
            components_path = save_path.replace('.png', '_components.png')
            fig2.savefig(components_path, dpi=300, bbox_inches='tight')
            print(f"PCA components plot saved to {components_path}")
            plt.close(fig2)

        # Save the main plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA embedding visualization saved to {save_path}")
        plt.close(fig)

        # Create additional analysis plots
        self.plot_embedding_analysis(pca_embeddings, pca, save_path.replace('.png', '_analysis.png'))

    def plot_embedding_analysis(self, pca_embeddings, pca, save_path):
        """Create additional analysis plots for the PCA embeddings"""
        print("Creating embedding analysis plots...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. PCA variance explained
        axes[0, 0].bar(range(1, 4), pca.explained_variance_ratio_, alpha=0.7)
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Explained Variance Ratio')
        axes[0, 0].set_title('PCA Explained Variance')
        axes[0, 0].set_xticks(range(1, 4))

        # 2. 3D scatter plot of PCA embeddings
        ax_3d = fig.add_subplot(2, 2, 2, projection='3d')
        scatter_3d = ax_3d.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], pca_embeddings[:, 2],
                                   c=np.arange(len(pca_embeddings)), cmap='viridis', alpha=0.6, s=1)
        ax_3d.set_xlabel('PC1')
        ax_3d.set_ylabel('PC2')
        ax_3d.set_zlabel('PC3')
        ax_3d.set_title('3D PCA Embedding Space')

        # 3. PC1 vs PC2
        scatter_12 = axes[1, 0].scatter(pca_embeddings[:, 0], pca_embeddings[:, 1],
                                        c=pca_embeddings[:, 2], cmap='viridis', alpha=0.6, s=1)
        axes[1, 0].set_xlabel('PC1')
        axes[1, 0].set_ylabel('PC2')
        axes[1, 0].set_title('PC1 vs PC2 (colored by PC3)')
        plt.colorbar(scatter_12, ax=axes[1, 0])

        # 4. Distribution of each PC
        axes[1, 1].hist(pca_embeddings[:, 0], bins=50, alpha=0.7, label='PC1', color='red')
        axes[1, 1].hist(pca_embeddings[:, 1], bins=50, alpha=0.7, label='PC2', color='green')
        axes[1, 1].hist(pca_embeddings[:, 2], bins=50, alpha=0.7, label='PC3', color='blue')
        axes[1, 1].set_xlabel('Component Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of PC Values')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Embedding analysis saved to {save_path}")
        plt.close()


def main():
    """Example usage"""
    generator = SatCLIPDatasetGenerator()

    # Generate dataset with 100k land points (default)
    output_file = generator.generate_dataset(
        n_points=1e6,  # Default to 100k points
        sampling_method="fibonacci",  # Default to fibonacci
        output_format="pt",  # or "csv"
        output_path="terranova_satclip_dataset",
        satclip_checkpoint=r"C:\Users\Carlos Rodriguez\PycharmProjects\biotime\scripts\satclip-vit16-l40.ckpt",  # Provide path to your SatCLIP checkpoint
        plot_results=True
    )

    output_file = generator.generate_dataset(
        n_points=1e6,  # Default to 100k points
        sampling_method="fibonacci",  # Default to fibonacci
        output_format="csv",  # or "csv"
        output_path="terranova_satclip_dataset",
        satclip_checkpoint=r"C:\Users\Carlos Rodriguez\PycharmProjects\biotime\scripts\satclip-vit16-l40.ckpt",
        # Provide path to your SatCLIP checkpoint
        plot_results=True
    )

    print(f"Dataset saved to: {output_file}")

    # Load and inspect the dataset
    if output_file.endswith('.pt'):
        dataset = torch.load(output_file)
        print(f"\nDataset Summary:")
        print(f"Dataset keys: {list(dataset.keys())}")
        print(f"Number of points: {len(dataset['coordinates'])}")
        print(f"Coordinates shape: {dataset['coordinates'].shape}")
        print(f"Embeddings shape: {dataset['satclip_embeddings'].shape}")
        print(f"Longitude range: {dataset['longitude'].min():.2f} to {dataset['longitude'].max():.2f}")
        print(f"Latitude range: {dataset['latitude'].min():.2f} to {dataset['latitude'].max():.2f}")
        print(f"Embedding dimension: {dataset['satclip_embeddings'].shape[1]}")


if __name__ == "__main__":
    # Install required packages:
    # pip install geopandas shapely pandas torch tqdm requests matplotlib cartopy

    main()