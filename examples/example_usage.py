"""
Example usage of the geospatial embedding generator.

This script demonstrates various ways to use the embedding generator
programmatically.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from get_embeddings import EmbeddingGenerator
import torch


def example_basic_usage():
    """Basic example with single location."""
    print("=" * 60)
    print("Example 1: Basic Usage - Single Location")
    print("=" * 60)

    # Initialize generator
    generator = EmbeddingGenerator()

    # New York City coordinates
    coordinates = [(40.7128, -74.0060)]

    # Generate embeddings
    embeddings = generator.generate_embeddings(coordinates)

    # Display results
    for encoder_name, emb in embeddings.items():
        print(f"\n{encoder_name.upper()}:")
        print(f"  Shape: {emb.shape}")
        print(f"  Norm: {torch.norm(emb).item():.4f}")
        print(f"  First 5 values: {emb[0][:5].tolist()}")


def example_multiple_locations():
    """Example with multiple locations."""
    print("\n" + "=" * 60)
    print("Example 2: Multiple Locations")
    print("=" * 60)

    generator = EmbeddingGenerator()

    # Famous cities
    coordinates = [
        (40.7128, -74.0060),   # New York City
        (34.0522, -118.2437),  # Los Angeles
        (51.5074, -0.1278),    # London
        (35.6762, 139.6503),   # Tokyo
        (-33.8688, 151.2093),  # Sydney
    ]

    city_names = ["NYC", "LA", "London", "Tokyo", "Sydney"]

    embeddings = generator.generate_embeddings(coordinates)

    print(f"\nGenerated embeddings for {len(coordinates)} cities")
    print(f"GeoCLIP shape: {embeddings['geoclip'].shape}")
    print(f"SatCLIP shape: {embeddings['satclip'].shape}")

    # Compute similarities between cities
    geoclip_emb = embeddings['geoclip']
    similarities = torch.mm(geoclip_emb, geoclip_emb.T)

    print("\nCosine similarities (GeoCLIP):")
    print("     " + "  ".join(f"{name:>6}" for name in city_names))
    for i, name in enumerate(city_names):
        print(f"{name:>6}", end=" ")
        for j in range(len(city_names)):
            print(f"{similarities[i, j].item():6.3f}", end=" ")
        print()


def example_save_and_load():
    """Example of saving and loading embeddings."""
    print("\n" + "=" * 60)
    print("Example 3: Save and Load Embeddings")
    print("=" * 60)

    import numpy as np

    generator = EmbeddingGenerator()
    coordinates = [(40.7128, -74.0060), (34.0522, -118.2437)]

    # Generate and save
    embeddings = generator.generate_embeddings(coordinates)
    output_path = Path(__file__).parent / "test_embeddings.npz"
    generator.save_embeddings(embeddings, str(output_path), coordinates)

    # Load
    loaded = np.load(output_path)
    print(f"\nSaved files: {list(loaded.keys())}")
    print(f"Coordinates shape: {loaded['coordinates'].shape}")
    print(f"GeoCLIP shape: {loaded['geoclip'].shape}")
    print(f"SatCLIP shape: {loaded['satclip'].shape}")

    # Clean up
    output_path.unlink()
    print(f"\nCleaned up test file: {output_path}")


def example_single_encoder():
    """Example using a single encoder."""
    print("\n" + "=" * 60)
    print("Example 4: Single Encoder Usage")
    print("=" * 60)

    from geoclip_encoder import GeoCLIPEncoder

    # Initialize only GeoCLIP
    encoder = GeoCLIPEncoder()

    coordinates = torch.Tensor([[40.7128, -74.0060], [34.0522, -118.2437]])
    embeddings = encoder.encode(coordinates)

    print(f"\nEncoder: {encoder.name}")
    print(f"Embedding dimension: {encoder.get_embedding_dim()}")
    print(f"Output shape: {embeddings.shape}")


def example_batch_processing():
    """Example of batch processing many locations."""
    print("\n" + "=" * 60)
    print("Example 5: Batch Processing")
    print("=" * 60)

    generator = EmbeddingGenerator()

    # Generate random coordinates
    import random
    random.seed(42)

    n_locations = 100
    coordinates = [
        (random.uniform(-90, 90), random.uniform(-180, 180))
        for _ in range(n_locations)
    ]

    print(f"\nProcessing {n_locations} random coordinates...")

    embeddings = generator.generate_embeddings(coordinates)

    print(f"[OK] Generated {n_locations} embeddings")
    print(f"  GeoCLIP: {embeddings['geoclip'].shape}")
    print(f"  SatCLIP: {embeddings['satclip'].shape}")


if __name__ == "__main__":
    try:
        example_basic_usage()
        example_multiple_locations()
        example_save_and_load()
        example_single_encoder()
        example_batch_processing()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
