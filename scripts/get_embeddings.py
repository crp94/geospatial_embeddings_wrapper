#!/usr/bin/env python3
"""
Geospatial Embedding Generator

A unified script for generating geospatial embeddings from latitude/longitude
coordinates using multiple models (GeoCLIP, SatCLIP, and extensible to more).

"""

import argparse
import json
import sys
import torch
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

# Add parent directory to path to import wrappers
sys.path.insert(0, str(Path(__file__).parent.parent))

from wrappers.embedding_encoder import GeoEmbeddingEncoder
from wrappers.geoclip_encoder import GeoCLIPEncoder
from wrappers.satclip_encoder import SatCLIPEncoder


class EmbeddingGenerator:
    """
    Main class for generating embeddings from multiple encoders.
    """

    def __init__(self, encoders: Optional[List[str]] = None, device: Optional[str] = None):
        """
        Initialize the embedding generator.

        Args:
            encoders: List of encoder names to use. Options: ['geoclip', 'satclip']
                     If None, all available encoders will be used.
            device: Device to run models on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoders: Dict[str, GeoEmbeddingEncoder] = {}

        # Determine which encoders to initialize
        available_encoders = {
            'geoclip': GeoCLIPEncoder,
            'satclip': SatCLIPEncoder
        }

        encoders_to_use = encoders or list(available_encoders.keys())

        # Initialize requested encoders
        for encoder_name in encoders_to_use:
            if encoder_name.lower() not in available_encoders:
                print(f"Warning: Unknown encoder '{encoder_name}'. Skipping.")
                continue

            try:
                encoder_class = available_encoders[encoder_name.lower()]
                print(f"Initializing {encoder_name.upper()} encoder...")
                self.encoders[encoder_name.lower()] = encoder_class(device=self.device)
                print(f"[OK] {encoder_name.upper()} encoder ready")
            except ImportError as e:
                print(f"[ERROR] Could not initialize {encoder_name.upper()}: {e}")
            except Exception as e:
                print(f"[ERROR] Error initializing {encoder_name.upper()}: {e}")

        if not self.encoders:
            raise RuntimeError("No encoders were successfully initialized!")

        print(f"\nActive encoders: {', '.join(self.encoders.keys())}")
        print(f"Device: {self.device}\n")

    def generate_embeddings(
        self,
        coordinates: List[tuple],
        return_numpy: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Generate embeddings for given coordinates using all active encoders.

        Args:
            coordinates: List of (latitude, longitude) tuples
            return_numpy: If True, return numpy arrays instead of torch tensors

        Returns:
            Dictionary mapping encoder names to embedding tensors/arrays
        """
        coords_tensor = torch.Tensor(coordinates)
        results = {}

        for encoder_name, encoder in self.encoders.items():
            print(f"Generating {encoder_name.upper()} embeddings...")
            embeddings = encoder.encode(coords_tensor)

            if return_numpy:
                embeddings = embeddings.numpy()

            results[encoder_name] = embeddings

        return results

    def save_embeddings(
        self,
        embeddings: Dict[str, torch.Tensor],
        output_path: str,
        coordinates: Optional[List[tuple]] = None
    ):
        """
        Save embeddings to a file.

        Args:
            embeddings: Dictionary of embeddings from generate_embeddings()
            output_path: Path to save the embeddings
            coordinates: Optional list of coordinates to save alongside embeddings
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to numpy for saving
        save_dict = {}
        for encoder_name, emb in embeddings.items():
            if isinstance(emb, torch.Tensor):
                save_dict[encoder_name] = emb.numpy()
            else:
                save_dict[encoder_name] = emb

        # Add coordinates if provided
        if coordinates is not None:
            save_dict['coordinates'] = np.array(coordinates)

        # Save based on file extension
        if output_path.suffix == '.npz':
            np.savez(output_path, **save_dict)
        elif output_path.suffix == '.pt':
            torch.save(embeddings, output_path)
        else:
            # Default to .npz
            output_path = output_path.with_suffix('.npz')
            np.savez(output_path, **save_dict)

        print(f"[OK] Embeddings saved to: {output_path}")


def main():
    """
    Main CLI interface for the embedding generator.
    """
    parser = argparse.ArgumentParser(
        description="Generate geospatial embeddings from latitude/longitude coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate embeddings for New York City and Los Angeles
  python get_embeddings.py --lat 40.7128 -74.0060 --lon 34.0522 -118.2437

  # Use only GeoCLIP encoder
  python get_embeddings.py --lat 40.7128 --lon -74.0060 --encoders geoclip

  # Read coordinates from a file
  python get_embeddings.py --input coordinates.txt --output embeddings.npz

  # Use specific device
  python get_embeddings.py --lat 40.7128 --lon -74.0060 --device cuda
        """
    )

    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument(
        '--lat', '--latitude',
        type=float,
        nargs='+',
        help='Latitude values (in degrees, -90 to 90)'
    )
    input_group.add_argument(
        '--lon', '--longitude',
        type=float,
        nargs='+',
        help='Longitude values (in degrees, -180 to 180)'
    )
    input_group.add_argument(
        '--input', '-i',
        type=str,
        help='Input file with coordinates (CSV or JSON format)'
    )

    # Model options
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument(
        '--encoders', '-e',
        type=str,
        nargs='+',
        choices=['geoclip', 'satclip'],
        help='Encoders to use (default: all available)'
    )
    model_group.add_argument(
        '--device', '-d',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to run models on (default: auto-detect)'
    )

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path (.npz or .pt)'
    )
    output_group.add_argument(
        '--print',
        action='store_true',
        help='Print embeddings to console'
    )

    args = parser.parse_args()

    # Parse coordinates
    coordinates = []

    if args.lat and args.lon:
        if len(args.lat) != len(args.lon):
            parser.error("Number of latitude and longitude values must match")
        coordinates = list(zip(args.lat, args.lon))

    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            parser.error(f"Input file not found: {args.input}")

        # Read coordinates from file
        if input_path.suffix == '.json':
            with open(input_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    coordinates = [(item['lat'], item['lon']) for item in data]
                else:
                    coordinates = data['coordinates']
        elif input_path.suffix in ['.csv', '.txt']:
            with open(input_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.replace(',', ' ').split()
                        if len(parts) >= 2:
                            lat, lon = float(parts[0]), float(parts[1])
                            coordinates.append((lat, lon))
        else:
            parser.error(f"Unsupported input file format: {input_path.suffix}")

    else:
        parser.error("Must provide either --lat/--lon or --input")

    if not coordinates:
        parser.error("No valid coordinates found")

    print(f"Processing {len(coordinates)} coordinate(s)...\n")

    # Initialize generator
    try:
        generator = EmbeddingGenerator(encoders=args.encoders, device=args.device)
    except Exception as e:
        print(f"Error initializing generator: {e}")
        return 1

    # Generate embeddings
    try:
        embeddings = generator.generate_embeddings(coordinates)
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return 1

    # Print results if requested
    if args.print:
        print("\n" + "="*60)
        print("EMBEDDINGS")
        print("="*60)
        for i, (lat, lon) in enumerate(coordinates):
            print(f"\nLocation {i+1}: ({lat:.4f}, {lon:.4f})")
            for encoder_name, emb in embeddings.items():
                print(f"  {encoder_name.upper()}: shape={emb[i].shape}, "
                      f"norm={torch.norm(emb[i]).item():.4f}")
                if len(coordinates) == 1:  # Print first few values for single location
                    print(f"    First 5 values: {emb[i][:5].tolist()}")

    # Save results if output path provided
    if args.output:
        generator.save_embeddings(embeddings, args.output, coordinates)

    print("\n[OK] Done!")
    return 0


if __name__ == "__main__":
    exit(main())
