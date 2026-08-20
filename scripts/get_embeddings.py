#!/usr/bin/env python3
"""Query registered geospatial encoders at latitude/longitude coordinates."""

import argparse
import json
import sys
import torch
import numpy as np
from typing import Any, Dict, Iterable, List, Optional
from pathlib import Path

# Add parent directory to path to import wrappers
sys.path.insert(0, str(Path(__file__).parent.parent))

from wrappers.embedding_encoder import GeoEmbeddingEncoder
from wrappers.registry import (
    DEFAULT_ENCODER_NAMES,
    get_encoder_class,
    list_encoder_names,
    normalize_encoder_name,
)


def parse_encoder_roots(values: Iterable[str] | None) -> dict[str, str]:
    """Parse ``encoder=path`` values, resolving registry aliases."""
    roots: dict[str, str] = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError(
                f"Invalid --encoder-root value '{value}'. Expected encoder=/path/to/data"
            )
        raw_name, raw_path = value.split("=", 1)
        if not raw_name or not raw_path:
            raise ValueError(
                f"Invalid --encoder-root value '{value}'. Expected encoder=/path/to/data"
            )
        roots[normalize_encoder_name(raw_name)] = raw_path
    return roots


def _coordinate_pair(value: Any, context: str) -> tuple[float, float]:
    """Convert a JSON coordinate item to the standard ``(lat, lon)`` pair."""
    if isinstance(value, dict):
        latitude = value.get("lat", value.get("latitude"))
        longitude = value.get("lon", value.get("longitude"))
        if latitude is None or longitude is None:
            raise ValueError(f"{context} must provide latitude/longitude (or lat/lon) fields")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        latitude, longitude = value
    else:
        raise ValueError(f"{context} must be a [latitude, longitude] pair or mapping")

    try:
        return float(latitude), float(longitude)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} contains non-numeric coordinates") from exc


def read_coordinates(input_path: str | Path) -> list[tuple[float, float]]:
    """Read JSON, CSV, or whitespace-delimited coordinate files strictly."""
    path = Path(input_path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
        if isinstance(data, dict):
            if "coordinates" in data:
                data = data["coordinates"]
            elif "latitude" in data and "longitude" in data:
                latitudes, longitudes = data["latitude"], data["longitude"]
                if len(latitudes) != len(longitudes):
                    raise ValueError("JSON latitude and longitude arrays must have the same length")
                data = list(zip(latitudes, longitudes))
            else:
                raise ValueError(
                    "JSON input must be a coordinate list, contain 'coordinates', "
                    "or contain 'latitude' and 'longitude' arrays"
                )
        if not isinstance(data, list):
            raise ValueError("JSON coordinates must be a list")
        return [_coordinate_pair(item, f"JSON coordinate {index + 1}") for index, item in enumerate(data)]

    if suffix not in {".csv", ".txt"}:
        raise ValueError(f"Unsupported input file format: {suffix or '<none>'}")

    rows: list[tuple[int, list[str]]] = []
    with path.open("r", encoding="utf-8") as file_handle:
        for line_number, raw_line in enumerate(file_handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            values = line.replace(",", " ").split()
            rows.append((line_number, values))

    if not rows:
        return []

    first_line, first_values = rows[0]
    try:
        float(first_values[0])
        float(first_values[1])
    except (IndexError, ValueError):
        normalized_headers = [value.lower() for value in first_values]
        try:
            latitude_index = next(
                index for index, value in enumerate(normalized_headers) if value in {"lat", "latitude"}
            )
            longitude_index = next(
                index for index, value in enumerate(normalized_headers) if value in {"lon", "longitude"}
            )
        except StopIteration as exc:
            raise ValueError(
                f"Line {first_line}: expected two numeric columns or lat/lon headers"
            ) from exc
        rows = rows[1:]
    else:
        latitude_index, longitude_index = 0, 1

    coordinates: list[tuple[float, float]] = []
    for line_number, values in rows:
        try:
            coordinates.append(
                (float(values[latitude_index]), float(values[longitude_index]))
            )
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Line {line_number}: invalid latitude/longitude values") from exc
    return coordinates


def validate_coordinates(coordinates: Iterable[tuple[float, float]]) -> torch.Tensor:
    """Return validated CPU float32 coordinates in the repository's lat/lon order."""
    return GeoEmbeddingEncoder.validate_coordinates(
        torch.as_tensor(list(coordinates), dtype=torch.float32)
    )


class EmbeddingGenerator:
    """
    Main class for generating embeddings from multiple encoders.
    """

    def __init__(
        self,
        encoders: Optional[List[str]] = None,
        device: Optional[str] = None,
        encoder_roots: Optional[dict[str, str]] = None,
    ):
        """
        Initialize the embedding generator.

        Args:
            encoders: Canonical registry names or aliases. If None, use the
                lightweight point-query defaults from the registry.
            device: Device to run models on ('cuda', 'cpu', or None for auto-detect)
            encoder_roots: Optional canonical encoder-name to local data/checkpoint
                specification mapping.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoders: Dict[str, GeoEmbeddingEncoder] = {}

        self.encoder_roots = encoder_roots or {}
        encoders_to_use = encoders or list(DEFAULT_ENCODER_NAMES)
        try:
            resolved_names = [normalize_encoder_name(name) for name in encoders_to_use]
        except KeyError as exc:
            raise ValueError(str(exc)) from exc

        if len(set(resolved_names)) != len(resolved_names):
            raise ValueError("Each encoder may be requested only once")

        # Initialize requested encoders
        for encoder_name in resolved_names:
            try:
                encoder_class = get_encoder_class(encoder_name)
                print(f"Initializing {encoder_name.upper()} encoder...")
                self.encoders[encoder_name] = encoder_class(
                    device=self.device,
                    data_root=self.encoder_roots.get(encoder_name),
                )
                print(f"[OK] {encoder_name.upper()} encoder ready")
            except Exception as exc:
                raise RuntimeError(
                    f"Could not initialize requested encoder '{encoder_name}': {exc}"
                ) from exc

        print(f"\nActive encoders: {', '.join(self.encoders.keys())}")
        print(f"Device: {self.device}\n")

    def generate_embeddings(
        self,
        coordinates: List[tuple],
        return_numpy: bool = False,
        year: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate embeddings for given coordinates using all active encoders.

        Args:
            coordinates: List of (latitude, longitude) tuples
            return_numpy: If True, return numpy arrays instead of torch tensors

        Returns:
            Dictionary mapping encoder names to embedding tensors/arrays
        """
        coords_tensor = validate_coordinates(coordinates)
        results = {}

        for encoder_name, encoder in self.encoders.items():
            print(f"Generating {encoder_name.upper()} embeddings...")
            embeddings = encoder.encode(coords_tensor, year=year)

            if not isinstance(embeddings, torch.Tensor):
                raise TypeError(f"{encoder_name} returned {type(embeddings).__name__}, not a tensor")
            if embeddings.ndim != 2 or embeddings.shape[0] != coords_tensor.shape[0]:
                raise ValueError(
                    f"{encoder_name} returned shape {tuple(embeddings.shape)} for "
                    f"{coords_tensor.shape[0]} coordinates"
                )

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

        coordinates_tensor = (
            validate_coordinates(coordinates) if coordinates is not None else None
        )
        # Keep the existing encoder-name keys while adding a documented common
        # schema to both file types.
        numpy_embeddings: dict[str, np.ndarray] = {}
        tensor_embeddings: dict[str, torch.Tensor] = {}
        for encoder_name, emb in embeddings.items():
            if isinstance(emb, torch.Tensor):
                tensor = emb.detach().cpu()
            else:
                tensor = torch.as_tensor(emb)
            tensor_embeddings[encoder_name] = tensor
            numpy_embeddings[encoder_name] = tensor.numpy()

        metadata = self._build_metadata(tensor_embeddings, coordinates_tensor)
        npz_payload: dict[str, np.ndarray] = dict(numpy_embeddings)
        pt_payload: dict[str, Any] = dict(tensor_embeddings)
        if coordinates_tensor is not None:
            coordinates_lonlat = coordinates_tensor[:, [1, 0]]
            coordinates_np = coordinates_tensor.numpy()
            npz_payload.update(
                {
                    "coordinates": coordinates_np,
                    "coordinates_latlon": coordinates_np,
                    "coordinates_lonlat": coordinates_lonlat.numpy(),
                    "latitude": coordinates_np[:, 0],
                    "longitude": coordinates_np[:, 1],
                }
            )
            pt_payload.update(
                {
                    "coordinates": coordinates_tensor,
                    "coordinates_latlon": coordinates_tensor,
                    "coordinates_lonlat": coordinates_lonlat,
                    "latitude": coordinates_tensor[:, 0],
                    "longitude": coordinates_tensor[:, 1],
                }
            )
        pt_payload["metadata"] = metadata
        npz_payload["metadata_json"] = np.asarray(json.dumps(metadata, default=str))

        # Save based on file extension
        if output_path.suffix == '.npz':
            np.savez(output_path, **npz_payload)
        elif output_path.suffix == '.pt':
            torch.save(pt_payload, output_path)
        else:
            # Default to .npz
            output_path = output_path.with_suffix('.npz')
            np.savez(output_path, **npz_payload)

        print(f"[OK] Embeddings saved to: {output_path}")

    def _build_metadata(
        self,
        embeddings: dict[str, torch.Tensor],
        coordinates: torch.Tensor | None,
    ) -> dict[str, Any]:
        """Create serializable metadata shared by NPZ and Torch outputs."""
        return {
            "format_version": 2,
            "n_points": 0 if coordinates is None else int(coordinates.shape[0]),
            "encoders": list(embeddings),
            "coordinate_order": {
                "coordinates": "lat_lon",
                "coordinates_latlon": "lat_lon",
                "coordinates_lonlat": "lon_lat",
            },
            "encoder_metadata": {
                name: encoder.get_metadata() for name, encoder in self.encoders.items()
            },
        }


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
  python get_embeddings.py --lat 40.7128 34.0522 --lon -74.0060 -118.2437

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
        metavar="ENCODER",
        help=(
            "Registry encoder names or aliases (default: geoclip satclip). "
            f"Canonical names: {', '.join(list_encoder_names())}"
        )
    )
    model_group.add_argument(
        '--device', '-d',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to run models on (default: auto-detect)'
    )
    model_group.add_argument(
        "--encoder-root",
        action="append",
        metavar="ENCODER=PATH",
        help="Local data/checkpoint path or encoder-specific specification; repeatable",
    )
    model_group.add_argument(
        "--year",
        type=int,
        help="Optional year for temporal encoders",
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

    # Parse and validate coordinates before initializing potentially expensive models.
    if args.input and (args.lat is not None or args.lon is not None):
        parser.error("Use either --lat/--lon or --input, not both")
    if args.lat is not None or args.lon is not None:
        if args.lat is None or args.lon is None:
            parser.error("Both --lat and --lon are required")
        if len(args.lat) != len(args.lon):
            parser.error("Number of latitude and longitude values must match")
        coordinates = list(zip(args.lat, args.lon))
    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            parser.error(f"Input file not found: {args.input}")
        try:
            coordinates = read_coordinates(input_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            parser.error(f"Could not parse {input_path}: {exc}")

    else:
        parser.error("Must provide either --lat/--lon or --input")

    if not coordinates:
        parser.error("No valid coordinates found")
    try:
        validate_coordinates(coordinates)
        encoder_roots = parse_encoder_roots(args.encoder_root)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Processing {len(coordinates)} coordinate(s)...\n")

    # Initialize generator
    try:
        generator = EmbeddingGenerator(
            encoders=args.encoders,
            device=args.device,
            encoder_roots=encoder_roots,
        )
    except Exception as e:
        print(f"Error initializing generator: {e}")
        return 1

    # Generate embeddings
    try:
        embeddings = generator.generate_embeddings(coordinates, year=args.year)
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
