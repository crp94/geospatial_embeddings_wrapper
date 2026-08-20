"""
Base interface for geospatial embedding encoders.

This module provides an extensible architecture for generating embeddings
from geographic coordinates using different models.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import torch


class GeoEmbeddingEncoder(ABC):
    """
    Abstract base class for geographic embedding encoders.

    All encoders should inherit from this class and implement the encode method.
    Coordinates are standardized to (latitude, longitude) format for input.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the encoder.

        Args:
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    @abstractmethod
    def encode(
        self, coordinates: torch.Tensor, year: Optional[int] = None
    ) -> torch.Tensor:
        """
        Encode geographic coordinates to embeddings.

        Args:
            coordinates: Tensor of shape (N, 2) where each row is [latitude, longitude]
            year: Optional calendar year for temporal embedding products

        Returns:
            Tensor of shape (N, embedding_dim) containing the embeddings
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of the embeddings.

        Returns:
            Integer representing the embedding dimension
        """
        pass

    def is_temporal(self) -> bool:
        """Whether the encoder exposes year-specific embeddings."""
        return False

    def get_available_years(self) -> list[int] | None:
        """Return available years for temporal encoders."""
        return None

    def validate_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Return a boolean mask indicating which embedding rows are valid.

        By default, any row containing NaN/Inf values is considered invalid.
        """
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected embeddings with shape (N, D), received {tuple(embeddings.shape)}"
            )
        return torch.isfinite(embeddings).all(dim=1)

    @staticmethod
    def validate_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
        """Validate the public ``(latitude, longitude)`` coordinate contract.

        This deliberately validates without moving data between devices or changing
        precision, so callers can use it at every public boundary without altering
        model behaviour.
        """
        if not isinstance(coordinates, torch.Tensor):
            raise TypeError("coordinates must be a torch.Tensor")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError(
                "Expected coordinates with shape (N, 2) in (latitude, longitude) order, "
                f"received {tuple(coordinates.shape)}"
            )
        if not torch.isfinite(coordinates).all():
            raise ValueError("Coordinates must contain only finite values")
        latitude, longitude = coordinates[:, 0], coordinates[:, 1]
        if ((latitude < -90) | (latitude > 90)).any():
            raise ValueError("Latitude values must be in the inclusive range [-90, 90]")
        if ((longitude < -180) | (longitude > 180)).any():
            raise ValueError("Longitude values must be in the inclusive range [-180, 180]")
        return coordinates

    def supports_coverage_sampling(self) -> bool:
        """Whether the encoder can sample candidate coordinates from its own coverage."""
        return False

    def get_sampling_oversample_factor(self) -> float:
        """Return a recommended oversample factor for candidate coordinate generation."""
        return 3.5

    def sample_candidate_coordinates(
        self,
        n_points: int,
        year: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample candidate coordinates as `(longitude, latitude)`.

        Encoders with explicit coverage metadata can override this to avoid expensive
        rejection sampling from the global land mask.
        """
        raise NotImplementedError(
            f"{self.name} does not implement coverage-aware coordinate sampling"
        )

    def get_metadata(self) -> dict[str, Any]:
        """Return serializable metadata describing this encoder."""
        return {
            "name": self.name,
            "embedding_dim": self.get_embedding_dim(),
            "input_coordinate_order": self.coordinate_order,
            "is_temporal": self.is_temporal(),
            "available_years": self.get_available_years(),
        }

    def encode_from_list(self, coordinates: list) -> torch.Tensor:
        """
        Encode coordinates from a list of (lat, lon) tuples.

        Args:
            coordinates: List of (latitude, longitude) tuples

        Returns:
            Tensor of shape (N, embedding_dim) containing the embeddings
        """
        coords_tensor = self.validate_coordinates(torch.tensor(coordinates, dtype=torch.float32))
        return self.encode(coords_tensor)

    def encode_single(self, latitude: float, longitude: float) -> torch.Tensor:
        """
        Encode a single coordinate pair.

        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees

        Returns:
            Tensor of shape (1, embedding_dim) containing the embedding
        """
        coords = self.validate_coordinates(
            torch.tensor([[latitude, longitude]], dtype=torch.float32)
        )
        return self.encode(coords)

    @property
    def coordinate_order(self) -> str:
        """Get the expected coordinate order for encode inputs."""
        return "lat_lon"

    @property
    def name(self) -> str:
        """Get the name of the encoder."""
        return self.__class__.__name__
