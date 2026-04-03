"""
Base interface for geospatial embedding encoders.

This module provides an extensible architecture for generating embeddings
from geographic coordinates using different models.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
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
        coords_tensor = torch.Tensor(coordinates)
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
        coords = torch.Tensor([[latitude, longitude]])
        return self.encode(coords)

    @property
    def coordinate_order(self) -> str:
        """Get the expected coordinate order for encode inputs."""
        return "lat_lon"

    @property
    def name(self) -> str:
        """Get the name of the encoder."""
        return self.__class__.__name__
