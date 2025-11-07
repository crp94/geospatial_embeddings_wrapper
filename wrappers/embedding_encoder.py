"""
Base interface for geospatial embedding encoders.

This module provides an extensible architecture for generating embeddings
from geographic coordinates using different models.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch
import numpy as np


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
    def encode(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Encode geographic coordinates to embeddings.

        Args:
            coordinates: Tensor of shape (N, 2) where each row is [latitude, longitude]

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
    def name(self) -> str:
        """Get the name of the encoder."""
        return self.__class__.__name__
