"""
GeoCLIP encoder wrapper.

This module wraps the GeoCLIP LocationEncoder to conform to the
GeoEmbeddingEncoder interface.
"""

import torch
from .embedding_encoder import GeoEmbeddingEncoder

try:
    from geoclip import LocationEncoder
    GEOCLIP_AVAILABLE = True
except ImportError:
    GEOCLIP_AVAILABLE = False


class GeoCLIPEncoder(GeoEmbeddingEncoder):
    """
    Encoder using GeoCLIP's LocationEncoder.

    GeoCLIP expects coordinates in (latitude, longitude) format.
    Uses Equal Earth projection and Gaussian Random Fourier Features.
    """

    def __init__(self, device: str = None):
        """
        Initialize GeoCLIP encoder.

        Args:
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detect)

        Raises:
            ImportError: If geoclip is not installed
        """
        if not GEOCLIP_AVAILABLE:
            raise ImportError(
                "GeoCLIP is not installed. Install it with: pip install geoclip"
            )

        super().__init__(device)
        self.encoder = LocationEncoder()
        self.encoder.to(self.device)
        self.encoder.eval()

    def encode(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Encode geographic coordinates to embeddings using GeoCLIP.

        Args:
            coordinates: Tensor of shape (N, 2) where each row is [latitude, longitude]

        Returns:
            Tensor of shape (N, 512) containing the embeddings
        """
        # GeoCLIP expects (lat, lon) format, which matches our standard
        coords = coordinates.to(self.device)

        with torch.no_grad():
            embeddings = self.encoder(coords)

        return embeddings.cpu()

    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of GeoCLIP embeddings.

        Returns:
            512 (GeoCLIP embedding dimension)
        """
        return 512

    @property
    def name(self) -> str:
        """Get the name of the encoder."""
        return "GeoCLIP"
