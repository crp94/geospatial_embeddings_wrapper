"""
SatCLIP encoder wrapper.

This module wraps the SatCLIP location encoder to conform to the
GeoEmbeddingEncoder interface. Handles coordinate system conversion
from (lat, lon) to (lon, lat) as required by SatCLIP.
"""

import torch
import sys
from pathlib import Path
from .embedding_encoder import GeoEmbeddingEncoder

# Add satclip directory and subdirectories to path if they exist
# satclip should be in the project root directory
SATCLIP_DIR = Path(__file__).parent.parent / "satclip"
if SATCLIP_DIR.exists():
    sys.path.insert(0, str(SATCLIP_DIR))
    # Add subdirectories for SatCLIP modules
    for subdir in SATCLIP_DIR.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('.'):
            sys.path.insert(0, str(subdir))

try:
    from huggingface_hub import hf_hub_download
    from load import get_satclip
    SATCLIP_AVAILABLE = True
except ImportError:
    SATCLIP_AVAILABLE = False


class SatCLIPEncoder(GeoEmbeddingEncoder):
    """
    Encoder using SatCLIP's location encoder.

    SatCLIP expects coordinates in (longitude, latitude) format with spherical harmonics.
    This class handles the conversion from standard (latitude, longitude) format.
    """

    def __init__(
        self,
        model_name: str = "microsoft/SatCLIP-ViT16-L40",
        checkpoint_name: str = "satclip-vit16-l40.ckpt",
        device: str = None,
        checkpoint_path: str = None
    ):
        """
        Initialize SatCLIP encoder.

        Args:
            model_name: Hugging Face model repository name
            checkpoint_name: Name of the checkpoint file
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detect)
            checkpoint_path: Optional path to a local checkpoint file (skips download)

        Raises:
            ImportError: If satclip or huggingface_hub is not installed
        """
        if not SATCLIP_AVAILABLE:
            raise ImportError(
                "SatCLIP dependencies are not available. "
                "Ensure satclip is in the project directory and "
                "install dependencies with: pip install huggingface_hub torch"
            )

        super().__init__(device)

        # Download or use provided checkpoint
        if checkpoint_path is None:
            print(f"Downloading SatCLIP model from {model_name}...")
            checkpoint_path = hf_hub_download(model_name, checkpoint_name)
            print(f"Model downloaded to: {checkpoint_path}")
        else:
            print(f"Using local checkpoint: {checkpoint_path}")

        # Load model (only loads location encoder by default)
        self.model = get_satclip(checkpoint_path, device=self.device)
        self.model.eval()

    def encode(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Encode geographic coordinates to embeddings using SatCLIP.

        Args:
            coordinates: Tensor of shape (N, 2) where each row is [latitude, longitude]

        Returns:
            Tensor of shape (N, 512) containing the embeddings

        Note:
            SatCLIP expects (longitude, latitude) format, so we flip the input coordinates.
        """
        # Convert from (lat, lon) to (lon, lat) for SatCLIP
        coords_lonlat = coordinates[:, [1, 0]]  # Swap columns
        coords_lonlat = coords_lonlat.double().to(self.device)

        with torch.no_grad():
            embeddings = self.model(coords_lonlat)

        return embeddings.detach().cpu()

    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of SatCLIP embeddings.

        Returns:
            256 (SatCLIP embedding dimension for location encoder)
        """
        return 256

    @property
    def name(self) -> str:
        """Get the name of the encoder."""
        return "SatCLIP"
