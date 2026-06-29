"""
Geospatial embedding encoder wrappers.

This package contains wrapper classes for various geospatial embedding models,
providing a unified interface for generating location embeddings.
"""

from .embedding_encoder import GeoEmbeddingEncoder
from .geoclip_encoder import GeoCLIPEncoder
from .satclip_encoder import SatCLIPEncoder

__all__ = [
    'GeoEmbeddingEncoder',
    'GeoCLIPEncoder',
    'SatCLIPEncoder',
    'LGNDClayEncoder',
    'CopernicusEmbedEncoder',
    'TesseraEmbeddingsEncoder',
    'GoogleSatelliteEmbeddingEncoder',
]


def __getattr__(name):
    """Lazily import optional raster-backed encoders and their heavy deps."""
    if name in {
        'LGNDClayEncoder',
        'CopernicusEmbedEncoder',
        'TesseraEmbeddingsEncoder',
        'GoogleSatelliteEmbeddingEncoder',
    }:
        from . import torchgeo_encoders

        return getattr(torchgeo_encoders, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
