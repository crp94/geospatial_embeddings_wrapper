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
]
