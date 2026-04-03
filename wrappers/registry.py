"""Encoder registry shared by CLI scripts."""

from __future__ import annotations

from .geoclip_encoder import GeoCLIPEncoder
from .satclip_encoder import SatCLIPEncoder
from .torchgeo_encoders import (
    CopernicusEmbedEncoder,
    GoogleSatelliteEmbeddingEncoder,
    LGNDClayEncoder,
    TesseraEmbeddingsEncoder,
)

ENCODER_REGISTRY = {
    "geoclip": GeoCLIPEncoder,
    "satclip": SatCLIPEncoder,
    "lgnd_clay": LGNDClayEncoder,
    "copernicus_embed": CopernicusEmbedEncoder,
    "tessera": TesseraEmbeddingsEncoder,
    "google_satellite_embedding": GoogleSatelliteEmbeddingEncoder,
}

ENCODER_ALIASES = {
    "geoclip": "geoclip",
    "satclip": "satclip",
    "clay": "lgnd_clay",
    "lgnd_clay": "lgnd_clay",
    "copernicus": "copernicus_embed",
    "copernicus_embed": "copernicus_embed",
    "copernicus-embed": "copernicus_embed",
    "tessera": "tessera",
    "google_satellite": "google_satellite_embedding",
    "google_satellite_embedding": "google_satellite_embedding",
    "gse": "google_satellite_embedding",
}


def list_encoder_names() -> tuple[str, ...]:
    """Return canonical encoder names."""
    return tuple(ENCODER_REGISTRY.keys())


def normalize_encoder_name(name: str) -> str:
    """Normalize aliases to canonical encoder names."""
    try:
        return ENCODER_ALIASES[name.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(ENCODER_REGISTRY.keys()))
        raise KeyError(f"Unknown encoder '{name}'. Available encoders: {available}") from exc


def get_encoder_class(name: str):
    """Return the encoder class for a canonical or alias name."""
    return ENCODER_REGISTRY[normalize_encoder_name(name)]
