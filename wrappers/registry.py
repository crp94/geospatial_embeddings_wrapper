"""Encoder registry shared by CLI scripts."""

from __future__ import annotations

import importlib
from typing import Any

ENCODER_REGISTRY = {
    "geoclip": "wrappers.geoclip_encoder:GeoCLIPEncoder",
    "satclip": "wrappers.satclip_encoder:SatCLIPEncoder",
    "lgnd_clay": "wrappers.torchgeo_encoders:LGNDClayEncoder",
    "copernicus_embed": "wrappers.torchgeo_encoders:CopernicusEmbedEncoder",
    "tessera": "wrappers.torchgeo_encoders:TesseraEmbeddingsEncoder",
    "google_satellite_embedding": "wrappers.torchgeo_encoders:GoogleSatelliteEmbeddingEncoder",
    "range_plus": "wrappers.location_model_encoders:RANGEEncoder",
    "range": "wrappers.location_model_encoders:RANGELegacyEncoder",
    "csp": "wrappers.location_model_encoders:CSPFMoWEncoder",
    "csp_fmow": "wrappers.location_model_encoders:CSPFMoWEncoder",
    "csp_fmow_unsuper": "wrappers.location_model_encoders:CSPFMoWUnsupervisedEncoder",
    "csp_inat": "wrappers.location_model_encoders:CSPINatEncoder",
    "csp_inat_unsuper": "wrappers.location_model_encoders:CSPINatUnsupervisedEncoder",
    "gtloc": "wrappers.location_model_encoders:GTLocEncoder",
    "torchspatial_direct": "wrappers.location_model_encoders:TorchSpatialDirectEncoder",
    "torchspatial_cartesian3d": "wrappers.location_model_encoders:TorchSpatialCartesian3DEncoder",
    "torchspatial_wrap": "wrappers.location_model_encoders:TorchSpatialWrapEncoder",
    "torchspatial_grid": "wrappers.location_model_encoders:TorchSpatialGridEncoder",
    "torchspatial_theory": "wrappers.location_model_encoders:TorchSpatialTheoryEncoder",
    "torchspatial_rff": "wrappers.location_model_encoders:TorchSpatialRFFEncoder",
}

# Small, model-backed defaults for interactive point queries and examples.  The
# full registry remains available to callers that explicitly select encoders.
DEFAULT_ENCODER_NAMES = ("geoclip", "satclip")

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
    "range": "range",
    "range+": "range_plus",
    "range_plus": "range_plus",
    "rangep": "range_plus",
    "csp": "csp",
    "csp_fmow": "csp_fmow",
    "csp_fmow_unsuper": "csp_fmow_unsuper",
    "csp_inat": "csp_inat",
    "csp_inat_unsuper": "csp_inat_unsuper",
    "gt-loc": "gtloc",
    "gtloc": "gtloc",
    "torchspatial_direct": "torchspatial_direct",
    "direct": "torchspatial_direct",
    "torchspatial_cartesian3d": "torchspatial_cartesian3d",
    "cartesian3d": "torchspatial_cartesian3d",
    "cartesian_3d": "torchspatial_cartesian3d",
    "torchspatial_wrap": "torchspatial_wrap",
    "wrap": "torchspatial_wrap",
    "torchspatial_grid": "torchspatial_grid",
    "grid": "torchspatial_grid",
    "space2vec_grid": "torchspatial_grid",
    "torchspatial_theory": "torchspatial_theory",
    "theory": "torchspatial_theory",
    "torchspatial_rff": "torchspatial_rff",
    "rff": "torchspatial_rff",
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


def _load_object(path: str) -> Any:
    module_name, object_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def get_encoder_class(name: str):
    """Return the encoder class for a canonical or alias name."""
    return _load_object(ENCODER_REGISTRY[normalize_encoder_name(name)])
