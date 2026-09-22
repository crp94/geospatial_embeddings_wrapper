"""Adapter for TerraNova, a foundation model for the Anthropocene.

TerraNova (Rodriguez-Pardo & Tavoni, 2026, arXiv:2607.29527) encodes location,
country, time and task with dedicated encoders and fuses them into a shared
spatiotemporal state.  The frozen backbone is published on the Hugging Face
Hub as ``crp94/terranova`` and loaded through the ``terranova`` Python package
(``pip install git+https://github.com/crp94/terranova-model``).

This repository's public contract is ``(latitude, longitude)`` input; TerraNova
natively expects ``[lon, lat]``, so the adapter flips the columns.  Two
registry entries are exposed:

* ``terranova``: the 256-d ``spatiotemporal`` space.  It depends on the query
  year, so the encoder is temporal over the documented range 1900-2035.
* ``terranova_spatial``: the 256-d ``spatial`` space, a location-only
  embedding that is independent of the year.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .embedding_encoder import GeoEmbeddingEncoder
from .location_model_encoders import _parse_data_root_spec

TERRANOVA_HF_REPO = "crp94/terranova"
TERRANOVA_YEAR_RANGE = (1900, 2035)
TERRANOVA_DEFAULT_YEAR = 2015
TERRANOVA_EMBEDDING_DIM = 256
TERRANOVA_CITATION = (
    "Rodriguez-Pardo, C. & Tavoni, M. (2026). TerraNova: A Foundation Model "
    "for the Anthropocene. arXiv:2607.29527."
)
TERRANOVA_INSTALL_HINT = (
    "The 'terranova' package is required for the TerraNova encoders. Install it with "
    "`pip install git+https://github.com/crp94/terranova-model` (or `pip install "
    "\".[models]\"` from this repository)."
)


def _import_terranova():
    try:
        from terranova import TerraNova
    except ImportError as exc:
        raise ImportError(TERRANOVA_INSTALL_HINT) from exc
    return TerraNova


class TerraNovaEncoder(GeoEmbeddingEncoder):
    """Adapter for the TerraNova ``spatiotemporal`` embedding space."""

    embedding_space = "spatiotemporal"
    display_name = "TerraNova"
    batch_size = 8192

    def __init__(self, device: str | None = None, data_root: str | None = None) -> None:
        super().__init__(device)
        spec = _parse_data_root_spec(data_root)
        self._repo = spec.get("repo") or spec.get("path") or TERRANOVA_HF_REPO
        self._revision = spec.get("revision")
        self._cache_dir = spec.get("cache")
        self._space = spec.get("space", self.embedding_space)
        self.batch_size = int(spec.get("batch_size", self.batch_size))

        TerraNova = _import_terranova()
        self.model = TerraNova.from_pretrained(
            self._repo,
            device=self.device,
            revision=self._revision,
            cache_dir=self._cache_dir,
        )

    def _check_year_range(self, years: np.ndarray) -> None:
        low, high = TERRANOVA_YEAR_RANGE
        if years.size and (years.min() < low or years.max() > high):
            raise ValueError(
                f"{self.name} supports years in the inclusive range [{low}, {high}], "
                f"received [{int(years.min())}, {int(years.max())}]"
            )

    def _resolve_year(self, year: int | None) -> int:
        if year is None:
            return TERRANOVA_DEFAULT_YEAR
        self._check_year_range(np.asarray([int(year)]))
        return int(year)

    def _embed(self, coordinates: torch.Tensor, year: int | np.ndarray) -> torch.Tensor:
        coords_lonlat = coordinates[:, [1, 0]].detach().cpu().double().numpy()
        embeddings = self.model.embed(
            coords=coords_lonlat,
            year=year,
            space=self._space,
            batch_size=self.batch_size,
        )
        return torch.as_tensor(np.asarray(embeddings), dtype=torch.float32)

    def encode(self, coordinates: torch.Tensor, year: int | None = None) -> torch.Tensor:
        return self._embed(coordinates, self._resolve_year(year))

    def encode_with_years(self, coordinates: torch.Tensor, years: Any) -> torch.Tensor:
        """One vectorised call: TerraNova accepts an ``[N]`` year vector natively."""
        year_array = self.validate_years(years, coordinates.shape[0])
        self._check_year_range(year_array)
        return self._embed(coordinates, year_array)

    def get_embedding_dim(self) -> int:
        return TERRANOVA_EMBEDDING_DIM

    def is_temporal(self) -> bool:
        return True

    def get_available_years(self) -> list[int] | None:
        low, high = TERRANOVA_YEAR_RANGE
        return list(range(low, high + 1))

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update(
            {
                "source_type": "terranova",
                "embedding_space": self._space,
                "hf_repo": self._repo,
                "hf_revision": self._revision,
                "backbone_hash": getattr(self.model, "backbone_hash", None),
                "year_range": list(TERRANOVA_YEAR_RANGE),
                "default_year": TERRANOVA_DEFAULT_YEAR,
                "native_coordinate_order": "lon_lat",
                "license": "CC-BY-4.0",
                "citation": TERRANOVA_CITATION,
            }
        )
        return metadata

    @property
    def name(self) -> str:
        return self.display_name


class TerraNovaSpatialEncoder(TerraNovaEncoder):
    """Adapter for the year-independent TerraNova ``spatial`` embedding space."""

    embedding_space = "spatial"
    display_name = "TerraNova-spatial"

    def is_temporal(self) -> bool:
        return False

    def get_available_years(self) -> list[int] | None:
        return None
