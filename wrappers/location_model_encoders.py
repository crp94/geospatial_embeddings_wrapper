"""Additional coordinate-only location embedding encoders.

This module contains two groups of adapters:

* optional wrappers around external open-weight projects (RANGE, CSP, GT-Loc)
* lightweight TorchSpatial-style baseline encoders with deterministic features

All public ``encode`` methods follow this repository's convention: input
coordinates are tensors of shape ``(N, 2)`` in ``(latitude, longitude)`` order.
Adapters flip to ``(longitude, latitude)`` only when the upstream model expects
that native convention.
"""

from __future__ import annotations

import importlib
import math
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .embedding_encoder import GeoEmbeddingEncoder


EXTERNAL_CACHE_ROOT = Path("./data_cache/external")
CSP_MODEL_URL = (
    "https://www.dropbox.com/scl/fi/rdig1ezmywm9avc8qubi6/model_dir.zip"
    "?rlkey=p8k16a5ifi69e08rnvu8rvt86&dl=1"
)
CSP_CHECKPOINTS = {
    "fmow": "model_dir/model_fmow/model_fmow_gridcell_0.0010_32_0.1000000_1_512_gelu_contsoftmax_ratio0.050_0.000050_1.000_1_0.100_TMP1.0000_1.0000_1.0000.pth.tar",
    "fmow_unsuper": "model_dir/model_fmow/model_fmow_gridcell_0.0010_32_0.1000000_1_512_gelu_UNSUPER-contsoftmax_0.000050_1.000_1_0.100_TMP1.0000_1.0000_1.0000.pth.tar",
    "inat": "model_dir/model_inat_2018/model_inat_2018_gridcell_0.0010_32_0.1000000_1_512_leakyrelu_contsoftmax_ratio0.050_0.000500_1.000_1_1.000_TMP20.0000_1.0000_1.0000.pth.tar",
    "inat_unsuper": "model_dir/model_inat_2018/model_inat_2018_gridcell_0.0010_32_0.1000000_1_512_leakyrelu_UNSUPER-contsoftmax_0.000500_1.000_1_1.000_TMP20.0000_1.0000_1.0000.pth.tar",
}


def _parse_data_root_spec(data_root: str | None) -> dict[str, str]:
    """Parse a plain path or ``key=value;key=value`` data_root spec."""
    if data_root is None:
        return {}
    if "=" not in data_root:
        return {"path": data_root}

    parsed: dict[str, str] = {}
    for part in data_root.split(";"):
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Invalid data_root spec segment '{part}'. Expected key=value."
            )
        key, value = part.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _ensure_repo(repo_url: str, cache_name: str, data_root: str | None = None) -> Path:
    """Return a local repository path, cloning atomically into the cache when needed."""
    if data_root:
        repo_path = Path(data_root).expanduser()
        if not repo_path.exists():
            raise RuntimeError(f"External repository path does not exist: {repo_path}")
        return repo_path

    repo_path = EXTERNAL_CACHE_ROOT / cache_name
    if repo_path.exists():
        return repo_path

    EXTERNAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        # Cloning into a sibling staging directory avoids leaving a partial
        # repository behind if the network fails or the process is interrupted.
        with tempfile.TemporaryDirectory(
            prefix=f".{cache_name}.", dir=EXTERNAL_CACHE_ROOT
        ) as staging_root:
            staged_repo = Path(staging_root) / cache_name
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(staged_repo)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                staged_repo.replace(repo_path)
            except FileExistsError:
                # A concurrent process populated the cache while this process
                # was cloning. Its completed cache is safe to reuse.
                if not repo_path.exists():
                    raise
    except Exception as exc:
        raise RuntimeError(
            f"Could not clone {repo_url} into {repo_path}. "
            "Pass a local repo with --encoder_root "
            f"{cache_name}=repo=/path/to/repo or install it on PYTHONPATH."
        ) from exc
    return repo_path


def _download_url_atomic(
    url: str,
    destination: Path,
    *,
    validator: Any | None = None,
    attempts: int = 3,
) -> None:
    """Download a URL to a validated temporary file then atomically promote it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for _attempt in range(1, attempts + 1):
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.name}.",
                suffix=".part",
                dir=destination.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            urllib.request.urlretrieve(url, temporary_path)
            if validator is not None:
                validator(temporary_path)
            temporary_path.replace(destination)
            return
        except Exception as exc:
            last_error = exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    raise RuntimeError(
        f"Could not download a valid file from {url} after {attempts} attempts."
    ) from last_error


def _validate_zip(path: Path) -> None:
    """Reject malformed or truncated ZIP archives before they enter the cache."""
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise zipfile.BadZipFile(f"CRC check failed for ZIP member: {bad_member}")


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _download_hf_file(
    repo_id: str,
    filename: str,
    repo_type: str = "model",
    local_dir: str | Path | None = None,
) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for automatic model downloads. "
            "Install requirements or pass an explicit checkpoint/db path."
        ) from exc

    return hf_hub_download(
        repo_id,
        filename,
        repo_type=repo_type,
        local_dir=str(local_dir) if local_dir is not None else None,
        local_dir_use_symlinks=False,
    )


class RANGEEncoder(GeoEmbeddingEncoder):
    """Adapter for MVRL RANGE / RANGE+ coordinate embeddings."""

    model_name = "RANGE+"
    batch_size = 1000
    repo_url = "https://github.com/mvrl/RANGE.git"

    def __init__(
        self,
        device: str | None = None,
        data_root: str | None = None,
        beta: float = 0.5,
    ) -> None:
        super().__init__(device)
        spec = _parse_data_root_spec(data_root)
        repo_path = _ensure_repo(
            self.repo_url,
            "RANGE",
            spec.get("repo") or spec.get("path") or os.environ.get("RANGE_REPO"),
        )
        _prepend_sys_path(repo_path)

        cache_dir = Path(spec.get("cache", "./data_cache/range"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = (
            spec.get("checkpoint")
            or spec.get("pretrained_path")
            or os.environ.get("RANGE_CHECKPOINT")
            or _download_hf_file(
                "microsoft/SatCLIP-ViT16-L40",
                "satclip-vit16-l40.ckpt",
                repo_type="model",
                local_dir=cache_dir,
            )
        )
        db_path = (
            spec.get("db")
            or spec.get("db_path")
            or os.environ.get("RANGE_DB")
            or _download_hf_file(
                "mvrl/RANGE-database",
                spec.get("db_name", "range_db_large.npz"),
                repo_type="dataset",
                local_dir=cache_dir,
            )
        )
        beta = float(spec.get("beta", beta))
        self.batch_size = int(spec.get("batch_size", self.batch_size))

        load_model = importlib.import_module("range.load_model").load_model
        self.model = load_model(
            model_name=self.model_name,
            pretrained_path=checkpoint_path,
            device=self.device,
            db_path=db_path,
            beta=beta,
        )
        self.model.eval()
        self._embedding_dim = int(getattr(self.model, "location_feature_dim", 1280))
        self._repo_path = str(repo_path)
        self._checkpoint_path = str(checkpoint_path)
        self._db_path = str(db_path)
        self._beta = beta

    def encode(
        self, coordinates: torch.Tensor, year: int | None = None
    ) -> torch.Tensor:
        coords_lonlat = coordinates[:, [1, 0]].double().to(self.device)
        with torch.no_grad():
            embeddings = self.model(coords_lonlat)
        return embeddings.detach().cpu().float()

    def get_embedding_dim(self) -> int:
        return self._embedding_dim

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update(
            {
                "source_type": "external_range",
                "external_repo": self.repo_url,
                "external_repo_path": self._repo_path,
                "checkpoint_path": self._checkpoint_path,
                "range_db_path": self._db_path,
                "native_coordinate_order": "lon_lat",
                "beta": self._beta,
            }
        )
        return metadata

    @property
    def name(self) -> str:
        return self.model_name


class RANGELegacyEncoder(RANGEEncoder):
    """Adapter for the original RANGE variant."""

    model_name = "RANGE"


class CSPEncoder(GeoEmbeddingEncoder):
    """Adapter for CSP pretrained location encoders via RANGE's CSP loader."""

    batch_size = 50000
    repo_url = "https://github.com/mvrl/RANGE.git"
    default_variant = "fmow"

    def _ensure_checkpoint(self, spec: dict[str, str]) -> tuple[str, str]:
        checkpoint_path = (
            spec.get("checkpoint")
            or spec.get("pretrained_path")
            or os.environ.get("CSP_CHECKPOINT")
        )
        if checkpoint_path is not None:
            return checkpoint_path, spec.get("variant", "custom")

        variant = spec.get("variant") or os.environ.get("CSP_VARIANT") or self.default_variant
        if variant not in CSP_CHECKPOINTS:
            known = ", ".join(sorted(CSP_CHECKPOINTS))
            raise RuntimeError(f"Unknown CSP variant '{variant}'. Known variants: {known}")

        cache_dir = Path(spec.get("cache", "./data_cache/csp")).expanduser()
        checkpoint = cache_dir / CSP_CHECKPOINTS[variant]
        if checkpoint.exists():
            return str(checkpoint), variant

        cache_dir.mkdir(parents=True, exist_ok=True)
        archive_path = cache_dir / "model_dir.zip"
        if archive_path.exists():
            try:
                _validate_zip(archive_path)
            except Exception:
                archive_path.unlink(missing_ok=True)

        if not archive_path.exists():
            try:
                _download_url_atomic(
                    CSP_MODEL_URL,
                    archive_path,
                    validator=_validate_zip,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Could not download a valid CSP pretrained checkpoint archive. "
                    "Pass a local checkpoint with --encoder_root "
                    "csp=checkpoint=/path/to/model.pth.tar."
                ) from exc

        try:
            with zipfile.ZipFile(archive_path) as archive:
                # Extract only the requested known member into a temporary file.
                # This avoids ZIP path traversal and guarantees an interrupted
                # extraction cannot leave a partial checkpoint in the cache.
                member = archive.getinfo(CSP_CHECKPOINTS[variant])
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{checkpoint.name}.",
                    suffix=".part",
                    dir=checkpoint.parent,
                    delete=False,
                ) as temporary_file:
                    temporary_checkpoint = Path(temporary_file.name)
                    try:
                        shutil.copyfileobj(source, temporary_file)
                        temporary_file.flush()
                        os.fsync(temporary_file.fileno())
                    except Exception:
                        temporary_checkpoint.unlink(missing_ok=True)
                        raise
                temporary_checkpoint.replace(checkpoint)
        except Exception as exc:
            raise RuntimeError(
                f"Could not extract CSP checkpoint '{CSP_CHECKPOINTS[variant]}' "
                f"from archive: {archive_path}"
            ) from exc

        if not checkpoint.exists():
            raise RuntimeError(
                f"CSP checkpoint variant '{variant}' was not found after extraction: {checkpoint}"
            )
        return str(checkpoint), variant

    def __init__(self, device: str | None = None, data_root: str | None = None) -> None:
        super().__init__(device)
        spec = _parse_data_root_spec(data_root)
        repo_path = _ensure_repo(
            self.repo_url,
            "RANGE",
            spec.get("repo") or spec.get("path") or os.environ.get("RANGE_REPO"),
        )
        checkpoint_path, variant = self._ensure_checkpoint(spec)

        _prepend_sys_path(repo_path)
        get_csp = importlib.import_module(
            "range.location_models.csp.load_csp"
        ).get_csp
        self.model = get_csp(path=checkpoint_path).to(self.device)
        self.model.eval()
        self._embedding_dim = int(getattr(self.model, "loc_emb_dim", 256))
        self._repo_path = str(repo_path)
        self._checkpoint_path = str(checkpoint_path)
        self._variant = variant

    def encode(
        self, coordinates: torch.Tensor, year: int | None = None
    ) -> torch.Tensor:
        coords_lonlat = coordinates[:, [1, 0]].float().to(self.device)
        with torch.no_grad():
            embeddings = self.model(coords_lonlat, return_feats=True)
        return embeddings.detach().cpu().float()

    def get_embedding_dim(self) -> int:
        return self._embedding_dim

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update(
            {
                "source_type": "external_csp",
                "external_repo": self.repo_url,
                "external_repo_path": self._repo_path,
                "checkpoint_path": self._checkpoint_path,
                "variant": self._variant,
                "native_coordinate_order": "lon_lat",
            }
        )
        return metadata

    @property
    def name(self) -> str:
        return "CSP"


class CSPFMoWEncoder(CSPEncoder):
    default_variant = "fmow"


class CSPFMoWUnsupervisedEncoder(CSPEncoder):
    default_variant = "fmow_unsuper"


class CSPINatEncoder(CSPEncoder):
    default_variant = "inat"


class CSPINatUnsupervisedEncoder(CSPEncoder):
    default_variant = "inat_unsuper"


class GTLocEncoder(GeoEmbeddingEncoder):
    """Adapter for the GT-Loc GPS branch."""

    batch_size = 50000
    repo_url = "https://github.com/dshatwell23/gtloc.git"

    def __init__(self, device: str | None = None, data_root: str | None = None) -> None:
        super().__init__(device)
        spec = _parse_data_root_spec(data_root)
        repo_path = _ensure_repo(
            self.repo_url,
            "gtloc",
            spec.get("repo") or spec.get("path") or os.environ.get("GTLOC_REPO"),
        )
        checkpoint_path = (
            spec.get("checkpoint")
            or spec.get("pretrained_path")
            or os.environ.get("GTLOC_CHECKPOINT")
        )
        if checkpoint_path is None:
            raise RuntimeError(
                "GTLoc requires the pretrained checkpoint. Download it with the "
                "upstream gdown command and pass "
                "--encoder_root gtloc=repo=/path/to/gtloc;checkpoint=/path/to/gtloc.pt "
                "or set GTLOC_CHECKPOINT."
        )

        _prepend_sys_path(repo_path / "src")
        existing_model_module = sys.modules.get("model")
        if existing_model_module is not None and not hasattr(existing_model_module, "__path__"):
            del sys.modules["model"]
        location_module = importlib.import_module("model.location_encoder")
        embedding_dim = int(spec.get("embedding_dim", 512))
        sigma = [
            float(value)
            for value in spec.get("loc_sigma", "1,16,256").split(",")
            if value
        ]
        self.model = location_module.LocationEncoder(
            sigma=sigma,
            embedding_dim=embedding_dim,
        ).to(self.device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()
        self._embedding_dim = embedding_dim
        self._repo_path = str(repo_path)
        self._checkpoint_path = str(checkpoint_path)
        self._sigma = sigma

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            state = (
                checkpoint.get("state_dict")
                or checkpoint.get("model_state_dict")
                or checkpoint.get("model")
                or checkpoint
            )
        else:
            state = checkpoint

        prefixes = (
            "location_encoder.",
            "module.location_encoder.",
            "model.location_encoder.",
        )
        location_state = {}
        for key, value in state.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    location_state[key[len(prefix) :]] = value
                    break
            else:
                if key.startswith("LocEnc") or key.startswith("capsule") or key.startswith("head"):
                    location_state[key] = value

        if not location_state:
            raise RuntimeError(
                "Could not find GTLoc location_encoder weights in checkpoint."
            )
        missing, unexpected = self.model.load_state_dict(location_state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected GTLoc location encoder keys: {unexpected}")
        if len(missing) == len(self.model.state_dict()):
            raise RuntimeError("GTLoc checkpoint did not load any location weights.")

    def encode(
        self, coordinates: torch.Tensor, year: int | None = None
    ) -> torch.Tensor:
        coords_latlon = coordinates.float().to(self.device)
        with torch.no_grad():
            embeddings = self.model(coords_latlon)
        return embeddings.detach().cpu().float()

    def get_embedding_dim(self) -> int:
        return self._embedding_dim

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update(
            {
                "source_type": "external_gtloc",
                "external_repo": self.repo_url,
                "external_repo_path": self._repo_path,
                "checkpoint_path": self._checkpoint_path,
                "native_coordinate_order": "lat_lon",
                "loc_sigma": self._sigma,
            }
        )
        return metadata

    @property
    def name(self) -> str:
        return "GTLoc"


class TorchSpatialBaselineEncoder(GeoEmbeddingEncoder):
    """Deterministic TorchSpatial-style coordinate feature baselines."""

    baseline_name = "torchspatial_grid"
    frequency_num = 16
    min_radius = 1.0
    max_radius = 10000.0
    batch_size = 250000

    def __init__(self, device: str | None = None, data_root: str | None = None) -> None:
        super().__init__(device)
        spec = _parse_data_root_spec(data_root)
        if "frequency_num" in spec:
            self.frequency_num = int(spec["frequency_num"])
        if "min_radius" in spec:
            self.min_radius = float(spec["min_radius"])
        if "max_radius" in spec:
            self.max_radius = float(spec["max_radius"])
        self._freqs = self._build_frequency_tensor()

    def _build_frequency_tensor(self) -> torch.Tensor:
        if self.frequency_num <= 1:
            values = np.array([1.0 / self.min_radius], dtype=np.float32)
        else:
            values = 1.0 / np.geomspace(
                self.min_radius,
                self.max_radius,
                num=self.frequency_num,
                dtype=np.float64,
            )
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _lonlat(self, coordinates: torch.Tensor) -> torch.Tensor:
        return coordinates[:, [1, 0]].float().to(self.device)

    def encode(
        self, coordinates: torch.Tensor, year: int | None = None
    ) -> torch.Tensor:
        lonlat = self._lonlat(coordinates)
        if self.baseline_name == "torchspatial_direct":
            embeddings = torch.deg2rad(lonlat)
        elif self.baseline_name == "torchspatial_cartesian3d":
            radians = torch.deg2rad(lonlat)
            lon = radians[:, 0:1]
            lat = radians[:, 1:2]
            embeddings = torch.cat(
                [torch.cos(lon) * torch.cos(lat), torch.sin(lon) * torch.cos(lat), torch.sin(lat)],
                dim=1,
            )
        elif self.baseline_name == "torchspatial_wrap":
            radians = torch.deg2rad(lonlat)
            lon = radians[:, 0:1]
            lat = radians[:, 1:2]
            embeddings = torch.cat(
                [torch.cos(lon), torch.sin(lon), torch.cos(lat), torch.sin(lat)],
                dim=1,
            )
        elif self.baseline_name == "torchspatial_theory":
            embeddings = self._encode_theory(lonlat)
        elif self.baseline_name == "torchspatial_rff":
            embeddings = self._encode_fourier(lonlat / torch.tensor([180.0, 90.0], device=self.device))
        else:
            embeddings = self._encode_fourier(lonlat)
        return embeddings.detach().cpu().float()

    def _encode_fourier(self, coords: torch.Tensor) -> torch.Tensor:
        angles = coords[:, :, None] * self._freqs[None, None, :]
        sin_cos = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)
        return sin_cos.reshape(coords.shape[0], -1)

    def _encode_theory(self, lonlat: torch.Tensor) -> torch.Tensor:
        unit_vectors = torch.tensor(
            [[1.0, 0.0], [-0.5, math.sqrt(3.0) / 2.0], [-0.5, -math.sqrt(3.0) / 2.0]],
            dtype=lonlat.dtype,
            device=lonlat.device,
        )
        projections = lonlat @ unit_vectors.T
        angles = projections[:, :, None] * self._freqs[None, None, :]
        sin_cos = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)
        return sin_cos.reshape(lonlat.shape[0], -1)

    def get_embedding_dim(self) -> int:
        if self.baseline_name == "torchspatial_direct":
            return 2
        if self.baseline_name == "torchspatial_cartesian3d":
            return 3
        if self.baseline_name == "torchspatial_wrap":
            return 4
        if self.baseline_name == "torchspatial_theory":
            return 3 * self.frequency_num * 2
        return 2 * self.frequency_num * 2

    def get_metadata(self) -> dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update(
            {
                "source_type": "torchspatial_baseline",
                "baseline_name": self.baseline_name,
                "native_coordinate_order": "lon_lat",
                "frequency_num": self.frequency_num,
                "min_radius": self.min_radius,
                "max_radius": self.max_radius,
            }
        )
        return metadata

    @property
    def name(self) -> str:
        return self.baseline_name


class TorchSpatialDirectEncoder(TorchSpatialBaselineEncoder):
    baseline_name = "torchspatial_direct"


class TorchSpatialCartesian3DEncoder(TorchSpatialBaselineEncoder):
    baseline_name = "torchspatial_cartesian3d"


class TorchSpatialWrapEncoder(TorchSpatialBaselineEncoder):
    baseline_name = "torchspatial_wrap"


class TorchSpatialGridEncoder(TorchSpatialBaselineEncoder):
    baseline_name = "torchspatial_grid"


class TorchSpatialTheoryEncoder(TorchSpatialBaselineEncoder):
    baseline_name = "torchspatial_theory"


class TorchSpatialRFFEncoder(TorchSpatialBaselineEncoder):
    baseline_name = "torchspatial_rff"
