# Geospatial Embeddings Wrapper

<p align="center">
  <img src="images/image.png" alt="Geospatial Embeddings Illustration" width="600">
</p>

Unified tooling for querying geospatial encoders at coordinates and building
reproducible land-only embedding datasets with explicit coordinate and provenance
metadata.

## Overview

This repository has two main entry points:

- `scripts/get_embeddings.py`: point-query CLI for every registered encoder
- `scripts/generate_dataset.py`: reproducible land-only dataset generator for
  coordinate models, raster products, and deterministic baselines

The dataset generator currently supports:

- `geoclip`
- `satclip`
- `lgnd_clay`
- `copernicus_embed`
- `tessera`
- `google_satellite_embedding`
- `range`
- `range_plus`
- `csp`
- `csp_fmow`
- `csp_fmow_unsuper`
- `csp_inat`
- `csp_inat_unsuper`
- `gtloc`
- `torchspatial_direct`
- `torchspatial_cartesian3d`
- `torchspatial_wrap`
- `torchspatial_grid`
- `torchspatial_theory`
- `torchspatial_rff`

All dataset outputs use the same coordinate conventions and named embedding
fields, even though the underlying models and products use different native
coordinate orders and storage layouts.

## What The Generator Does

`scripts/generate_dataset.py`:

- samples candidate coordinates with Fibonacci sphere sampling
- uses a seeded Fibonacci sampler, or reusable coordinates supplied by the user
- filters to land using a selected Natural Earth polygon resolution
- queries one or more encoders
- drops invalid rows for the selected encoder set
- writes `.pt`, `.csv`, or chunked `.zarr` datasets
- writes location plots and ICA-to-RGB embedding plots

For temporal products, the generator can emit one output per year.

Requested encoders are initialized fail-fast: if any one cannot be created, the
run stops before writing a partial dataset.

## Coordinate Conventions

Inside this repo, the standard coordinate input format is always:

- `(latitude, longitude)`

Saved dataset files include both explicit coordinate layouts:

- `coordinates` and `coordinates_latlon`: `(lat, lon)`
- `coordinates_lonlat`: `(lon, lat)`
- separate `latitude` and `longitude` tensors/vectors

For `.pt` outputs, the metadata block records these conventions explicitly.
The point-query CLI uses the same layouts in both `.pt` and `.npz` outputs.

## Installation

### Prerequisites

- Python 3.10+
- `pip`
- optional CUDA GPU for faster `geoclip` / `satclip`

### Setup

```bash
git clone https://github.com/crp94/geospatial_embeddings_wrapper.git
cd geospatial_embeddings_wrapper

python -m venv .venv
source .venv/bin/activate

# Full installation, including raster products, model-backed encoders, and plots.
pip install ".[all]"

# The original requirements-file workflow remains supported.
# pip install -r requirements.txt
```

`satclip/` is vendored in this repository already. You do not need to clone it separately.

The base package installs the shared API and the Parquet engine needed for the
Google Satellite Embedding index. Install only the feature groups you need with
`pip install ".[raster]"`, `pip install ".[models]"`, or
`pip install ".[visualization]"`. Zarr output is included in the full install
and requirements file; install `pip install ".[storage]"` when starting from
the minimal package. The installed command-line entry points are
`geospatial-embeddings` and `geospatial-embeddings-dataset`.

## Supported Encoders

### Point-query CLI

`scripts/get_embeddings.py` accepts every canonical name and alias in
`wrappers/registry.py`. Its lightweight defaults remain `geoclip satclip`.
Use `--encoder-root ENCODER=PATH` repeatedly for local data or checkpoint
specifications, and pass `--year` when querying a temporal encoder.

Example:

```bash
python scripts/get_embeddings.py \
  --lat 40.7128 34.0522 \
  --lon -74.0060 -118.2437 \
  --encoders geoclip satclip \
  --output embeddings.npz
```

Coordinates are always `(latitude, longitude)`, must be finite, and must be in
the inclusive ranges `[-90, 90]` and `[-180, 180]`. Supply either paired
`--lat`/`--lon` values or `--input`, never both. JSON input accepts coordinate
pairs or `lat`/`lon` objects; CSV and text input accept the first two numeric
columns or columns headed `lat`/`lon` (or `latitude`/`longitude`).

### Dataset Generator

`scripts/generate_dataset.py` supports:

- `geoclip`
- `satclip`
- `lgnd_clay`
- `copernicus_embed`
- `tessera`
- `google_satellite_embedding`
- `range`
- `range_plus`
- `csp`
- `csp_fmow`
- `csp_fmow_unsuper`
- `csp_inat`
- `csp_inat_unsuper`
- `gtloc`
- `torchspatial_direct`
- `torchspatial_cartesian3d`
- `torchspatial_wrap`
- `torchspatial_grid`
- `torchspatial_theory`
- `torchspatial_rff`

Notes:

- `geoclip` and `satclip` are model-backed encoders
- `lgnd_clay`, `copernicus_embed`, `tessera`, and
  `google_satellite_embedding` are raster-backed products
- `copernicus_embed` is TorchGeo-backed and auto-downloads its raster
- `tessera` can run through `geotessera` without a local root
- `google_satellite_embedding` can run against the public AEF annual index without a local root
- `range` and `range_plus` wrap the open-weight RANGE models
- `csp` wraps the open-weight Contrastive Spatial Pre-Training location encoder
- `csp_fmow`, `csp_fmow_unsuper`, `csp_inat`, and `csp_inat_unsuper` select the other published CSP variants
- `gtloc` wraps the open-weight GT-Loc GPS branch
- `torchspatial_*` encoders are deterministic coordinate-feature baselines and do not require pretrained weights
- land-only behavior is enforced by this generator layer, not by every source dataset

### Checkpoints For New Models

The new coordinate-only wrappers do not vendor pretrained weights in this repository. Users should download upstream checkpoints and point the generator at them when needed:

- `range` / `range_plus`: pass a local SatCLIP checkpoint and RANGE database with `checkpoint=...;db=...` if the Hugging Face auto-download is not available or if you want a specific local copy
- `csp*`: pass a local CSP `.pth.tar` checkpoint with `checkpoint=...` if the Dropbox auto-download is not available or if you want a specific CSP variant/checkpoint
- `gtloc`: always requires a local GT-Loc checkpoint via `checkpoint=...`
- `torchspatial_*`: no checkpoint is required

Use `--encoder_root` for these paths. It accepts either `encoder=/plain/path` or a semicolon-separated spec such as `encoder=repo=/path/to/repo;checkpoint=/path/to/model.pt`.

## Dataset Generation Examples

Generate a static 100k land-only dataset with the two model-backed encoders:

```bash
python scripts/generate_dataset.py \
  --n_points 100000 \
  --encoders geoclip satclip \
  --device cuda \
  --output_path outputs/example_static
```

Make a sampled coordinate set reproducible and reuse it across separate encoder
runs. `--coordinates_in` uses the supplied points exactly: rows outside an
encoder's coverage cause a clear error rather than being silently replaced.

```bash
# Sample once and retain the exact selected land points.
python scripts/generate_dataset.py \
  --n_points 100000 \
  --encoders geoclip \
  --seed 20260820 \
  --coordinates_out outputs/shared_coordinates.npz \
  --output_path outputs/geoclip

# Query another encoder at precisely those locations.
python scripts/generate_dataset.py \
  --encoders satclip \
  --coordinates_in outputs/shared_coordinates.npz \
  --output_path outputs/satclip
```

For a large multi-encoder run, Zarr streams batches into chunked arrays rather
than retaining the complete embedding matrix in memory:

```bash
python scripts/generate_dataset.py \
  --n_points 500000 \
  --encoders geoclip satclip \
  --seed 20260820 \
  --output_format zarr \
  --no_plot \
  --output_path outputs/land_embeddings
```

Generate a 2024 land-only dataset for the 3 temporal/raster products:

```bash
python scripts/generate_dataset.py \
  --n_points 100000 \
  --encoders copernicus_embed tessera google_satellite_embedding \
  --years 2024 \
  --output_path outputs/example_2024
```

Generate the full 5-product set as separate runs:

```bash
python scripts/generate_dataset.py --n_points 500000 --encoders geoclip --device cuda --output_path outputs/land_only_500k/geoclip_land_500k
python scripts/generate_dataset.py --n_points 500000 --encoders satclip --device cuda --output_path outputs/land_only_500k/satclip_land_500k
python scripts/generate_dataset.py --n_points 500000 --encoders copernicus_embed --output_path outputs/land_only_500k/copernicus_land_500k
python scripts/generate_dataset.py --n_points 500000 --encoders tessera --years 2024 --output_path outputs/land_only_500k/tessera_land_500k
python scripts/generate_dataset.py --n_points 500000 --encoders google_satellite_embedding --years 2024 --output_path outputs/land_only_500k/google_satellite_embedding_land_500k
```

Generate CSV output without plots:

```bash
python scripts/generate_dataset.py \
  --n_points 50000 \
  --encoders geoclip \
  --output_format csv \
  --no_plot \
  --output_path outputs/geoclip_csv
```

CSV expands every embedding dimension into a separate column, so reserve it for
small, interoperability-focused exports. Prefer `.pt` for ordinary PyTorch
workflows and `.zarr` for large datasets or incremental reads.

Generate the additional coordinate-only encoders:

```bash
python scripts/generate_dataset.py \
  --n_points 500000 \
  --encoders range_plus \
  --encoder_root 'range_plus=checkpoint=/path/to/satclip.ckpt;db=/path/to/range_db_large.npz' \
  --device cuda \
  --output_path outputs/land_only_500k/range_plus_land_500k

python scripts/generate_dataset.py \
  --n_points 500000 \
  --encoders csp \
  --encoder_root 'csp=checkpoint=/path/to/csp_model.pth.tar' \
  --device cuda \
  --output_path outputs/land_only_500k/csp_land_500k

python scripts/generate_dataset.py \
  --n_points 500000 \
  --encoders csp_inat \
  --encoder_root 'csp_inat=checkpoint=/path/to/csp_inat_model.pth.tar' \
  --device cuda \
  --output_path outputs/land_only_500k/csp_inat_land_500k

python scripts/generate_dataset.py \
  --n_points 500000 \
  --encoders gtloc \
  --encoder_root 'gtloc=repo=/path/to/gtloc;checkpoint=/path/to/gtloc.pt' \
  --device cuda \
  --output_path outputs/land_only_500k/gtloc_land_500k

python scripts/generate_dataset.py \
  --n_points 500000 \
  --encoders torchspatial_direct torchspatial_cartesian3d torchspatial_wrap \
             torchspatial_grid torchspatial_theory torchspatial_rff \
  --output_path outputs/land_only_500k/torchspatial_baselines_land_500k
```

## Outputs and Provenance

### `.pt` output

Each saved `.pt` dataset contains:

- `metadata`
- `latitude`
- `longitude`
- `coordinates`
- `coordinates_latlon`
- `coordinates_lonlat`
- one `*_embeddings` tensor per encoder

Example keys:

```python
[
    "metadata",
    "latitude",
    "longitude",
    "coordinates",
    "coordinates_latlon",
    "coordinates_lonlat",
    "geoclip_embeddings",
]
```

### `metadata`

The metadata block includes:

- selected encoders
- year
- number of points
- coordinate order declarations
- encoder-specific metadata such as embedding dimension and available years
- sampling seed and maximum sampling-attempt limit
- land-mask source, resolution, and Antarctica policy

The same metadata is saved as `metadata_json` in point-query `.npz` output and
as the `metadata` group attribute in Zarr output.

### `.zarr` output

Zarr stores the same named arrays as `.pt` (`latitude`, `longitude`, all three
coordinate layouts, and `<encoder>_embeddings`) in independently chunked,
float32 arrays. It is the preferred format for hundreds of thousands of points
or high-dimensional/multi-encoder datasets because data can be read in chunks.
Plots are skipped for streamed Zarr sampling; create visualizations from a
deterministic subset in a separate analysis step.

### Point-query output compatibility

Point queries retain their legacy top-level encoder keys/arrays. Both `.pt` and
`.npz` now also retain the submitted coordinates under `coordinates`,
`coordinates_latlon`, `coordinates_lonlat`, `latitude`, and `longitude`, so
outputs can be joined safely with generated datasets.

### Plots

When plotting is enabled, the generator writes:

- `*_locations.png`: sampled land coordinates
- `*_<encoder>_ica.png`: embeddings projected to RGB with ICA

The ICA fit is done on a capped subsample and transformed in batches, so large outputs remain tractable.

## Temporal Products

The following products are temporal in this repo:

- `tessera`
- `google_satellite_embedding`

`copernicus_embed` is treated as a fixed annual product with reference year `2021`.

If you pass `--years`, the generator creates one output per requested year:

```bash
python scripts/generate_dataset.py \
  --n_points 100000 \
  --encoders tessera google_satellite_embedding \
  --years 2023 2024 \
  --output_path outputs/temporal_pair
```

This produces:

- `outputs/temporal_pair_2023.pt`
- `outputs/temporal_pair_2024.pt`

The date-specific output names use the selected extension (or `.zarr` directory).

## Architecture

The shared encoder contract is defined in `wrappers/embedding_encoder.py`.

The main implementation split is:

- `wrappers/geoclip_encoder.py`
- `wrappers/satclip_encoder.py`
- `wrappers/torchgeo_encoders.py`
- `wrappers/location_model_encoders.py`
- `wrappers/registry.py`

Canonical encoder names and aliases are centralized in `wrappers/registry.py`.

## Testing

Run the test suite with:

```bash
python -m unittest discover -s tests -v
```

Quick syntax check:

```bash
python -m py_compile scripts/get_embeddings.py scripts/generate_dataset.py wrappers/*.py tests/test_*.py
```

## Practical Notes

- Use `--seed` to reproduce the generator's candidate sampling and valid-row
  selection, subject to stable encoder/data versions. The saved metadata records
  the seed and land-mask policy.
- Use `--coordinates_out` followed by `--coordinates_in` when comparing separate
  encoder runs. This is stronger than matching a seed: it guarantees the exact
  same submitted coordinates.
- Without `--include_antarctica`, the land filter excludes points at or below
  latitude `-60` before testing against Natural Earth polygons. This preserves
  the historical policy. Choose `--land_resolution 110m`, `50m`, or `10m`
  deliberately: resolution affects coastlines and small islands. Polygon
  boundaries are not considered land (`contains` semantics).
- Sampling has a bounded `--max_sampling_attempts` limit (default 100) and
  reports candidate/valid counts on failure. If coverage is sparse, reduce the
  requested size, select a compatible year, adjust land-mask policy, or raise
  the limit intentionally.
- `geoclip` and `satclip` are much faster than the large raster-backed products.
- `tessera` and `google_satellite_embedding` may need substantial network and disk activity on first use.
- `google_satellite_embedding` uses the public AEF annual index and remote GeoTIFF access. Its index is downloaded to a temporary file, validated as Parquet, and atomically promoted into the cache; a corrupt cached index is automatically replaced.

## Licensing Notes

Licensing is not uniform across the supported products.

- `geoclip` package: MIT
- `satclip`: MIT
- `tessera` embeddings in TorchGeo: CC0-1.0
- `copernicus_embed`: CC-BY-4.0
- `google_satellite_embedding`: CC-BY-4.0

For the CC-BY products, attribution is required.

## Repository Layout

```text
geospatial_embeddings_wrapper/
├── images/
├── outputs/
├── satclip/
├── scripts/
│   ├── generate_dataset.py
│   └── get_embeddings.py
├── tests/
├── wrappers/
│   ├── embedding_encoder.py
│   ├── geoclip_encoder.py
│   ├── satclip_encoder.py
│   ├── torchgeo_encoders.py
│   └── registry.py
├── README.md
└── requirements.txt
```
