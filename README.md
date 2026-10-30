# similarity

Linkage attacks on whole-slide images (WSIs): given a slide, find the slides of the same patient by comparing image features.
The code was written by [RationAI](https://github.com/RationAI) (Masaryk University) and partners for **PRIVAGAMS**, a project funded in the 1st OSCARS Open Call (grant agreement No. 101129751), and for the public challenge [Benchmarking WSI Privacy](https://www.kaggle.com/competitions/benchmarking-wsi-privacy).

## What is where

| Path | Content |
| --- | --- |
| `src/` | Slide-similarity pipeline: tile embeddings (pathology foundation models) aggregated into slide-level descriptors (mean pooling, Fisher vectors, VLAD) for retrieval and re-identification experiments. |
| `notebooks/analyze.ipynb` | Embedding-quality analysis: re-identification rates, cross-scanner robustness, preprocessing and mpp impact, scaling. |
| `memorisation/` | Memorisation audit of synthetic patches against training tiles (UNI2-h features, FAISS), and an edge-based near-duplicate check. See its README. |
| `steganography/` | Script that hides one image inside another. See its README. |

## Pipeline

1. **Tile embeddings** — `src/tile_embeddings.py`
   Tiles each WSI at a given mpp, optionally removes background, normalizes
   H&E staining and/or applies CLAHE, then embeds 224×224 tiles with a
   pathology foundation model. Writes one parquet shard set per slide
   (`slide_id=<encoded path>/` partitions).

2. **Slide embeddings** — `src/slide_embeddings.py`
   Aggregates tile embeddings into fixed-length slide descriptors: mean
   pooling, Fisher vectors (GMM) and VLAD (K-Means), each computed on raw
   and spatially-smoothed "super tile" embeddings, over multiple seeded
   runs.

3. **Analysis** — `notebooks/analyze.ipynb`
   Embedding-quality experiments: re-identification success rates, cross-
   scanner robustness, preprocessing and mpp impact, and scaling analysis.

### Encoders

| `--encoder` | Model |
|-------------|-------|
| 0 | GigaPath |
| 1 | Virchow2 |
| 2 | UNI2-h |
| 3 | midnight-12k |
| 4 | SimCLRv2 (ResNet152x4) |

Weights are downloaded from the Hugging Face Hub on first use.

## Installation

Python 3.12 or newer. The dependencies are in `pyproject.toml` and locked in `uv.lock`:

```sh
uv sync
```

OpenSlide and libvips must be installed on the system. The `memorisation/`
scripts are standalone and document their own dependencies (FAISS, `fire`,
…); they are not part of the `similarity` package.

## Setup (Musica cluster)

```
mamba create -n similarity-env python=3.12.9 -y
mamba activate similarity-env
mamba install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
mamba install -c conda-forge libvips openslide libgcc-ng -y
uv pip install -e .
```

## Usage

All commands are run from the repository root.

```bash
# 1. tile embeddings for a folder of WSIs (encoder 1 = Virchow2)
python -m src.tile_embeddings --slide-path /data/slides --save-path /data/tile_embeddings --encoder 1

# 2. slide descriptors (mean/Fisher/VLAD, several seeded runs)
python -m src.slide_embeddings --slide-path /data/tile_embeddings --save-path /data/slide_embeddings --num-runs 20
```

Preprocessing toggles for tile embeddings: `--rmbg` (drop non-tissue tiles),
`--normalize` (stain normalization), `--clahe` (contrast enhancement),
`--mpp` (tiling resolution, default 0.5 = 20×), `--overwrite`.

## Related resources

- Challenge: <https://www.kaggle.com/competitions/benchmarking-wsi-privacy>
- Challenge data: BBMRI-ERIC WSI archive, <https://cbioportal.wsi-archive.bbmri-eric.eu/archive/> (Life Science Login required).
- Risk assessment this code builds on: J. Buchta, *Assessing Privacy Risks in Real and Synthetic Whole-Slide Images via Foundation Models*, diploma thesis, Masaryk University, 2026, which extends Holub et al., *Privacy risks of whole-slide image sharing in digital pathology*, Nature Communications 14, 2577 (2023), <https://doi.org/10.1038/s41467-023-37991-y>.

## Licence

No licence file has been added yet.
