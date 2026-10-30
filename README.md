# RationAI Slide Similarity

Whole-slide-image (WSI) similarity pipeline: embed whole-slide images with
pathology foundation models and aggregate tile embeddings into fixed-length
slide descriptors for retrieval and re-identification experiments.

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

## Repository layout

```
src/tile_embeddings.py     # WSI -> tile embeddings
src/slide_embeddings.py    # tile -> slide descriptors
src/feature_extractors.py  # encoder zoo
notebooks/analyze.ipynb    # embedding-quality analysis
```
