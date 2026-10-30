# RationAI Slide Similarity

Whole-slide-image (WSI) retrieval pipeline for the RationAI competition:
given a probe slide, retrieve the visually similar slides (same patient /
tissue block) from a gallery.

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
   runs. `src/slide_embeddings_competition.py` is a leaner VLAD-only variant
   used for the competition submission.

3. **Retrieval evaluation / submission** — `src/competition_results.py`,
   `src/competition_submission.py`
   Rank each probe's gallery by cosine similarity, sweep similarity
   thresholds on the train set (official metric: mean per-example F1) and
   emit a `submission.csv` for the test set at the chosen threshold.
   Shared loading/ranking helpers live in `src/retrieval.py`.

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

# 2a. slide descriptors (mean/Fisher/VLAD, several seeded runs)
python -m src.slide_embeddings --slide-path /data/tile_embeddings --save-path /data/slide_embeddings --num-runs 20

# 2b. or: VLAD-only competition variant
python -m src.slide_embeddings_competition --slide-path /data/tile_embeddings --save-path /data/vlad

# 3. evaluate on the train split (threshold sweep, F1 report)
python -m src.competition_results --embeddings-dir /data/vlad

# 4. write the test-set submission at the best threshold
python -m src.competition_submission --embeddings-dir /data/vlad --threshold 0.60
```

Preprocessing toggles for tile embeddings: `--rmbg` (drop non-tissue tiles),
`--normalize` (stain normalization), `--clahe` (contrast enhancement),
`--mpp` (tiling resolution, default 0.5 = 20×), `--overwrite`.

## Repository layout

```
data/                          # competition splits + slide metadata
notebooks/analyze.ipynb        # embedding-quality analysis (scanners, preprocessing, scaling)
src/tile_embeddings.py         # WSI -> tile embeddings
src/slide_embeddings.py        # tile -> slide descriptors
src/slide_embeddings_competition.py
src/feature_extractors.py      # encoder zoo
src/retrieval.py               # shared embedding loading / gallery ranking
src/competition_results.py     # train-set F1 evaluation
src/competition_submission.py  # test-set submission writer
```
