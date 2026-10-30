# similarity

Linkage attacks on whole-slide images (WSIs): given a slide, find the slides of the same patient by comparing image features.
The code was written by [RationAI](https://github.com/RationAI) (Masaryk University) and partners for **PRIVAGAMS**, a project funded in the 1st OSCARS Open Call (grant agreement No. 101129751), and for the public challenge [Benchmarking WSI Privacy](https://www.kaggle.com/competitions/benchmarking-wsi-privacy).

## What is where

| Path | Content | Branch |
| --- | --- | --- |
| `src/`, `examples/`, `tests/` | Early prototype: slide thumbnails are encoded with a pretrained network (ResNet-18, VGG-16, Prov-GigaPath or SimCLR) and compared by cosine similarity; the top-k most similar slides are returned. | `master` |
| `pretrained/` | SimCLR and ResNet-50 weights (about 340 MB) used by the SimCLR extractor. | `master` |
| `memorisation/` | Memorisation audit of synthetic patches against training tiles (UNI2-h features, FAISS), and an edge-based near-duplicate check. See its README. | `master` |
| `steganography/` | Script that hides one image inside another. See its README. | `master` |
| Tile and slide embeddings, VLAD descriptors, challenge baseline | Virchow2 tile embeddings, VLAD slide descriptors and cosine ranking for the challenge, with the challenge input files. | `feature/slide-encoder` |

**The challenge baseline is not on `master`.**
It lives on the branch `feature/slide-encoder` and has not been merged.
That branch has these entry points: `src/tile_embeddings.py`, `src/slide_embeddings.py`, `src/slide_embeddings_competition.py` (Virchow2 tile embeddings combined into a VLAD descriptor per slide) and `src/competition_results.py` (ranking of gallery slides by cosine similarity), together with `train.csv`, `train_gt.csv`, `test.csv` and `ration_ai_competition.csv`, the files of the challenge.
Its README contains only the cluster set-up.
Several scripts there contain paths of the authors' cluster, and it depends on a package from a private GitLab server (`rationai-staining`), so it cannot be installed from outside without changes.

## The prototype on `master`

The package takes slides from a folder, reads a thumbnail of each, encodes it, and compares all slides with each other.

```python
from src.__main__ import main
from src.datasources import ThumbnailDatasource
from src.feature_extractors import resnet18

df, top_k, top_k_sim = main(
    ThumbnailDatasource(["/path/to/slides"], image_extent=224), resnet18(), k=1
)
```

- `df` has one row per slide, with its `name` and `features`.
- `top_k` holds, for each slide, the indices of its `k` most similar other slides.
- `examples/` contains two scripts that compute the recall of this search on the authors' scans; they use the authors' paths (`/mnt/data/scans/...`) and show how to evaluate the result.
- Any slide format that OpenSlide or libvips reads is accepted (`.svs`, `.mrxs`, `.ndpi`, `.tif` and others).

A GPU is required (the feature extractor runs on CUDA), and the feature extraction runs through Ray.
`tests/` holds scripts that need the authors' scans; they are not run in CI, which only runs the lint workflow.

## Installation

Python 3.12 or newer. The dependencies are in `pyproject.toml` and locked in `uv.lock`:

```sh
uv sync
```

OpenSlide and libvips must be installed on the system. The `gigapath` extractor downloads its weights from Hugging Face on first use.

## Related resources

- Challenge: <https://www.kaggle.com/competitions/benchmarking-wsi-privacy>
- Challenge data: BBMRI-ERIC WSI archive, <https://cbioportal.wsi-archive.bbmri-eric.eu/archive/> (Life Science Login required).
- Risk assessment this code builds on: J. Buchta, *Assessing Privacy Risks in Real and Synthetic Whole-Slide Images via Foundation Models*, diploma thesis, Masaryk University, 2026, which extends Holub et al., *Privacy risks of whole-slide image sharing in digital pathology*, Nature Communications 14, 2577 (2023), <https://doi.org/10.1038/s41467-023-37991-y>.

## Licence

No licence file has been added yet.
