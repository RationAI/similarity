# Memorisation audit

Tools for checking whether synthetic histopathology patches reproduce tissue of the training slides.
Written by Bernd Saurugger (TU Wien) for the PRIVAGAMS project. They work on image files only; the generator that produced the patches is not part of this repository.

## Method

1. `build_wsi_index.py` cuts the training slides into non-overlapping tiles of 128 µm, encodes each tile with the UNI2-h pathology foundation model and stores the features in a FAISS index (multi-GPU).
2. `search_wsi_index.py` encodes the synthetic patches the same way and finds, for each, the most similar training tile by cosine similarity. It writes `uni2_wsi_search_results.csv`.
3. `visualize_top10.py` shows the closest matches side by side, and `visualize_memorisation.ipynb` plots the similarity distribution (histogram, cumulative curve, per-slide boxplot, spatial map).

Thresholds used in the notebook: 0.85 for a morphological match and 0.95 for memorisation.

## Edge-based check

`run_bb_attack.py` clusters generated images by their Sobel edge maps with FAISS to find near-duplicates among the generated images themselves. See `README_BB_ATTACK.md`. No results of this check are documented.

## Results

`results/` holds the outputs of one run on 1,000 generated patches (`uni2_wsi_search_results.csv`) and of an earlier multi-image search (`multi_wsi_search_results.csv`, L2 distance).
For the first file: mean cosine similarity 0.78, maximum 0.89, 103 patches at 0.85 or above, none at 0.90 or above.
These files list names of internal training slides; check this before publishing the repository.

## Running

The scripts read from a data folder set by the environment variable `DATA` (default `~/data`):

| Folder | Content |
| --- | --- |
| `$DATA/annoPaperScanns` | training slides (`.svs`) |
| `$DATA/samples_uncond` | generated patches (`.png`) |
| `$DATA/indices` | FAISS index and metadata written by step 1 |

The scripts need a GPU, `torch`, `timm` (with access to the `MahmoodLab/UNI2-h` weights on Hugging Face), `faiss`, `openslide-python`, `pandas`, `Pillow`, `matplotlib`, and for the edge check `scipy` and `fire`.
`processing_utils.py` is the edge-map helper of the edge check, copied unchanged except for the import in `run_bb_attack.py`.
These scripts are not yet part of the `similarity` package and have not been run from this repository.
