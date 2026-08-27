"""
Virchow2 → VLAD slide-level descriptors.

Reads per-slide parquet shards (2560-d Virchow2 embeddings, tissue-only),
fits PCA + K-Means on a representative sample, then computes a power-
normalised VLAD vector for every slide.

Usage:
    python vlad_slides.py \
        --slide-path /data/virchow2_slides \
        --save-path  /data/vlad_output
"""

import argparse
import gc

import numpy as np
import pandas as pd
import ray
import torch
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Config (override via CLI)
# ---------------------------------------------------------------------------
PCA_DIMS      = 512
VLAD_CLUSTERS = 128
SAMPLE_PER_SLIDE = 100
LOAD_WINDOW  = 16          # CPU slides in-flight while GPU computes
CDIST_CHUNK  = 500_000   # rows per cdist block (keeps peak GPU mem low)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_slide_embeddings(slide_dir: Path) -> np.ndarray:
    """Concatenate all parquet shards in *slide_dir* → (N, D) float32."""
    shards = sorted(slide_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet files in {slide_dir}")
    embs = np.concatenate([
        np.vstack(pq.read_table(s, columns=["embedding"]).column("embedding").to_numpy())
        for s in shards
    ]).astype(np.float32)
    return embs


def hard_assign(X_norm: torch.Tensor, centers: torch.Tensor,
                chunk: int = CDIST_CHUNK) -> torch.Tensor:
    """Nearest-centroid index per row, computed in chunks to limit GPU memory."""
    N = X_norm.shape[0]
    idx = torch.empty(N, dtype=torch.long, device=X_norm.device)
    for i in range(0, N, chunk):
        idx[i:i + chunk] = torch.argmin(
            torch.cdist(X_norm[i:i + chunk], centers), dim=1
        )
    return idx


def compute_vlad(embs: np.ndarray,
                 pca_comps: torch.Tensor,
                 pca_mean: torch.Tensor,
                 vlad_centers: torch.Tensor,
                 device: torch.device) -> np.ndarray:
    """PCA → L2-norm → hard-assign VLAD → power-norm → L2-norm."""
    print(f"  embs.shape={embs.shape}  pca_comps.shape={tuple(pca_comps.shape)}, pca_mean.shape={tuple(pca_mean.shape)}, vlad_centers.shape={tuple(vlad_centers.shape)}")
    X = torch.from_numpy(embs).to(device)

    # project + normalise
    X_pca = (X - pca_mean) @ pca_comps.t()
    X_norm = X_pca / (X_pca.norm(dim=1, keepdim=True) + 1e-8)

    # hard-assign (chunked)
    idx = hard_assign(X_norm, vlad_centers)

    # accumulate residuals WITHOUT materialising a one-hot matrix
    K, D = vlad_centers.shape
    V      = torch.zeros(K, D, device=device)
    V.index_add_(0, idx, X_norm)                       # V[k] += x_j  for j→k
    counts = torch.bincount(idx, minlength=K).float()
    V -= counts.unsqueeze(1) * vlad_centers            # subtract centroid

    # power-norm + L2
    v = V.flatten().cpu().numpy()
    v = np.sign(v) * np.sqrt(np.abs(v))
    v /= (np.linalg.norm(v) + 1e-8)

    del X, X_pca, X_norm, idx, V, counts
    torch.cuda.empty_cache()
    return v


# ---------------------------------------------------------------------------
# Ray workers  (top-level so they are picklable)
# ---------------------------------------------------------------------------

@ray.remote(num_cpus=1)
def _sample_slide(slide_dir: str, n: int) -> np.ndarray:
    """Load a slide and return *n* random rows (for PCA/KMeans fitting)."""
    embs = load_slide_embeddings(Path(slide_dir))
    idx  = np.random.choice(len(embs), min(n, len(embs)), replace=False)
    return embs[idx]


@ray.remote(num_cpus=1)
def _load_slide(slide_dir: str):
    """Load a full slide → (slide_id, embeddings)."""
    d = Path(slide_dir)
    return d.name.replace("slide_id=", ""), load_slide_embeddings(d)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Virchow2 → VLAD slide descriptors")
    parser.add_argument("--slide-path", type=str, required=True,
                        help="Root dir containing slide_id=…/ folders")
    parser.add_argument("--save-path",  type=str, required=True)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--pca-dims",    type=int, default=PCA_DIMS)
    parser.add_argument("--vlad-clusters",type=int, default=VLAD_CLUSTERS)
    parser.add_argument("--load-window",  type=int, default=LOAD_WINDOW)
    args = parser.parse_args()

    set_seed(args.seed)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- discover slides ------------------------------------------------
    slide_dirs = sorted(
        d for d in Path(args.slide_path).iterdir()
        if d.is_dir() and d.name.startswith("slide_id=")
    )
    if not slide_dirs:
        raise SystemExit(f"No slide folders found in {args.slide_path}")
    print(f"Found {len(slide_dirs)} slides")

    ray.init(num_cpus=16, ignore_reinit_error=True)

    # ---- Phase 1: fit PCA + K-Means on a representative sample ----------
    print("Collecting fit sample …")
    sample_refs = [
        _sample_slide.remote(str(d), SAMPLE_PER_SLIDE) for d in slide_dirs
    ]
    chunks = [ray.get(r) for r in tqdm(sample_refs, desc="Sampling")]
    sample = np.concatenate(chunks)
    del chunks, sample_refs
    gc.collect()

    print(f"Fitting PCA({args.pca_dims}) + KMeans({args.vlad_clusters}) …")
    pca = PCA(n_components=args.pca_dims, random_state=args.seed).fit(sample)
    s_pca = pca.transform(sample)
    s_pca /= (np.linalg.norm(s_pca, axis=1, keepdims=True) + 1e-8)

    kmeans = MiniBatchKMeans(
        n_clusters=args.vlad_clusters,
        n_init=5,
        random_state=args.seed,
    ).fit(s_pca)

    pca_comps    = torch.from_numpy(pca.components_.astype(np.float32)).to(device)
    pca_mean     = torch.from_numpy(pca.mean_.astype(np.float32)).to(device)
    vlad_centers = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(device)
    del sample, s_pca
    gc.collect()
    print("Fit complete.")

    # ---- Phase 2: pipelined VLAD inference ------------------------------
    print("Computing VLAD descriptors …")
    pending: list[ray.ObjectRef] = []
    f_iter = iter(slide_dirs)

    for _ in range(min(args.load_window, len(slide_dirs))):
        pending.append(_load_slide.remote(str(next(f_iter))))

    with tqdm(total=len(slide_dirs), desc="VLAD") as pbar:
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            for ref in done:
                slide_id, embs = ray.get(ref)

                out_file = save_dir / f"{slide_id}.parquet"
                if not out_file.exists():
                    v = compute_vlad(
                        embs, pca_comps, pca_mean, vlad_centers, device
                    )
                    pd.DataFrame(
                        {"slide_id": [slide_id], "vlad": [v.tolist()]}
                    ).to_parquet(out_file)
                    del v

                del embs
                pbar.update(1)

                try:
                    pending.append(_load_slide.remote(str(next(f_iter))))
                except StopIteration:
                    pass

    ray.shutdown()
    print(f"\nDone.  {len(slide_dirs)} slides → {save_dir}")


if __name__ == "__main__":
    main()
