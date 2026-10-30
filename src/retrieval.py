"""Shared helpers for slide-embedding retrieval experiments."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

SLIDE_RE = re.compile(r"(SLD-[0-9a-fA-F]+\.(?:svs|parquet))")


def extract_slide_id(filename: str) -> str:
    m = SLIDE_RE.search(filename)
    if not m:
        raise ValueError(f"Cannot extract slide ID from: {filename}")
    return m.group(1).removesuffix(".parquet")


def load_embeddings(
    parquet_dir: Path, embedding_col: str = "vlad"
) -> dict[str, np.ndarray]:
    """Load one embedding per slide from <slide_id>.svs.parquet files.

    If ``embedding_col`` is missing, the first object column is used.
    Multi-row files are mean-pooled into a single vector.
    """
    embeddings: dict[str, np.ndarray] = {}
    files = sorted(parquet_dir.glob("*.svs.parquet"))
    if not files:
        raise FileNotFoundError(f"No *.svs.parquet in {parquet_dir}")

    df0 = pd.read_parquet(files[0])
    print(f"  files      : {len(files)}")
    print(f"  first      : {files[0].name}  →  {extract_slide_id(files[0].name)}")
    print(f"  columns    : {list(df0.columns)}")
    print(f"  shape      : {df0.shape}")

    for fpath in files:
        sid = extract_slide_id(fpath.name)
        df = pd.read_parquet(fpath)

        if embedding_col in df.columns:
            col = df[embedding_col]
        else:
            obj = df.select_dtypes("object").columns
            if len(obj) == 0:
                raise ValueError(f"No vector column in {fpath.name}")
            col = df[obj[0]]

        if df.shape[0] == 1:
            vec = np.asarray(col.iloc[0], dtype=np.float32)
        else:
            vec = np.stack(
                [np.asarray(v, dtype=np.float32) for v in col]
            ).mean(axis=0)
        embeddings[sid] = vec

    print(f"  → {len(embeddings)} embeddings loaded\n")
    return embeddings


def rank_gallery(
    probe_id: str, gallery: list[str], embs: dict[str, np.ndarray]
) -> list[tuple[str, float]]:
    """Return [(slide_id, cosine_sim), …] sorted by descending similarity."""
    pv = embs[probe_id]
    pn = np.linalg.norm(pv)
    scored = []
    for sid in gallery:
        if sid not in embs:
            continue
        gv = embs[sid]
        sim = float(np.dot(pv, gv) / (pn * np.linalg.norm(gv) + 1e-12))
        scored.append((sid, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def report_missing(all_slides: set[str], embs: dict[str, np.ndarray]) -> None:
    missing = all_slides - set(embs.keys())
    if missing:
        print(
            f"  ⚠ {len(missing)} slides missing from embeddings "
            f"(e.g. {sorted(missing)[:3]})"
        )
    else:
        print(f"  ✓ all {len(all_slides)} slides found\n")
