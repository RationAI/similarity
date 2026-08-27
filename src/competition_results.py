import re
import pandas as pd
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────
PARQUET_DIR   = Path("/data/fs201316/jb88526/rationai_competition_slides")
QUERY_CSV     = Path("/home/jb88526/similarity/train.csv")
TRUTH_CSV     = Path("/home/jb88526/similarity/train_gt.csv")
EMBEDDING_COL = "vlad"
SIM_THRESHOLD = 0.20          # will be optimised below
# ──────────────────────────────────────────────────

SLIDE_RE = re.compile(r"(SLD-[0-9a-fA-F]+\.(?:svs|parquet))")


def extract_slide_id(filename: str) -> str:
    m = SLIDE_RE.search(filename)
    if not m:
        raise ValueError(f"Cannot extract slide ID from: {filename}")
    return m.group(1).removesuffix(".parquet")


def load_embeddings(parquet_dir: Path) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}
    files = sorted(parquet_dir.glob("*.svs.parquet"))
    if not files:
        raise FileNotFoundError(f"No *.svs.parquet in {parquet_dir}")

    df0 = pd.read_parquet(files[0])
    sid0 = extract_slide_id(files[0].name)
    print(f"  files      : {len(files)}")
    print(f"  first      : {files[0].name}  →  {sid0}")
    print(f"  columns    : {list(df0.columns)}")
    print(f"  shape      : {df0.shape}")

    for fpath in files:
        sid = extract_slide_id(fpath.name)
        df  = pd.read_parquet(fpath)

        if EMBEDDING_COL in df.columns:
            col = df[EMBEDDING_COL]
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


def rank_gallery(probe_id, gallery, embs):
    """Return [(slide_id, cosine_sim), …] sorted desc."""
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


def get_retrieved(r, threshold):
    """Set of slide IDs with sim >= threshold."""
    return set(
        sid for sid, s in zip(r["ranked_ids"], r["ranked_scores"])
        if s >= threshold
    )


def f1(r, threshold):
    """Per-example F1 between retrieved set and ground truth."""
    retrieved = set(
        sid for sid, s in zip(r["ranked_ids"], r["ranked_scores"])
        if s >= threshold
    )
    true = r["match_set"]

    tp = len(retrieved & true)
    fp = len(retrieved - true)
    fn = len(true - retrieved)

    # both empty → perfect
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score  = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0.0
    return f1_score


def main():
    embs = load_embeddings(PARQUET_DIR)
    qdf  = pd.read_csv(QUERY_CSV)
    tdf  = pd.read_csv(TRUTH_CSV)
    qdf["example_id"] = qdf["example_id"].astype(int)
    tdf["example_id"] = tdf["example_id"].astype(int)

    all_slides = set()
    for g in qdf["gallery"]:
        all_slides.update(g.split())
    all_slides.update(qdf["probe_slide"].tolist())
    missing = all_slides - set(embs.keys())
    if missing:
        print(f"  ⚠ {len(missing)} slides missing from embeddings "
              f"(e.g. {list(missing)[:3]})")
    else:
        print(f"  ✓ all {len(all_slides)} slides found\n")

    # ── rank every example ──
    rows = []
    for _, q in qdf.iterrows():
        probe   = q["probe_slide"].strip()
        gallery = q["gallery"].split()
        ranked  = rank_gallery(probe, gallery, embs)

        rows.append({
            "example_id": q["example_id"],
            "probe_slide": probe,
            "ranked_ids":    [sid for sid, _ in ranked],
            "ranked_scores": [sim for _, sim in ranked],
        })

    res = pd.DataFrame(rows)
    res = res.merge(tdf, on="example_id", how="left")
    res["match_set"] = res["matching_slide_ids"].apply(
        lambda v: set(v.split()) if pd.notna(v) else set()
    )
    res["n_match"] = res["match_set"].apply(len)

    # ── F1 sweep over thresholds ──
    print("\n  F1 by threshold:")
    best_t, best_f1 = 0.0, -1.0
    for t in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
              0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        f1s = res.apply(lambda r: f1(r, t), axis=1)
        mean_f1 = f1s.mean()
        marker = ""
        if mean_f1 > best_f1:
            best_t, best_f1 = t, mean_f1
            marker = "  ← best"
        print(f"    sim ≥ {t:.2f}  →  F1 = {mean_f1:.4f}{marker}")

    # ── final metrics at best threshold ──
    SIM_T = best_t
    res["f1"] = res.apply(lambda r: f1(r, SIM_T), axis=1)

    # extra diagnostics
    def recall_at_k(r):
        k = r["n_match"]
        if k == 0:
            return 0
        top_k = set(r["ranked_ids"][:k])
        return len(top_k & r["match_set"])

    res["n_hits_adaptive"] = res.apply(recall_at_k, axis=1)
    res["n_hits_thresh"] = res.apply(
        lambda r: len(get_retrieved(r, SIM_T) & r["match_set"]), axis=1
    )
    res["exact"] = res.apply(
        lambda r: int(get_retrieved(r, SIM_T) == r["match_set"]), axis=1
    )

    def jaccard(r):
        retrieved = get_retrieved(r, SIM_T)
        union = retrieved | r["match_set"]
        return len(retrieved & r["match_set"]) / len(union) if union else 0.0

    res["jaccard"] = res.apply(jaccard, axis=1)

    # ── summary ──
    tot        = int(res["n_match"].sum())
    n_examples = len(res)
    a          = int(res["n_hits_adaptive"].sum())
    b          = int(res["n_hits_thresh"].sum())
    n_exact    = int(res["exact"].sum())

    print("\n" + "=" * 60)
    print(f"  OPTIMAL THRESHOLD = {SIM_T:.2f}")
    print("=" * 60)
    print(f"\n  {'Metric':<35} {'Score':>10} {'Total':>8} {'Rate':>8}")
    print(f"  {'─'*63}")
    print(f"  {'★ Mean F1 (official metric)':<35} "
          f"{res['f1'].mean():>10.4f}")
    print(f"  {'A: Recall@K (K=n_match)':<35} "
          f"{a:>10} {tot:>8} {a/tot:>7.1%}")
    print(f"  {f'B: hits  sim ≥ {SIM_T:.2f}':<35} "
          f"{b:>10} {tot:>8} {b/tot:>7.1%}")
    print(f"  {f'C: exact match (sim ≥ {SIM_T:.2f})':<35} "
          f"{n_exact:>10} {n_examples:>8} {n_exact/n_examples:>7.1%}")
    print(f"  {f'D: avg Jaccard (sim ≥ {SIM_T:.2f})':<35} "
          f"{res['jaccard'].mean():>10.3f}")
    print()

    # ── save ──
    out = Path("retrieval_results.csv")
    res[["example_id", "probe_slide", "n_match",
         "n_hits_adaptive", "n_hits_thresh",
         "exact", "jaccard", "f1"]].to_csv(out, index=False)
    print(f"  Saved → {out.resolve()}")


if __name__ == "__main__":
    main()
