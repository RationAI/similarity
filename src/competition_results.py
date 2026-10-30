"""Evaluate slide retrieval on the competition training set.

Loads per-slide embeddings, ranks every probe's gallery by cosine
similarity, sweeps similarity thresholds and reports mean F1 (the
official competition metric) plus diagnostics at the best threshold.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.retrieval import load_embeddings, rank_gallery, report_missing

THRESHOLDS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
              0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def get_retrieved(r, threshold):
    """Set of slide IDs with sim >= threshold."""
    return {
        sid for sid, s in zip(r["ranked_ids"], r["ranked_scores"])
        if s >= threshold
    }


def f1(r, threshold):
    """Per-example F1 between retrieved set and ground truth."""
    retrieved = get_retrieved(r, threshold)
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
    parser = argparse.ArgumentParser(description="Evaluate retrieval F1 on the train set")
    parser.add_argument("--embeddings-dir", type=Path, required=True,
                        help="Dir with <slide_id>.svs.parquet embedding files")
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--gt-csv", type=Path, default=Path("data/train_gt.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("retrieval_results.csv"))
    args = parser.parse_args()

    embs = load_embeddings(args.embeddings_dir)
    qdf  = pd.read_csv(args.train_csv)
    tdf  = pd.read_csv(args.gt_csv)
    qdf["example_id"] = qdf["example_id"].astype(int)
    tdf["example_id"] = tdf["example_id"].astype(int)

    all_slides = set()
    for g in qdf["gallery"]:
        all_slides.update(g.split())
    all_slides.update(qdf["probe_slide"].tolist())
    report_missing(all_slides, embs)

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
    for t in THRESHOLDS:
        mean_f1 = res.apply(lambda r: f1(r, t), axis=1).mean()
        marker = ""
        if mean_f1 > best_f1:
            best_t, best_f1 = t, mean_f1
            marker = "  ← best"
        print(f"    sim ≥ {t:.2f}  →  F1 = {mean_f1:.4f}{marker}")

    # ── final metrics at best threshold ──
    sim_t = best_t
    res["f1"] = res.apply(lambda r: f1(r, sim_t), axis=1)

    # extra diagnostics
    def recall_at_k(r):
        k = r["n_match"]
        if k == 0:
            return 0
        top_k = set(r["ranked_ids"][:k])
        return len(top_k & r["match_set"])

    res["n_hits_adaptive"] = res.apply(recall_at_k, axis=1)
    res["n_hits_thresh"] = res.apply(
        lambda r: len(get_retrieved(r, sim_t) & r["match_set"]), axis=1
    )
    res["exact"] = res.apply(
        lambda r: int(get_retrieved(r, sim_t) == r["match_set"]), axis=1
    )

    def jaccard(r):
        retrieved = get_retrieved(r, sim_t)
        union = retrieved | r["match_set"]
        return len(retrieved & r["match_set"]) / len(union) if union else 0.0

    res["jaccard"] = res.apply(jaccard, axis=1)

    # ── summary ──
    tot         = int(res["n_match"].sum())
    n_examples  = len(res)
    a           = int(res["n_hits_adaptive"].sum())
    b           = int(res["n_hits_thresh"].sum())
    n_exact     = int(res["exact"].sum())

    print("\n" + "=" * 60)
    print(f"  OPTIMAL THRESHOLD = {sim_t:.2f}")
    print("=" * 60)
    print(f"\n  {'Metric':<35} {'Score':>10} {'Total':>8} {'Rate':>8}")
    print(f"  {'─'*63}")
    print(f"  {'★ Mean F1 (official metric)':<35} "
          f"{res['f1'].mean():>10.4f}")
    print(f"  {'A: Recall@K (K=n_match)':<35} "
          f"{a:>10} {tot:>8} {a/tot:>7.1%}")
    print(f"  {f'B: hits  sim ≥ {sim_t:.2f}':<35} "
          f"{b:>10} {tot:>8} {b/tot:>7.1%}")
    print(f"  {f'C: exact match (sim ≥ {sim_t:.2f})':<35} "
          f"{n_exact:>10} {n_examples:>8} {n_exact/n_examples:>7.1%}")
    print(f"  {f'D: avg Jaccard (sim ≥ {sim_t:.2f})':<35} "
          f"{res['jaccard'].mean():>10.3f}")
    print()

    # ── save ──
    res[["example_id", "probe_slide", "n_match",
         "n_hits_adaptive", "n_hits_thresh",
         "exact", "jaccard", "f1"]].to_csv(args.output_csv, index=False)
    print(f"  Saved → {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
