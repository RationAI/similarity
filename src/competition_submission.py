"""Generate a competition submission file from slide embeddings.

Ranks each test probe's gallery by cosine similarity and emits every
slide above the similarity threshold as the predicted match set.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.retrieval import load_embeddings, rank_gallery, report_missing


def main():
    parser = argparse.ArgumentParser(description="Create test-set submission CSV")
    parser.add_argument("--embeddings-dir", type=Path, required=True,
                        help="Dir with <slide_id>.svs.parquet embedding files")
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("submission.csv"))
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Similarity threshold (use the best from the training F1 sweep)")
    args = parser.parse_args()

    embs = load_embeddings(args.embeddings_dir)
    tdf  = pd.read_csv(args.test_csv)
    tdf["example_id"] = tdf["example_id"].astype(int)

    all_slides = set()
    for g in tdf["gallery"]:
        all_slides.update(g.split())
    all_slides.update(tdf["probe_slide"].tolist())
    report_missing(all_slides, embs)

    rows = []
    for _, q in tdf.iterrows():
        probe   = q["probe_slide"].strip()
        gallery = q["gallery"].split()
        ranked  = rank_gallery(probe, gallery, embs)

        predicted = [sid for sid, sim in ranked if sim >= args.threshold]

        rows.append({
            "example_id": q["example_id"],
            "matching_slide_ids": " ".join(predicted) if predicted else "",
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.output_csv, index=False)
    print(f"\n  Wrote {len(out)} rows → {args.output_csv.resolve()}")
    print(f"  Threshold: {args.threshold}")


if __name__ == "__main__":
    main()
