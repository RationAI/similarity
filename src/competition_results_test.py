import re
import pandas as pd
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────
PARQUET_DIR   = Path("/data/fs201316/jb88526/rationai_competition_slides")
TEST_CSV      = Path("/home/jb88526/similarity/test.csv")
OUTPUT_CSV    = Path("submission.csv")
EMBEDDING_COL = "vlad"
SIM_THRESHOLD = 0.25          # use the best value from your training F1 sweep
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


def main():
    embs = load_embeddings(PARQUET_DIR)
    tdf  = pd.read_csv(TEST_CSV)
    tdf["example_id"] = tdf["example_id"].astype(int)

    all_slides = set()
    for g in tdf["gallery"]:
        all_slides.update(g.split())
    all_slides.update(tdf["probe_slide"].tolist())
    missing = all_slides - set(embs.keys())
    if missing:
        print(f"  ⚠ {len(missing)} slides missing from embeddings "
              f"(e.g. {list(missing)[:3]})")
    else:
        print(f"  ✓ all {len(all_slides)} slides found\n")

    rows = []
    for _, q in tdf.iterrows():
        probe   = q["probe_slide"].strip()
        gallery = q["gallery"].split()
        pv      = embs[probe]
        pn      = np.linalg.norm(pv)

        predicted = []
        for sid in gallery:
            if sid not in embs:
                continue
            gv = embs[sid]
            sim = float(np.dot(pv, gv) / (pn * np.linalg.norm(gv) + 1e-12))
            if sim >= SIM_THRESHOLD:
                predicted.append(sid)

        rows.append({
            "example_id": q["example_id"],
            "matching_slide_ids": " ".join(predicted) if predicted else "",
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Wrote {len(out)} rows → {OUTPUT_CSV.resolve()}")
    print(f"  Threshold: {SIM_THRESHOLD}")


if __name__ == "__main__":
    main()
