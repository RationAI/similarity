import os

import faiss
import fire
import numpy as np
import pandas as pd
import processing_utils
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def load_images_and_edge_vectors(samples_folder="samples", img_size=256):
    img_files = sorted(
        [
            f
            for f in os.listdir(samples_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    edge_vectors = []
    edge_scores = []
    filenames = []
    for f in img_files:
        path = os.path.join(samples_folder, f)
        img = (
            Image.open(path)
            .convert("RGB")
            .resize((img_size, img_size), resample=Image.LANCZOS)
        )
        img_np = np.array(img).astype("float32")
        edge_map = processing_utils.get_edge_img(
            img_np / 255.0
        ).squeeze()  # (img_size,img_size)
        score = edge_map.mean()
        vec = edge_map.flatten().astype("float32")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        edge_vectors.append(vec)
        edge_scores.append(score)
        filenames.append(f)
    return np.stack(edge_vectors), filenames, np.array(edge_scores)


def mse(a, b):
    return np.mean((a - b) ** 2)


def build_similarity_graph(edge_vectors, similarity_threshold=0.99, k=20):
    d = edge_vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(edge_vectors)

    D, I = index.search(edge_vectors, k)

    rows = []
    cols = []
    sims = []

    for i in range(len(edge_vectors)):
        for rank in range(1, k):  # Skip self-match
            j = I[i, rank]
            sim = D[i, rank]
            if sim >= similarity_threshold:
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
                sims.append(sim)  # for both directions
                sims.append(sim)

    graph = coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(len(edge_vectors), len(edge_vectors))
    )
    return graph, I, D


def cluster_images(graph):
    n_components, labels = connected_components(csgraph=graph, directed=False)
    return n_components, labels


def run_bb_attack(
    samples_folder="samples",
    out_parquet_file="bb_attack_clusters.parquet",
    similarity_threshold=0.99,
    k=10,
    min_cluster_size=2,
    verb=True,
):
    edge_vectors, filenames, edge_scores = load_images_and_edge_vectors(samples_folder)
    if verb:
        print(f"Loaded {len(filenames)} images and computed edge vectors.")

    graph, neighbors_idx, neighbors_sim = build_similarity_graph(
        edge_vectors, similarity_threshold=similarity_threshold, k=k
    )
    if verb:
        print(f"Built similarity graph with {graph.nnz // 2} edges.")

    n_components, labels = cluster_images(graph)
    if verb:
        print(f"Found {n_components} clusters.")

    # For fast access from (i,j) to similarity
    sim_dict = {}
    for i in range(len(edge_vectors)):
        for rank in range(1, k):
            j = neighbors_idx[i, rank]
            sim = neighbors_sim[i, rank]
            sim_dict[(i, j)] = sim

    data = []
    for cid in range(n_components):
        members = np.where(labels == cid)[0]
        if len(members) < min_cluster_size:
            continue
        ref_vec = edge_vectors[members[0]]
        for idx in members:
            mse_val = mse(edge_vectors[idx], ref_vec)
            # similarity to reference (if i==ref_idx then sim=1.0)
            if idx == members[0]:
                sim_to_ref = 1.0
            else:
                sim_to_ref = sim_dict.get(
                    (idx, members[0]), sim_dict.get((members[0], idx), 0.0)
                )
            data.append(
                {
                    "filename": filenames[idx],
                    "cluster": cid,
                    "edge_score": edge_scores[idx],
                    "mse_to_ref": mse_val,
                    "faiss_similarity_to_ref": sim_to_ref,
                }
            )

    if not data:
        if verb:
            print("No clusters found with the specified minimum cluster size.")
        return

    df = pd.DataFrame(data)
    df.to_parquet(out_parquet_file)

    if verb:
        print(
            f"Saved {len(df)} images from {len(df['cluster'].unique())} clusters to {out_parquet_file}"
        )
        for cid in df["cluster"].unique():
            cluster_files = df[df["cluster"] == cid]["filename"].tolist()
            print(f"Cluster {cid} ({len(cluster_files)} imgs):")
            for fn in cluster_files:
                print(f"  {fn}")


if __name__ == "__main__":
    fire.Fire(run_bb_attack)
