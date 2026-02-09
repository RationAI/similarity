import torch
import ray
import ray.data
import os
import pandas as pd
import numpy as np
import re
import argparse
from scipy.stats import spearmanr
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import KDTree
import sklearn.preprocessing as preprocessing
from sklearn.decomposition import PCA

def load_parquet(path): 
    data = pd.read_parquet(path)
    return data

def load_embeddings(slide_path, file_name):
    print(slide_path)
    dirs_to_process = sorted([
        d for d in os.listdir(slide_path) 
        if os.path.isdir(os.path.join(slide_path, d)) and not d.startswith(".")
    ])
    embeddings: List[torch.Tensor] = []
    labels = []

    for dir in dirs_to_process:
        embeddings_path = os.path.join(slide_path, dir, file_name)
        tensor = torch.tensor(load_parquet(embeddings_path).embedding[0], dtype=torch.float32)
        embeddings.append(tensor)
        labels.append(dir)
    
    emb_matrix = torch.stack(embeddings, dim=0)

    return (emb_matrix, labels)

def save_simmilarity(sim_matrix, labels, name):

    sim_matrix_mod = sim_matrix.clone()
    sim_matrix_mod.fill_diagonal_(-2.0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        annot=True,
        fmt=".2f",                    
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": f"{name} similarity"},
        annot_kws={"fontsize": 10}
    )
    plt.title(f"Pairwise {name} similarity of slide cuts embeddings")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    return 0

def remove_pc1_and_rescale(vlad_vectors):
    """
    1. Removes Scanner Bias (PC1)
    2. Rescales the output so similarities look 'normal' (0 to 1).
    """
    # 1. Convert to Numpy
    if isinstance(vlad_vectors, torch.Tensor):
        X = vlad_vectors.cpu().numpy()
    else:
        X = vlad_vectors.copy()

    # --- STEP 1: Power Norm ---
    # Standard square root to dampen peaks
    X = np.sign(X) * np.sqrt(np.abs(X))

    # --- STEP 2: PCA De-Noising ---
    # Fit on the 20 slides
    n_components = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # === THE FIX: Zero out PC1 ===
    # This removes the one dominant 'Scanner' signal
    X_pca[:, 0] = 0.0  # Remove PC1
    #X_pca[:, 1] = 0.0  # Remove PC2
    
    # Inverse transform to get back to original space
    X_clean = pca.inverse_transform(X_pca)

    # --- STEP 3: L2 Normalize ---
    X_final = torch.from_numpy(X_clean).float()
    X_final = F.normalize(X_final, p=2, dim=1)
    
    return X_final

def get_scaled_similarity_matrix(embeddings):
    """
    Computes Cosine Similarity and stretches the values 
    so the best matches are ~1.0 and worst are ~0.0.
    """
    # 1. Standard Cosine Sim (-1 to 1)
    # But because we removed PC1, values might be tiny (e.g. 0.05 to 0.15)
    raw_sim = embeddings @ embeddings.T
    
    # 2. Smart Rescaling
    # We ignore the diagonal (self-matches) for calculating min/max
    mask = ~torch.eye(raw_sim.shape[0], dtype=bool)
    valid_scores = raw_sim[mask]
    
    min_val = valid_scores.min()
    max_val = valid_scores.max()

    # Linear Stretch: (x - min) / (max - min)
    # This preserves the exact ranking but fixes the percentages
    scaled_sim = (raw_sim - min_val) / (max_val - min_val)
    
    # Fix the diagonal to exactly 1.0
    scaled_sim.fill_diagonal_(1.0)
    
    return scaled_sim


def calculate_sims(embedding_matrix, labels, path):
    # ==========================================
    # EXECUTION
    # ==========================================

    # 1. Clean the vectors (Delete PC1)
    clean_emb = remove_pc1_and_rescale(embedding_matrix)

    # 2. Get the "Nice Looking" Matrix
    sim_matrix = get_scaled_similarity_matrix(clean_emb)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        annot=True,
        fmt=".2f",                    
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"fontsize": 10}
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(path + "/similarity_heatmap.png", dpi=300, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(
        description="Creates slide embedding from parquet files with usage of VLAD encoder"
    )
    
    parser.add_argument(
        '--slide-path', 
        type=str, 
        required=True,
        help='Absolute path to folder, with multiple subfolders with parquet files'
    )
    args = parser.parse_args()
    path = args.slide_path

    embedding_matrix, labels = load_embeddings(path, "slide_vlad.parquet")
    calculate_sims(embedding_matrix, labels, path)

if __name__ == "__main__":
    main()
