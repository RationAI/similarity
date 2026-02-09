import torch
import ray
import os
import pandas as pd
import numpy as np
import re
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

def load_tiles_embeddings(slide_path):
    dirs_to_process = sorted([
        d for d in os.listdir(slide_path) 
        if os.path.isdir(os.path.join(slide_path, d)) and not d.startswith(".")
    ])
    embeddings: List[List[torch.Tensor]] = []
    labels = []

    for dir in dirs_to_process:
        embeddings_path = os.path.join(slide_path, dir, "tiles.parquet")
        embedding_array = np.array(load_parquet(embeddings_path).embedding.tolist())
        embedding_matrix = torch.tensor(embedding_array, dtype=torch.float32)
        embeddings.append(embedding_matrix)
        labels.append(dir)
    

    return (embeddings, labels)

def l1_simmilarity(slide_path, file_name):
    emb_matrix, labels = load_embeddings(slide_path, file_name)
    l1_distance_matrix = torch.cdist(emb_matrix, emb_matrix, p=1)

    D_min = l1_distance_matrix.min()
    D_max = l1_distance_matrix.max()

    # 2. Normalizace matice na rozsah [0, 1]
    # (Odečteme minimum a vydělíme rozsahem)
    D_range = D_max - D_min
    # Ošetření případu, kdy D_range je nula (např. matice plná stejných hodnot)
    if D_range == 0:
        similarity_matrix = torch.ones_like(l1_distance_matrix)
    else:
        D_norm = (l1_distance_matrix - D_min) / D_range
        
        # 3. Inverze (Odečtení od 1)
        similarity_matrix = 1 - D_norm

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


def calculate_sims(embedding_matrix, labels):
    # ==========================================
    # EXECUTION
    # ==========================================

    # 1. Clean the vectors (Delete PC1)
    clean_emb = remove_pc1_and_rescale(embedding_matrix)

    # 2. Get the "Nice Looking" Matrix
    sim_matrix = get_scaled_similarity_matrix(clean_emb)


