import torch
import ray
import ray.data
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

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
import pyarrow.parquet as pq

def load_parquet(path): 
    data = pd.read_parquet(path)
    return data


def load_tiles_embeddings(slide_path):
    dirs_to_process = sorted([
        d for d in os.listdir(slide_path) 
        if os.path.isdir(os.path.join(slide_path, d)) and not d.startswith(".")
    ])
    embeddings: List[List[torch.Tensor]] = {}
    coords = {}
    labels = []

    for dir in dirs_to_process:
        embeddings_path = os.path.join(slide_path, dir, "tiles.parquet")
        data = load_parquet(embeddings_path)
        embedding_array = np.array(data.embedding.tolist())
        embedding_matrix = torch.tensor(embedding_array, dtype=torch.float32)
        embeddings[dir] = embedding_matrix
        coords[dir] = [*zip(data.x_coord.tolist(), data.y_coord.tolist())]
        labels.append(dir)

    return (embeddings, labels, coords)

class WSI_VLAD_Encoder:
    def __init__(self, n_clusters=16, embedding_dim=1536):
        """
        n_clusters (K): The size of the vocabulary (usually 8 to 32).
                        Higher = more detail, but larger final vector.
        embedding_dim (D): The size of your input tile embeddings.
        """
        self.k = n_clusters
        self.d = embedding_dim
        # MiniBatchKMeans is much faster for large datasets
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=256, random_state=42)
        self.is_fitted = False

    def fit(self, sample_embeddings):
        """
        Step 1: Build the Codebook (Vocabulary).
        
        Input: A large numpy array of shape (N_samples, D).
        NOTE: Do not pass ALL tiles from ALL slides. Just take a random 
        10% sample from your dataset to train the dictionary.
        """
        print(f"Fitting K-Means vocabulary on {sample_embeddings.shape[0]} tiles...")
        self.kmeans.fit(sample_embeddings)
        self.is_fitted = True
        print("Vocabulary fitted.")

    def _spatial_smoothing(self, embeddings, coords, radius=300, alpha=0.5):
        """
        Internal helper: Mixes neighbor info into tiles.
        radius: Distance in pixels to look for neighbors.
        alpha: 0.5 means 50% original feature, 50% neighbor average.
        """
        tree = KDTree(coords)
        neighbors_list = tree.query_ball_point(coords, r=radius)
        
        smoothed_embeddings = np.zeros_like(embeddings)
        
        for i, neighbors in enumerate(neighbors_list):
            if len(neighbors) > 1:
                # Average of neighbors
                context = np.mean(embeddings[neighbors], axis=0)
                smoothed_embeddings[i] = (1 - alpha) * embeddings[i] + alpha * context
            else:
                smoothed_embeddings[i] = embeddings[i]
                
        return smoothed_embeddings

    def transform(self, tile_embeddings, tile_coords, spatial_smoothing=True):
        """
        Step 2: Convert one WSI (thousands of tiles) into ONE vector.
        
        tile_embeddings: (N, D) numpy array
        tile_coords: (N, 2) numpy array of x, y positions
        """
        if not self.is_fitted:
            raise ValueError("You must call .fit() before .transform()")

        # 1. Spatial Smoothing (The context step)
        if spatial_smoothing:
            data = self._spatial_smoothing(tile_embeddings, tile_coords)
        else:
            data = tile_embeddings

        # 2. Predict Clusters (Assign tiles to vocabulary words)
        predicted_labels = self.kmeans.predict(data)
        cluster_centers = self.kmeans.cluster_centers_

        # 3. VLAD Calculation (Accumulate Residuals)
        # Final shape will be (K * D)
        vlad_vector = np.zeros((self.k, self.d))

        for i in range(self.k):
            # Get all tiles belonging to cluster i
            mask = (predicted_labels == i)
            if np.sum(mask) > 0:
                cluster_tiles = data[mask]
                center = cluster_centers[i]
                
                # The Core Math: Sum of (Tile - Center)
                residual = cluster_tiles - center
                vlad_vector[i] = np.sum(residual, axis=0)
            # Else: If no tiles match this cluster, it stays zeros (which is correct)

        # Flatten to 1D
        vlad_vector = vlad_vector.flatten()

        # 4. Normalization (Crucial for Similarity Search)
        
        # A. Power Normalization (Standard in VLAD papers)
        # This reduces the impact of visual bursts (common patterns)
        vlad_vector = np.sign(vlad_vector) * np.sqrt(np.abs(vlad_vector))
        
        # B. L2 Normalization
        # Makes the vector length = 1, so Cosine Similarity works perfectly
        vlad_vector = preprocessing.normalize(vlad_vector.reshape(1, -1), norm='l2').flatten()

        return vlad_vector

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
    slide_path = args.slide_path

    print(slide_path)
    emb, labels, coords = load_tiles_embeddings(slide_path)
    print(labels)

    vlad = WSI_VLAD_Encoder()

    to_fit = []

    for l in labels:
        indices = torch.randperm(emb[l].size(0))[:350]
        random_sample = emb[l][indices]
        to_fit.append(random_sample)

    combined_samples = torch.cat(to_fit, dim=0)
    cs_numpy = combined_samples.numpy()

    vlad.fit(cs_numpy)

    slide_embeddings = {}

    for l in labels:
        wsi_vector = vlad.transform(emb[l].numpy(), coords[l])
        slide_embeddings[l] = wsi_vector

    for l in labels:
        wsi_embedding = slide_embeddings[l]
        label = l
        output_df = pd.DataFrame({
            'slide_id': label,
            'embedding': [wsi_embedding.tolist()],
        })
        output_df.to_parquet(f"{slide_path}/{label}/slide_vlad.parquet", index=False)

if __name__ == "__main__":
    main()
