import ray
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KDTree
import torch
from pathlib import Path
import time

# --- 1. PROSTOROVÉ ZPRACOVÁNÍ (SUPER-TILES) ---

def create_super_tiles(df, tile_size=224):
    """
    Zprůměruje embeddingy v mřížce 3x3 (SuperTiles).
    """
    coords = df[['x_coord', 'y_coord']].values
    embs = np.stack(df['embedding'].values)
    
    # Radius nastaven tak, aby našel sousedy v mřížce (včetně diagonál)
    # 1.5 * tile_size pokryje okolí 3x3
    tree = KDTree(coords)
    indices = tree.query_radius(coords, r=1.5 * tile_size)
    
    super_embs = np.zeros_like(embs)
    for i, idx_list in enumerate(indices):
        super_embs[i] = np.mean(embs[idx_list], axis=0)
        
    return super_embs

# --- 2. AGREGAČNÍ ALGORITMY ---

def compute_soft_vlad(embeddings, centroids, sigma=1.0):
    """
    Soft-assignment VLAD (Vector of Locally Aggregated Descriptors).
    """
    # Vzdálenosti (N, K)
    dists = np.linalg.norm(embeddings[:, np.newaxis] - centroids, axis=2)
    
    # Soft-weights
    weights = np.exp(-sigma * dists**2)
    weights /= (weights.sum(axis=1, keepdims=True) + 1e-12)
    
    K, D = centroids.shape
    vlad = np.zeros((K, D))
    
    for k in range(K):
        res = (embeddings - centroids[k]) * weights[:, k:k+1]
        vlad[k] = np.sum(res, axis=0)
        
    # Normalizace
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad)) # Power norm
    vlad_flat = vlad.flatten()
    return vlad_flat / (np.linalg.norm(vlad_flat) + 1e-6)

def compute_fisher_vector(embeddings, gmm):
    """
    Fisher Vector encoding pomocí GMM (Gaussian Mixture Model).
    """
    means = gmm.means_
    covs = gmm.covariances_ # Diagonální
    priors = gmm.weights_
    N = embeddings.shape[0]
    K, D = means.shape

    # Pravděpodobnosti příslušnosti ke komponentám (N, K)
    resps = gmm.predict_proba(embeddings)
    
    # Gradienty
    u_k = np.zeros((K, D))
    v_k = np.zeros((K, D))
    
    for k in range(K):
        diff = embeddings - means[k]
        u_k[k] = np.sum(resps[:, k:k+1] * diff, axis=0) / (N * np.sqrt(priors[k]))
        v_k[k] = np.sum(resps[:, k:k+1] * (diff**2 / covs[k] - 1), axis=0) / (N * np.sqrt(2 * priors[k]))

    fv = np.concatenate([u_k.flatten(), v_k.flatten()])
    
    # Normalizace
    fv = np.sign(fv) * np.sqrt(np.abs(fv))
    return fv / (np.linalg.norm(fv) + 1e-6)

# --- 4. MAIN PIPELINE ---

def aggregate_all_methods(group_df, vlad_model, gmm_model):
    """
    Zpracuje jeden slide všemi 4 kombinacemi najednou.
    """
    slide_id = group_df["slide_id"].iloc[0]
    raw_embs = np.stack(group_df['embedding'].values)
    
    # 1. Připravíme SuperTiles (prostorové vyhlazení)
    st_embs = create_super_tiles(group_df)
    
    # Pomocné funkce pro výpočet (předpokládám, že je máš v kódu definované)
    vlad_centroids = vlad_model.cluster_centers_
    
    results = {
        "slide_id": slide_id,
        # VLAD kombinace
        "vlad_raw": compute_soft_vlad(raw_embs, vlad_centroids),
        "vlad_super": compute_soft_vlad(st_embs, vlad_centroids),
        # Fisher kombinace
        "fisher_raw": compute_fisher_vector(raw_embs, gmm_model),
        "fisher_super": compute_fisher_vector(st_embs, gmm_model)
    }
    
    return pd.DataFrame([results])

def main():
    # TODO: change on musica
    INPUT_DIR = "../output/"
    SAMPLE_SIZE = 500000 
    vlad_clusters = 256
    fisher_clusters = 64

    ray.init(ignore_reinit_error=True)
    
    print(f"📂 Načítám data z {INPUT_DIR}...")
    ds = ray.data.read_parquet(INPUT_DIR)
    
    # --- KROK 1: TRÉNOVÁNÍ MODELŮ (CODEBOOKS) ---
    print(f"🧠 Trénuji modely na vzorku dat (pro VLAD i Fisher)...")
    sample_df = ds.random_sample(0.1).limit(SAMPLE_SIZE).to_pandas()
    train_embs = create_super_tiles(sample_df)
    
    # VLAD model (K-Means)
    vlad_model = MiniBatchKMeans(n_clusters=vlad_clusters, batch_size=2048, n_init=3)
    vlad_model.fit(train_embs)
    
    # Fisher model (GMM)
    gmm_model = GaussianMixture(n_components=fisher_clusters, covariance_type='diag', max_iter=50)
    gmm_model.fit(train_embs)
    
    # --- KROK 2: DISTRIBUOVANÁ AGREGACE ---
    print(f"🚀 Spouštím hromadnou agregaci (4 metody) pomocí map_groups...")
    
    # Spustíme výpočet pro všechny kombinace najednou
    final_ds = ds.groupby("slide_id").map_groups(
        lambda df: aggregate_all_methods(df, vlad_model, gmm_model)
    )
    
    print("⌛ Probíhá výpočet na klastru... (toto může trvat déle, počítáme 4 deskriptory)")
    final_data_df = final_ds.to_pandas()
    
    # --- KROK 3: ULOŽENÍ ---
    save_name = "descriptors_comparison_all.pkl"
    final_data_df.to_pickle(save_name)
    
    print(f"✅ Hotovo! Výsledky uloženy do {save_name}")
    print(f"Zpracováno slidů: {len(final_data_df)}")
    print(f"Dostupné sloupce: {list(final_data_df.columns)}")

if __name__ == "__main__":
    main()