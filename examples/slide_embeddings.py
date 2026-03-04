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
    weights /= weights.sum(axis=1, keepdims=True)
    
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

# --- 3. RAY ACTOR PRO PARALELNÍ AGREGACI ---

@ray.remote
class SlideAggregator:
    def __init__(self, method, model):
        self.method = method
        self.model = model

    def aggregate(self, slide_id, df):
        # Nejdřív vytvoříme SuperTiles
        st_embeddings = create_super_tiles(df)
        
        if self.method == "soft_vlad":
            # Pro VLAD používáme jen středy (centroids)
            centroids = self.model.means_ if hasattr(self.model, 'means_') else self.model.cluster_centers_
            vector = compute_soft_vlad(st_embeddings, centroids)
        else:
            vector = compute_fisher_vector(st_embeddings, self.model)
            
        return {"slide_id": slide_id, "descriptor": vector}

# --- 4. MAIN PIPELINE ---

def main():
    # Nastavení
    INPUT_DIR = "../output/"
    METHOD = "soft_vlad"  # "soft_vlad" nebo "fisher"
    NUM_CLUSTERS = 16 if METHOD == "fisher" else 64
    SAMPLE_SIZE = 50000   # Kolik dlaždic použít pro trénink "slovníku"
    
    ray.init(ignore_reinit_error=True)
    
    print(f"📂 Načítám data z {INPUT_DIR}...")
    ds = ray.data.read_parquet(INPUT_DIR)
    
    # --- KROK 1: TRÉNOVÁNÍ MODELU (CODEBOOK) ---
    print(f"🧠 Trénuji {METHOD.upper()} model na vzorku dat...")
    # Vezmeme náhodný vzorek dlaždic ze všech slidů
    sample_ds = ds.random_sample(0.1).limit(SAMPLE_SIZE)
    sample_df = sample_ds.to_pandas()
    
    # Vytvoříme SuperTiles i pro tréninkový vzorek, aby model znal "vyhlazená" data
    train_embeddings = create_super_tiles(sample_df)
    
    if METHOD == "soft_vlad":
        model = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, batch_size=2048, n_init=3)
    else:
        model = GaussianMixture(n_components=NUM_CLUSTERS, covariance_type='diag', max_iter=50)
    
    model.fit(train_embeddings)
    
    # --- KROK 2: DISTRIBUOVANÁ AGREGACE ---
    print(f"🚀 Spouštím agregaci po slidech...")
    
    # Vytvoříme pool aktorů
    aggregator = SlideAggregator.remote(METHOD, model)
    
    results = []
    # Groupby rozdělí dataset podle slide_id a zpracuje každý slide zvlášť
    for slide_id, slide_ds in ds.groupby("slide_id"):
        # Převedeme jeden slide na pandas (vejde se do RAM)
        slide_df = slide_ds.to_pandas()
        # Pošleme aktorovi ke zpracování
        results.append(aggregator.aggregate.remote(slide_id, slide_df))
    
    # Počkáme na všechny výsledky
    final_data = ray.get(results)
    
    # --- KROK 3: ULOŽENÍ ---
    output_df = pd.DataFrame(final_data)
    save_name = f"descriptors_{METHOD}_k{NUM_CLUSTERS}.pkl"
    output_df.to_pickle(save_name)
    
    print(f"✅ Hotovo! Výsledky uloženy do {save_name}")
    print(f"Dimenze výsledného vektoru: {len(final_data[0]['descriptor'])}")

if __name__ == "__main__":
    main()