import ray
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import os
import random
import time
import gc
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KDTree
from pathlib import Path
from tqdm import tqdm
# --- 1. SKUTEČNĚ PAMĚŤOVĚ ŠETRNÉ NAČÍTÁNÍ ---
def safe_read_parquet(path):
    table = pq.read_table(path)
    coords = np.column_stack([
        table.column('x_coord').to_numpy(),
        table.column('y_coord').to_numpy()
    ]).astype(np.float32)
    
    # KLÍČOVÁ ZMĚNA: Žádné to_pylist(). 
    # Vytáhneme raw data z Arrow bufferu přímo do NumPy pole.
    # .values.to_numpy() u ListArray v Arrow vrátí zploštělé pole všech floatů.
    combined = table.column('embedding').combine_chunks()
    flattened_embs = combined.values.to_numpy()
    
    # Zjistíme dimenzi (pravděpodobně 768 nebo 1024)
    dim = len(flattened_embs) // len(coords)
    embs = flattened_embs.reshape(len(coords), dim).astype(np.float32)
    
    del table, combined, flattened_embs
    return coords, embs

def compute_soft_vlad(embeddings, centroids, sigma=1.0):
    """Vektorizovaný výpočet VLAD agregace."""
    if len(embeddings) == 0: 
        return np.zeros(centroids.shape[0] * centroids.shape[1], dtype=np.float32)
    
    # Efektivní výpočet vzdáleností přes dot product
    dots = np.dot(embeddings, centroids.T)
    emb_sq = np.sum(embeddings**2, axis=1, keepdims=True)
    cen_sq = np.sum(centroids**2, axis=1)
    dists_sq = emb_sq + cen_sq - 2 * dots
    
    # Soft-assignment váhy
    weights = np.exp(-sigma * dists_sq)
    weights /= (np.sum(weights, axis=1, keepdims=True) + 1e-12)
    
    # Výpočet reziduí (K, D) bez loopů
    V = np.dot(weights.T, embeddings) - (weights.sum(axis=0)[:, np.newaxis] * centroids)
    
    vlad_flat = V.flatten()
    # Power normalization
    vlad_flat = np.sign(vlad_flat) * np.sqrt(np.abs(vlad_flat))
    
    # L2 normalization
    norm = np.linalg.norm(vlad_flat)
    return (vlad_flat / (norm + 1e-6)).astype(np.float32)

def compute_fisher_vector(embeddings, means, covs, priors):
    """Vektorizovaný Fisher Vector bez 3D matic a loopů."""
    if len(embeddings) == 0: 
        return np.zeros(2 * means.shape[0] * means.shape[1], dtype=np.float32)
    
    N, D = embeddings.shape
    K = means.shape[0]
    inv_covs = 1.0 / (covs + 1e-6)
    
    # Výpočet responsibilit (N, K) přes log-likelihood trik
    # dists = (x-m)^2 / s = x^2/s - 2xm/s + m^2/s
    dots = np.dot(embeddings, (means * inv_covs).T)
    emb_sq = np.dot(embeddings**2, inv_covs.T)
    means_sq = np.sum(means**2 * inv_covs, axis=1)
    
    dists_sq = emb_sq - 2 * dots + means_sq
    resps = np.exp(-0.5 * dists_sq) * priors / (np.sqrt(np.prod(covs, axis=1)) + 1e-6)
    resps /= (resps.sum(axis=1, keepdims=True) + 1e-12)
    
    # Gradienty u_k a v_k (vše vektorizovaně)
    resps_sum = resps.sum(axis=0)[:, np.newaxis]
    
    # u_k: První řád (středy)
    u_k = (np.dot(resps.T, embeddings) - resps_sum * means) 
    u_k /= (N * np.sqrt(priors)[:, np.newaxis] + 1e-6)
    
    # v_k: Druhý řád (rozptyly)
    # v_k = sum(gamma * [(x-mu)^2 / sigma - 1])
    # (x-mu)^2 = x^2 - 2x*mu + mu^2
    term2 = np.dot(resps.T, embeddings**2) - 2 * means * np.dot(resps.T, embeddings) + resps_sum * means**2
    v_k = (term2 * inv_covs - resps_sum) / (N * np.sqrt(2 * priors)[:, np.newaxis] + 1e-6)
    
    fv = np.concatenate([u_k.flatten(), v_k.flatten()])
    # Power + L2 normalization
    fv = np.sign(fv) * np.sqrt(np.abs(fv))
    return (fv / (np.linalg.norm(fv) + 1e-6)).astype(np.float32)

def create_super_tiles(coords, embs, r=336):
    """Bleskový výpočet super-tiles pomocí NumPy indexování."""
    if len(coords) < 2: return embs
    tree = KDTree(coords)
    
    # Najdeme indexy 10 nejbližších sousedů pro všechny body najednou
    _, indices = tree.query(coords, k=10)
    
    # embs[indices] vytvoří matici (N, 10, D)
    # np.mean přes osu 1 spočítá průměr těch 10 sousedů pro každou dlaždici
    return np.mean(embs[indices], axis=1).astype(np.float32)

@ray.remote
def process_single_slide(f_path, v_c, f_m, f_c, f_p, out_dir):
    """Worker funkce pro paralelní běh."""
    try:
        slide_id = os.path.basename(f_path).replace("slide_id=", "")
        out_path = Path(out_dir) / f"{slide_id.replace('/', '_')}.parquet"
        
        # Přeskočit, pokud už hotovo
        if out_path.exists(): 
            return slide_id

        p_files = list(Path(f_path).glob("*.parquet"))
        c_list, e_list = [], []
        for pf in p_files:
            c, e = safe_read_parquet(pf)
            c_list.append(c)
            e_list.append(e)
        
        coords = np.concatenate(c_list)
        embs = np.concatenate(e_list)
        
        # Výpočty (tvůj super-rychlý kód)
        st_embs = create_super_tiles(coords, embs)
        
        res = {
            "slide_id": slide_id,
            "vlad_raw": compute_soft_vlad(embs, v_c),
            "vlad_super": compute_soft_vlad(st_embs, v_c),
            "fisher_raw": compute_fisher_vector(embs, f_m, f_c, f_p),
            "fisher_super": compute_fisher_vector(st_embs, f_m, f_c, f_p)
        }
        
        pd.DataFrame([res]).to_parquet(out_path)
        return slide_id
    except Exception as e:
        return f"Error {slide_id}: {e}"

def main():
    INPUT_DIR = "/data/fs201053/jb88526/privagams_enc2_mpp05_enhanced"
    OUTPUT_DIR = "/data/fs201053/jb88526/privagams_enc2_mpp05_enhanced_slide"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Inicializace Ray - omezíme na 20 CPU (nebo kolik máš alokováno)
    if not ray.is_initialized():
        ray.init(num_cpus=20)

    all_folders = sorted([str(f) for f in Path(INPUT_DIR).iterdir() if f.is_dir() and f.name.startswith("slide_id=")])
    
    # 1. TRÉNOVÁNÍ (Sekvenční, jen na začátku)
    print("🧠 Trénuji codebooky...")
    sample_folders = random.sample(all_folders, min(25, len(all_folders)))
    sample_list = [safe_read_parquet(list(Path(f).glob("*.parquet"))[0])[1][:2000] for f in sample_folders if list(Path(f).glob("*.parquet"))]
    all_sample = np.concatenate(sample_list)
    
    vlad_m = MiniBatchKMeans(n_clusters=256, batch_size=4096).fit(all_sample)
    gmm_m = GaussianMixture(n_components=64, covariance_type='diag').fit(all_sample)
    
    # 2. PŘÍPRAVA PRO SDÍLENOU PAMĚŤ (ray.put)
    v_c_ref = ray.put(vlad_m.cluster_centers_.astype(np.float32))
    f_m_ref = ray.put(gmm_m.means_.astype(np.float32))
    f_c_ref = ray.put(gmm_m.covariances_.astype(np.float32))
    f_p_ref = ray.put(gmm_m.weights_.astype(np.float32))
    
    del all_sample, sample_list
    gc.collect()

    # 3. PARALELNÍ SPOUŠTĚNÍ
    print(f"🚀 Startuji paralelní zpracování {len(all_folders)} slidů...")
    
    # Vytvoříme seznam úkolů
    result_refs = [
        process_single_slide.remote(f_path, v_c_ref, f_m_ref, f_c_ref, f_p_ref, OUTPUT_DIR) 
        for f_path in all_folders
    ]

    # Sledujeme progress pomocí tqdm
    results = []
    with tqdm(total=len(result_refs)) as pbar:
        while len(result_refs) > 0:
            done_refs, result_refs = ray.wait(result_refs, num_returns=1)
            results.append(ray.get(done_refs[0]))
            pbar.update(1)

    print(f"✨ Vše hotovo. Výsledky v: {OUTPUT_DIR}")
    ray.shutdown()

if __name__ == "__main__":
    main()