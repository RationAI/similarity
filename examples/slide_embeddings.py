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

# --- FIXACE SEEDU PRO REPRODUKOVATELNOST ---
random.seed(42)
np.random.seed(42)

# --- 1. ŠETRNÉ NAČÍTÁNÍ ---
def safe_read_parquet(path):
    table = pq.read_table(path)
    coords = np.column_stack([
        table.column('x_coord').to_numpy(),
        table.column('y_coord').to_numpy()
    ]).astype(np.float32)
    
    combined = table.column('embedding').combine_chunks()
    flattened_embs = combined.values.to_numpy()
    
    dim = len(flattened_embs) // len(coords)
    embs = flattened_embs.reshape(len(coords), dim).astype(np.float32)
    
    del table, combined, flattened_embs
    return coords, embs

def compute_soft_vlad(embeddings, centroids, sigma=1.0):
    if len(embeddings) == 0: 
        return np.zeros(centroids.shape[0] * centroids.shape[1], dtype=np.float32)
    
    # Vzdálenosti k centrům
    dots = np.dot(embeddings, centroids.T)
    emb_sq = np.sum(embeddings**2, axis=1, keepdims=True)
    cen_sq = np.sum(centroids**2, axis=1)
    dists_sq = np.maximum(emb_sq + cen_sq - 2 * dots, 0)
    
    # Klasický Soft-assignment
    weights = np.exp(-sigma * dists_sq)
    weights /= (np.sum(weights, axis=1, keepdims=True) + 1e-12)
    
    # Rezidua (rozdíl mezi dlaždicí a klastrem)
    V = np.dot(weights.T, embeddings) - (weights.sum(axis=0)[:, np.newaxis] * centroids)
    
    vlad_flat = V.flatten()
    vlad_flat = np.sign(vlad_flat) * np.sqrt(np.abs(vlad_flat)) # Power norm
    
    norm = np.linalg.norm(vlad_flat)
    return (vlad_flat / (norm + 1e-6)).astype(np.float32)

# --- 3. RYCHLÝ FISHER VECTOR ---
def compute_fisher_vector(embeddings, means, covs, priors):
    if len(embeddings) == 0: 
        return np.zeros(2 * means.shape[0] * means.shape[1], dtype=np.float32)
    
    N, D = embeddings.shape
    inv_covs = 1.0 / (covs + 1e-6)
    
    dots = np.dot(embeddings, (means * inv_covs).T)
    emb_sq = np.dot(embeddings**2, inv_covs.T)
    means_sq = np.sum(means**2 * inv_covs, axis=1)
    
    dists_sq = np.maximum(emb_sq - 2 * dots + means_sq, 0)
    resps = np.exp(-0.5 * dists_sq) * priors / (np.sqrt(np.prod(covs, axis=1)) + 1e-6)
    resps /= (resps.sum(axis=1, keepdims=True) + 1e-12)
    
    resps_sum = resps.sum(axis=0)[:, np.newaxis]
    u_k = (np.dot(resps.T, embeddings) - resps_sum * means) / (N * np.sqrt(priors)[:, np.newaxis] + 1e-6)
    
    term2 = np.dot(resps.T, embeddings**2) - 2 * means * np.dot(resps.T, embeddings) + resps_sum * means**2
    v_k = (term2 * inv_covs - resps_sum) / (N * np.sqrt(2 * priors)[:, np.newaxis] + 1e-6)
    
    fv = np.concatenate([u_k.flatten(), v_k.flatten()])
    fv = np.sign(fv) * np.sqrt(np.abs(fv))
    return (fv / (np.linalg.norm(fv) + 1e-6)).astype(np.float32)

# --- 4. SUPER-TILES (RADIUS) ---
def create_super_tiles(coords, embs, k=10):
    if len(coords) < k: return embs
    tree = KDTree(coords)
    _, indices = tree.query(coords, k=k)
    return np.mean(embs[indices], axis=1).astype(np.float32)

@ray.remote
def process_single_slide(f_path, v_c, f_m, f_c, f_p, out_dir):
    try:
        slide_id = os.path.basename(f_path).replace("slide_id=", "")
        out_path = Path(out_dir) / f"{slide_id.replace('/', '_')}.parquet"
        
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
        
        # Výpočet super-tiles
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
        return f"Error {slide_id}: {str(e)}"

# --- 5. HLAVNÍ FUNKCE ---
def main():
    INPUT_DIR = "/data/fs201053/jb88526/privagams_enc2_mpp05_enhanced"
    OUTPUT_DIR = "/data/fs201053/jb88526/privagams_enc2_mpp05_enhanced_slide_v3" # Nový adresář pro v3
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    if not ray.is_initialized():
        ray.init(num_cpus=20)

    all_folders = sorted([str(f) for f in Path(INPUT_DIR).iterdir() if f.is_dir() and f.name.startswith("slide_id=")])
    
    print("🧠 Trénuji stabilní codebooky...")
    sample_folders = all_folders[::5] 
    sample_list = []
    for f in sample_folders:
        p_files = list(Path(f).glob("*.parquet"))
        if p_files:
            _, e = safe_read_parquet(p_files[0])
            # Bereme jen 500 náhodných dlaždic z každého slidu, ať je to pestré
            idx = np.random.choice(len(e), min(500, len(e)), replace=False)
            sample_list.append(e[idx])
    
    all_sample = np.concatenate(sample_list)
    
    # 64 klastrů je pro tuhle úlohu "sweet spot"
    vlad_m = MiniBatchKMeans(n_clusters=64, batch_size=4096, random_state=42, n_init=3).fit(all_sample)
    gmm_m = GaussianMixture(n_components=32, covariance_type='diag', random_state=42).fit(all_sample)

    # Sdílení v Ray paměti
    v_c_ref = ray.put(vlad_m.cluster_centers_.astype(np.float32))
    f_m_ref = ray.put(gmm_m.means_.astype(np.float32))
    f_c_ref = ray.put(gmm_m.covariances_.astype(np.float32))
    f_p_ref = ray.put(gmm_m.weights_.astype(np.float32))
    
    del all_sample, sample_list
    gc.collect()

    print(f"🚀 Startuji paralelní zpracování {len(all_folders)} slidů...")
    result_refs = [
        process_single_slide.remote(f_path, v_c_ref, f_m_ref, f_c_ref, f_p_ref, OUTPUT_DIR) 
        for f_path in all_folders
    ]

    results = []
    with tqdm(total=len(result_refs)) as pbar:
        while len(result_refs) > 0:
            done_refs, result_refs = ray.wait(result_refs, num_returns=1)
            results.append(ray.get(done_refs[0]))
            pbar.update(1)

    print(f"✨ Hotovo. Výsledky v: {OUTPUT_DIR}")
    ray.shutdown()

if __name__ == "__main__":
    main()