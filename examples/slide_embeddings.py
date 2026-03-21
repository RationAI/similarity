
import os
# MUSÍ BÝT PŘED IMPORTEM NUMPY/SKLEARN
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["OPENBLAS_NUM_THREADS"] = "1"

import ray
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
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
    
    raw_col = table.column('embedding').to_pylist()
    embs_all = np.array(raw_col, dtype=np.float32)
    
    if embs_all.ndim == 3:
        embs_all = embs_all.reshape(-1, embs_all.shape[-1])

    # Vrátíme surový embedding a souřadnice
    # (Ořezání na CLS/Patches uděláme až při výpočtu)
    if len(embs_all) > len(coords):
        embs_all = embs_all[:len(coords)]
    elif len(embs_all) < len(coords):
        coords = coords[:len(embs_all)]
        
    return coords, embs_all

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

import numpy as np

def compute_fisher_variants(embeddings, means, covs, priors, suffix=""):
    N, D = embeddings.shape
    # Soft assignment
    # Místo původního resps výpočtu:
    dists = -0.5 * np.sum((embeddings[:, np.newaxis, :] - means)**2 / (covs + 1e-3), axis=2)
    # Stabilní výpočet pravděpodobností
    max_dists = np.max(dists, axis=1, keepdims=True)
    resps = np.exp(dists - max_dists) 
    resps /= (resps.sum(axis=1, keepdims=True) + 1e-12)
    resps_sum = resps.sum(axis=0)

    # u_k (středy)
    u_k = (np.dot(resps.T, embeddings) - resps_sum[:, np.newaxis] * means)
    u_k /= (N * np.sqrt(priors)[:, np.newaxis] + 1e-8)

    # v_k (rozptyl - s regulací 1e-3)
    v_k = np.zeros_like(u_k)
    for k in range(len(priors)):
        diff = embeddings - means[k]
        v_k[k] = np.dot(resps[:, k], (diff**2 / (covs[k] + 1e-3)) - 1.0)
    v_k /= (N * np.sqrt(2 * priors)[:, np.newaxis] + 1e-8)

    def finalize(v):
        v = np.sign(v) * np.sqrt(np.abs(v))
        norm = np.linalg.norm(v)
        return v / (norm + 1e-8) if norm > 1e-8 else v

    s = f"_{suffix}" if suffix else ""
    return {
        f"fisher_mean{s}": finalize(u_k.flatten()),
        f"fisher_var{s}": finalize(v_k.flatten()),
        f"fisher_robust{s}": finalize(np.concatenate([u_k.flatten(), v_k.flatten()]))
    }

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
def process_single_slide(f_path, gmm_refs_dict, vlad_refs_dict, out_dir):
    try:
        slide_id = os.path.basename(f_path).replace("slide_id=", "")
        out_path = Path(out_dir) / f"{slide_id.replace('/', '_')}.parquet"
        
        if out_path.exists(): 
            return slide_id

        # Změna na iterdir, aby to našlo soubory i bez přípony .parquet (časté u Ray Datasetů)
        p_files = [f for f in Path(f_path).iterdir() if f.is_file() and not f.name.startswith(".")]
        if not p_files:
            return f"Skipped {slide_id}: No files found"

        c_list, e_list = [], []
        for pf in p_files:
            c, e = safe_read_parquet(pf) 
            c_list.append(c)
            e_list.append(e)
        
        coords = np.concatenate(c_list)
        embs_full = np.concatenate(e_list)
        
        res = {"slide_id": slide_id}
        dim = embs_full.shape[1]

        # Rozdělení podle dimenze modelu
        if dim == 3072: # Midnight
            configs = {"cls": embs_full[:, :1536], "patch": embs_full[:, 1536:], "hybrid": embs_full}
        elif dim == 2560: # Virchow
            configs = {"cls": embs_full[:, :1280], "patch": embs_full[:, 1280:], "hybrid": embs_full}
        else: # UNI2-h / GigaPath (1536)
            configs = {"default": embs_full}

        for name, data in configs.items():
            st_data = create_super_tiles(coords, data)

            # --- FISHER (GMM) ---
            if name in gmm_refs_dict:
                # OPRAVA: Načtení referencí ze slovníku a jejich dereference přes ray.get
                m_ref, cov_ref, p_ref = gmm_refs_dict[name]
                m, cov, p = ray.get([m_ref, cov_ref, p_ref]) # Dereference najednou pro rychlost
                
                res.update(compute_fisher_variants(data, m, cov, p, suffix=name))
                res.update(compute_fisher_variants(st_data, m, cov, p, suffix=f"{name}_super"))

            # --- VLAD (KMeans) ---
            if name in vlad_refs_dict:
                # OPRAVA: Dereference i pro VLAD středy
                v_centers_ref = vlad_refs_dict[name]
                v_centers = ray.get(v_centers_ref)
                
                res[f"vlad_{name}"] = compute_soft_vlad(data, v_centers)
                res[f"vlad_{name}_super"] = compute_soft_vlad(st_data, v_centers)

        # Ukládání souboru
        if len(res) > 1:
            pd.DataFrame([res]).to_parquet(out_path)
            # Volitelně: print(f"✅ Uloženo: {out_path.name}") 
            return slide_id
        else:
            return f"Skipped {slide_id}: No data calculated"

    except Exception as e:
        # V Ray je lepší chybu vyhodit, aby se zobrazila v hlavním logu
        print(f"🔥 Error processing {f_path}: {e}")
        raise e

# --- 5. HLAVNÍ FUNKCE ---
def compute(input, output):
    # Změň cestu podle toho, co zrovna procesuješ (Virchow/UNI2/Midnight)
    INPUT_DIR = input
    OUTPUT_DIR = output
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    if not ray.is_initialized():
        ray.init(num_cpus=5, object_store_memory=50 * 1024**3)

    all_folders = sorted([str(f) for f in Path(INPUT_DIR).iterdir() if f.is_dir() and f.name.startswith("slide_id=")])
    
    # Sběr vzorků
    sample_list = []
    for f in all_folders[:50]: 
        p_files = list(Path(f).glob("*.parquet"))
        if p_files:
            _, e = safe_read_parquet(p_files[0])
            idx = np.random.choice(len(e), min(2000, len(e)), replace=False)
            sample_list.append(e[idx])
    
    all_sample = np.concatenate(sample_list)
    dim = all_sample.shape[1]
    half = dim // 2
    
    # Definice úloh pro trénink
    if dim > 2000:
        train_tasks = {"cls": all_sample[:, :half], "patch": all_sample[:, half:], "hybrid": all_sample}
    else:
        train_tasks = {"default": all_sample}

    gmm_refs = {}
    vlad_refs = {}

    for name, data in train_tasks.items():
        print(f"--- Trénuji Codebooky pro větev: {name} (dim {data.shape[1]}) ---")
        
        # 1. Trénink K-Means pro VLAD (64 klastrů, n_init=1 pro rychlost)
        vlad_m = MiniBatchKMeans(n_clusters=64, batch_size=4096, random_state=42, n_init=1).fit(data)
        vlad_refs[name] = ray.put(vlad_m.cluster_centers_.astype(np.float32))

        # 2. Trénink GMM pro Fishera (32 komponent, reg 1e-3)
        gmm = GaussianMixture(
            n_components=32, 
            covariance_type='diag',
            max_iter=100,
            reg_covar=1e-3,
            random_state=42,
            init_params='kmeans'
        ).fit(data)
        
        gmm_refs[name] = (
            ray.put(gmm.means_.astype(np.float32)),
            ray.put(gmm.covariances_.astype(np.float32)),
            ray.put(gmm.weights_.astype(np.float32))
        )

    del all_sample, train_tasks
    gc.collect()

    print(f"🚀 Startuji zpracování {len(all_folders)} slidů...")
    result_refs = [
        process_single_slide.remote(f_path, gmm_refs, vlad_refs, OUTPUT_DIR) 
        for f_path in all_folders
    ]

    results = []
    with tqdm(total=len(result_refs)) as pbar:
        while len(result_refs) > 0:
            done_refs, result_refs = ray.wait(result_refs, num_returns=1)
            results.append(ray.get(done_refs[0]))
            pbar.update(1)

    ray.shutdown()
    print("✨ Hotovo. Máš v Parquetech VLAD i Fisher pro všechny varianty.")

def main():
    # GIGAPATH
    #input_path = "/data/fs201053/jb88526/privagams_enc1_mpp10_rmbg"  # Např. "/data/virchow/slides"
    #output_path = "/data/fs201053/jb88526/privagams_enc1_mpp10_rmbg_slides_final"  # Např. "/data/virchow/embeddings"
    #compute(input_path, output_path)
#
    ##VIRCHOW
    #input_path = "/data/fs201053/jb88526/privagams_enc1_mpp10_rmbg_clahe"  # Např. "/data/virchow/slides"
    #output_path = "/data/fs201053/jb88526/privagams_enc1_mpp10_rmbg_clahe_slides_final"  # Např. "/data/virchow/embeddings"
    #compute(input_path, output_path)
    
    #UNI2-H
    input_path = "/data/fs201053/jb88526/privagams_enc1_mpp10_rmbg_norm"  # Např. "/data/virchow/slides"
    output_path = "/data/fs201053/jb88526/privagams_enc1_mpp10_rmbg_norm_slides_final"  # Např. "/data/virchow/embeddings"
    compute(input_path, output_path)

if __name__ == "__main__":
    main()