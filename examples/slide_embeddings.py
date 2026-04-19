import os
import torch
import gc
import ray
import random
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import hnswlib

# --- 1. KONFIGURACE ---
MASTER_SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# --- 2. PROSTOROVÁ AGREGACE (CPU) ---
def create_hierarchical_super_tiles(coords, embs, k_local=10, k_global=50):
    if len(coords) < k_global: return embs
    N, D = coords.shape
    index = hnswlib.Index(space='l2', dim=coords.shape[1])
    index.init_index(max_elements=N, ef_construction=100, M=16)
    index.add_items(coords)
    
    # Lokální (detail)
    idx_l, dist_l = index.knn_query(coords, k=k_local)
    w_l = np.exp(-dist_l / (2 * np.mean(dist_l) + 1e-8))
    w_l /= np.sum(w_l, axis=1, keepdims=True)
    loc_e = np.sum(embs[idx_l] * w_l[:, :, np.newaxis], axis=1)
    
    # Globální (architektura)
    idx_g, dist_g = index.knn_query(coords, k=k_global)
    w_g = np.exp(-dist_g / (2 * np.mean(dist_g) + 1e-8))
    w_g /= np.sum(w_g, axis=1, keepdims=True)
    glob_e = np.sum(embs[idx_g] * w_g[:, :, np.newaxis], axis=1)
    
    return (0.4 * loc_e + 0.6 * glob_e).astype(np.float32)

@ray.remote(num_cpus=2)
def load_and_prepare_worker(f_path):
    try:
        slide_id = os.path.basename(f_path).replace("slide_id=", "")
        p_files = list(Path(f_path).glob("*.parquet"))
        if not p_files: return None
        c_l, e_l = [], []
        for pf in p_files:
            t = pq.read_table(pf)
            c_l.append(np.column_stack([t.column('x_coord').to_numpy(), t.column('y_coord').to_numpy()]).astype(np.float32))
            e_l.append(np.vstack(t.column('embedding').to_numpy()).astype(np.float32))
        c, e = np.concatenate(c_l), np.concatenate(e_l)
        return slide_id, e, create_hierarchical_super_tiles(c, e)
    except Exception as e: return f"Error: {e}"

# --- 3. KOMPLEXNÍ GPU AGREGÁTOR ---
@ray.remote(num_gpus=1)
class FullGPUWorker:
    def __init__(self, pca_comps, pca_mean, gmm_params, vlad_centers):
        self.device = torch.device("cuda")
        self.pca_comps = torch.from_numpy(pca_comps).to(self.device)
        self.pca_mean = torch.from_numpy(pca_mean).to(self.device)
        m, c, p = gmm_params
        self.gmm_m = torch.from_numpy(m).to(self.device)
        self.gmm_c = torch.from_numpy(c).to(self.device) + 1e-6
        self.vlad_c = torch.from_numpy(vlad_centers).to(self.device)

    def compute_slide(self, slide_id, data_raw, data_super, out_path):
        try:
            res = {"slide_id": slide_id}
            chunk_size = 1000  
            
            for name, data in [("default", data_raw), ("default_super", data_super)]:
                # Převod na tensor a normalizace
                X_full = torch.from_numpy(data).to(self.device)
                X_full = X_full / (torch.norm(X_full, dim=1, keepdim=True) + 1e-8)
                N, D = X_full.shape
                K_gmm = self.m.shape[0]
                K_vlad = self.vlad_centers.shape[0]

                # --- 1. MEAN POOLING ---
                with torch.no_grad():
                    mean_emb = torch.mean(X_full, dim=0).cpu().numpy()
                    res[f"mean_{name}"] = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

                # --- 2. FISHER VECTOR (Chunked) ---
                sum_resp = torch.zeros(K_gmm, device=self.device)
                sum_u_k = torch.zeros((K_gmm, D), device=self.device)
                
                with torch.no_grad():
                    for i in range(0, N, chunk_size):
                        X = X_full[i : i + chunk_size]
                        diff = X.unsqueeze(1) - self.m.unsqueeze(0) 
                        log_exps = -0.5 * torch.sum(diff**2 / self.c, dim=2)
                        resps = torch.softmax(log_exps, dim=1) 
                        
                        sum_resp += resps.sum(dim=0)
                        sum_u_k += torch.matmul(resps.t(), X)
                        del diff, log_exps, resps 

                    u_k = sum_u_k - (sum_resp.unsqueeze(1) * self.m)
                    fv = u_k.flatten().cpu().numpy()
                    fv = np.sign(fv) * np.sqrt(np.abs(fv))
                    res[f"fisher_{name}"] = fv / (np.linalg.norm(fv) + 1e-8)

                # --- 3. VLAD (Chunked) ---
                all_idx = []
                with torch.no_grad():
                    salience = torch.norm(X_norm - torch.mean(X_norm, dim=0), dim=1)
                    weights = torch.softmax(salience / 0.02, dim=0)
                    emb = torch.sum(X_norm * weights.unsqueeze(1), dim=0).cpu().numpy()
                    res[f"ultimate_boost_{name}"] = emb / (np.linalg.norm(emb) + 1e-8)

                # 3. VLAD
                with torch.no_grad():
                    dists = torch.cdist(X_norm, self.vlad_c)
                    idx = torch.argmin(dists, dim=1)
                    oh = torch.nn.functional.one_hot(idx, num_classes=self.vlad_c.shape[0]).float()
                    V = torch.matmul(oh.t(), X_norm) - (oh.sum(dim=0).unsqueeze(1) * self.vlad_c)
                    v = V.flatten().cpu().numpy()
                    v = np.sign(v) * np.sqrt(np.abs(v))
                    res[f"vlad_{name}"] = v / (np.linalg.norm(v) + 1e-8)
                
                del X_full, oh, sum_x_vlad, V
                torch.cuda.empty_cache()

            pd.DataFrame([res]).to_parquet(out_path)
            return True
        except Exception as e: return str(e)

# --- 4. MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide-path', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)
    args = parser.parse_args()
    set_seed(MASTER_SEED)
    ray.init(num_cpus=16)

    all_folders = sorted([str(f) for f in Path(args.slide_path).iterdir() if f.is_dir() and f.name.startswith("slide_id=")])
    
    print("--- Fáze 1: Načítání a Hierarchický Pooling ---")
    loaded_data = [r for r in ray.get([load_and_prepare_worker.remote(f) for f in all_folders]) if isinstance(r, tuple)]

    print("--- Fáze 2: Trénink slovníků (PCA, GMM, KMeans) ---")
    sample = np.concatenate([d[1][np.random.choice(len(d[1]), min(500, len(d[1])), replace=False)] for d in loaded_data[:40]])
    pca = PCA(n_components=128, random_state=MASTER_SEED).fit(sample)
    sample_pca = pca.transform(sample)
    sample_pca /= (np.linalg.norm(sample_pca, axis=1, keepdims=True) + 1e-8)
    
    vlad_centers = MiniBatchKMeans(n_clusters=64, n_init=1, random_state=MASTER_SEED).fit(sample_pca).cluster_centers_
    gmm = GaussianMixture(n_components=32, covariance_type='diag', random_state=MASTER_SEED).fit(sample_pca)
    gmm_params = (gmm.means_.astype(np.float32), gmm.covariances_.astype(np.float32), gmm.weights_.astype(np.float32))

    print("--- Fáze 3: GPU Produkce všech metod ---")
    worker = FullGPUWorker.remote(pca.components_.astype(np.float32), pca.mean_.astype(np.float32), gmm_params, vlad_centers.astype(np.float32))
    
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    for s_id, raw, sup in tqdm(loaded_data):
        ray.get(worker.compute_slide.remote(s_id, raw, sup, save_dir / f"{s_id}.parquet"))

if __name__ == "__main__":
    main()