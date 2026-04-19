import os, torch, gc, ray
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
import hnswlib 
import argparse

# Inicializace Ray - automaticky si vezme dostupné prostředky
# Vynucené vypnutí starého Raye, pokud existuje
if ray.is_initialized():
    ray.shutdown()

# Inicializace s čistým štítem
ray.init(num_cpus=16, ignore_reinit_error=True, include_dashboard=False)

# Malý trik: vyčištění paměti GPU hned na startu
torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.cuda.ipc_collect()
# --- 1. POMOCNÉ FUNKCE (Čtení a Sousedé na CPU) ---

def safe_read_parquet(path):
    table = pq.read_table(path)
    coords = np.column_stack([table.column('x_coord').to_numpy(), table.column('y_coord').to_numpy()]).astype(np.float32)
    embs_all = np.vstack(table.column('embedding').to_numpy()).astype(np.float32)
    if embs_all.ndim == 3: embs_all = embs_all.reshape(-1, embs_all.shape[-1])
    min_len = min(len(embs_all), len(coords))
    return coords[:min_len], embs_all[:min_len]

def create_super_tiles_cpu(coords, embs, k=10):
    if len(coords) < k: return embs
    N, D = coords.shape
    index = hnswlib.Index(space='l2', dim=D)
    index.init_index(max_elements=N, ef_construction=100, M=16)
    index.add_items(coords)
    indices, _ = index.knn_query(coords, k=k)
    return np.mean(embs[indices], axis=1).astype(np.float32)

# --- 2. RAY WORKERS (Paralelní části) ---

@ray.remote(num_cpus=2) # Načítání a HNSW na CPU
def load_and_prepare(f_path, out_dir):
    try:
        slide_id = os.path.basename(f_path).replace("slide_id=", "")
        out_path = Path(out_dir) / f"{slide_id.replace('/', '_')}.parquet"
        if out_path.exists(): return "Skip"

        # Načtení všech parquetů ve složce slidu
        p_files = list(Path(f_path).glob("*.parquet"))
        if not p_files: return None

        c_list, e_list = [], []
        for pf in p_files:
            c, e = safe_read_parquet(pf)
            c_list.append(c); e_list.append(e)
        
        coords = np.concatenate(c_list)
        data = np.concatenate(e_list)
        
        # Sousedé bleskově na CPU
        data_super = create_super_tiles_cpu(coords, data)
        return slide_id, data, data_super, out_path
    except Exception as e:
        return f"Error: {e}"

@ray.remote(num_gpus=1)
class GPUWorker:
    def __init__(self, gmm_params, vlad_centers):
        self.device = torch.device("cuda")
        m, c, p = gmm_params
        self.m = torch.from_numpy(m).to(self.device)
        self.c = torch.from_numpy(c).to(self.device) + 1e-6
        self.p = torch.from_numpy(p).to(self.device)
        self.vlad_centers = torch.from_numpy(vlad_centers).to(self.device)

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
                    for i in range(0, N, chunk_size):
                        X = X_full[i : i + chunk_size]
                        dists = torch.cdist(X, self.vlad_centers)
                        all_idx.append(torch.argmin(dists, dim=1))
                    
                    idx = torch.cat(all_idx)
                    oh = torch.nn.functional.one_hot(idx, num_classes=K_vlad).float()
                    
                    sum_x_vlad = torch.matmul(oh.t(), X_full)
                    counts = oh.sum(dim=0).unsqueeze(1)
                    V = sum_x_vlad - (counts * self.vlad_centers)
                    
                    v = V.flatten().cpu().numpy()
                    v = np.sign(v) * np.sqrt(np.abs(v))
                    res[f"vlad_{name}"] = v / (np.linalg.norm(v) + 1e-8)
                
                del X_full, oh, sum_x_vlad, V
                torch.cuda.empty_cache()

            pd.DataFrame([res]).to_parquet(out_path)
            return f"Done: {slide_id}"
            
        except Exception as e:
            # Pokud se stane chyba, zkusíme aspoň vyčistit VRAM pro další slide
            torch.cuda.empty_cache()
            return f"GPU Error {slide_id}: {str(e)}"
# --- 3. HLAVNÍ LOGIKA ---

def compute(input_dir, output_dir):
    print(f"\n--- RAY PIPELINE START: {input_dir} ---")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_folders = sorted([str(f) for f in Path(input_dir).iterdir() if f.is_dir() and f.name.startswith("slide_id=")])
    
    # TRÉNINK (CPU)
    sample_list = []
    for f in all_folders[:50]:
        p_files = list(Path(f).glob("*.parquet"))
        if p_files:
            _, e = safe_read_parquet(p_files[0])
            e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
            sample_list.append(e[np.random.choice(len(e), min(2000, len(e)), replace=False)])
    
    all_sample = np.concatenate(sample_list)
    v_m = MiniBatchKMeans(n_clusters=64, batch_size=1024, n_init=1).fit(all_sample)
    g = GaussianMixture(n_components=32, covariance_type='diag', max_iter=100).fit(all_sample)
    
    gmm_params = (g.means_.astype(np.float32), g.covariances_.astype(np.float32), g.weights_.astype(np.float32))
    vlad_centers = v_m.cluster_centers_.astype(np.float32)
    
    # Inicializace GPU herce
    worker = GPUWorker.remote(gmm_params, vlad_centers)

    # 1. Spustíme VŠECHNY loadery naráz (Ray si je bude dávkovat podle CPU jader)
    # Každý loader si vezme 2 CPU (podle @ray.remote(num_cpus=2))
    loader_futures = [load_and_prepare.remote(f, output_dir) for f in all_folders]

    # 2. Použijeme ray.wait, abychom brali to, co je zrovna hotové
    with tqdm(total=len(all_folders), desc="Pipeline") as pbar:
        while loader_futures:
            # Vezmeme slidy, které už CPU dožvýkalo (vratí jeden hotový a zbytek čekajících)
            ready_list, loader_futures = ray.wait(loader_futures, num_returns=1)
            
            result = ray.get(ready_list[0])
            if isinstance(result, tuple): # Úspěšně načteno a HNSW hotovo
                s_id, raw, sup, out_p = result
                # GPU teď dostane "čistou práci" bez čekání na disk
                # Tady ray.get() necháme, aby GPU jelo jeden po druhém a nepřeplnilo VRAM
                status = ray.get(worker.compute_slide.remote(s_id, raw, sup, out_p))
                print(status)
            else:
                print(result) # Může být "Skip" nebo chyba načítání

            pbar.update(1)

def main():
    parser = argparse.ArgumentParser(
        description="Creates tile and slide embeddings for WSI"
    )
    
    parser.add_argument(
        '--slide-path', 
        type=str, 
        required=True,
        help='Absolute path to WSI. can be an directory with WSIs.'
    )
    
    parser.add_argument(
        '--save-path', 
        type=str, 
        default='./',
        help='Path for saving parquet files, folder for each WSI will be created automatically.'
    )

    args = parser.parse_args()

    input_path = args.slide_path.rstrip("/")
    output_path = args.save_path.rstrip("/")
    compute(input_path, output_path)

if __name__ == "__main__":
    main()