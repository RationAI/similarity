import argparse
import gc
import os
import random
from pathlib import Path

import hnswlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ray
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def create_hierarchical_super_tiles(
    coords, embs, k_local=10, k_global=50, batch_size=5000
):
    N, D = embs.shape
    if N < k_global:
        return embs.astype(np.float32)

    index = hnswlib.Index(space="l2", dim=coords.shape[1])
    index.init_index(max_elements=N, ef_construction=100, M=16)
    index.add_items(coords)

    out_embs = np.zeros((N, D), dtype=np.float32)

    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        b_coords = coords[i:end]

        idx_l, dist_l = index.knn_query(b_coords, k=k_local)
        w_l = np.exp(-dist_l / (2 * (np.mean(dist_l) + 1e-8)))
        w_l /= np.sum(w_l, axis=1, keepdims=True) + 1e-8
        loc_e = np.array(
            [
                np.sum(embs[idx_l[j]] * w_l[j, :, np.newaxis], axis=0)
                for j in range(len(b_coords))
            ]
        )

        idx_g, dist_g = index.knn_query(b_coords, k=k_global)
        w_g = np.exp(-dist_g / (2 * (np.mean(dist_g) + 1e-8)))
        w_g /= np.sum(w_g, axis=1, keepdims=True) + 1e-8
        glob_e = np.array(
            [
                np.sum(embs[idx_g[j]] * w_g[j, :, np.newaxis], axis=0)
                for j in range(len(b_coords))
            ]
        )

        out_embs[i:end] = 0.4 * loc_e + 0.6 * glob_e

    del index
    return out_embs


@ray.remote(num_cpus=2)
def load_and_prepare_worker(f_path, tmp_dir):
    try:
        slide_id = os.path.basename(f_path).replace("slide_id=", "")
        save_path = Path(tmp_dir) / f"{slide_id}_prepared.parquet"

        if save_path.exists():
            try:
                pq.read_metadata(save_path)
                return slide_id, str(save_path)
            except:
                save_path.unlink()

        p_files = list(Path(f_path).glob("*.parquet"))
        if not p_files:
            return None

        c_l, e_l = [], []
        for pf in p_files:
            t = pq.read_table(pf)
            c_l.append(
                np.column_stack(
                    [t.column("x_coord").to_numpy(), t.column("y_coord").to_numpy()]
                ).astype(np.float32)
            )
            e_batch = np.vstack(t.column("embedding").to_numpy()).astype(np.float32)
            e_l.append(e_batch)

        c = np.concatenate(c_l)
        e = np.concatenate(e_l)

        super_e = create_hierarchical_super_tiles(c, e)

        dim = e.shape[1]
        table = pa.table(
            {
                "embedding": pa.FixedSizeListArray.from_arrays(e.flatten(), dim),
                "super_embedding": pa.FixedSizeListArray.from_arrays(
                    super_e.flatten(), dim
                ),
            }
        )
        pq.write_table(table, save_path)

        del c, e, super_e, table
        gc.collect()

        return slide_id, str(save_path)
    except Exception as ex:
        return f"Error v {f_path}: {ex}"


@ray.remote(num_gpus=1)
class FullGPUWorker:
    def __init__(self, pca_comps, pca_mean, gmm_params, vlad_centers):
        self.device = torch.device("cuda")
        self.pca_comps = torch.from_numpy(pca_comps).to(self.device)
        self.pca_mean = torch.from_numpy(pca_mean).to(self.device)
        m, c, p = gmm_params
        self.gmm_m = torch.from_numpy(m).to(self.device)
        self.gmm_c = torch.from_numpy(c).to(self.device)
        self.vlad_c = torch.from_numpy(vlad_centers).to(self.device)

    def compute_slide(self, slide_id, p_path, out_path):
        try:
            table = pq.read_table(p_path)
            raw_np = np.vstack(table.column("embedding").to_numpy()).astype(np.float32)
            sup_np = np.vstack(table.column("super_embedding").to_numpy()).astype(
                np.float32
            )

            res = {"slide_id": slide_id}
            for name, data in [("default", raw_np), ("default_super", sup_np)]:
                X = torch.from_numpy(data).to(self.device)
                X_pca = torch.matmul(X - self.pca_mean, self.pca_comps.t())
                X_norm = X_pca / (torch.norm(X_pca, dim=1, keepdim=True) + 1e-8)

                # Mean Pooling
                mean_p = torch.mean(X_norm, dim=0).cpu().numpy()
                res[f"mean_{name}"] = mean_p / (np.linalg.norm(mean_p) + 1e-8)

                # Fisher Vector (GMM)
                diff = X_norm.unsqueeze(1) - self.gmm_m.unsqueeze(0)
                log_exps = -0.5 * torch.sum(diff**2 / (self.gmm_c + 1e-6), dim=2)
                resps = torch.softmax(log_exps, dim=1)
                u_k = torch.matmul(resps.t(), X_norm) - (
                    resps.sum(dim=0).unsqueeze(1) * self.gmm_m
                )
                fv = u_k.flatten().cpu().numpy()
                fv = np.sign(fv) * np.sqrt(np.abs(fv))
                res[f"fisher_{name}"] = fv / (np.linalg.norm(fv) + 1e-8)

                # VLAD
                dists = torch.cdist(X_norm, self.vlad_c)
                idx = torch.argmin(dists, dim=1)
                oh = torch.nn.functional.one_hot(
                    idx, num_classes=self.vlad_c.shape[0]
                ).float()
                V = torch.matmul(oh.t(), X_norm) - (
                    oh.sum(dim=0).unsqueeze(1) * self.vlad_c
                )
                v = V.flatten().cpu().numpy()
                v = np.sign(v) * np.sqrt(np.abs(v))
                res[f"vlad_{name}"] = v / (np.linalg.norm(v) + 1e-8)

                del X, X_pca, X_norm
                torch.cuda.empty_cache()

            pd.DataFrame([res]).to_parquet(out_path)
            return True
        except Exception as e:
            return str(e)


# --- 3. MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--num-runs", type=int, default=20)
    args = parser.parse_args()

    tmp_dir = Path(args.save_path) / "_tmp_prepared_embeddings"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ray.init(num_cpus=44, object_store_memory=10**9 * 15)

    all_folders = sorted(
        [
            str(f)
            for f in Path(args.slide_path).iterdir()
            if f.is_dir() and f.name.startswith("slide_id=")
        ]
    )

    prepared_paths = []
    pending = []
    f_iter = iter(all_folders)

    MAX_PREPARE_CONCURRENT = 128

    for _ in range(min(MAX_PREPARE_CONCURRENT, len(all_folders))):
        pending.append(load_and_prepare_worker.remote(next(f_iter), tmp_dir))

    with tqdm(total=len(all_folders), desc="Příprava dat") as pbar:
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            for ref in done:
                res = ray.get(ref)
                if isinstance(res, tuple):
                    prepared_paths.append(res)
                else:
                    print(f"\Error: {res}")
                pbar.update(1)
                try:
                    pending.append(
                        load_and_prepare_worker.remote(next(f_iter), tmp_dir)
                    )
                except StopIteration:
                    pass

    for run_idx in range(args.num_runs):
        current_seed = random.randint(0, 2**32 - 1)
        print(f"\n>>> Spouštíme RUN {run_idx+1}/{args.num_runs} s náhodným seedem {current_seed}")
        set_seed(current_seed)
        run_save_dir = Path(args.save_path) / f"run_seed_{current_seed}"
        run_save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> RUN {run_idx + 1} (Seed: {current_seed})")

        # training PCA/GMM
        sample_list = []
        for _, p_path in prepared_paths[:40]:
            t = pq.read_table(p_path, columns=["embedding"])
            e_s = np.vstack(t.column("embedding").to_numpy())
            idx = np.random.choice(len(e_s), min(500, len(e_s)), replace=False)
            sample_list.append(e_s[idx])

        sample = np.concatenate(sample_list)
        pca = PCA(n_components=128, random_state=current_seed).fit(sample)
        s_pca = pca.transform(sample)
        s_pca /= np.linalg.norm(s_pca, axis=1, keepdims=True) + 1e-8

        vlad_c = (
            MiniBatchKMeans(n_clusters=64, n_init=1, random_state=current_seed)
            .fit(s_pca)
            .cluster_centers_
        )
        gmm = GaussianMixture(
            n_components=32, covariance_type="diag", random_state=current_seed
        ).fit(s_pca)
        gmm_p = (
            gmm.means_.astype(np.float32),
            gmm.covariances_.astype(np.float32),
            gmm.weights_.astype(np.float32),
        )

        worker = FullGPUWorker.remote(
            pca.components_.astype(np.float32),
            pca.mean_.astype(np.float32),
            gmm_p,
            vlad_c.astype(np.float32),
        )

        for s_id, p_path in tqdm(prepared_paths, desc=f"Inference Seed {current_seed}"):
            out_file = run_save_dir / f"{s_id}.parquet"
            if not out_file.exists():
                success = ray.get(worker.compute_slide.remote(s_id, p_path, out_file))
                if success is not True:
                    print(f"Error {s_id}: {success}")

        ray.kill(worker)
        gc.collect()


if __name__ == "__main__":
    main()
