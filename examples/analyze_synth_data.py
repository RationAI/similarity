import os
import glob
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from pathlib import Path

def load_data_for_pair(root_path, encoder, variant):
    search_pattern = f"**/orig_embeddings_{encoder}_{variant}.parquet"
    orig_files = glob.glob(os.path.join(root_path, search_pattern), recursive=True)
    
    patient_data = {}
    for f_orig in orig_files:
        f_gen = f_orig.replace("orig_embeddings", "gen_embeddings")
        if not os.path.exists(f_gen): continue
        
        p_id = os.path.basename(os.path.dirname(os.path.dirname(f_orig)))
        
        try:
            # Načtení a okamžitá konverze na float32 a normalizace (šetří čas později)
            o_embs = np.vstack(pd.read_parquet(f_orig)['embedding'].values).astype('float32')
            g_embs = np.vstack(pd.read_parquet(f_gen)['embedding'].values).astype('float32')
            
            faiss.normalize_L2(o_embs)
            faiss.normalize_L2(g_embs)
            
            patient_data[p_id] = {'orig': o_embs, 'gen': g_embs}
        except: continue
    return patient_data

def evaluate_attack_fast(orig_embs, gen_embs, gen_pids, orig_pids):
    """Vektorizované vyhodnocení útoku."""
    d = orig_embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(orig_embs)
    
    # Vyhledáme 5 nejbližších sousedů
    _, indices = index.search(gen_embs, 5)
    
    # Převedeme pids na numpy pole pro rychlé indexování
    orig_pids_arr = np.array(orig_pids)
    gen_pids_arr = np.array(gen_pids)
    
    # Získáme matici predikovaných ID (každý řádek jsou ID 5 sousedů)
    pred_pids = orig_pids_arr[indices]
    
    # Top-1: Shoda na první pozici
    t1_hits = np.sum(pred_pids[:, 0] == gen_pids_arr)
    
    # Top-5: Shoda v kterémkoliv z 5 sloupců
    t5_hits = np.sum(np.any(pred_pids == gen_pids_arr[:, None], axis=1))
    
    n = len(gen_embs)
    return (t1_hits / n) * 100, (t5_hits / n) * 100

def main():
    root_path = "/data/fs201053/jb88526/reidentification_test_complete_v2"
    output_dir = Path("/home/jb88526/similarity/data")
    output_dir.mkdir(exist_ok=True)

    encoders = ["gigapath", "virchow2", "uni2h", "midnight12k", "simclrv2"]
    variants = ["raw", "norm"]
    patient_counts = [5, 10, 20, 40, 60, 80]
    tile_counts = [1, 5, 10, 50, 100, 250, 500]

    all_scaling_p, all_scaling_t = [], []

    for enc in encoders:
        for var in variants:
            data = load_data_for_pair(root_path, enc, var)
            pids = sorted(list(data.keys()))
            if not pids: continue
            
            # Předpřipravíme si data, abychom je v cyklech neslepovali zbytečně
            # Vytvoříme seznamy embeddingů a odpovídajících PID pro celou sadu
            all_o_list = [data[p]['orig'] for p in pids]
            all_g_list = [data[p]['gen'] for p in pids]
            
            # Mapování PID na řádky (pro rychlé filtrování)
            o_pids_full = np.concatenate([[p] * len(data[p]['orig']) for p in pids])
            g_pids_full = np.concatenate([[p] * len(data[p]['gen']) for p in pids])

            # --- ANALÝZA 1: SCALING PATIENTS ---
            for n in tqdm(patient_counts, desc=f" {enc}-{var} Patients"):
                if n > len(pids): continue
                
                # Vezmeme jen prvních N pacientů
                subset_pids_set = set(pids[:n])
                mask_o = np.isin(o_pids_full, list(subset_pids_set))
                mask_g = np.isin(g_pids_full, list(subset_pids_set))
                
                # Efektivní konkatenace jen potřebné části
                curr_orig = np.concatenate(all_o_list[:n])
                curr_gen = np.concatenate(all_g_list[:n])
                
                t1, t5 = evaluate_attack_fast(curr_orig, curr_gen, g_pids_full[mask_g], o_pids_full[mask_o])
                all_scaling_p.append({"Encoder": enc, "Var": var, "N_Patients": n, "Top1": t1, "Top5": t5})

            # --- ANALÝZA 2: SCALING TILES ---
            # Pro scaling tiles (všechny pacienti, ale jen 't' dlaždic v galerii)
            for t in tqdm(tile_counts, desc=f" {enc}-{var} Tiles"):
                # Ořežeme embeddingy v originálech na 't' dlaždic
                curr_orig = np.concatenate([emb[:t] for emb in all_o_list])
                curr_gen = np.concatenate(all_g_list)
                
                # Upravíme PIDs pro galerii (každý pacient tam má teď přesně 't' záznamů)
                o_pids_t = np.concatenate([[p] * min(t, len(data[p]['orig'])) for p in pids])
                
                t1, t5 = evaluate_attack_fast(curr_orig, curr_gen, g_pids_full, o_pids_t)
                all_scaling_t.append({"Encoder": enc, "Var": var, "N_Tiles": t, "Top1": t1, "Top5": t5})

    pd.DataFrame(all_scaling_p).to_csv(output_dir / "scaling_patients_all.csv", index=False)
    pd.DataFrame(all_scaling_t).to_csv(output_dir / "scaling_tiles_all.csv", index=False)

if __name__ == "__main__":
    main()