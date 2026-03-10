import faiss
import numpy as np
import pandas as pd
import glob
import os
import urllib.parse
import re
from pathlib import Path
from collections import defaultdict

def analyze_similarity_decay(embeddings_dir, vector_type="vlad_super"):
    files = sorted(glob.glob(str(Path(embeddings_dir) / "*.parquet")))
    vectors = []
    metadata = [] # List of tuples: (folder_id, prefix, slice_num)

    print(f"📂 Načítám a parsuji {len(files)} souborů...")
    
    for f in files:
        # Dekódování názvu
        decoded_name = urllib.parse.unquote(os.path.basename(f))
        
        # 1. Složka (např. 43)
        folder_match = re.search(r'/(\d+)/', decoded_name)
        # 2. Slice ID (např. DORS a 5)
        slice_match = re.search(r'([a-zA-Z]+)(\d+)\.', decoded_name)
        
        if folder_match and slice_match:
            folder_id = folder_match.group(1)
            prefix = slice_match.group(1)
            slice_num = int(slice_match.group(2))
            
            df = pd.read_parquet(f)
            vectors.append(df[vector_type].iloc[0])
            metadata.append({
                'folder': folder_id,
                'prefix': prefix,
                'num': slice_num,
                'name': decoded_name
            })

    # Příprava Faiss
    xb = np.array(vectors).astype('float32')
    faiss.normalize_L2(xb)
    d = xb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(xb)

    # Hledáme 20 nejbližších, abychom měli šanci najít shodu i u vzdálenějších řezů
    D, I = index.search(xb, 20)

    # Statistiky: {diff: [seznam_shod_0_nebo_1]}
    # Chceme zjistit: "Když hledám nejpodobnější k DORS5, je to DORS (5+diff)?"
    results = defaultdict(list)

    for i in range(len(metadata)):
        query = metadata[i]
        
        # Pro každou možnou vzdálenost (diff) v datasetu se podíváme, 
        # jestli ji náš Top-1 soused trefil.
        # Ale intuitivnější pro tvůj graf je: 
        # "V jaké vzdálenosti byl nalezen nejpodobnější soused?"
        
        neighbor_idx = I[i][1] # Ten úplně nejpodobnější (Top-1)
        neighbor = metadata[neighbor_idx]
        
        # Kontrolujeme jen stejnou tkáň (např. DORS k DORS)
        if query['prefix'] == neighbor['prefix']:
            diff = abs(query['num'] - neighbor['num'])
            results[diff].append(1)
            
    # Výpis tabulky
    print(f"\n📊 Úspěšnost identifikace podle vzdálenosti řezu (Top-1 Match):")
    print(f"{'Diff (vzdálenost)':<18} | {'Počet případů':<15} | {'Úspěšnost %'}")
    print("-" * 55)

    total_matches = len(metadata)
    sorted_diffs = sorted(results.keys())
    
    for d in sorted_diffs:
        count = len(results[d])
        percentage = (count / total_matches) * 100
        bar = "█" * int(percentage / 2)
        print(f"Diff = {d:<11} | {count:<15} | {percentage:>6.1f}%  {bar}")

# --- SPUŠTĚNÍ ---
#EMB_DIR = "/data/fs201053/jb88526/privagams_enc2_mpp05_enhanced_slide"
#print(EMB_DIR)
#analyze_similarity_decay(EMB_DIR, vector_type="vlad_raw")
#analyze_similarity_decay(EMB_DIR, vector_type="vlad_super")
#analyze_similarity_decay(EMB_DIR, vector_type="fisher_raw")
#analyze_similarity_decay(EMB_DIR, vector_type="fisher_super")
#
#EMB_DIR = "/data/fs201053/jb88526/privagams_enc2_mpp10_enhanced_slide"
#print(EMB_DIR)
#analyze_similarity_decay(EMB_DIR, vector_type="vlad_raw")
#analyze_similarity_decay(EMB_DIR, vector_type="vlad_super")
#analyze_similarity_decay(EMB_DIR, vector_type="fisher_raw")
#analyze_similarity_decay(EMB_DIR, vector_type="fisher_super")
#
#EMB_DIR = "/data/fs201053/jb88526/privagams_enc2_mpp20_enhanced_slide"
#print(EMB_DIR)
#analyze_similarity_decay(EMB_DIR, vector_type="vlad_raw")
#analyze_similarity_decay(EMB_DIR, vector_type="vlad_super")
#analyze_similarity_decay(EMB_DIR, vector_type="fisher_raw")
#analyze_similarity_decay(EMB_DIR, vector_type="fisher_super")

def analyze_similarity_decay_top5(embeddings_dir, vector_type="vlad_super", k=5):
    files = sorted(glob.glob(str(Path(embeddings_dir) / "*.parquet")))
    vectors = []
    metadata = [] 

    print(f"📂 Načítám a parsuji {len(files)} souborů...")
    
    for f in files:
        decoded_name = urllib.parse.unquote(os.path.basename(f))
        
        # Extrakce ID a prefixu (přizpůsobeno tvé struktuře)
        folder_match = re.search(r'/(\d+)/', decoded_name)
        slice_match = re.search(r'([a-zA-Z]+)(\d+)\.', decoded_name)
        
        if slice_match:
            # Pokud folder_id v názvu není, použijeme default nebo prefix
            folder_id = folder_match.group(1) if folder_match else "0"
            prefix = slice_match.group(1)
            slice_num = int(slice_match.group(2))
            
            df = pd.read_parquet(f)
            if vector_type in df.columns:
                vectors.append(df[vector_type].iloc[0].astype('float32'))
                metadata.append({
                    'folder': folder_id,
                    'prefix': prefix,
                    'num': slice_num,
                    'name': decoded_name
                })

    if not vectors:
        print(f"⚠️ Žádná data pro {vector_type} v {embeddings_dir}")
        return

    # Příprava Faiss (IndexFlatIP pro Cosine Similarity na normalizovaných vektorech)
    xb = np.array(vectors).astype('float32')
    faiss.normalize_L2(xb)
    d = xb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(xb)

    # Hledáme k+1 sousedů (první je vždy dotaz samotný)
    D, I = index.search(xb, k + 1)

    # Statistiky: {diff: počet_úspěšných_top5_zásahů}
    results = defaultdict(int)
    # Celkový počet možných dvojic pro dané diff
    total_possible_for_diff = defaultdict(int)

    # Předvýpočet existujících řezů pro kontrolu
    registry = defaultdict(set)
    for m in metadata:
        registry[m['prefix']].add(m['num'])

    # Pro každý dotaz zkontrolujeme Top-K sousedy
    for i in range(len(metadata)):
        query = metadata[i]
        
        # Získáme indexy Top-K sousedů (vynecháme první, což je query self)
        neighbor_indices = I[i][1:k+1]
        top_k_neighbors = [metadata[idx] for idx in neighbor_indices]
        
        # Pro každé rozumné diff (1 až 10)
        for diff in range(0, 11):
            target_plus = query['num'] + diff
            target_minus = query['num'] - diff
            
            # Existuje vůbec takový soused v datech?
            exists = (target_plus in registry[query['prefix']]) or (target_minus in registry[query['prefix']])
            
            if exists:
                total_possible_for_diff[diff] += 1
                # Je některý z Top-K sousedů ten správný?
                found = False
                for nb in top_k_neighbors:
                    if nb['prefix'] == query['prefix'] and (nb['num'] == target_plus or nb['num'] == target_minus):
                        found = True
                        break
                if found:
                    results[diff] += 1
            
    # Výpis tabulky
    print(f"\n📊 Úspěšnost identifikace (Top-{k} Match) - {vector_type}:")
    print(f"{'Diff (vzdálenost)':<18} | {'Nalezeno':<15} | {'Úspěšnost %'}")
    print("-" * 55)

    for d in range(0, 11):
        count = results[d]
        total = total_possible_for_diff[d]
        if total > 0:
            percentage = (count / total) * 100
            bar = "█" * int(percentage / 2)
            print(f"Diff = {d:<11} | {count:<15} | {percentage:>6.1f}%  {bar}")

# --- SPUŠTĚNÍ ---
# Příklad pro jednu složku, můžeš zopakovat pro ostatní jako v tvém skriptu
EMB_DIRS = [
    "/data/fs201053/jb88526/privagams_enc2_mpp05_enhanced_slide_v3",
]

for edir in EMB_DIRS:
    print(f"\n{'='*60}\nADRESÁŘ: {edir}\n{'='*60}")
    analyze_similarity_decay_top5(edir, vector_type="vlad_raw", k=5)
    analyze_similarity_decay_top5(edir, vector_type="vlad_super", k=5)
    analyze_similarity_decay_top5(edir, vector_type="fisher_raw", k=5)
    analyze_similarity_decay_top5(edir, vector_type="fisher_super", k=5)