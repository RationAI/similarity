import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import glob
from tqdm import tqdm
import time

def integrity_check(REF_FILE, PARQUET_DIR):
    # 2. Načtení reference
    # Předpokládám sloupce 'path' a 'expected_tiles'
    ref_df = pd.read_csv(REF_FILE)

    print(f"Kontroluji {len(ref_df)} slajdů...")

    results = []

    for _, row in tqdm(ref_df.iterrows()):
        slide_path = row['slide_id']
        slide_dir_name = slide_path.replace("/local","slide_id=/data/fs201053/bs37803").replace("/", "%2F")
        expected = row['found']
        
        # Najdi odpovídající parquet (můžeš mít víc souborů v adresáři pro jeden slajd)
        parquet_files = glob.glob(f"{PARQUET_DIR}/{slide_dir_name}/*.parquet")

        if not parquet_files:
            results.append({"slide": slide_dir_name, "status": "CHYBÍ", "diff": expected})
            continue
        
        # Spočítej řádky ve všech parquetech pro tento slajd
        actual = 0
        for f in parquet_files:
            metadata = pq.read_metadata(f)
            actual += metadata.num_rows
        
        if actual == expected:
            # Volitelně: results.append({"slide": slide_name, "status": "OK"})
            pass 
        else:
            results.append({
                "slide": slide_dir_name, 
                "status": "CHYBA", 
                "expected": expected, 
                "actual": actual,
                "diff": expected - actual
            })

    # 3. Vyhodnocení
    errors_df = pd.DataFrame(results)
    if errors_df.empty:
        print("✅ Všechny soubory sedí na počet dlaždic!")
    else:
        print("❌ Nalezeny neshody:")
        print(errors_df)
        errors_df.to_csv(f"integrity_errors{time.time()}.csv", index=False)


ref_files = [
    "/data/fs201053/jb88526/integrity_baseline_mpp0.5.csv",
    "/data/fs201053/jb88526/integrity_baseline_mpp1.0.csv",
    "/data/fs201053/jb88526/integrity_baseline_mpp2.0.csv"
]
parq_files = [
    "/data/fs201053/jb88526/privagams_enc2_mpp05_enhanced",
    "/data/fs201053/jb88526/privagams_enc2_mpp10_enhanced",
    "/data/fs201053/jb88526/privagams_enc2_mpp20_enhanced"
]

for ref, parq in zip(ref_files, parq_files):
    integrity_check(ref, parq)

