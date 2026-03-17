import ray
import os
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import numpy as np
import urllib.parse
import time
import shutil

# Importy z tvého balíku
from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles, read_slide_tiles

def tiling(row):
    slide_id = str(Path(row["path"]))
    res = []
    for x, y in grid_tiles(
        slide_extent=(row["extent_x"], row["extent_y"]),
        tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
        stride=(row["stride_x"], row["stride_y"]),
        last="keep",
    ):
        res.append({
            "tile_x": int(x),
            "tile_y": int(y),
            "path": str(row["path"]),
            "slide_id": slide_id,
            "level": int(row.get("level", 0)),
            "tile_extent_x": int(row["tile_extent_x"]),
            "tile_extent_y": int(row["tile_extent_y"]),
        })
    return res

class SimpleTissueCounter:
    def __init__(self):
        # Inicializace proběhne jednou na každém workeru (aktorovi)
        import numpy as np
        import cv2
        self.np = np
        self.cv2 = cv2
        self.tile_size = 224

    def __call__(self, batch: dict) -> dict:
        import pandas as pd
        tiles = batch["tile"]
        keep_mask = []

        for tile in tiles:
            # 1. Resize
            if tile.shape[0] != self.tile_size:
                tile = self.cv2.resize(
                    tile, 
                    (self.tile_size, self.tile_size), 
                    interpolation=self.cv2.INTER_AREA
                )
            
            # 2. Výpočet sytosti
            tile_sample = tile[::2, ::2].astype(self.np.float32) / 255.0
            c_max = self.np.max(tile_sample, axis=-1)
            c_min = self.np.min(tile_sample, axis=-1)
            delta = c_max - c_min
            
            saturation = self.np.where(c_max > 0, delta / c_max, 0)
            is_tissue = self.np.mean(saturation > 0.15) > 0.05
            keep_mask.append(is_tissue)

        # --- OPRAVA TADY ---
        # Odstraníme "tile" z batche, protože Pandas neumí 4D sloupce (batch obrázků)
        # a my ty obrázky v Parquetu stejně nechceme (šetříme místo)
        output_batch = {k: v for k, v in batch.items() if k != "tile"}
        
        # Teď už DataFrame projde, protože obsahuje jen 1D pole (souřadnice, cesty atd.)
        df = pd.DataFrame(output_batch)
        
        # Vrátíme jen dlaždice, které prošly filtrem tkáně
        return df[keep_mask]

def get_processing_tasks(root_path: Path):
    tasks = []
    tiff_tasks = []
    input_path = Path(root_path)
    extensions = {".svs", ".tiff", ".mrxs"}

    if input_path.is_dir():
        for path in input_path.rglob("*"):
            if path.suffix.lower() in extensions and "_COPY" not in path.name:
                if path.suffix.lower() == ".tiff":
                    tiff_tasks.append(str(path))
                else:
                    tasks.append(str(path))
    return tasks, tiff_tasks

def main():
    # Creates tiles and save it into parqet files to not need to run is_tissue and read all tiles from drives
    #also creates number of tiles for each slide to check integrity
    base_data_path = "/data/fs201053/jb88526"
    parquet_root = "/data/fs201053/bs37803/annoPaperScanns"    
    
    # Inicializace Ray s větší pamětí pro object store (pokud je dostupná)
    ray.init(num_cpus=88, object_store_memory=150 * 1024 * 1024 * 1024)
    print(f"Ray resources: {ray.cluster_resources()}")

    slide_paths, tiff_paths = get_processing_tasks(Path(parquet_root))
    slide_paths = []
    print(f"Total tasks to process: {len(slide_paths) + len(tiff_paths)}")

    for mpp in [2.0, 1.0, 0.5]:
        mpp_start_time = time.time()
        print(f"\n--- Zpracovávám MPP: {mpp} ---")
        
        ds_list = []
        if slide_paths:
            ds_slide = read_slides(slide_paths, mpp=mpp, tile_extent=224, stride=224)
            ds_list.append(ds_slide)

        if tiff_paths:
            extent = 1792 if mpp == 2.0 else (896 if mpp == 1.0 else 448)
            ds_tiff = read_slides(tiff_paths, level=0, tile_extent=extent, stride=extent)
            ds_list.append(ds_tiff)

        if not ds_list: continue

        ds = ds_list[0].union(ds_list[1]) if len(ds_list) > 1 else ds_list[0]
        
        # Pipeline
        ds = ds.flat_map(tiling)
        ds = ds.repartition(num_blocks=500)
        
        # Čtení dlaždic
        ds = ds.map_batches(read_slide_tiles, batch_size=32, num_cpus=1, concurrency=30)
        
        # Filtrace tkáně
        results = ds.map_batches(SimpleTissueCounter, batch_size=32, num_cpus=2, concurrency=25)
        
        # --- NOVÝ ZPŮSOB ULOŽENÍ ---
        temp_dir = os.path.join(base_data_path, f"temp_mpp_{mpp}")
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        
        print(f"Zapisuji výsledky (index masku) do {temp_dir}...")
        results.write_parquet(temp_dir)
        
        # Načtení výsledků zpět pro vytvoření finálního baseline CSV
        # (Teď už je to jen malá tabulka souřadnic, Pandas to zvládne hravě)
        final_results = pd.read_parquet(temp_dir)
        
        # 1. Uložíme počty pro kontrolu integrity
        baseline_df = final_results.groupby("slide_id").size().reset_index(name="found")
        baseline_path = os.path.join(base_data_path, f"integrity_baseline_mpp{mpp}_tiff.csv")
        baseline_df.to_csv(baseline_path, index=False)
        
        # 2. Uložíme i ty souřadnice jako "masku" (to je ten tvůj nápad)
        mask_path = os.path.join(base_data_path, f"tissue_mask_mpp{mpp}_tiff.parquet")
        final_results.to_parquet(mask_path, index=False)

        print(f"✅ Baseline uložen: {baseline_path}")
        print(f"✅ Tissue maska (souradnice) uložena: {mask_path}")
        print(f"--- MPP {mpp} hotovo za {time.time() - mpp_start_time:.2f}s ---")

    ray.shutdown()

if __name__ == "__main__":
    main()