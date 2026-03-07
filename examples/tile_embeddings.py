import torch
import ray
import os
import argparse
import pandas as pd
import numpy as np
import time
import logging
import ray.data
from pathlib import Path
import multiprocessing

from PIL import Image
from typing import Any
from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles, read_slide_tiles
from rationai.staining import ColorConversion, normalize_staining
from dataclasses import dataclass, asdict
import urllib.parse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas.errors

# Vynucení existence atributu, který Ray postrádá
if not hasattr(pd.errors, "SettingWithCopyWarning"):
    class SettingWithCopyWarning(Warning):
        pass
    pd.errors.SettingWithCopyWarning = SettingWithCopyWarning

import pandas.core.common as pcc
if not hasattr(pcc, "SettingWithCopyWarning"):
    pcc.SettingWithCopyWarning = pd.errors.SettingWithCopyWarning

@dataclass
class Config:

    slide_path: str
    save_path: str

    # --- MODEL SETTINGS ---
    encoder: int
    device: str
    model_dtype: torch.dtype
    
    # --- TILING & WSI ---
    tile_size: int
    mpp: float
    batch_size: int
    num_workers: int
    
    # --- PREPROCESSING TOGGLES ---
    rmBackground: bool
    normalize: bool
    clahe: bool
    overwrite: bool
    
    # --- CONSTANTS ---
    # Vectors for coloring (Hematoxylin, Eosin)
    stain_vectors: tuple

    def display(self):
        print("\n" + "="*30)
        print("SLIDE PROCESSING CONFIG")
        print("="*30)
        print("encoders: 0: Gigapath, 1: Virchow2, 2: UNI2-h, 3: midnight-12k")
        
        config_dict = asdict(self)
        
        for key, value in config_dict.items():
            if isinstance(value, torch.dtype):
                value = str(value)
            elif isinstance(value, np.ndarray):
                value = f"Array {value.shape}"
            
            print(f"{key:<15}: {value}")
        
        print("="*30 + "\n")

def tiling(row: dict[str, Any]) -> list[dict[str, Any]]:
    slide_id = str(Path(row["path"]))
    
    return [
        {
            "tile_x": x,
            "tile_y": y,
            "path": row["path"],
            "slide_id": slide_id,
            "level": row["level"],
            "tile_extent_x": row["tile_extent_x"],
            "tile_extent_y": row["tile_extent_y"],
        }
        for x, y in grid_tiles(
            slide_extent=(row["extent_x"], row["extent_y"]),
            tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
            stride=(row["stride_x"], row["stride_y"]),
            last="keep",
        )
    ]

class TileEncoderActor:
    def __init__(self, DEVICE: torch.device, MODEL_DTYPE: torch.dtype, 
                 NORMALIZE=False, CLAHE=False, RM_BG=False, MAKE_IMAGE=False, ENCODER=0):
        # --- LOKÁLNÍ IMPORTY (Uvnitř) ---
        import torch
        import cv2
        # Tady si je ulož do self, pokud je potřebuješ v metodách
        self.torch = torch
        self.cv2 = cv2

        self.device = DEVICE
        self.model_dtype = MODEL_DTYPE
        self.NORMALIZE = NORMALIZE
        self.CLAHE = CLAHE
        self.MAKE_IMAGE = MAKE_IMAGE
        self.RM_BG = RM_BG
        
        self.saved_samples = 0 
        self.batch_count = 0
        
        # Stain vektory pro normalizaci
        self.STAIN_VECTORS = np.array([
            [0.64429328, 0.71655047, 0.26684416], # Hematoxylin
            [0.03448942, 0.6508934,  0.75845514]  # Eosin
        ])

        # Inicializace CLAHE objektu jednou pro celý život aktor (obrovská úspora CPU)
        if self.CLAHE:
            self.clahe_obj = self.cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Načtení konkrétního modelu
        from src.feature_extractors import gigapathTile, virchow2, UNI2h, midnight12k
        encoders = [gigapathTile, virchow2, UNI2h, midnight12k]
        selected_encoder_func = encoders[ENCODER]
        
        tile_encoder, transform = selected_encoder_func()
        self.tile_encoder = tile_encoder.to(self.device).to(self.model_dtype).eval()
        self.transform = transform

    def is_tissue(self, tile: np.ndarray, threshold: float = 0.05) -> bool:
        """Optimalizovaná verze detekce tkáně s downsamplingem."""
        # Každý 2. pixel stačí pro odhad sytosti (4x rychlejší)
        tile_sample = tile[::2, ::2].astype(np.float32) / 255.0
        
        c_max = np.max(tile_sample, axis=-1)
        c_min = np.min(tile_sample, axis=-1)
        delta = c_max - c_min
        
        saturation = np.where(c_max > 0, delta / c_max, 0)
        tissue_mask = saturation > 0.15
        return np.mean(tissue_mask) > threshold

    def apply_clahe_fast(self, img_np):
        """CLAHE přímo na numpy poli bez zbytečných konverzí."""
        lab = self.cv2.cvtColor(img_np, self.cv2.COLOR_RGB2LAB)
        l, a, b = self.cv2.split(lab)
        l_updated = self.clahe_obj.apply(l)
        lab_updated = self.cv2.merge((l_updated, a, b))
        return self.cv2.cvtColor(lab_updated, self.cv2.COLOR_LAB2RGB)

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        start_batch = time.time()
        
        transformed_tiles = []
        keep_indices = []
        
        # Optimalizovaný přístup k datům v DataFrame
        tiles = batch['tile']
        x_coords = batch['tile_x']
        y_coords = batch['tile_y']
        slide_ids = batch['slide_id']

        start_pre = time.time()
        
        for i in range(len(tiles)):
            tile_data = tiles[i]
            
            # 1. Rychlý test na tkáň hned na začátku
            if self.RM_BG and not self.is_tissue(tile_data):
                continue

            current_tile = tile_data

            # 2. Normalizace (pokud vrací PIL, převedeme na numpy pro CLAHE)
            if self.NORMALIZE:
                current_tile = normalize_staining(
                    current_tile, ColorConversion.RGB2HER.matrix, 
                    self.STAIN_VECTORS[0], self.STAIN_VECTORS[1]
                )
                if not isinstance(current_tile, np.ndarray):
                    current_tile = np.array(current_tile)

            # 3. CLAHE (v numpy)
            if self.CLAHE:
                current_tile = self.apply_clahe_fast(current_tile)

            # 4. Finalizace pro model (převod na PIL a Tensor)
            pil_img = Image.fromarray(current_tile).convert("RGB")
            tensor = self.transform(pil_img)
            
            transformed_tiles.append(tensor)
            keep_indices.append(i)

        end_pre = time.time()

        if not transformed_tiles:
            return {
                    "slide_id": np.array([], dtype=object),
                    "x_coord": np.array([], dtype=np.int64),
                    "y_coord": np.array([], dtype=np.int64),
                    "embedding": np.array([], dtype=object),
                }

        # --- GPU INFERENCE ---
        start_gpu = time.time()
        # Vytvoříme batch na GPU asynchronně, pokud možno
        batch_tensor = self.torch.stack(transformed_tiles).to(self.device, non_blocking=True).to(self.model_dtype)

        with self.torch.no_grad():
            embeddings_tensor = self.tile_encoder(batch_tensor)

        # Převod zpět na CPU numpy
        embeddings_array = embeddings_tensor.cpu().to(torch.float32).numpy()
        end_gpu = time.time()

        # Statistiky pro monitoring
        self.batch_count += 1
        if self.batch_count % 10 == 0:
            total_time = time.time() - start_batch
            print(f"\n[Worker {os.getpid()}] Batch {self.batch_count}:")
            print(f"  - Preprocessing: {(end_pre - start_pre):.3f}s")
            print(f"  - GPU Inference: {(end_gpu - start_gpu):.3f}s")
            print(f"  - Total:         {total_time:.3f}s")

        # Sestavení výsledného DataFrame
        output_df = pd.DataFrame({
            'slide_id': slide_ids[keep_indices],
            'x_coord': x_coords[keep_indices],
            'y_coord': y_coords[keep_indices],
            'embedding': list(embeddings_array),
        })

        # Explicitní úklid
        del embeddings_tensor
        del batch_tensor
        del transformed_tiles
        
        return output_df

def get_processing_tasks(config: Config):
    tasks = []
    input_path = Path(config.slide_path)
    output_base = Path(config.save_path)
    extensions = {".svs", ".tiff", ".mrxs"}

    if input_path.is_dir():
        for path in input_path.rglob("*"):
            if path.suffix.lower() in extensions and "_COPY" not in path.name:
                # 1. Vytvoření identifikátoru slide_id (stejně jako to dělá Ray)
                # Ray cesty escapuje (např. / se změní na %2F), musíme to simulovat
                slide_id_val = urllib.parse.quote(str(path), safe="")
                
                # 2. Cesta, kam Ray ukládá data pro tento konkrétní slide
                # Formát: save_path/slide_id=...
                check_dir = output_base / f"slide_id={slide_id_val}"
                
                # 3. Kontrola: Existuje složka a obsahuje aspoň jeden parquet?
                is_done = check_dir.exists() and any(check_dir.glob("*.parquet"))
                
                if not is_done:
                    rel_path = path.relative_to(input_path).parent
                    save_dir = output_base / rel_path
                    tasks.append((str(path), str(save_dir)))
                else:
                    print(f"Skipping already processed slide: {path.name}")
                    pass
    else:
        # Pro jeden soubor (zjednodušená kontrola)
        tasks.append((str(input_path), str(output_base)))

    print(f"Total tasks to process: {len(tasks)}")
    return tasks

def parse_args() -> Config:
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

    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=512,
        help='Batch size for one worker. Program computes itself if it can handle more workers with this batch size.'
    )

    parser.add_argument(
        '--workers', 
        type=int, 
        default=0,
        help='Limits workers to this number.'
    )

    parser.add_argument(
        '--overwrite', 
        action='store_true',
        help='It will override previously generated files'
    )

    parser.add_argument(
        '--mpp',
        type=float,
        default=0.5,
        help='Microns per pixel for tiling. Default is 0.5, which is common for 20x magnification.'
    )

    parser.add_argument(
        '--encoder',
        type=int,
        default=0,
        help="Choose the encoder you want to use: 0: Gigapath, 1: Virchow2, 2: UNI2-h, 3: midnight-12k"
    )
    
    parser.add_argument(
        '--rmbg',
        action='store_true',
        help='Removes the background of the processed images before computing embeddings.'
    )

    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Normalize the colors of the processed images before computing embeddings.'
    )

    parser.add_argument(
        '--clahe',
        action='store_true',
        help='Use CLAHE on the processed images before computing embeddings.'
    )

    args = parser.parse_args()

    config = Config(
        slide_path = args.slide_path.rstrip("/"),
        save_path = args.save_path.rstrip("/"),
        encoder= args.encoder,
        device= torch.device("cuda"),
        model_dtype=torch.bfloat16,
        tile_size = 224,
        mpp = args.mpp,
        batch_size = args.batch_size,
        num_workers = args.workers,
        overwrite=args.overwrite,
        normalize=args.normalize,
        rmBackground=args.rmbg,
        clahe= args.clahe,
        stain_vectors= (
        (0.64429328, 0.71655047, 0.26684416), # Hematoxylin
        (0.03448942, 0.6508934,  0.75845514)  # Eosin
        )
    )
    config.display()
    return config


def estimate_workers(config: Config) -> float:
    """for 256 batch size"""
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    vram_per_actor = {
        0: 4.0, # GigaPath
        1: 2.5, # Virchow2
        2: 5.0, # UNI2-h
        3: 4.5  # Midnight-12k 
    }.get(config.encoder, 1)
    
    return int(total_vram * 0.90 // vram_per_actor)

def main() -> None:
    config = parse_args()

    if config.num_workers <= 1: 
            config.num_workers = estimate_workers(config)
            print(f"🤖 Auto-scaling: Detected {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB VRAM.")
            print(f"🚀 Deployment: Using {config.num_workers} parallel workers for encoder {config.encoder}.")

    # Path to the library that fixed your 'jpeg12' error
    #PRELOAD_LIB = "/home/jb88526/.conda/envs/similarity-env/lib/libjpeg.so.8"
    #
    #runtime_env = {
    #    "working_dir": ".",
    #    "excludes": ["*"],
    #    "env_vars": {
    #        "LD_PRELOAD": PRELOAD_LIB,
    #        "LD_LIBRARY_PATH": f"/home/jb88526/.conda/envs/similarity-env/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
    #    }
    #}
    runtime_env = {}
    logging.getLogger("ray").setLevel(logging.ERROR)
    logging.getLogger("ray.data").setLevel(logging.ERROR)
    logging.getLogger("ray._private.state_accelerator_v2").setLevel(logging.ERROR)

    ray.init(runtime_env=runtime_env, logging_level=logging.ERROR, configure_logging=True, object_store_memory=60 * 1024**3) #TODO make bigger on h100?

    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.max_pending_blocks = 100 # Extrémně málo, ale u MPP 0.5 nutné
    # Vypne ukládání na disk úplně - pokud dojde RAM, Ray raději počká (backpressure)
    ctx.execution_options.spill_threshold = 0.99

    try:
        start_time = time.time()
        tasks = get_processing_tasks(config)
        print(tasks)
        return
        all_slide_paths = [t[0] for t in tasks]
        
        print(f"Starting parallel processing of {len(all_slide_paths)} slides...")

        ds = read_slides(all_slide_paths, mpp=config.mpp, tile_extent=config.tile_size, stride=config.tile_size)

        total_cpus = 20 # musica cpu count
        # Rezervujeme 20 % jader pro I/O a režii, zbytek rozdělíme mezi GPU workery
        cpus_per_worker = max(1, int((total_cpus * 0.8) / config.num_workers))
        cpus_concurrency = max(1,int(total_cpus*0.2))

        ds = ds.flat_map(tiling)
        ds = ds.map_batches(read_slide_tiles, batch_size=config.batch_size, num_cpus=1, concurrency=cpus_concurrency)

        results = ds.map_batches(
            TileEncoderActor,
            fn_constructor_kwargs={
                "DEVICE": config.device,
                "MODEL_DTYPE": config.model_dtype,
                "NORMALIZE": config.normalize,
                "RM_BG": config.rmBackground,
                "CLAHE": config.clahe,
                "ENCODER": config.encoder
            },
            num_cpus=cpus_per_worker,
            num_gpus=1.0 / config.num_workers,
            compute=ray.data.ActorPoolStrategy(size=config.num_workers),
            batch_size=config.batch_size
        )

        # 2. Samotný zápis
        results.write_parquet(
            config.save_path, 
            partition_cols=["slide_id"], 
        )

        print(f"✅ Finished in {time.time() - start_time:.2f} seconds")
        print(f"Time per slide: {(time.time() - start_time) / len(all_slide_paths):.2f} seconds")
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()

    #slide_path = '/mnt/data/scans/AI scans/Comparison_of_scanners/breast/FLASH2021_6802-01-T.mrxs'
    # /home/jovyan/prov-gigapath/demo/outputs_jirka/parquets
    #python -m examples.tile_embeddings --slide-path "/mnt/data/MOU/breast/comparison_of_scanners/" --save-path "/home/jovyan/output" --mpp 2.0 --encoder 1 --rmbg --normalize --clahe