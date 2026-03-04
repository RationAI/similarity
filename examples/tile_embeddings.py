import openslide # needed for ratiopath
import pyvips
import torch
import ray
import os
import argparse
import pandas as pd
import numpy as np
import requests
import time
import logging
import ray.data
import shutil
import psutil
import cv2
from pathlib import Path

from PIL import Image
from typing import Any
from torchvision import transforms
from ratiopath.ray import read_slides
from ratiopath.tiling.utils import row_hash
from ratiopath.tiling import grid_tiles, read_slide_tiles
from src.feature_extractors import gigapathTile, virchow2, UNI2h, midnight12k
from rationai.staining import ColorConversion, normalize_staining, estimate_stain_vectors
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A
from dataclasses import dataclass, asdict

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
    slide_id = Path(row["path"]).stem
    
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
    def __init__(self, DEVICE: torch.device, MODEL_DTYPE: torch.dtype, NORMALIZE=False, CLAHE=False, RM_BG=False, MAKE_IMAGE=False, ENCODER=0):
        self.saved_samples = 0 
        self.device = DEVICE
        self.model_dtype = MODEL_DTYPE
        self.STAIN_VECTORS = np.array([
            [0.64429328, 0.71655047, 0.26684416], # Hematoxylin
            [0.03448942, 0.6508934,  0.75845514]  # Eosin
        ])
        self.NORMALIZE = NORMALIZE
        self.CLAHE = CLAHE
        self.MAKE_IMAGE = MAKE_IMAGE
        self.RM_BG = RM_BG

        encoders = [gigapathTile, virchow2, UNI2h, midnight12k]
        selected_encoder_func = encoders[ENCODER]
        tile_encoder, transform = selected_encoder_func()

        self.tile_encoder = tile_encoder.to(self.device).to(self.model_dtype).eval()

        self.transform = transform

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        x_coords = batch['tile_x']
        y_coords = batch['tile_y']
        transformed_tiles = []
        keep_indices = []
        
        for i, tile_data in enumerate(batch['tile']):
            if self.is_tissue(tile_data) or not self.RM_BG:

                img = Image.fromarray(tile_data).convert("RGB")

                if self.NORMALIZE:
                    img = normalize_staining(
                        img, ColorConversion.RGB2HER.matrix, self.STAIN_VECTORS[0], self.STAIN_VECTORS[1]
                    )
                    normalized = img

                if self.CLAHE:
                    img = self.apply_clahe(img)
                    enhanced = img

                pil_image = Image.fromarray(img)
                tensor = self.transform(pil_image)
                transformed_tiles.append(tensor)
                keep_indices.append(i)

                if self.saved_samples < 5 and self.MAKE_IMAGE: # Uložíme jen prvních 5 dlaždic pro kontrolu
                    plt.figure(figsize=(12, 4))
                    plt.subplot(131); plt.imshow(tile_data); plt.title("Original (Raw)")
                    plt.subplot(132); plt.imshow(normalized); plt.title("RationAI Norm")
                    plt.subplot(133); plt.imshow(enhanced); plt.title("Norm + CLAHE")
                    plt.savefig(f"debug_tile_{self.saved_samples}_{x_coords[i]}_{y_coords[i]}.png")
                    plt.close()
                    self.saved_samples += 1

        if not transformed_tiles:
            return {
                    "slide_id": np.array([], dtype=object),
                    "x_coord": np.array([], dtype=np.int64),
                    "y_coord": np.array([], dtype=np.int64),
                    "embedding": np.array([], dtype=object),
                }

        batch_tensor = torch.stack(transformed_tiles)
        final_input_tensor = batch_tensor.to(self.device).to(self.model_dtype)

        with torch.no_grad():
            embeddings_tensor = self.tile_encoder(final_input_tensor)

        embeddings_array = embeddings_tensor.cpu().to(torch.float32).numpy()
        
        del embeddings_tensor
        del final_input_tensor
        torch.cuda.empty_cache()

        slide_ids = np.array(batch['slide_id'])
        tile_x = np.array(batch['tile_x'])
        tile_y = np.array(batch['tile_y'])
        
        output_df = pd.DataFrame({
            'slide_id': slide_ids[keep_indices],
            'x_coord': tile_x[keep_indices],
            'y_coord': tile_y[keep_indices],
            'embedding': list(embeddings_array),
        })
        return output_df

    def is_tissue(self, tile: np.ndarray, threshold: float = 0.05) -> bool:
        """
        Vrací True, pokud dlaždice obsahuje dostatek 'barevných' pixelů (tkáně).
        tile: (H, W, 3) v RGB, uint8 (0-255)
        """
        # Převod RGB na Sytost (Saturation) v rámci HSV
        # S = (max(R,G,B) - min(R,G,B)) / max(R,G,B)
        
        tile_float = tile.astype(np.float32) / 255.0
        c_max = np.max(tile_float, axis=-1)
        c_min = np.min(tile_float, axis=-1)
        delta = c_max - c_min
        
        # Vyhneme se dělení nulou u černé/šedé
        saturation = np.where(c_max > 0, delta / c_max, 0)
        
        # Dlaždice je tkáň, pokud má víc než 5 % pixelů sytost > 0.15
        # (tyto hodnoty jsou v patologii standardem pro H&E)
        tissue_mask = saturation > 0.15
        return np.mean(tissue_mask) > threshold

    def apply_clahe(self, img_np):
        # CLAHE se standardně provádí v LAB prostoru na 'L' kanálu (jas), 
        # aby se nerozbily barvy
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # clipLimit: jak moc "agresivní" kontrast bude (2.0 je standard)
        # tileGridSize: na jak velké čtverce se dlaždice rozdělí
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_updated = clahe.apply(l)
        
        lab_updated = cv2.merge((l_updated, a, b))
        return cv2.cvtColor(lab_updated, cv2.COLOR_LAB2RGB)

def get_processing_tasks(config: Config):
    tasks = []
    input_path = Path(config.slide_path)
    output_base = Path(config.save_path)

    extensions = {".svs", ".tiff", ".mrxs"}

    if input_path.is_dir():
        for path in input_path.rglob("*"):
            if path.suffix.lower() in extensions and "_COPY" not in path.name:
                rel_path = path.relative_to(input_path).parent
                save_dir = output_base / rel_path
                
                tasks.append((str(path), str(save_dir)))
    else:
        # Pokud je vstupem jen jeden soubor
        tasks.append((str(input_path), str(output_base)))

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
        default=256,
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


def estimate_vram_requirement(config: Config) -> float:
    # Midnight je lehčí, dejme mu menší základ
    model_weights = {
        0: 2.2, # GigaPath
        1: 0.8, # Virchow2
        2: 0.8, # UNI2-h
        3: 0.5  # Midnight-12k (pokud je to ViT-B, 0.5GB stačí)
    }.get(config.encoder, 0.8)
    
    # Agresivnější výpočet aktivací (méně rezervy)
    # Pro 224x224 a batch 256 je to cca 1.5 - 2 GB
    batch_vram = (config.batch_size / 256) * 2.0
    
    # CUDA a Ray režie
    cuda_overhead = 1.0
    
    # Snížíme bezpečnostní rezervu z 1.3 na 1.15
    total_needed = (model_weights + batch_vram + cuda_overhead) * 1.15
    
    return total_needed

def auto_scale_workers(config: Config) -> int:
    if not torch.cuda.is_available(): return 1
    
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    needed_per_worker = estimate_vram_requirement(config)
    
    # Necháme si 10% rezervu, aby GPU "nedýchala naposledy"
    usable_vram = total_vram * 0.9 
    
    num_workers = int(usable_vram // needed_per_worker)
    return max(1, num_workers)

def main() -> None:
    config = parse_args()

    if config.num_workers <= 1: 
            config.num_workers = auto_scale_workers(config)
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

    ray.init(runtime_env=runtime_env, logging_level=logging.ERROR, configure_logging=True)

    try:
        start_time = time.time()
        tasks = get_processing_tasks(config)
        all_slide_paths = [t[0] for t in tasks][:10]
        
        print(f"Starting parallel processing of {len(all_slide_paths)} slides...")

        ds = read_slides(all_slide_paths, mpp=config.mpp, tile_extent=config.tile_size, stride=config.tile_size)

        ds = ds.flat_map(tiling).map_batches(read_slide_tiles)

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
            num_cpus=0.2,
            num_gpus=1.0 / config.num_workers,
            compute=ray.data.ActorPoolStrategy(size=config.num_workers),
            batch_size=config.batch_size
        )

        results.write_parquet(config.save_path, partition_cols=["slide_id"])

        print(f"✅ Finished in {time.time() - start_time:.2f} seconds")
        print(f"Time per slide: {(time.time() - start_time) / len(all_slide_paths):.2f} seconds")
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()

    #slide_path = '/mnt/data/scans/AI scans/Comparison_of_scanners/breast/FLASH2021_6802-01-T.mrxs'
    # /home/jovyan/prov-gigapath/demo/outputs_jirka/parquets
    #python -m examples.slide_embeddings --slide-path "/mnt/data/MOU/breast/comparison_of_scanners/FLASH2021_6802-01-T.mrxs" --save-path "/home/jovyan/output" --mpp 2.0 --encoder 3 --rmbg --normalize --clahe