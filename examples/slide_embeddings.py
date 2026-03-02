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

from PIL import Image
from typing import Any
from torchvision import transforms
from ratiopath.ray import read_slides
from ratiopath.tiling.utils import row_hash
from ratiopath.tiling import grid_tiles, read_slide_tiles
from src.feature_extractors import gigapathTile
from rationai.staining import ColorConversion, normalize_staining, estimate_stain_vectors
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A

def tiling(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "tile_x": x,
            "tile_y": y,
            "path": row["path"],
            "slide_id": row["id"],
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

def load_metadata(slide_path, MPP=0.5):   
    slides = read_slides(slide_path, mpp=MPP, tile_extent=224, stride=224)
    slides = slides.map(row_hash)

    tiles = slides.flat_map(tiling).repartition(target_num_rows_per_block=4096)

    tissue_tiles = tiles.map_batches(
        read_slide_tiles,
    )

    return (slides, tissue_tiles)

def load_parquet(path): 
    data = pd.read_parquet(path)
    return data

class TileEncoderActor:
    def __init__(self, DEVICE: torch.device, MODEL_DTYPE: torch.dtype, NORMALIZE=True, ENHANCE=True, MAKE_IMAGE=False):
        self.saved_samples = 0 
        self.device = DEVICE
        self.model_dtype = MODEL_DTYPE
        self.STAIN_VECTORS = np.array([
            [0.64429328, 0.71655047, 0.26684416], # Hematoxylin
            [0.03448942, 0.6508934,  0.75845514]  # Eosin
        ])
        self.NORMALIZE = NORMALIZE
        self.ENHANCE = ENHANCE
        self.MAKE_IMAGE = MAKE_IMAGE

        tile_encoder = gigapathTile()
        tile_encoder = tile_encoder.to(self.device)
        tile_encoder = tile_encoder.to(self.model_dtype)
        tile_encoder.eval()
        self.tile_encoder = tile_encoder

        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        x_coords = batch['tile_x']
        y_coords = batch['tile_y']
        transformed_tiles = []
        keep_indices = []
        
        for i, tile_data in enumerate(batch['tile']):
            if self.is_tissue(tile_data):

                img = Image.fromarray(tile_data).convert("RGB")

                if self.NORMALIZE:
                    img = normalize_staining(
                        img, ColorConversion.RGB2HER.matrix, self.STAIN_VECTORS[0], self.STAIN_VECTORS[1]
                    )
                    normalized = img

                if self.ENHANCE:
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


def create_tile_embeddings(slide_path, DEVICE, MODEL_DTYPE, tile_size, BATCH_SIZE, NUM_WORKERS, save_path, MPP):
    slide_metadata, tiles_metadata = load_metadata(slide_path, MPP)
    ray.data.DataContext.get_current().execution_options.verbose_progress = False

    #conda_lib = f"{os.environ.get('CONDA_PREFIX')}/lib/libjpeg.so.8"
    #brew_lib = "/home/linuxbrew/.linuxbrew/lib/libjpeg.so.8"
    #
    ## Use whichever one actually exists
    #PRELOAD_LIB_PATH = conda_lib if os.path.exists(conda_lib) else brew_lib
#
    #runtime_env = {
    #    "env_vars": {
    #        "LD_LIBRARY_PATH": f"{os.path.dirname(PRELOAD_LIB_PATH)}:{os.environ.get('LD_LIBRARY_PATH', '')}",
    #        "LD_PRELOAD": PRELOAD_LIB_PATH
    #    },
    #    "excludes": ["*"] # Keep your excludes from before!
    #}

    result_ds = tiles_metadata.map_batches(
        TileEncoderActor,
        fn_constructor_kwargs={
            "DEVICE": DEVICE,
            "MODEL_DTYPE": MODEL_DTYPE
        },
        num_gpus=1.0/NUM_WORKERS,
        batch_size=BATCH_SIZE,
        compute=ray.data.ActorPoolStrategy(size=NUM_WORKERS),
        runtime_env=runtime_env,
    )
    
    result_ds.repartition(1).write_parquet(save_path)
    part_file = os.path.join(save_path, os.listdir(save_path)[0]) # vezme první soubor ve složce
    shutil.move(part_file, save_path + "/tiles.parquet")
    return True


def save_tile_embeddings(save_path, tiles_df):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tiles_df.to_parquet(save_path + "/tiles.parquet", index=False)

def process_slide(slide_path, save_path, DEVICE, MODEL_DTYPE, NUM_WORKERS, BATCH_SIZE, OVERRIDE, MPP):
    slide_name = "Unknown"
    try:
        slide_metadata, tile_metadata = load_metadata(slide_path, MPP)
        slide_name = slide_metadata.take(1)[0]["path"].split('/')[-1].split('.')[0]

        if(OVERRIDE or not os.path.exists(save_path + '/' + slide_name)):
            tile_embeddings = create_tile_embeddings(slide_path, DEVICE, MODEL_DTYPE, 256, BATCH_SIZE, NUM_WORKERS, save_path + '/' + slide_name, MPP )
        else:
            print("\nTile embeddings already exists. Skipping", flush=True)

    except Exception as e:
        print(f"\n" + "!"*50)
        print(f"CHYBA: Slide {slide_name} nebylo možné zpracovat.")
        print(f"Cesta: {slide_path}")
        print(f"Chyba: {str(e)}")
        print("!"*50 + "\n", flush=True)
        current_wsi_save_path = os.path.join(save_path, slide_name)
        if slide_name != "Unknown" and os.path.exists(current_wsi_save_path):
            import shutil
            shutil.rmtree(current_wsi_save_path)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Creates tile and slide embeddings for WSI with help of Gigapath"
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
        help='Limits workers to this number, will not use heuristics for workers.'
    )

    parser.add_argument(
        '--overwrite', 
        type=bool, 
        default=False,
        help='If True, it will override previously generated files'
    )

    parser.add_argument(
        '--mpp',
        type=float,
        default=0.5,
        help='Microns per pixel for tiling. Default is 0.5, which is common for 20x magnification.'
    )
    args = parser.parse_args()

    #slide_path = '/mnt/data/scans/AI scans/Comparison_of_scanners/breast/FLASH2021_6802-01-T.mrxs'
    #python -m examples.slide_embeddings --slide-path "/mnt/data/MOU/breast/comparison_of_scanners/FLASH2021_6802-01-T.mrxs" --save-path "/home/jovyan/output" -mpp 2.0
    # /home/jovyan/prov-gigapath/demo/outputs_jirka/parquets
    slide_path = args.slide_path
    save_path = args.save_path.rstrip('/')
    BATCH_SIZE = args.batch_size
    OVERRIDE= args.overwrite
    MODEL_DTYPE = torch.bfloat16
    DEVICE = torch.device("cuda")
    NUM_WORKERS = args.workers
    SLIDE_COUNT = 0
    MPP = args.mpp

    # disclaimer: based on testing, can be wrong
    MODEL_SIZE_GB = 4.8
    ONE_BATCH_SIZE_GB = 0.0113  # size of 1 tile 256x256
    OVERHEAD = 1 # for pytorch and os stuff

    VRAM_PER_WORKER = ((BATCH_SIZE * ONE_BATCH_SIZE_GB) + MODEL_SIZE_GB)

    TOTAL_VRAM = torch.cuda.get_device_properties(DEVICE).total_memory / 1024**3

    if (NUM_WORKERS == 0):
        NUM_WORKERS = int((TOTAL_VRAM) // VRAM_PER_WORKER)
        if NUM_WORKERS == 0:
            NUM_WORKERS = 1

    print(f"Number of workers: {NUM_WORKERS:.2f}")
    print(f"total vram: {TOTAL_VRAM:.2f}")
    print(f"vram per worker: {VRAM_PER_WORKER:.2f}")

    start_time = time.time()
    if os.path.isdir(slide_path):
        for root, dirs, files in os.walk(slide_path):
            if not root.split("/")[-1].startswith("."):
                for file in files:
                    if (file.endswith(".svs") or file.endswith(".tiff")) or file.endswith(".mrxs") and not "_COPY" in file:
                        root_folder = root.split("/")[-1]
                        absolute_path = f"{root}/{file}"
                        save_path_current = f"{save_path}/{root_folder}"
                        print("Working on: "+ absolute_path, flush=True)

                        SLIDE_COUNT += 1
                        process_slide(absolute_path, save_path_current, DEVICE, MODEL_DTYPE, NUM_WORKERS, BATCH_SIZE, OVERRIDE, MPP)
    else:
        process_slide(slide_path, save_path, DEVICE, MODEL_DTYPE, NUM_WORKERS, BATCH_SIZE, OVERRIDE, MPP)
        SLIDE_COUNT += 1
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"=============================================")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Slide processed: {SLIDE_COUNT} slides")
    print(f"Number of workers: {NUM_WORKERS}")
    print(f"total vram: {TOTAL_VRAM:.2f}")
    print(f"vram per worker: {VRAM_PER_WORKER:.2f}")
    print(f"=============================================")

if __name__ == "__main__":
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
    main()
