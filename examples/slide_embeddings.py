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

def print_memory_stats():
    # Paměť aktuálního procesu
    process = psutil.Process(os.getpid())
    mem_rss = process.memory_info().rss / (1024 ** 3)  # v GB
    
    # Celková paměť na uzlu (dostupná pro tvůj job)
    node_mem = psutil.virtual_memory()
    
    print("-" * 30)
    print(f"RAM Usage (Process): {mem_rss:.2f} GB")
    print(f"RAM Usage (Node Total): {node_mem.percent}% used")
    print(f"RAM Available (Node): {node_mem.available / (1024**3):.2f} GB")
    print("-" * 30, flush=True)

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
    slides = read_slides(slide_path, mpp=MPP, tile_extent=256, stride=256)
    slides = slides.map(row_hash)

    tiles = slides.flat_map(tiling).repartition(target_num_rows_per_block=4096)

    tissue_tiles = tiles.map_batches(
        read_slide_tiles,
    )#.filter(lambda row: row["tile"].std() > 8)

    return (slides, tissue_tiles)

def load_parquet(path): 
    data = pd.read_parquet(path)
    return data

def create_slide_embeddings_service(slide_metadata, tiles_df, MODEL_DTYPE, DEVICE):
    embeddings_list_of_arrays = tiles_df['embedding'].to_list() 
    embeddings_numpy = np.stack(embeddings_list_of_arrays).astype(np.float32)

    x_coords = tiles_df['x_coord'].to_numpy()
    y_coords = tiles_df['y_coord'].to_numpy()
    coords_numpy = np.stack([x_coords, y_coords], axis=1).astype(np.float32)

    host = "http://rayservice-models-gigapath-serve-svc.rationai-jobs-ns.svc.cluster.local:8000"
    L = coords_numpy.shape[0]

    payload = embeddings_numpy.tobytes() + coords_numpy.tobytes() 
    url = f"{host}/gigapath-slide-encoder/{L}" 

    slide_metadata_df = slide_metadata.to_pandas()
    r = requests.post( url, data=payload, headers={"Content-Type": "application/octet-stream"}, timeout=600, ) 

    try: 
        slide_metadata_df['embedding'] = r.json()["embeddings"]
    except Exception: 
        print("ERROR: gigapath service not answering.")
        raise

    return slide_metadata_df

class TileEncoderActor:
    def __init__(self, DEVICE: torch.device, MODEL_DTYPE: torch.dtype, STAIN_VECTORS: np.array):
        self.device = DEVICE
        self.model_dtype = MODEL_DTYPE
        self.stain_vectors = STAIN_VECTORS

        tile_encoder = gigapathTile()
        tile_encoder = tile_encoder.to(self.device)
        tile_encoder = tile_encoder.to(self.model_dtype)
        tile_encoder.eval()
        self.tile_encoder = tile_encoder

        self.transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        x_coords = batch['tile_x']
        y_coords = batch['tile_y']
        transformed_tiles = []
        
        for tile_data in batch['tile']:
            img = Image.fromarray(tile_data).convert("RGB")

            # Only example values, real values should be computed from a reference region.
            target1 = self.stain_vectors[0]
            target2 = self.stain_vectors[1]

            normalized = normalize_staining(
                img, ColorConversion.RGB2HER.matrix, target1, target2
            )

            pil_image = Image.fromarray(normalized).convert("RGB") # convert RGB??
            tensor = self.transform(pil_image)
            transformed_tiles.append(tensor)
            
        batch_tensor = torch.stack(transformed_tiles)
        final_input_tensor = batch_tensor.to(self.device).to(self.model_dtype)

        with torch.no_grad():
            embeddings_tensor = self.tile_encoder(final_input_tensor)

        embeddings_array = embeddings_tensor.cpu().to(torch.float32).numpy()
        embeddings_list = list(embeddings_array)
        
        del embeddings_tensor
        del final_input_tensor
        torch.cuda.empty_cache()

        output_df = pd.DataFrame({
            'slide_id': batch["slide_id"],
            'x_coord': x_coords,
            'y_coord': y_coords,
            'embedding': embeddings_list,
        })
        return output_df

def create_tile_embeddings(slide_path, DEVICE, MODEL_DTYPE, tile_size, BATCH_SIZE, NUM_WORKERS, save_path, MPP):
    slide_metadata, tiles_metadata = load_metadata(slide_path, MPP)
    ray.data.DataContext.get_current().execution_options.verbose_progress = False

    conda_lib = f"{os.environ.get('CONDA_PREFIX')}/lib/libjpeg.so.8"
    brew_lib = "/home/linuxbrew/.linuxbrew/lib/libjpeg.so.8"
    
    # Use whichever one actually exists
    PRELOAD_LIB_PATH = conda_lib if os.path.exists(conda_lib) else brew_lib

    runtime_env = {
        "env_vars": {
            "LD_LIBRARY_PATH": f"{os.path.dirname(PRELOAD_LIB_PATH)}:{os.environ.get('LD_LIBRARY_PATH', '')}",
            "LD_PRELOAD": PRELOAD_LIB_PATH
        },
        "excludes": ["*"] # Keep your excludes from before!
    }

    img = openslide.OpenSlide(slide_path).get_thumbnail((1000, 1000))
    estimated_stain_vectors = estimate_stain_vectors(img)
    img.close()

    result_ds = tiles_metadata.map_batches(
        TileEncoderActor,
        fn_constructor_kwargs={
            "DEVICE": DEVICE,
            "MODEL_DTYPE": MODEL_DTYPE,
            "STAIN_VECTORS": estimated_stain_vectors,
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

def save_slide_embeddings(save_path, slide_df):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    slide_df.to_parquet(save_path + "/slide.parquet", index=False)

def process_slide(slide_path, save_path, DEVICE, MODEL_DTYPE, NUM_WORKERS, BATCH_SIZE, OVERRIDE, MPP):
    slide_name = "Unknown"
    try:
        slide_metadata, tile_metadata = load_metadata(slide_path, MPP)
        slide_name = slide_metadata.take(1)[0]["path"].split('/')[-1].split('.')[0]

        if(OVERRIDE or not os.path.exists(save_path + '/' + slide_name)):
            tile_embeddings = create_tile_embeddings(slide_path, DEVICE, MODEL_DTYPE, 256, BATCH_SIZE, NUM_WORKERS, save_path + '/' + slide_name, MPP )
        else:
            print("\nTile embeddings already exists. Skipping", flush=True)
            #tile_embeddings = load_parquet(save_path + '/' + slide_name)

        if (OVERRIDE or not os.path.exists(save_path + '/' + slide_name +"/slide.parquet")):
            pass
            #slide_embeddings = create_slide_embeddings_service(slide_metadata, tile_embeddings, MODEL_DTYPE, DEVICE)
            #save_slide_embeddings(save_path + '/' + slide_name, slide_embeddings)    
        else: 
            print("\nSlide embeddings already exists. Skipping")

    except Exception as e:
        # Tady je to klíčové: zapíšeme chybu, ale neukončíme skript
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

    print_memory_stats()

    #slide_path = '/mnt/data/scans/AI scans/Comparison_of_scanners/breast/FLASH2021_6802-01-T.mrxs'
    #python -m examples.slide_embeddings --slide-path "/mnt/data/MOU/breast/comparison_of_scanners" --save-path "/mnt/projects/ri_scale/privagams"
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
                    if (file.endswith(".svs") or file.endswith(".tiff")) and not "_COPY" in file:
                        root_folder = root.split("/")[-1]
                        absolute_path = f"{root}/{file}"
                        save_path_current = f"{save_path}/{root_folder}"
                        print("Working on: "+ absolute_path, flush=True)
                        print_memory_stats()

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
    PRELOAD_LIB = "/home/jb88526/.conda/envs/similarity-env/lib/libjpeg.so.8"
    
    runtime_env = {
        "working_dir": ".",
        "excludes": ["*"],
        "env_vars": {
            "LD_PRELOAD": PRELOAD_LIB,
            "LD_LIBRARY_PATH": f"/home/jb88526/.conda/envs/similarity-env/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
        }
    }

    logging.getLogger("ray").setLevel(logging.ERROR)
    logging.getLogger("ray.data").setLevel(logging.ERROR)
    logging.getLogger("ray._private.state_accelerator_v2").setLevel(logging.ERROR)

    ray.init(runtime_env=runtime_env, logging_level=logging.ERROR, configure_logging=True)
    main()
