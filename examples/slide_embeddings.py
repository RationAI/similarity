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

from PIL import Image
from typing import Any
from torchvision import transforms
from ratiopath.ray import read_slides
from ratiopath.tiling.utils import row_hash
from ratiopath.tiling import grid_tiles, read_slide_tiles
from src.feature_extractors import gigapathTile
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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


def load_metadata(slide_path):   
    slides = read_slides(slide_path, mpp=0.25, tile_extent=256, stride=256)
    slides = slides.map(row_hash)

    tiles = slides.flat_map(tiling).repartition(target_num_rows_per_block=4096)

    tissue_tiles = tiles.map_batches(
        read_slide_tiles,
    ).filter(lambda row: row["tile"].std() > 8)

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

    host = "http://rayservice-models-serve-svc.rationai-jobs-ns.svc.cluster.local:8000"
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
    def __init__(self, DEVICE: torch.device, MODEL_DTYPE: torch.dtype):
        self.device = DEVICE
        self.model_dtype = MODEL_DTYPE

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
            pil_image = Image.fromarray(tile_data)
            tensor = self.transform(pil_image)
            transformed_tiles.append(tensor)
            
        batch_tensor = torch.stack(transformed_tiles)
        final_input_tensor = batch_tensor.to(self.device).to(self.model_dtype)

        with torch.no_grad():
            embeddings_tensor = self.tile_encoder(final_input_tensor)
            vram_used = torch.cuda.max_memory_allocated(device=self.device)

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

def create_tile_embeddings(slide_path, DEVICE, MODEL_DTYPE, tile_size, BATCH_SIZE, NUM_WORKERS):
    slide_metadata, tiles_metadata = load_metadata(slide_path)

    # needed for right integration of workers
    PRELOAD_LIB_PATH = "/home/linuxbrew/.linuxbrew/lib/libjpeg.so.8" 
    ld_library_path = "/home/linuxbrew/.linuxbrew/lib"

    runtime_env = {
        "env_vars": {
            "LD_LIBRARY_PATH": f"{ld_library_path}:{os.environ.get('LD_LIBRARY_PATH', '')}",
            "LD_PRELOAD": PRELOAD_LIB_PATH
        }
    }

    result_ds = tiles_metadata.map_batches(
        TileEncoderActor,
        fn_constructor_kwargs={
            "DEVICE": DEVICE,
            "MODEL_DTYPE": MODEL_DTYPE,
        },
        num_gpus=1.0/NUM_WORKERS,
        batch_size=BATCH_SIZE,
        compute=ray.data.ActorPoolStrategy(size=NUM_WORKERS),
        runtime_env=runtime_env,
    )
    return result_ds.to_pandas()

def save_tile_embeddings(save_path, tiles_df):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    tiles_df.to_parquet(save_path + "/tiles.parquet", index=False)

def save_slide_embeddings(save_path, slide_df):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    slide_df.to_parquet(save_path + "/slide.parquet", index=False)

def process_slide(slide_path, save_path, DEVICE, MODEL_DTYPE, NUM_WORKERS, BATCH_SIZE, OVERRIDE):
    slide_metadata, tile_metadata = load_metadata(slide_path)
    slide_name = slide_metadata.take(1)[0]["path"].split('/')[-1].split('.')[0]
    if(OVERRIDE or not os.path.exists(save_path + '/' + slide_name)):
        tile_embeddings = create_tile_embeddings(slide_path, DEVICE, MODEL_DTYPE, 256, BATCH_SIZE, NUM_WORKERS)
        save_tile_embeddings(save_path + '/' + slide_name, tile_embeddings)
    else:
        print("\nTile embeddings already exists. Skipping")
        tile_embeddings = load_parquet(save_path + '/' + slide_name)

    if (OVERRIDE or not os.path.exists(save_path + '/' + slide_name +"/slide.parquet")):
        slide_embeddings = create_slide_embeddings_service(slide_metadata, tile_embeddings, MODEL_DTYPE, DEVICE)
        save_slide_embeddings(save_path + '/' + slide_name, slide_embeddings)    
    else: 
        print("\nSlide embeddings already exists. Skipping")

def get_simmilarity(slide_path):
    embeddings: List[torch.Tensor] = [] 
    for dir in os.listdir(slide_path):
        if os.path.isdir(os.path.join(slide_path, dir)):
            embeddings_path = os.path.join(slide_path, dir, "slide.parquet")
            tensor = torch.tensor(load_parquet(embeddings_path).embedding[0], dtype=torch.float32)
            embeddings.append(tensor)

    labels = os.listdir(slide_path)
    emb_matrix = torch.stack(embeddings, dim=0)
    X_norm = F.normalize(emb_matrix, p=2, dim=1)
    cosine_sim = X_norm @ X_norm.T
    df = pd.DataFrame(cosine_sim, index=labels, columns=labels)
    df.to_csv("similarity_matrix.csv", float_format="%.12f", index=True)


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
    args = parser.parse_args()


    #slide_path = '/mnt/data/scans/AI scans/Comparison_of_scanners/breast/FLASH2021_6802-01-T.mrxs'
    slide_path = args.slide_path
    save_path = args.save_path.rstrip('/')
    BATCH_SIZE = args.batch_size
    OVERRIDE= args.overwrite
    MODEL_DTYPE = torch.bfloat16
    DEVICE = torch.device("cuda")
    NUM_WORKERS = args.workers
    SLIDE_COUNT = 0

    ### TODO REMOVE IF YOU NEED TO GENERATE EMBEDDINGS
    get_simmilarity(save_path)
    return


    # disclaimer: based on testing, can be wrong
    MODEL_SIZE_GB = 4.8
    ONE_BATCH_SIZE_GB = 0.0113  # size of 1 tile 256x256
    OVERHEAD = 2 # for pytorch and os stuff

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
        for slide_name in os.listdir(slide_path):
            absolute_path = os.path.join(slide_path, slide_name)
            if os.path.isdir(absolute_path):
                continue
            SLIDE_COUNT += 1
            process_slide(absolute_path, save_path, DEVICE, MODEL_DTYPE, NUM_WORKERS, BATCH_SIZE, OVERRIDE)
    else:
        process_slide(slide_path, save_path, DEVICE, MODEL_DTYPE, NUM_WORKERS, BATCH_SIZE, OVERRIDE)
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
    main()
