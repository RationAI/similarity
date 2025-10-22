import openslide
import pyvips
import torch
import ray
import os
import argparse
import pandas as pd
import numpy as np
import requests
import time

from tqdm import tqdm
from PIL import Image
from typing import Any
from torchvision import transforms
from ratiopath.ray import read_slides
from ratiopath.tiling.utils import row_hash
from ratiopath.tiling import grid_tiles, read_slide_tiles
from src.feature_extractors import gigapathTile


def get_sensible_worker_config():
    """
    Pokusí se detekovat vlastnosti systému a navrhnout
    rozumnou výchozí konfiguraci pro počet workerů.
    """
    if not torch.cuda.is_available():
        return 1, 4096

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"Detekována GPU s {vram_gb:.1f} GB VRAM.")
    estimated_vram_per_worker_gb = 5
    available_vram_for_workers = vram_gb - 1.0 
    num_workers = int(available_vram_for_workers / estimated_vram_per_worker_gb)
    
    num_workers = max(1, num_workers) 

    rows_per_block = 4096 * (num_workers // 2)
    rows_per_block = max(4096, min(32768, rows_per_block))

    print(f"Navrhovaná konfigurace: {num_workers} workerů/GPU, {rows_per_block} řádků/blok.")
    return num_workers, rows_per_block

def profile_and_get_auto_batch_size(model_class, model_dtype, num_workers):
    """
    Spustí 'suchý běh' pro změření reálné spotřeby VRAM a určí optimální batch_size.
    """
    if not torch.cuda.is_available():
        return 128

    print("Spouštím profilování VRAM pro automatické nastavení batch_size...")

    # Připravíme model a dummy data
    device = torch.device("cuda")
    model = model_class().to(device).to(model_dtype).eval()
    
    # Velikost dávky, na které budeme testovat (musí být malá)
    profile_batch_size = 32 
    dummy_input = torch.randn(profile_batch_size, 3, 224, 224, device=device, dtype=model_dtype)

    # Změříme paměť obsazenou jen modelem
    torch.cuda.empty_cache()
    vram_model_only = torch.cuda.memory_allocated(device)

    # Spustíme forward pass a změříme PEAK paměť
    with torch.no_grad():
        model(dummy_input)
    
    vram_peak_with_batch = torch.cuda.max_memory_allocated(device)
    del dummy_input, model # Uvolníme paměť
    torch.cuda.empty_cache()

    # Vypočítáme, kolik paměti navíc spotřebovala jedna dávka (včetně aktivací)
    vram_for_one_batch = vram_peak_with_batch - vram_model_only
    vram_per_item = vram_for_one_batch / profile_batch_size

    # Nyní uděláme finální výpočet
    vram_total = torch.cuda.get_device_properties(0).total_memory
    vram_reserved = 1.0 * (1024**3) # Menší rezerva stačí, protože měření je přesnější
    vram_available = vram_total - vram_reserved
    vram_for_all_models = vram_model_only * num_workers
    
    vram_for_all_batches = vram_available - vram_for_all_models
    if vram_for_all_batches <= 0:
        raise MemoryError("Není dostatek VRAM ani pro nahrání všech modelů.")
        
    vram_per_worker_for_batch = vram_for_all_batches / num_workers
    
    max_batch_size = int(vram_per_worker_for_batch / vram_per_item)
    
    # Zarovnání na nejbližší (nižší) mocninu 2
    safe_batch_size = 2**int(np.log2(max_batch_size))
    
    print(f"Profilování dokončeno. Navrhovaný BATCH_SIZE_PER_WORKER: {safe_batch_size}")
    return max(32, safe_batch_size)

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


def load_metadata(slide_path, ROWS_PER_BLOCK):   
    slides = read_slides(slide_path, mpp=0.25, tile_extent=256, stride=256)
    slides = slides.map(row_hash, num_cpus=0.1, memory=3 * 1024**3)

    tiles = slides.flat_map(tiling, num_cpus=0.2, memory=3 * 1024**3).repartition(
        target_num_rows_per_block=ROWS_PER_BLOCK
    )

    tissue_tiles = tiles.map_batches(
        read_slide_tiles, num_cpus=1, memory=5 * 1024**3
    ).filter(lambda row: row["tile"].std() > 8)

    tissue_tiles = tissue_tiles.drop_columns(
        ["tile", "level", "tile_extent_x", "tile_extent_y"], memory = 3 * 1024**3
    )
    return (slides, tissue_tiles)

def load_parquet(path): 
    data = pd.read_parquet(path)
    return data

def create_slide_embeddings_service(slide_metadata, tiles_df, MODEL_DTYPE, device):
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
    def __init__(self, device: torch.device, model_dtype: torch.dtype, slide_path: str):
        self.device = device
        self.model_dtype = model_dtype
        self.slide = pyvips.Image.new_from_file(slide_path)

        tile_encoder = gigapathTile()
        tile_encoder = tile_encoder.to(device)
        tile_encoder = tile_encoder.to(model_dtype)
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
        coords = zip(x_coords, y_coords)

        batch_of_inputs = []
        
        for x, y in coords:
            patch_vips = self.slide.extract_area(int(x), int(y), 256, 256) 
            patch_array = np.asarray(patch_vips.numpy())[:, :, :3] 

            patch_pil = Image.fromarray(patch_array)
            sample_input = self.transform(patch_pil)
            batch_of_inputs.append(sample_input)

        batch_tensor = torch.stack(batch_of_inputs) 
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

def create_tile_embeddings(slide_path, device, model_dtype, tile_size, BATCH_SIZE, NUM_WORKERS, ROWS_PER_BLOCK):
    slide_metadata, tiles_metadata = load_metadata(slide_path, ROWS_PER_BLOCK)

    # needed for right pyvips integration of workers
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
            "device": device,
            "model_dtype": model_dtype,
            "slide_path": slide_path,
        },
        num_gpus=1.0/NUM_WORKERS if device.type == "cuda" else 0,
        memory=7*1024**3,
        batch_size=BATCH_SIZE,
        compute=ray.data.ActorPoolStrategy(size=NUM_WORKERS),
        runtime_env=runtime_env,
    )
    print("Processing batches...")
    all_result_dfs = []
    for result_batch in result_ds.iter_batches(batch_format="pandas"):
        all_result_dfs.append(result_batch)
    print("Processing finished.")

    if not all_result_dfs:
        return pd.DataFrame()

    return pd.concat(all_result_dfs, ignore_index=True)

def save_tile_embeddings(save_path, tiles_df):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    tiles_df.to_parquet(save_path + "/tiles.parquet", index=False)

def save_slide_embeddings(save_path, slide_df):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    slide_df.to_parquet(save_path + "/slide.parquet", index=False)

def process_slide(slide_path, save_path, device, MODEL_DTYPE, NUM_WORKERS, ROWS_PER_BLOCK, BATCH_SIZE):
    slide_metadata, tile_metadata = load_metadata(slide_path, ROWS_PER_BLOCK)
    slide_name = slide_metadata.take(1)[0]["path"].split('/')[-1].split('.')[0]
    if(not os.path.exists(save_path + '/' + slide_name)):
        print("\nStarting tile embeddings")
        tile_embeddings = create_tile_embeddings(slide_path, device, MODEL_DTYPE, 256, BATCH_SIZE, NUM_WORKERS, ROWS_PER_BLOCK)
        print("\nSaving tile embeddings")
        save_tile_embeddings(save_path + '/' + slide_name, tile_embeddings)
    else:
        print("\nTile embeddings already exists. Skipping")
        tile_embeddings = load_parquet(save_path + '/' + slide_name)

    if (not os.path.exists(save_path + '/' + slide_name +"/slide.parquet")):
        print("\nStarting slide embeddings")
        slide_embeddings = create_slide_embeddings_service(slide_metadata, tile_embeddings, MODEL_DTYPE, device)
        print("\nSaving slide embeddings")
        save_slide_embeddings(save_path + '/' + slide_name, slide_embeddings)    
    else: 
        print("\nSlide embeddings already exists. Skipping")

def main() -> None:
    parser = argparse.ArgumentParser(description="Creates tile and slide embeddings for WSI with help of Gigapath")
    
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

    args = parser.parse_args()


    #slide_path = '/mnt/data/scans/AI scans/Comparison_of_scanners/breast/FLASH2021_6802-01-T.mrxs'
    slide_path = args.slide_path
    save_path = args.save_path.rstrip('/')
    device = torch.device("cuda")
    MODEL_DTYPE = torch.bfloat16

    NUM_WORKERS, ROWS_PER_BLOCK = get_sensible_worker_config()
    BATCH_SIZE = profile_and_get_auto_batch_size(gigapathTile, MODEL_DTYPE, NUM_WORKERS)


    start_time = time.time()

    if os.path.isdir(slide_path):
        for slide_name in tqdm(os.listdir(slide_path)):
            absolute_path = os.path.join(slide_path, slide_name)
            print(absolute_path)
            if os.path.isdir(absolute_path):
                continue
            process_slide(absolute_path, save_path, device, MODEL_DTYPE, NUM_WORKERS, ROWS_PER_BLOCK, BATCH_SIZE)
        return

    process_slide(slide_path, save_path, device, MODEL_DTYPE, NUM_WORKERS, ROWS_PER_BLOCK, BATCH_SIZE)

    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"=============================================")
    print(f"Celkový čas zpracování: {elapsed_time:.2f} sekund")
    print(f"=============================================")


if __name__ == "__main__":
    main()
