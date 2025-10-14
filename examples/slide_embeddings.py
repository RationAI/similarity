from src.feature_extractors import gigapathTile, gigapathSlide
import torch
import pyvips
from torchvision import transforms
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

from typing import Any

from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles, read_slide_tiles
from ratiopath.tiling.utils import row_hash
import ray
import os
import pandas as pd


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


def make_parquet(slide_path, save_path=""):
    full_save_path = save_path + slide_path.split("/")[-1].split(".")[0]

    if len(os.listdir(full_save_path)) != 0:
        print("Parquets already created, skipping.")
        return
        
    slides = read_slides(slide_path, mpp=0.25, tile_extent=256, stride=256)


    slides = slides.map(row_hash, num_cpus=0.1, memory=128 * 1024**2)
    slides.write_parquet(full_save_path)

    tiles = slides.flat_map(tiling, num_cpus=0.2, memory=128 * 1024**2).repartition(
        target_num_rows_per_block=4096
    )

    tissue_tiles = tiles.map_batches(
        read_slide_tiles, num_cpus=1, memory=4 * 1024**3
    ).filter(lambda row: row["tile"].std() > 8)

    tissue_tiles = tissue_tiles.drop_columns(
        ["tile", "level", "tile_extent_x", "tile_extent_y"]
    )
    tissue_tiles.write_parquet(full_save_path + "/tiles/")

def load_parquet(path): 
    if not ray.is_initialized():
        ray.init() 
    return ray.data.read_parquet(path)


def encode_tiles_gigapath(batch: pd.DataFrame) -> pd.DataFrame:
    # Získání původních klíčů pro spojení
    slide_path = batch['path']
    coords = torch.tensor([batch['x_coord'], batch['y_coord']])
    batch_of_inputs = []
    all_tile_embeddings = []

    TILE_SIZE=256

    slide = pyvips.Image.new_from_file(slide_path)

    tile_encoder = gigapathTile()
    tile_encoder = tile_encoder.to(device)
    tile_encoder = tile_encoder.to(torch.bfloat16)
    tile_encoder.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    for x, y in coords:
        patch_vips = slide.extract_area(x, y, TILE_SIZE, TILE_SIZE)
        patch_array = np.asarray(patch_vips.numpy())[:, :, :3] 

        patch_pil = Image.fromarray(patch_array)
        sample_input_tensor = transform(patch_pil) 

        batch_of_inputs.append(sample_input_tensor)

        # 3. Inference
        with torch.no_grad():
            embeddings_tensor = tile_encoder(sample_input_tensor).squeeze()

        all_tile_embeddings.append(embeddings_tensor)

    # 4. Vrácení výsledků
    # Vytvoření nového DataFrame s původními klíči a novými embeddingy
    output_df = pd.DataFrame({
        'slide_id': slide_ids,
        'x_coord': x_coords,
        'y_coord': y_coords,
        'embedding': list(all_tile_embeddings), # Uložení embeddingu jako list (pro Parquet)
    })
    
    # Užitečné pro kontrolu v logu
    print(f"Zpracováno {len(batch)} dlaždic, vytvořeno {embeddings_array.shape[1]}-rozměrných embeddingů.")

    return output_df

def create_tile_embeddings(slide_path, parquet_dataset):
    device = torch.device("cuda")
    MODEL_DTYPE = torch.bfloat16

    slide_metadata = parquet_dataset.take(1)[0]
    ## slide_metadata have slide_id none, tiles not
    tiles_metadata = parquet_dataset.filter(
        lambda row: 'slide_id' in row
    )
    TILE_SIZE = 256
    BATCH_SIZE = 256

    batch_of_inputs = []
    batch_of_coords = []
    all_tile_embeddings = []
    all_coords = []

    embeddings_ds = tiles_metadata.map_batches(
        encode_tiles_gigapath,
        batch_size=128,
    )

    # 3. Spusťte a uložte (zde se spouští celá pipeline)
    embeddings_ds.count()
    # Můžete použít ds.count() nebo ds.write_parquet(), aby se vynutilo provedení
    return

def main() -> None:
    slide_path = '/mnt/data/scans/AI scans/Comparison_of_scanners/breast/FLASH2021_6802-01-T.mrxs'
    
    make_parquet("/mnt/data/scans/AI scans/Comparison_of_scanners/breast/FLASH2021_6802-01-T.mrxs")
    dataset = load_parquet("/home/jovyan/similarity/FLASH2021_6802-01-T")

    create_tile_embeddings(slide_path, dataset)

if __name__ == "__main__":
    main()
