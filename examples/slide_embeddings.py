import pyvips
import torch
import ray
import os
import argparse
import pandas as pd
import numpy as np

from PIL import Image
from typing import Any
from torchvision import transforms
from ratiopath.ray import read_slides
from ratiopath.tiling.utils import row_hash
from ratiopath.tiling import grid_tiles, read_slide_tiles
from src.feature_extractors import gigapathTile, gigapathSlide


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
    slides = slides.map(row_hash, num_cpus=0.1, memory=128 * 1024**2)

    tiles = slides.flat_map(tiling, num_cpus=0.2, memory=128 * 1024**2).repartition(
        target_num_rows_per_block=4096
    )

    tissue_tiles = tiles.map_batches(
        read_slide_tiles, num_cpus=1, memory=4 * 1024**3
    ).filter(lambda row: row["tile"].std() > 8)

    tissue_tiles = tissue_tiles.drop_columns(
        ["tile", "level", "tile_extent_x", "tile_extent_y"]
    )
    return (slides, tissue_tiles)

def load_parquet(path): 
    data = pd.read_parquet(path)
    return data

def print_parquets(path): 
    p = load_parquet(path)
    print(p.count())
    print(p.take(1))
    return 

def create_slide_embeddings(slide_metadata, tiles_df, MODEL_DTYPE, device):
    slide_encoder = gigapathSlide()
    slide_encoder = slide_encoder.to(MODEL_DTYPE)
    slide_encoder = slide_encoder.to(device)

    embeddings_list_of_arrays = tiles_df['embedding'].to_list() 
    embeddings_numpy = np.stack(embeddings_list_of_arrays)
    embeddings_tensor = torch.from_numpy(embeddings_numpy).to(MODEL_DTYPE).to(device)
    final_input_tensor = embeddings_tensor.unsqueeze(0)


    x_coords = tiles_df['x_coord'].to_numpy()
    y_coords = tiles_df['y_coord'].to_numpy()
    coords_numpy = np.stack([x_coords, y_coords], axis=1).astype(np.int64)
    coords_tensor = torch.from_numpy(coords_numpy).to(torch.int64) 
    final_coords_tensor = coords_tensor.unsqueeze(0).to(device)

    slide_embedding = slide_encoder(final_input_tensor, final_coords_tensor)[0].squeeze().cpu().to(torch.float32).detach().numpy()

    del slide_encoder
    torch.cuda.empty_cache()

    slide_metadata_df = slide_metadata.to_pandas()
    slide_metadata_df['embedding'] = pd.Series([slide_embedding])

    return slide_metadata_df # Vrací standardní Pandas DataFrame




def encode_tiles_gigapath(batch: pd.DataFrame, slide: pyvips.Image, tile_encoder, transform, device, MODEL_DTYPE, TILE_SIZE):
    
    x_coords = batch['tile_x']
    y_coords = batch['tile_y']
    coords = zip(x_coords, y_coords)
    
    batch_of_inputs = []
    
    for x, y in coords:
        patch_vips = slide.extract_area(int(x), int(y), TILE_SIZE, TILE_SIZE) 
        patch_array = np.asarray(patch_vips.numpy())[:, :, :3] 

        patch_pil = Image.fromarray(patch_array)
        sample_input = transform(patch_pil)
        batch_of_inputs.append(sample_input)

    batch_tensor = torch.stack(batch_of_inputs) 
    final_input_tensor = batch_tensor.to(device).to(MODEL_DTYPE)

    with torch.no_grad():
        embeddings_tensor = tile_encoder(final_input_tensor).squeeze()

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

def create_tile_embeddings(slide_path, device, MODEL_DTYPE, BATCH_SIZE, TILE_SIZE):
    tile_encoder = gigapathTile()
    tile_encoder = tile_encoder.to(device)
    tile_encoder = tile_encoder.to(torch.bfloat16)
    tile_encoder.eval()

    slide = pyvips.Image.new_from_file(slide_path)
    
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    slide_metadata, tiles_metadata = load_metadata(slide_path)
    
    batch_iterator = tiles_metadata.iter_batches(batch_size=BATCH_SIZE)
    all_result_dfs = []
    
    for b in batch_iterator:
        e = encode_tiles_gigapath(b, slide, tile_encoder, transform, device, MODEL_DTYPE, TILE_SIZE)
        all_result_dfs.append(e)
    
    output = pd.concat(all_result_dfs, ignore_index=True)

    del slide
    del tile_encoder
    torch.cuda.empty_cache()

    return output

def save_tile_embeddings(save_path, tiles_df):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    tiles_df.to_parquet(save_path + "/tiles.parquet", index=False)

def save_slide_embeddings(save_path, slide_df):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    slide_df.to_parquet(save_path + "/slide.parquet", index=False)

def process_slide(slide_path, save_path, device, MODEL_DTYPE):
    slide_metadata, tile_metadata = load_metadata(slide_path)
    slide_name = slide_metadata.take(1)[0]["path"].split('/')[-1].split('.')[0]
    if(not os.path.exists(save_path + '/' + slide_name)):
        print("\nStarting tile embeddings")
        tile_embeddings = create_tile_embeddings(slide_path, device, MODEL_DTYPE, 256, 256)
        print("\nSaving tile embeddings")
        save_tile_embeddings(save_path + '/' + slide_name, tile_embeddings)
    else:
        print("\nTile embeddings already exists. Skipping")
        tile_embeddings = load_parquet(save_path + '/' + slide_name)

    if (not os.path.exists(save_path + '/' + slide_name +"/slide.parquet")):
        print("\nStarting slide embeddings")
        slide_embeddings = create_slide_embeddings(slide_metadata, tile_embeddings, MODEL_DTYPE, device)
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

    if os.path.isdir(slide_path):
        for slide in tqdm(os.listdir(slide_path)):
            process_slide(slide, save_path, device, model)
        return


if __name__ == "__main__":
    main()
