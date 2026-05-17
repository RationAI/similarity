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
from rationai.staining import StandardConversions, normalize_staining
from dataclasses import dataclass, asdict
import urllib.parse

import pandas.errors

# For Ray to be happy and not throw error
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

def get_processing_tasks(config: Config):
    tasks = []
    tiff_tasks = []
    input_path = Path(config.slide_path)
    output_base = Path(config.save_path)
    extensions = {".svs", ".tiff", ".mrxs"}

    if input_path.is_dir():
        for path in input_path.rglob("*"):
            if path.suffix.lower() in extensions and "_COPY" not in path.name:
                
                slide_id_val = urllib.parse.quote(str(path), safe="")
                
                check_dir = output_base / f"slide_id={slide_id_val}"
                
                is_done = check_dir.exists() and any(check_dir.glob("*.parquet"))
                
                if not is_done:
                    rel_path = path.relative_to(input_path).parent
                    save_dir = output_base / rel_path
                    if path.suffix.lower() == ".tiff":
                        tiff_tasks.append((str(path), str(save_dir)))
                    else:
                        tasks.append((str(path), str(save_dir)))
                else:
                    print(f"Skipping already processed slide: {path.name}")
                    pass

    print(f"Total tasks to process: {len(tasks)+len(tiff_tasks)}")
    return tasks, tiff_tasks

class CPUPreprocessActor:
    def __init__(self, ENCODER=0, NORMALIZE=False, CLAHE=False, RM_BG=False):
        import torch
        import cv2
        import numpy as np
        from PIL import Image
        from rationai.staining import StandardConversions, normalize_staining
        from src.feature_extractors import gigapathTile, virchow2, UNI2h, midnight12k, simclrv2

        self.cv2 = cv2
        self.np = np
        self.Image = Image
        self.normalize_staining = normalize_staining
        self.StandardConversions = StandardConversions
        
        self.NORMALIZE = NORMALIZE
        self.CLAHE = CLAHE
        self.RM_BG = RM_BG
        self.tile_size = 224
        
        self.STAIN_VECTORS = self.np.array([
            [0.64429328, 0.71655047, 0.26684416],
            [0.03448942, 0.6508934,  0.75845514]
        ])

        if self.CLAHE:
            self.clahe_obj = self.cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        encoders = [gigapathTile, virchow2, UNI2h, midnight12k, simclrv2]
        _, self.transform = encoders[ENCODER]()

    def is_tissue(self, tile: np.ndarray, threshold: float = 0.05) -> bool:
        tile_sample = tile[::2, ::2].astype(self.np.float32) / 255.0
        c_max = self.np.max(tile_sample, axis=-1)
        c_min = self.np.min(tile_sample, axis=-1)
        delta = c_max - c_min
        saturation = self.np.divide(delta, c_max, out=self.np.zeros_like(delta), where=c_max > 0)
        return self.np.mean(saturation > 0.15) > threshold

    def apply_clahe_fast(self, img_np):
        lab = self.cv2.cvtColor(img_np, self.cv2.COLOR_RGB2LAB)
        l, a, b = self.cv2.split(lab)
        l_updated = self.clahe_obj.apply(l)
        return self.cv2.cvtColor(self.cv2.merge((l_updated, a, b)), self.cv2.COLOR_LAB2RGB)

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:        
        transformed_tiles = []
        keep_indices = []
        
        tiles = batch['tile']
        x_coords = batch['tile_x']
        y_coords = batch['tile_y']
        slide_ids = batch['slide_id']
        
        for i in range(len(tiles)):
            tile_data = tiles[i]
            if tile_data.shape[0] != self.tile_size:
                tile_data = self.cv2.resize(tile_data, (self.tile_size, self.tile_size), interpolation=self.cv2.INTER_AREA)

            if self.RM_BG and not self.is_tissue(tile_data):
                continue

            if self.NORMALIZE:
                tile_data = self.normalize_staining(tile_data, self.StandardConversions.RGB2HER.matrix, self.STAIN_VECTORS[0], self.STAIN_VECTORS[1])
                if not isinstance(tile_data, self.np.ndarray): tile_data = self.np.array(tile_data)

            if self.CLAHE:
                tile_data = self.apply_clahe_fast(tile_data)

            pil_img = self.Image.fromarray(tile_data).convert("RGB")
            tensor = self.transform(pil_img)
            transformed_tiles.append(tensor.numpy())
            keep_indices.append(i)

        if not transformed_tiles:
            return pd.DataFrame({
                            "slide_id": pd.Series([], dtype=str),
                            "x_coord": pd.Series([], dtype=int),
                            "y_coord": pd.Series([], dtype=int),
                            "tile": pd.Series([], dtype=object),
                        })

        return pd.DataFrame({
            'slide_id': slide_ids[keep_indices],
            'x_coord': x_coords[keep_indices],
            'y_coord': y_coords[keep_indices],
            'tile': transformed_tiles,
        })

class GPUPredictor:
    def __init__(self, encoder_idx, device, dtype):
        import torch
        import numpy as np
        from src.feature_extractors import gigapathTile, virchow2, UNI2h, midnight12k, simclrv2
        self.device = torch.device(device)
        self.dtype = dtype
        self.torch = torch
        self.np = np
        
        encoders = [gigapathTile, virchow2, UNI2h, midnight12k, simclrv2]
        model, _ = encoders[encoder_idx]()
        self.model = model.to(self.device).to(self.dtype).eval()
        self.encoder_idx = encoder_idx

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        if len(batch.get('slide_id', [])) == 0:
            return pd.DataFrame({
                "slide_id": pd.Series([], dtype=str),
                "x_coord": pd.Series([], dtype=int),
                "y_coord": pd.Series([], dtype=int),
                "embedding": pd.Series([], dtype=object),
            })

        data_stack = self.np.stack(batch['tile'])
        batch_tensor = self.torch.as_tensor(data_stack).to(self.device, non_blocking=True).to(self.dtype)

        with self.torch.no_grad():
            output = self.model(batch_tensor)
            
        embeddings_array = output.cpu().to(self.torch.float16).numpy()
        
        del output, batch_tensor

        return pd.DataFrame({
            'slide_id': batch['slide_id'],
            'x_coord': batch['x_coord'],
            'y_coord': batch['y_coord'],
            'embedding': list(embeddings_array),
        })

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
        help="Choose the encoder you want to use: 0: Gigapath, 1: Virchow2, 2: UNI2-h, 3: midnight-12k, 4: SimCLRv2"
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

def main() -> None:
    config = parse_args()

    ray.init(logging_level=logging.ERROR, configure_logging=True, object_store_memory=50 * 1024**3)

    # to not spill on disk and slow down whole computation
    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.max_pending_blocks = 20
    ctx.execution_options.spill_threshold = 0.99
    ctx.prefetch_capacity = 20

    try:
        start_time = time.time()
        tasks, tiff_tasks = get_processing_tasks(config)
        slide_paths = [t[0] for t in tasks]
        tiff_paths = [t[0] for t in tiff_tasks]

        print(slide_paths)
        print(tiff_paths)
        
        ds_slide = None
        ds_tiff = None

        if slide_paths:
            ds_slide = read_slides(slide_paths, mpp=config.mpp, tile_extent=config.tile_size, stride=config.tile_size)
        if tiff_paths: # I hate you Jakub
            if config.mpp == 0.5:
                ds_tiff = read_slides(tiff_paths, level=0, tile_extent=448, stride=448)
            elif config.mpp == 1.0:
                ds_tiff = read_slides(tiff_paths, level=0, tile_extent=896, stride=896)
            else:
                ds_tiff = read_slides(tiff_paths, level=0, tile_extent=1792, stride=1792)

        if ds_slide and ds_tiff:
            ds = ds_slide.union(ds_tiff)
        elif ds_slide:
            ds = ds_slide
        elif ds_tiff:
            ds = ds_tiff
        else:
            print("no slides to process.")
            ray.shutdown()
            return

        cpus_concurrency = 20 # for data reading

        ds = ds.flat_map(tiling)
        ds = ds.map_batches(read_slide_tiles, batch_size=128, num_cpus=1, concurrency=cpus_concurrency)

        ds = ds.map_batches(
            CPUPreprocessActor,
            fn_constructor_kwargs={
                "NORMALIZE": config.normalize,
                "CLAHE": config.clahe,
                "RM_BG": config.rmBackground,
                "ENCODER": config.encoder},
            compute=ray.data.ActorPoolStrategy(size=10),
            num_cpus=1,
            batch_size=32
        )

        # --- 2. GPU INFERENCE (1 worker na H100) ---
        results = ds.map_batches(
            GPUPredictor,
            fn_constructor_kwargs={
                "encoder_idx": config.encoder, 
                "device": "cuda", 
                "dtype": config.model_dtype
            },
            compute=ray.data.ActorPoolStrategy(size=1),
            num_gpus=1,
            num_cpus=4,
            batch_size=256
        )

        results.write_parquet(
            config.save_path, 
            partition_cols=["slide_id"], 
        )

        print(f"Finished in {time.time() - start_time:.2f} seconds")
        print(f"Time per slide: {(time.time() - start_time) / (len(slide_paths) + len(tiff_paths)):.2f} seconds")
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
