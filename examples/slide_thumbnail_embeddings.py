import os
import torch
import ray
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from openslide import OpenSlide
from PIL import Image
from pathlib import Path
import urllib.parse
import argparse

from src.feature_extractors import gigapathTile, virchow2, UNI2h, midnight12k, simclrv2

def get_args():
    parser = argparse.ArgumentParser(description="Extrakce embeddingů z celých slidů (full downscale).")
    
    # Cesty
    parser.add_argument("--input", type=str, required=True,
                        help="Cesta ke složce se slidy.")
    parser.add_argument("--output", type=str, required=True,
                        help="Cesta pro uložení výsledného Parquet souboru.")
    
    # Hardware / Ray
    parser.add_argument("--cpus", type=int, default=16, help="Počet CPU jader pro Ray.")
    parser.add_argument("--gpus", type=int, default=1, help="Počet GPU pro Ray.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size pro inference.")
    
    return parser.parse_args()

# Globální proměnné pro Predictor (budou nastaveny v main)
DEVICE = "cuda"
DTYPE = torch.float16

class EmbeddingPredictor:
    def __init__(self):
        self.device = torch.device(DEVICE)
        self.dtype = DTYPE
        self.encoder_names = ["gigapath", "virchow2", "uni2h", "midnight", "simclrv2"]
        loaders = [gigapathTile, virchow2, UNI2h, midnight12k, simclrv2]
        
        self.models = []
        self.transforms = []
        
        print("Načítám modely do GPU...")
        for loader in loaders:
            model, transform = loader()
            model = model.to(self.device).to(self.dtype).eval()
            self.models.append(model)
            self.transforms.append(transform)

    def __call__(self, batch):
        new_batch = {"slide_id": batch["slide_id"], "mode": batch["mode"]}
        for name in self.encoder_names:
            new_batch[f"emb_{name}"] = []

        for img_array in batch["image"]:
            pil_img = Image.fromarray(img_array)
            for name, model, transform in zip(self.encoder_names, self.models, self.transforms):
                tensor = transform(pil_img).unsqueeze(0).to(self.device).to(self.dtype)
                with torch.no_grad():
                    output = model(tensor)
                    if isinstance(output, dict):
                        emb = output.get("global_pool", output.get("pooler", next(iter(output.values()))))
                    else:
                        emb = output
                    new_batch[f"emb_{name}"].append(emb.cpu().numpy().astype(np.float32).flatten())
        return new_batch

def process_full_downscale(row: dict):
    path = row["item"]
    try:
        slide = OpenSlide(path)
        full_img = slide.get_thumbnail((224, 224)).convert('RGB')
        
        if full_img.size != (224, 224):
            full_img = full_img.resize((224, 224), Image.BILINEAR)

        return [{
            "slide_id": path,
            "image": np.array(full_img),
            "mode": "full_downscale"
        }]
    except Exception as e:
        return []

def main():
    args = get_args()

    if not ray.is_initialized():
        ray.init(num_cpus=args.cpus, num_gpus=args.gpus)
    

    exts = {".svs", ".tiff", ".mrxs", ".ndpi"}
    all_paths = [str(p) for p in Path(args.input).rglob("*") if p.suffix.lower() in exts]
    print(f"Nalezeno {len(all_paths)} slidů.")

    ds = ray.data.from_items(all_paths)
    
    # 1. CPU část - předáváme argumenty pomocí fn_args
    ds = ds.flat_map(
        process_full_downscale
    )
    
    # 2. GPU část
    results = ds.map_batches(
        EmbeddingPredictor,
        batch_size=args.batch_size, 
        num_gpus=args.gpus,
        concurrency=args.gpus # Typicky chceme 1 instanci modelu na 1 GPU
    )
    
    results.write_parquet(args.output)
    print(f"Hotovo. Výsledky uloženy do: {args.output}")

if __name__ == "__main__":
    main()