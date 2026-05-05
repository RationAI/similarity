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

# --- KONFIGURACE ---
INPUT_PATH = "/data/fs201053/bs37803/annoPaperScanns"
SAVE_PATH = "./tissue_embeddings_full_downscale.parquet"
PREVIEW_DIR = "./previews_full_downscale"  # Složka pro kontrolu
DEVICE = "cuda"
DTYPE = torch.float16

from src.feature_extractors import gigapathTile, virchow2, UNI2h, midnight12k, simclrv2

# --- OPTIMALIZOVANÝ PREDICTOR JAKO ACTOR ---
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

# --- CPU FUNKCE PRO RYCHLÝ DOWNSCALE + UKÁZKY ---
def process_full_downscale(row: dict):
    path = row["item"]
    try:
        slide = OpenSlide(path)
        
        # Bleskové načtení celého obrazu
        full_img = slide.get_thumbnail((224, 224)).convert('RGB')
        
        if full_img.size != (224, 224):
            full_img = full_img.resize((224, 224), Image.BILINEAR)

        # --- UKÁZKY ---
        # Uložíme náhled, pokud index (vytvořený z názvu) odpovídá vzorku
        # Abychom neukládali tisíce souborů, uložíme jen prvních 10 unikátních slidů
        # (Využijeme jednoduchý globální counter v rámci workeru není možný, tak použijeme náhodu)
        if np.random.rand() < 0.05: # Uloží cca 5% všech slidů pro kontrolu
            os.makedirs(PREVIEW_DIR, exist_ok=True)
            safe_name = urllib.parse.quote(path, safe="").replace("%", "_")[-60:]
            full_img.save(os.path.join(PREVIEW_DIR, f"{safe_name}_full.png"))

        return [{
            "slide_id": path,
            "image": np.array(full_img),
            "mode": "full_downscale"
        }]
    except Exception as e:
        return []

def main():
    if not ray.is_initialized():
        ray.init(num_cpus=44, num_gpus=1)
    
    if not os.path.exists(PREVIEW_DIR):
        os.makedirs(PREVIEW_DIR)

    exts = {".svs", ".tiff", ".mrxs", ".ndpi"}
    all_paths = [str(p) for p in Path(INPUT_PATH).rglob("*") if p.suffix.lower() in exts]
    print(f"Nalezeno {len(all_paths)} slidů. Ukládám ukázky do {PREVIEW_DIR}")

    ds = ray.data.from_items(all_paths)
    
    # 1. CPU část (Thumbnailing)
    ds = ds.flat_map(process_full_downscale)
    
    # 2. GPU část (Inference)
    results = ds.map_batches(
        EmbeddingPredictor,
        batch_size=16, 
        num_gpus=1,
        concurrency=1 
    )
    
    results.write_parquet(SAVE_PATH)
    print(f"Hotovo. Výsledky uloženy do: {SAVE_PATH}")

if __name__ == "__main__":
    main()