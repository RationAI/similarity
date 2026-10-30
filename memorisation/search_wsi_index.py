#!/usr/bin/env python3
import torch
import faiss
import pickle
import cv2
from PIL import Image
from torchvision import transforms as T
import timm
import numpy as np
import os
import pandas as pd
import torch.nn as nn
from timm.layers import SwiGLUPacked
from timm.data import resolve_data_config, create_transform
import logging
logging.disable(logging.CRITICAL)  # Clean Logs

def get_model(device):
    timm_kwargs = {
        'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'embed_dim': 1536, 'mlp_ratio': 5.33334, 'num_classes': 0,
        'no_embed_class': True, 'mlp_layer': SwiGLUPacked, 'act_layer': nn.SiLU,
        'reg_tokens': 8, 'init_values': 1e-5, 'dynamic_img_size': True
    }
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    model.eval()
    model.to(device)
    model = model.half()
    return model

def get_transform(model):
    data_config = resolve_data_config(model.pretrained_cfg, model=model)
    return create_transform(**data_config)

def main():
    device = torch.device("cuda")
    num_gpus = torch.cuda.device_count()  # 4!
    print(f"🚀 Using {num_gpus} GPUs")

    index_dir = os.path.expanduser(os.environ.get('DATA', '~/data') + '/indices/multi_wsi_uni2.index')
    index_meta = os.path.expanduser(os.environ.get('DATA', '~/data') + '/indices/multi_wsi_uni2_meta.pkl')

    # 1) FAISS CPU-Index laden (ANTI-OOM!)
    index = faiss.read_index(index_dir)
    print(f"✅ UNI2-h Index: {index.ntotal} tiles (dim 1536) – CPU-only")

    with open(index_meta, "rb") as f:
        meta = pickle.load(f)
    wsi_data_list = meta['wsi_data']
    labels = meta['labels']
    wsi_paths = meta['wsi_paths']

    # 2) Multi-GPU Model + Transform (einmalig)
    model_single = get_model(device)
    transform = get_transform(model_single)
    model = nn.DataParallel(model_single)  # ← 4 GPUs!

    # 3) Samples laden
    samples_dir = os.path.expanduser(os.environ.get('DATA', '~/data') + '/samples_uncond')
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    sample_files = [os.path.join(samples_dir, f) for f in sorted(os.listdir(samples_dir)) 
                    if os.path.splitext(f.lower())[1] in exts]
    print(f"Found {len(sample_files)} samples")

    # 4) BATCHED Search (32 pro Batch → null Overhead)
    BATCH_SIZE = 32
    all_results = []
    for i in range(0, len(sample_files), BATCH_SIZE):
        batch_files = sample_files[i:i+BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}: {len(batch_files)} samples")

        batch_tensors = []
        for path in batch_files:
            img = cv2.imread(path)
            if img is None: continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)
            tensor = transform(pil).unsqueeze(0).half().to(device, non_blocking=True)
            batch_tensors.append(tensor)

        if not batch_tensors: continue
        batch_tensor = torch.cat(batch_tensors, dim=0)  # [N, 3, 224, 224]

        # Multi-GPU Inference
        with torch.no_grad():
            batch_emb = model(batch_tensor).cpu().numpy().astype("float32")  # [N, 1536]

        faiss.normalize_L2(batch_emb)
        D_cos, I_cos = index.search(batch_emb, k=1)  # Roh → echte Cosines!


        # Results dekodieren
        for j, global_idx in enumerate(I_cos.flatten()):
            d_cos = D_cos[j][0]
            wsi_idx, local_pos = labels[global_idx]
            wsi_path = wsi_paths[wsi_idx]
            wsi_data = wsi_data_list[wsi_idx]
            x, y = local_pos
            mpp = wsi_data['mpp']
            x_um, y_um = x * mpp, y * mpp

            all_results.append({
                "sample_file": os.path.basename(batch_files[j]),
                "wsi_folder": os.path.basename(os.path.dirname(wsi_path)),
                "wsi_file": os.path.basename(wsi_path),
                "best_pos_px_x": x, "best_pos_px_y": y,
                "best_pos_um_x": round(x_um), "best_pos_um_y": round(y_um),
                "cosine_similarity": round(d_cos, 4),
            })

    # 5) CSV + Stats
    df = pd.DataFrame(all_results)
    df = df.sort_values("cosine_similarity", ascending=False)
    df.to_csv("uni2_wsi_search_results.csv", index=False)
    print(f"\n✅ Saved {len(df)} results!")
    print(df.head())
    print(f"🏆 Mean Cosine: {df['cosine_similarity'].mean():.4f} ± {df['cosine_similarity'].std():.4f}")

if __name__ == "__main__":
    main()
