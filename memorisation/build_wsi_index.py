#!/usr/bin/env python3
import torch
import numpy as np
import faiss
import openslide
from PIL import Image
from torchvision import transforms as T
import timm
import time
import os
import pickle
from timm.data import resolve_data_config, create_transform
import torch.nn as nn
from timm.layers import SwiGLUPacked
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import logging

# Setup logging FIRST
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose logging from libraries
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
logging.getLogger("timm").setLevel(logging.CRITICAL)
logging.getLogger("PIL").setLevel(logging.CRITICAL)


def get_model(device):
    """Load UNI2-h model on GPU"""
    timm_kwargs = {
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'embed_dim': 1536,
        'mlp_ratio': 5.33334,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': SwiGLUPacked,
        'act_layer': nn.SiLU,
        'reg_tokens': 8,
        'init_values': 1e-5,
        'dynamic_img_size': True
    }
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    model.eval()
    model.to(device)
    model = model.half()
    return model


def get_transform(model):
    """Get preprocessing transform"""
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    data_config = resolve_data_config(actual_model.pretrained_cfg, model=actual_model)
    return create_transform(**data_config)


def process_wsi_tiling(wsi_path, target_um=128):
    """CPU-only: Extract tile coordinates from WSI"""
    try:
        wsi = openslide.OpenSlide(wsi_path)
        mpp = float(wsi.properties["openslide.mpp-x"])
        level = 0
        tile_size = int(target_um / mpp)
        
        width, height = wsi.level_dimensions[level]
        tile_coords = []
        
        for y in range(0, height - tile_size + 1, tile_size):
            for x in range(0, width - tile_size + 1, tile_size):
                tile_coords.append((x, y))
        
        wsi.close()
        
        logger.info(f"✅ {os.path.basename(wsi_path)}: {len(tile_coords)} tiles @ {mpp:.4f}μm/px")
        
        return {
            'wsi_path': wsi_path,
            'mpp': mpp,
            'tile_size': tile_size,
            'tile_coords': tile_coords,
            'level': level
        }
    except Exception as e:
        logger.error(f"❌ Failed to tile {wsi_path}: {e}")
        return None


def extract_embeddings_batched(wsi_data, model, transform, device, batch_size=512, chunk_size=2000):
    """GPU-accelerated: Read tiles from WSI + extract embeddings"""
    wsi_path = wsi_data['wsi_path']
    tile_coords = wsi_data['tile_coords']
    mpp = wsi_data['mpp']
    tile_size = wsi_data['tile_size']
    level = wsi_data['level']
    
    wsi = openslide.OpenSlide(wsi_path)
    
    all_embeddings = []
    all_positions = []
    
    for i in range(0, len(tile_coords), chunk_size):
        chunk_coords = tile_coords[i:i + chunk_size]
        chunk_tiles = [
            wsi.read_region((x, y), level, (tile_size, tile_size)).convert("RGB")
            for x, y in chunk_coords
        ]
        
        for j in range(0, len(chunk_tiles), batch_size):
            batch_tiles = chunk_tiles[j:j + batch_size]
            batch_coords = chunk_coords[j:j + batch_size]
            
            batch = torch.stack([transform(t) for t in batch_tiles]).half().to(device)
            
            with torch.no_grad():
                emb = model(batch)
                emb = emb.cpu().numpy().astype('float32')
            
            for e, coord in zip(emb, batch_coords):
                all_embeddings.append(e)
                all_positions.append(coord)
    
    wsi.close()
    
    logger.info(f"  ✅ {os.path.basename(wsi_path)}: {len(all_embeddings)} embeddings extracted")
    
    return {
        'wsi_path': wsi_path,
        'mpp': mpp,
        'tile_size': tile_size,
        'positions': all_positions,
        'embeddings': np.stack(all_embeddings).astype('float32')
    }


def extract_wsi_on_gpu(args):
    """
    Extract one WSI on one specific GPU (runs in separate process)
    args = (wsi_idx, wsi_data, gpu_id)
    """
    wsi_idx, wsi_data, gpu_id = args
    
    # Set GPU affinity for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    device = torch.device("cuda:0")
    
    # Load model in this process
    model = get_model(device)
    transform = get_transform(model)
    
    # Extract embeddings
    result = extract_embeddings_batched(
        wsi_data, model, transform, device,
        batch_size=512, chunk_size=2000
    )
    
    logger.info(f"[GPU {gpu_id}] ✅ {os.path.basename(result['wsi_path'])}: {len(result['embeddings'])} embeddings")
    
    return result


def main():
    # Set spawn method for CUDA multiprocessing
    mp.set_start_method('spawn', force=True)
    
    num_gpus = torch.cuda.device_count()
    logger.info(f"🚀 Using {num_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
    
    # ===== STAGE 1: CPU Multiprocessing for Tiling =====
    data_dir = os.environ.get('DATA', '~/data') + '/annoPaperScanns'
    data_dir = os.path.expandvars(os.path.expanduser(data_dir))
    
    wsi_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.svs'):
                wsi_paths.append(os.path.join(root, file))
    
    logger.info(f"🔍 Found {len(wsi_paths)} WSIs")
    for p in wsi_paths[:3]:
        logger.info(f"  {p}")
    
    # Parallel tiling with 16 CPUs
    num_workers = 16
    logger.info(f"\n📦 Stage 1: Tiling {len(wsi_paths)} WSIs with {num_workers} CPUs...")
    
    with mp.Pool(num_workers) as pool:
        tiling_results = pool.map(process_wsi_tiling, wsi_paths)
    
    tiling_results = [r for r in tiling_results if r is not None]
    logger.info(f"✅ Tiling complete: {len(tiling_results)}/{len(wsi_paths)} successful\n")
    
    # ===== STAGE 2: GPU Feature Extraction (4 GPUs in parallel) =====
    logger.info(f"🚀 Stage 2: Parallel WSI extraction on {num_gpus} GPUs...")
    logger.info(f"   Each GPU processes WSIs independently\n")
    
    all_wsi_data = []
    master_embeddings = []
    master_labels = []
    
    # Prepare task list: (wsi_idx, wsi_data, gpu_id)
    tasks = []
    for idx, wsi_data in enumerate(tiling_results):
        gpu_id = idx % num_gpus  # Round-robin assignment to GPUs
        tasks.append((idx, wsi_data, gpu_id))
    
    # Process in parallel with ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_gpus, mp_context=mp.get_context('spawn')) as executor:
        results = list(executor.map(extract_wsi_on_gpu, tasks))
    
    logger.info(f"\n✅ Feature extraction complete for all WSIs\n")
    
    # Collect results
    for idx, result in enumerate(results):
        all_wsi_data.append(result)
        master_embeddings.append(result['embeddings'])
        
        for pos in result['positions']:
            master_labels.append((idx, pos))
    
    # ===== STAGE 3: Build FAISS Index =====
    logger.info("🔧 Building FAISS Cosine index...")

    master_emb = np.vstack(master_embeddings).astype('float32')
    logger.info(f"  Total embeddings: {master_emb.shape[0]} × {master_emb.shape[1]}")

    d = master_emb.shape[1]  # 1536

    # IP für Cosine Similarity
    master_index = faiss.IndexFlatIP(d)

    # Normalisieren (der "lange" Teil, einmalig!) – modifiziert master_emb IN PLACE
    logger.info(f"  Normalizing {master_emb.shape[0]} vectors...")
    faiss.normalize_L2(master_emb)
    logger.info("  Adding to index...")

    master_index.add(master_emb)
    logger.info(f"✅ Index ready: {master_index.ntotal} tiles (Cosine Similarity)")


        
    # ===== STAGE 4: Save Index =====
    index_dir = os.environ.get('DATA', '~/data') + '/indices'
    index_dir = os.path.expandvars(os.path.expanduser(index_dir))
    os.makedirs(index_dir, exist_ok=True)
    
    index_path = os.path.join(index_dir, "multi_wsi_uni2.index")
    faiss.write_index(master_index, index_path)
    logger.info(f"✅ FAISS index saved: {index_path}")
    
    meta = {
        'wsi_data': all_wsi_data,
        'labels': master_labels,
        'wsi_paths': [r['wsi_path'] for r in all_wsi_data]
    }
    meta_path = os.path.join(index_dir, "multi_wsi_uni2_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    logger.info(f"✅ Metadata saved: {meta_path}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ SUCCESS: 4-GPU Parallel Multi-WSI UNI2-h Index Complete")
    logger.info(f"   Tiles: {master_index.ntotal}")
    logger.info(f"   WSIs: {len(all_wsi_data)}")
    logger.info(f"   GPUs: {num_gpus}")
    logger.info(f"   Output: {index_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
