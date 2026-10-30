#!/usr/bin/env python3
import pandas as pd
import openslide
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def extract_exact_tile(wsi_path, x, y, tile_size=509, level=0):
    """Nur EXAKTES Tile ohne Pad"""
    wsi = openslide.OpenSlide(wsi_path)
    tile = wsi.read_region((x, y), level, (tile_size, tile_size)).convert('RGB')
    wsi.close()
    return tile.resize((224, 224))  # Standard-Vergleichsgröße


def main():
    # Pfade wie in deinen Scripts
    data_dir = os.path.expanduser(os.environ.get('DATA', '~/data') + '/annoPaperScanns')
    samples_dir = os.path.expanduser(os.environ.get('DATA', '~/data') + '/samples_uncond')
    output_dir = os.path.expanduser(os.environ.get('DATA', '~/data') + '/sample_compare')
    
    df = pd.read_csv('uni2_wsi_search_results.csv')
    top10 = df.nlargest(10, 'cosine_similarity').reset_index(drop=True)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.ioff()
    
    for i, (_, row) in enumerate(top10.iterrows()):
        print(f"{i+1}/10: {row['sample_file']} Cosine={row['cosine_similarity']:.4f}")
        
        sample_path = os.path.join(samples_dir, str(row['sample_file']))  # ← str() Fix
        sample = Image.open(sample_path).convert('RGB').resize((224, 224))
        
        wsi_path = os.path.join(data_dir, str(row['wsi_folder']), str(row['wsi_file']))  # ← str() Fix
        match_tile = extract_exact_tile(wsi_path, int(row['best_pos_px_x']), int(row['best_pos_px_y']))  # ← int() Fix
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.imshow(sample); ax1.set_title(str(row['sample_file'])); ax1.axis('off')  # ← str() Fix
        ax2.imshow(match_tile); ax2.set_title(f"Cosine={row['cosine_similarity']:.4f}"); ax2.axis('off')
        
        sample_base = os.path.splitext(str(row['sample_file']))[0]  # ← str() Fix
        plt.savefig(os.path.join(output_dir, f'{i+1:02d}_{sample_base}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✅ {output_dir}/ ready!")



if __name__ == "__main__":
    main()
