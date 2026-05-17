import openslide
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def read_clean_with_bg(slide, x, y, level, size):
    """
    Načte region a zajistí, že průhledné oblasti budou bílé.
    """
    img = slide.read_region((x, y), level, size)
    # Vytvoření bílého plátna
    bg = Image.new("RGB", img.size, (255, 255, 255))
    
    # Pokud má obrázek Alpha kanál (průhlednost), použijeme ho jako masku
    if img.mode == 'RGBA':
        bg.paste(img, mask=img.split()[3])
    else:
        bg.paste(img)
    return bg

def get_tissue_bbox(slide, level=4):
    """Najde bounding box tkáně a eliminuje vliv průhlednosti."""
    # Načtení náhledu na bílé pozadí
    dims = slide.level_dimensions[level]
    img = read_clean_with_bg(slide, 0, 0, level, dims).convert('L')
    thumb_np = np.array(img)
    
    # Maska tkáně (vše co není bílé pozadí 255 a není úplná černá 0)
    mask = (thumb_np < 235) & (thumb_np > 15)
    
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    
    ds = slide.level_downsamples[level]
    return (int(x0 * ds), int(y0 * ds), int((x1 - x0) * ds), int((y1 - y0) * ds))

def plot_comparison_images(flash_path, level=3):
    midi_path = flash_path.replace("FLASH", "MIDI")
    sf = openslide.OpenSlide(flash_path)
    sm = openslide.OpenSlide(midi_path)

    ds = sf.level_downsamples[level]

    # 1. Bounding box pro FLASH
    bbox_f = get_tissue_bbox(sf, level=level+1)
    if bbox_f:
        xf, yf, wf, hf = bbox_f
        target_size_f = (int(wf / ds), int(hf / ds))
        crop_f = read_clean_with_bg(sf, xf, yf, level, target_size_f)
    else:
        crop_f = None

    # 2. Bounding box pro MIDI (samostatně, aby nebyla ořízlá)
    bbox_m = get_tissue_bbox(sm, level=level+1)
    if bbox_m:
        xm, ym, wm, hm = bbox_m
        target_size_m = (int(wm / ds), int(hm / ds))
        crop_m = read_clean_with_bg(sm, xm, ym, level, target_size_m)
    else:
        crop_m = None

    # Vykreslení
    if crop_f and crop_m:
        # --- NASTAVENÍ PÍSMA PRO DIPLOMKU (LATEX / COMPUTER MODERN) ---
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 15,
            "axes.titlesize": 16
        })
        # V názvech pak nepoužívej r"\textbf{...}", ale klasický text (bude v patkovém fontu)

        # Vytvoření figury s minimálními mezerami (wspace=0.02 dává 2% šířky jako mezeru mezi obrázky)
        fig, ax = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'wspace': 0.02})
        
        ax[0].imshow(crop_f)
        ax[0].set_title("FLASH scanner") # \textbf funguje díky usetex=True
        
        ax[1].imshow(crop_m)
        ax[1].set_title("MIDI scanner")
        
        for a in ax: 
            a.axis('off')
            # Odstranění vnitřních neviditelných okrajů okolo os
            a.xaxis.set_major_locator(plt.NullLocator())
            a.yaxis.set_major_locator(plt.NullLocator())

        # Uložení BEZ bílých okrajů (pad_inches=0)
        plt.savefig(
            "v1_cropped_fix.png", 
            bbox_inches='tight', 
            pad_inches=0, 
            dpi=300 # Vyšší rozlišení pro tisk v diplomce
        )
        plt.show()

    sf.close()
    sm.close()

# --- SPUŠTĚNÍ ---
flash_path = "/mnt/data/MOU/breast/comparison_of_scanners/FLASH2021_6802-01-T.mrxs"
plot_comparison_images(flash_path, level=3)