import sys
import os
import pathlib
import random
import openslide
import torch
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
from tqdm import tqdm
from omegaconf import omegaconf
from PIL import Image

# --- CESTY ---
sys.path.append('/home/jb88526/similarity') 
DATA_ROOT = '/data/fs201053/bs37803/annoPaperScanns' 
OUTPUT_ROOT = '/data/fs201053/jb88526/reidentification_test_complete_v2'

from sample import karras_sample
from util.script_util import get_model_and_diffusion
from util.img_util import unnormalize_img, pil_to_np
from src.feature_extractors import gigapathTile, virchow2, UNI2h, midnight12k, simclrv2

# --- RATIONAI STAINING IMPORT ---
from rationai.staining import StandardConversions, normalize_staining

def set_seed(seed=42):
    # 1. Základní Python a NumPy
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. PyTorch (CPU i GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # pro případ multi-GPU

    # 3. Vynucení deterministických algoritmů v CUDA
    # POZOR: Toto může mírně snížit výkon, ale zajistí bitovou shodu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Pro novější verze PyTorch (vynutí chybu, pokud operace nemá det. alternativu)
    # torch.use_deterministic_algorithms(True)

class Model:
    @torch.no_grad()
    def __init__(self, cfg=None, model_path=None):
        model, diffusion = get_model_and_diffusion(cfg, model_path)
        model.eval()
        self.patch_size = cfg.dataset.img_size
        self.model = model
        self.cfg = cfg
        self.diffusion = diffusion

    def get_semantic_code(self, img):
        return self.model.encode(img)['cond']

    def sample(self, semantic_code, steps=50):
        model_kwargs = {'cond': semantic_code}
        return karras_sample(self.diffusion, self.model,
                            (semantic_code.shape[0], 3, self.patch_size, self.patch_size),
                            steps=steps, model_kwargs=model_kwargs, device='cuda',
                            clip_denoised=self.cfg.sampler.clip_denoised, sampler='heun',
                            sigma_min=self.cfg.sampler.sigma_min, sigma_max=self.cfg.sampler.sigma_max)

def get_all_encoders():
    # Načte Gigapath, Virchow2, UNI2, UNI2h
    fns = [gigapathTile, virchow2, UNI2h, midnight12k, simclrv2]
    models = []
    for fn in fns:
        m, _ = fn()
        models.append(m.cuda().eval().to(torch.bfloat16))
    return models

def get_random_active_patches(slide, n_patches, patch_size_pixels, target_size, threshold=220):
    w, h = slide.dimensions
    patches = []
    attempts = 0
    max_attempts = n_patches * 20 
    while len(patches) < n_patches and attempts < max_attempts:
        attempts += 1
        x = random.randint(0, w - patch_size_pixels)
        y = random.randint(0, h - patch_size_pixels)
        check_region = slide.read_region((x, y), 0, (patch_size_pixels, patch_size_pixels)).convert('L')
        if np.mean(np.array(check_region)) < threshold:
            full_patch = slide.read_region((x, y), 0, (patch_size_pixels, patch_size_pixels)).convert('RGB')
            resized_patch = full_patch.resize((target_size, target_size))
            patches.append(torch.tensor(pil_to_np(resized_patch)))
    if not patches: return None
    return torch.stack(patches)

def apply_rationai_staining(batch_tensor):
    """Aplikuje rationAI normalizaci na batch tensorů (GPU -> CPU -> GPU)"""
    # 1. Převod z GPU tensoru [-1, 1] na numpy uint8 [0, 255]
    batch_np = unnormalize_img(batch_tensor) # vrací numpy array [B, H, W, C]
    
    # Standardní vektory pro rationAI (H&E)
    STAIN_VECTORS = np.array([
        [0.64429328, 0.71655047, 0.26684416],
        [0.03448942, 0.6508934,  0.75845514]
    ])
    
    norm_tiles = []
    for i in range(batch_np.shape[0]):
        try:
            # rationAI magie
            tile_norm = normalize_staining(
                batch_np[i], 
                StandardConversions.RGB2HER.matrix, 
                STAIN_VECTORS[0], 
                STAIN_VECTORS[1]
            )
            # Pokud vrátí PIL, převedeme zpět
            if not isinstance(tile_norm, np.ndarray):
                tile_norm = np.array(tile_norm)
            norm_tiles.append(tile_norm)
        except:
            # Pokud selže, použijeme původní (aby se nerozbil batch)
            norm_tiles.append(batch_np[i])
            
    # 2. Převod zpět na tensor [B, C, H, W] na GPU
    norm_array = np.stack(norm_tiles)
    norm_tensor = torch.from_numpy(norm_array).permute(0, 3, 1, 2).float().cuda()
    # Normalizace zpět do rozsahu [0, 1] nebo [-1, 1] dle potřeby enkodérů (většinou 0-1 po transform)
    return norm_tensor / 255.0

def main():
    set_seed()
    cfg = omegaconf.OmegaConf.load(pathlib.Path(__file__).parent / 'cfg' / 'tcga_brca.yaml')
    gen_model = Model(cfg, 'model.pt')
    enc_models = get_all_encoders()
    enc_names = ["gigapath", "virchow2", "uni2h", "midnight12k", "simclrv2"]

    patient_dirs = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    
    for p_id in tqdm(patient_dirs, desc="Pacienti"):
        p_path = os.path.join(DATA_ROOT, p_id)
        svs_files = [f for f in os.listdir(p_path) if f.lower().endswith('.svs')]
        if not svs_files: continue
        
        slide_output_dir = os.path.join(OUTPUT_ROOT, p_id, svs_files[0])
        os.makedirs(slide_output_dir, exist_ok=True)

        # Kontrola existence posledního souboru pro skip
        if os.path.exists(os.path.join(slide_output_dir, "gen_embeddings_simclrv2_norm.parquet")):
            continue

        try:
            slide = openslide.open_slide(os.path.join(p_path, svs_files[0]))
            test_patches = get_random_active_patches(slide, n_patches=500, 
                                                   patch_size_pixels=256, 
                                                   target_size=cfg.dataset.img_size)
            if test_patches is None: continue

            storage = {f"{name}_{var}": {"orig": [], "gen": []} 
                       for name in enc_names for var in ["raw", "norm"]}

            batch_size = 32 
            for i in range(0, len(test_patches), batch_size):
                batch_orig = test_patches[i : i + batch_size].cuda()
                
                with torch.no_grad():
                    # 1. GENERAVÁNÍ
                    codes = gen_model.get_semantic_code(batch_orig)
                    batch_gen = gen_model.sample(codes, steps=50)

                    # 2. RAW VARIANTY (Center Crop)
                    b_orig_raw = TF.center_crop(batch_orig, [224, 224])
                    b_gen_raw = TF.center_crop(batch_gen, [224, 224])

                    # 3. EXTRAKCE RAW
                    for idx, (name, model) in enumerate(zip(enc_names, enc_models)):
                        e_o = model(b_orig_raw.to(torch.bfloat16)).cpu().to(torch.float16).numpy()
                        e_g = model(b_gen_raw.to(torch.bfloat16)).cpu().to(torch.float16).numpy()
                        storage[f"{name}_raw"]["orig"].extend(e_o.tolist())
                        storage[f"{name}_raw"]["gen"].extend(e_g.tolist())

                    # 4. NORMALIZACE (rationAI)
                    b_orig_norm = apply_rationai_staining(b_orig_raw)
                    b_gen_norm = apply_rationai_staining(b_gen_raw)

                    # 5. EXTRAKCE NORM
                    for idx, (name, model) in enumerate(zip(enc_names, enc_models)):
                        e_o_n = model(b_orig_norm.to(torch.bfloat16)).cpu().to(torch.float16).numpy()
                        e_g_n = model(b_gen_norm.to(torch.bfloat16)).cpu().to(torch.float16).numpy()
                        storage[f"{name}_norm"]["orig"].extend(e_o_n.tolist())
                        storage[f"{name}_norm"]["gen"].extend(e_g_n.tolist())

            # 6. ULOŽENÍ
            for key, data in storage.items():
                pd.DataFrame({"embedding": data["orig"]}).to_parquet(
                    os.path.join(slide_output_dir, f"orig_embeddings_{key}.parquet")
                )
                pd.DataFrame({"embedding": data["gen"]}).to_parquet(
                    os.path.join(slide_output_dir, f"gen_embeddings_{key}.parquet")
                )
            
        except Exception as e:
            print(f"CHYBA u {p_id}: {e}")

if __name__ == '__main__':
    main()