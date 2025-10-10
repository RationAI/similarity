import timm
from PIL import Image
from torchvision import transforms
import torch
from gigapath import slide_encoder
import timm
import torch.nn as nn
import pyvips
from src.feature_extractors import gigapathTile, gigapathSlide


slide_path = '/mnt/data/scans/AI scans/Comparison_of_scanners/breast/FLASH2021_6802-01-T.mrxs'
device = torch.device("cuda")
MODEL_DTYPE = torch.bfloat16
TILE_SIZE = 256
slide = pyvips.Image.new_from_file(slide_path)

tile_encoder = gigapathTile()
tile_encoder = tile_encoder.to(torch.device("cuda"))
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


# 1. Initialize Storage
all_tile_embeddings = []
all_coords = []
MIN_TISSUE_THRESHOLD = 0.1 # Optional: Skip tiles that are mostly white/empty

# 2. Get Slide Dimensions (Level 0)
level_0_width = slide.width
level_0_height = slide.height

print(f"Starting tiling for slide size: {level_0_width} x {level_0_height}")

# 3. Iteration Loop
for y in tqdm(range(0, level_0_height, TILE_SIZE)):
    for x in range(0, level_0_width, TILE_SIZE):
        
        # Check if the tile goes outside the slide boundary
        if x + TILE_SIZE > level_0_width or y + TILE_SIZE > level_0_height:
            continue
        
        try:
            # A. Extract the 256x256 patch (L0 coordinates)
            patch_vips = slide.extract_area(x, y, TILE_SIZE, TILE_SIZE)
            
            # B. Force computation and convert to a NumPy array (RGB)
            # This is the step that reads the pixels and takes time!
            patch_array = np.asarray(patch_vips.numpy())[:, :, :3] 

            # C. Simple Tissue Filter (Optional but highly recommended for WSIs)
            # Calculate the average color (simple measure to exclude white background)
            if np.mean(patch_array) > 220: # Example threshold for white background
                 # If the tile is too bright (mostly background), skip it
                 continue

            # D. Convert to PIL and apply transformations
            patch_pil = Image.fromarray(patch_array)
            sample_input = transform(patch_pil).unsqueeze(0).to(device).to(MODEL_DTYPE)

            # E. Run Tile Encoder
            with torch.no_grad():
                tile_embedding = tile_encoder(sample_input).squeeze()

            # F. Store results
            tile_embedding_cpu = tile_embedding.detach().cpu()
            all_tile_embeddings.append(tile_embedding_cpu)
            all_coords.append(torch.tensor([x, y]))
            # Inside the tiling loop, after using sample_input:
            del sample_input          # removes the input batch
            del tile_embedding        # removes the GPU tensor (if you kept a reference)
            torch.cuda.empty_cache()  # releases cached memory back to the allocator
            
        except pyvips.Error as e:
            # Catch errors that occur when reading corrupt or empty regions
            # print(f"Skipping tile at ({x}, {y}) due to VIPS error: {e}")
            continue

# 4. Final Data Aggregation
L = len(all_tile_embeddings)
if L > 0:
    print(f"\n--- Aggregation Complete: {L} valid tiles found ---")
    
    # [1, L, D]: Tile embeddings tensor for the slide encoder
    final_tile_embed = torch.stack(all_tile_embeddings).unsqueeze(0) 
    
    # [1, L, 2]: Coordinates tensor for the slide encoder
    final_coords = torch.stack(all_coords).unsqueeze(0).to(device).to(MODEL_DTYPE)
    
    print(f"final_tile_embed shape: {final_tile_embed.shape}")
    print(f"final_coords shape: {final_coords.shape}")

    # Now you can run your slide_encoder
    # slide_level_output = slide_encoder(final_tile_embed, final_coords).squeeze()
    