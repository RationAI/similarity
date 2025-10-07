import torch
from pathlib import Path
from src.feature_extractors import gigapathTile, gigapathSlide

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
