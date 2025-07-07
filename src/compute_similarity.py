from src.feature_extractors import vgg16  # , simclr
import torch
from typing import overload, Iterable


@torch.no_grad()
@overload
def compute_similarity(
    images: torch.Tensor,
    model: torch.nn.Module = vgg16(),
    f: torch.nn.Module = torch.nn.CosineSimilarity(),
) -> torch.Tensor:
    model.eval()
    features = model(images).flatten()
    return f(features[:, :, None], features.T[None, :, :])


@torch.no_grad()
@overload
def compute_similarity(
    source: torch.utils.data.DataLoader,
    model: torch.nn.Module = vgg16(),
    f: torch.nn.Module = torch.nn.CosineSimilarity(),
) -> Iterable[torch.Tensor]:
    model.eval()
    for batch in source:
        features = model(batch).flatten()
        yield f(features[:, :, None], features.T[None, :, :])
