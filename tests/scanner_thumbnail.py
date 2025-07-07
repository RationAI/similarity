from openslide import OpenSlide
from constants import (
    SCANNER_X_CASE_1,
    SCANNER_X_CASE_2,
    SCANNER_X_CASE_3,
    SCANNER_Y_CASE_1,
    SCANNER_Y_CASE_2,
    SCANNER_Y_CASE_3,
)
import albumentations as A
import torch
from pathlib import Path
from src.compute_similarity import compute_similarity


def scanner_thumbnail() -> None:
    transforms = A.Compose(A.Resize(224, 224, p=1), A.Normalize(), A.ToTensor())

    def load_thumbnail(path: str | Path) -> torch.Tensor:
        with OpenSlide(path) as slide:
            thumbnail = slide.get_thumbnail((224, 224))
            return transforms(image=thumbnail)["image"]

    batch = torch.tensor(
        map(
            load_thumbnail,
            [
                SCANNER_X_CASE_1,
                SCANNER_X_CASE_2,
                SCANNER_X_CASE_3,
                SCANNER_Y_CASE_1,
                SCANNER_Y_CASE_2,
                SCANNER_Y_CASE_3,
            ],
        )
    )
    sim_score = compute_similarity(batch)
    print(sim_score)
