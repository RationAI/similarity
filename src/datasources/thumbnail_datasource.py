import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pyvips
from pathlib import Path
from sys import getsizeof
from typing import Iterator

import pyarrow
from ray.data.block import Block
from ray.data.datasource import FileBasedDatasource

FILE_EXTENSIONS = [
    "svs",
    "tif",
    "dcm",
    "ndpi",
    "vms",
    "vmu",
    "scn",
    "mrxs",
    "tiff",
    "svslide",
    "bif",
    "czi",
]


class ThumbnailDatasource(FileBasedDatasource):
    def __init__(
        self,
        paths: str | list[str],
        *,
        image_extent: int | tuple[int, int],
        **file_based_datasource_kwargs,
    ) -> None:
        super().__init__(
            paths, file_extensions=FILE_EXTENSIONS, **file_based_datasource_kwargs
        )

        self.image_extent = np.broadcast_to(image_extent, 2)
        self.transform = A.Compose(
            [
                A.Resize(*self.image_extent),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    def _read_stream(self, f: pyarrow.NativeFile, path: str) -> Iterator[Block]:
        from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder

        # Use pyvips to create a thumbnail. This is much faster than
        # loading the full image into memory, especially for formats like .svs or .tiff.
        image = pyvips.Image.thumbnail(path, self.image_extent[0])

        # Convert pyvips image to a numpy array and ensure it's in RGB format.
        # We also drop the alpha channel if it exists.
        image_np = image.numpy()[:, :, :3]

        # Apply the Albumentations transformations
        img = self.transform(image=image_np)["image"]

        builder = DelegatingBlockBuilder()
        item = {
            "path": path,
            "name": Path(path).stem,
            "img": img,
        }
        builder.add(item)
        yield builder.build()

    def _rows_per_file(self) -> int:  # type: ignore[override]
        return 1

    def estimate_inmemory_data_size(self) -> int | None:
        paths = self._paths()
        if not paths:
            return 0

        # Create a sample item to calculate the base size of a single row.
        sample_item = {"path": "", "img": np.zeros([3, *self.image_extent])}

        # Calculate the size of the dictionary structure, keys, and fixed-size values.
        base_row_size = getsizeof(sample_item)
        for k, v in sample_item.items():
            base_row_size += getsizeof(k)
            base_row_size += getsizeof(v)

        # Calculate the total size of all path strings.
        total_path_size = sum(getsizeof(p) for p in paths)

        # The total estimated size is the base size for each row plus the total size of paths.
        return base_row_size * len(paths) + total_path_size
