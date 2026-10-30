# Image helpers copied unchanged from path_diffusion/util/img_util.py.
import numpy as np
import torch


def unnormalize_img(img):
    """
    Converts a normalized tensor image (range [-1, 1]) to unnormalized numpy array (range [0, 255]).

    Args:
        img (torch.Tensor): Input tensor image with values in range [-1, 1].
                           Can be a single image with shape (C, H, W) or a batch of images (B, C, H, W).

    Returns:
        numpy.ndarray: Unnormalized image(s) with values in range [0, 255] in uint8 format.
                      For single images: shape (H, W, C)
                      For batched images: shape (B, H, W, C)
    """
    img = 127.5 * (img + 1)

    # Handle both single images and batched images
    if len(img.shape) == 4:  # Batched images: (B, C, H, W)
        # Convert to (B, H, W, C) format for numpy
        img = img.to(torch.uint8).cpu().numpy().transpose(0, 2, 3, 1)
    else:  # Single image: (C, H, W)
        # Convert to (H, W, C) format for numpy
        img = img.to(torch.uint8).cpu().numpy().transpose(1, 2, 0)

    return img


def pil_to_np(patch_PIL):
    patch_PIL = patch_PIL.convert("RGB")
    patch_PIL = np.array(patch_PIL).astype(np.float32) / 127.5 - 1
    patch_PIL = np.transpose(patch_PIL, [2, 0, 1])

    return patch_PIL
