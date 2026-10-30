import openslide
import pathlib
import torch
import numpy as np

from omegaconf import omegaconf
from PIL import Image

from img_util import unnormalize_img, pil_to_np

class Steganography:
    def _int_to_bin(self, rgb):
        """Convert an integer tuple to a binary (string) tuple.

        :param rgb: An integer tuple like (220, 110, 96)
        :return: A string tuple like ("00101010", "11101011", "00010110")
        """
        r, g, b = rgb
        return f'{r:08b}', f'{g:08b}', f'{b:08b}'

    def _bin_to_int(self, rgb):
        """Convert a binary (string) tuple to an integer tuple.

        :param rgb: A string tuple like ("00101010", "11101011", "00010110")
        :return: Return an int tuple like (220, 110, 96)
        """
        r, g, b = rgb
        return int(r, 2), int(g, 2), int(b, 2)

    def _merge_rgb(self, rgb1, rgb2):
        """Merge two RGB tuples.

        :param rgb1: An integer tuple like (220, 110, 96)
        :param rgb2: An integer tuple like (240, 95, 105)
        :return: An integer tuple with the two RGB values merged.
        """
        r1, g1, b1 = self._int_to_bin(rgb1)
        r2, g2, b2 = self._int_to_bin(rgb2)
        rgb = r1[:4] + r2[:4], g1[:4] + g2[:4], b1[:4] + b2[:4]
        return self._bin_to_int(rgb)

    def _unmerge_rgb(self, rgb):
        """Unmerge RGB.

        :param rgb: An integer tuple like (220, 110, 96)
        :return: An integer tuple with the two RGB values merged.
        """
        r, g, b = self._int_to_bin(rgb)#
        
        # Extract the last 4 bits (corresponding to the hidden image)
        # Concatenate 4 zero bits because we are working with 8 bit
        new_rgb = r[4:] + '0000', g[4:] + '0000', b[4:] + '0000'
        return self._bin_to_int(new_rgb)
    
    def hide(self, image1, image2):
        """Merge image2 into image1.

        :param image1: First image
        :param image2: Second image
        :return: A new merged image.
        """
        image1 = unnormalize_img(image1.squeeze(0))
        image2 = unnormalize_img(image2.squeeze(0))
        
        # Get the pixel map of the two images
        new_image = np.zeros_like(image1)
        new_image[:, :] = image1

        h, w = image2.shape[:2]

        new_image[:h, :w] = np.array([
            self._merge_rgb(image1[i, j], image2[i, j]) for i in range(h) for j in range(w)
        ]).reshape(h, w, 3)  # Assuming RGB images

        return Image.fromarray(new_image.astype(np.uint8))

    def exfiltrate(self, image):
        """Exfiltrate an image.

        :param image: The input image.
        :return: The hidden image.
        """
        image = Image.open(image)
        pixel_map = image.load()

        # Create the new image and load the pixel map
        new_image = Image.new(image.mode, image.size)
        new_map = new_image.load()

        for i in range(image.size[0]):
            for j in range(image.size[1]):
                new_map[i, j] = self._unmerge_rgb(pixel_map[i, j])

        return new_image

def get_patches(slide, positions, img_patch_sizes, patch_size):
    test_patches = []
    for pos, size in zip(positions, img_patch_sizes):
        test_patch = slide.read_region(pos, 0, size).resize((patch_size, patch_size))
        test_patch = torch.tensor(pil_to_np(test_patch))
        test_patches.append(test_patch)
    return torch.stack(test_patches)

def main():
    slide_1 = openslide.open_slide('TCGA-S3-AA15-01Z-00-DX1.A2456A4A-E6E8-4429-8F09-B997AA497BB0.svs')
    positions_1 = [(70000, 25000)]
    positions_2 = [(43000, 18000)]
    img_patch_size = [(4096, 4096)]

    # Create a new image that will be outputted
    cfg = omegaconf.OmegaConf.load(pathlib.Path(__file__).parent / 'cfg' / 'tcga_brca.yaml')
    pixel_map1 = get_patches(slide_1, positions_1, img_patch_size, cfg.dataset.img_size)
    pixel_map2 = get_patches(slide_1, positions_2, img_patch_size, cfg.dataset.img_size)
    img1 = unnormalize_img(pixel_map1.squeeze(0))
    img2 = unnormalize_img(pixel_map2.squeeze(0))

    # Save the extracted patches
    Image.fromarray(img1).save("output/original_1.png")
    Image.fromarray(img2).save("output/origingal_2.png")
    Steganography().hide(pixel_map1, pixel_map2).save("output/hidden.png")
    Steganography().exfiltrate("output/merged.png").save("output/exfiltrated.png")

if __name__ == '__main__':
    main()

