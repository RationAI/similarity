# Steganography script

`hiding.py` hides one image inside another by keeping the four most significant bits of each colour channel of the first image and storing the four most significant bits of the second image in the least significant bits (`hide`), and recovers an approximation of the hidden image (`exfiltrate`). Written by Bernd Saurugger for PRIVAGAMS.

`img_util.py` holds the two image helpers it needs, copied unchanged from the `path_diffusion` repository.

The `main()` demo reads slides and a configuration file from the original project and does not run on its own; the `Steganography` class can be used directly.
