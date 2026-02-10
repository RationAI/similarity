# Musica cluster setup
```
mamba create -n similarity-env python=3.12.9 -y
mamba activate similarity-env
mamba install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
mamba install -c conda-forge libvips openslide libgcc-ng -y
uv pip install -e .
```
