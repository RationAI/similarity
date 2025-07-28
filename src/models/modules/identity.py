import torch.nn as nn

import torch


class Identity(nn.Module):
    def __init__(self) -> None:
        super(Identity, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
