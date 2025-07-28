import torch.nn as nn
from src.models.modules import Identity
import torchvision
import torch


class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self) -> None:
        super(SimCLR, self).__init__()

        self.encoder = torchvision.models.resnet50()
        self.n_features = self.encoder.fc.in_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, 64, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
