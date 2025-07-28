import timm
import torch.nn as nn
import torch
import torchvision.models as models


from src.models import SimCLR


def resnet18(
    weights: models.ResNet18_Weights | None = models.ResNet18_Weights.IMAGENET1K_V1,
) -> nn.Module:
    model = models.resnet18(weights=weights)
    return nn.Sequential(
        *(list(model.children())[:-2]),
        nn.AdaptiveAvgPool2d(1)
    )


def vgg16(
    weights: models.VGG16_BN_Weights | None = models.VGG16_BN_Weights.IMAGENET1K_V1,
) -> nn.Module:
    model = models.vgg16_bn(weights=weights)
    return nn.Sequential(
        *(list(model.features.children())),
        nn.AdaptiveAvgPool2d(1)
    )


def gigapath() -> nn.Module:
    return timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)


def simclr() -> nn.Module:
    model = SimCLR()
    model.load_state_dict(torch.load("pretrained/simclr.tar"))
    return model
