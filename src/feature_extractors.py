import timm
import transformers
import torch.nn as nn
import torchvision.models as models


def resnet18(
    weights: models.ResNet18_Weights | None = models.ResNet18_Weights.IMAGENET1K_V1,
) -> nn.Module:
    resnet18 = models.resnet18(weights=weights)
    return nn.Sequential(
        *(list(resnet18.children())[:-2]),
    )


def vgg16(
    weights: models.VGG16_BN_Weights | None = models.VGG16_BN_Weights.IMAGENET1K_V1,
) -> nn.Module:
    vgg16 = models.vgg16_bn(weights=weights)
    return nn.Sequential(
        *(list(vgg16.features.children())),
    )


def gigapath() -> nn.Module:
    return timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)


def simclr() -> nn.Module:
    return transformers.from_pretrained("lightly-ai/simclrv2-imagenet1k-r50_1x_sk1")
