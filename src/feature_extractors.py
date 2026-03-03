import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
import torch.nn as nn
import torch
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import v2
from src.models import SimCLR
from transformers import AutoImageProcessor, AutoModel


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


def gigapathTile() -> nn.Module:
    """This model expects 224x224 tiles"""
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    return model, transform

def virchow2() -> nn.Module:
    """This model expects 224x224 tiles"""
    model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform


def UNI2h() -> nn.Module:
    """This model expects 224x224 tiles"""
    timm_kwargs = {
        'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
        'num_classes': 0, 'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 
        'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 'dynamic_img_size': True
    }
    
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    
    config = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**config)
    
    return model, transform

class MidnightWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Zavoláme originální model
        outputs = self.model(x)
        # Vytáhneme last_hidden_state (batch, 197, 768)
        last_hidden_state = outputs.last_hidden_state
        
        # Extrakce podle dokumentace Kaiko-AI
        cls_token = last_hidden_state[:, 0, :]
        patch_tokens_mean = last_hidden_state[:, 1:, :].mean(dim=1)
        
        # Vrátíme sjednocený embedding (batch, 1536)
        return torch.cat([cls_token, patch_tokens_mean], dim=-1)

def midnight12k() -> nn.Module:
    """This model expects 224x224 tiles"""
    transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    )
    model = AutoModel.from_pretrained('kaiko-ai/midnight')
    wrapped_model = MidnightWrapper(model)
    return wrapped_model, transform

def simclr() -> nn.Module:
    model = SimCLR()
    model.load_state_dict(torch.load("pretrained/simclr.tar"))
    return model
