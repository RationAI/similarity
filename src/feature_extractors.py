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

class Virchow2Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        class_token = output[:, 0]
        patch_tokens = output[:, 5:]
        patched_avg = patch_tokens.mean(dim=1)
        return torch.cat([class_token, patched_avg], dim=-1)

def virchow2():
    model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    wrapped_model = Virchow2Wrapper(model)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return wrapped_model, transform

class UNI2hWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if len(output.shape) == 3:
            return output.mean(dim=1)
        return output

def UNI2h():
    timm_kwargs = {
        'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
        'num_classes': 0, 'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 
        'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 'dynamic_img_size': True
    }
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    wrapped_model = UNI2hWrapper(model)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return wrapped_model, transform

class GigaPathWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if len(output.shape) == 3:
            output = output.squeeze(1)
        return output

def gigapathTile():
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    wrapped_model = GigaPathWrapper(model)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return wrapped_model, transform

class MidnightWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        last_hidden_state = outputs.last_hidden_state
        
        cls_token = last_hidden_state[:, 0, :]
        patch_tokens_mean = last_hidden_state[:, 1:, :].mean(dim=1)
        
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

def simclrv2() -> nn.Module:
    """
    SimCLRv2 (ResNet152x4) implementation based on the Nature article.
    Expects 224x224 tiles and uses 100% fine-tuned labels.
    """
    import timm
    import torch.nn as nn
    from torchvision import transforms

    model_name = "hf-hub:timm/resnetv2_152x4_bit.goog_in21k_ft_in1k"
    
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        model.eval()
        print(f"INFO: Successfully loaded SimCLRv2 (R152x4) with {model.num_features} output features.")
        
    except Exception as e:
        print(f"ERROR: Failed to load {model_name}: {e}")
        raise e

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    return model, transform