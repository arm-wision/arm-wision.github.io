import torch
import torch.nn as nn
import timm

class PlantConvNeXt(nn.Module):
    def __init__(self, model_name='convnextv2_huge.fcmae_ft_in22k_in1k_384', input_res=448):
        super(PlantConvNeXt, self).__init__()
        
        # Load ConvNeXt-V2 from timm
        # Huge version: 2048-dim features
        self.backbone = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=0, 
            img_size=input_res
        )
        
        self.feature_dim = self.backbone.num_features

    def forward(self, x):
        return self.backbone(x)
