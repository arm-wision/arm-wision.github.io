import torch
import torch.nn as nn
from .bioclip import PlantBioCLIP
from .dinov2 import PlantDINOv2
from .convnext import PlantConvNeXt

class PlantEnsemble(nn.Module):
    def __init__(self, num_classes=7800, input_res=448, 
                 bioclip_name='hf-hub:imageomics/bioclip', 
                 dinov2_name='vit_giant_patch14_dinov2.lvd142m', 
                 convnext_name='convnextv2_huge.fcmae_ft_in22k_in1k_384'):
        """
        Triple Threat Ensemble for PlantCLEF 2026.
        1. BioCLIP (Taxonomic Foundation)
        2. DINOv2 (Geometric/Structural Features)
        3. ConvNeXt-V2 (Convolutional/Local Context)
        """
        super(PlantEnsemble, self).__init__()
        
        print(f"Initializing Triple Threat Ensemble ({input_res}px)...")
        
        # 1. Backbones
        self.bioclip = PlantBioCLIP(checkpoint=bioclip_name, input_res=input_res)
        self.dinov2 = PlantDINOv2(model_name=dinov2_name, input_res=input_res)
        self.convnext = PlantConvNeXt(model_name=convnext_name, input_res=input_res)
        
        # 2. Dimensions
        # BioCLIP-L (1024) + DINOv2-G (1536) + ConvNeXt-H (2048) = 4608
        self.fusion_dim = self.bioclip.feature_dim + self.dinov2.feature_dim + self.convnext.feature_dim
        print(f"Fused Feature Dimension: {self.fusion_dim}")
        
        # 3. Deep Fusion Head
        # Higher capacity head to handle the massive input dimension
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 2048),
            nn.LayerNorm(2048), # LayerNorm often more stable than BatchNorm for ViT outputs
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # Extract features
        feat_bio = self.bioclip(x)
        feat_dino = self.dinov2(x)
        feat_conv = self.convnext(x)
        
        # Concatenate: [Batch, Fusion_Dim]
        combined = torch.cat([feat_bio, feat_dino, feat_conv], dim=1)
        
        # Final classification
        logits = self.classifier(combined)
        return logits

    def freeze_backbones(self):
        """Freeze all three backbones."""
        for param in self.bioclip.parameters():
            param.requires_grad = False
        for param in self.dinov2.parameters():
            param.requires_grad = False
        for param in self.convnext.parameters():
            param.requires_grad = False
        print("All 3 backbones frozen.")

    def unfreeze_backbones(self):
        """Unfreeze all three backbones."""
        for param in self.bioclip.parameters():
            param.requires_grad = True
        for param in self.dinov2.parameters():
            param.requires_grad = True
        for param in self.convnext.parameters():
            param.requires_grad = True
        print("All 3 backbones unfrozen.")
