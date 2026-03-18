import torch
import torch.nn as nn
from bioclip_model import PlantBioCLIP
from dinov2_model import PlantDINOv2

class PlantEnsemble(nn.Module):
    def __init__(self, num_classes=7800, input_res=448):
        super(PlantEnsemble, self).__init__()
        
        print(f"Initializing Ensemble with resolution {input_res}...")
        self.bioclip = PlantBioCLIP(input_res=input_res)
        self.dinov2 = PlantDINOv2(input_res=input_res)
        
        # Fusion Head: 768 (BioCLIP-B/16) + 1024 (DINOv2-L/14) = 1792
        self.fusion_dim = self.bioclip.feature_dim + self.dinov2.feature_dim
        print(f"Fusion dimension: {self.fusion_dim}")
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract features from both backbones
        feat_bio = self.bioclip(x)
        feat_dino = self.dinov2(x)
        
        # Concatenate features
        combined = torch.cat([feat_bio, feat_dino], dim=1)
        
        # Final classification
        logits = self.classifier(combined)
        return logits

    def freeze_backbones(self):
        """Freeze all parameters in both backbones."""
        for param in self.bioclip.parameters():
            param.requires_grad = False
        for param in self.dinov2.parameters():
            param.requires_grad = False
        print("Backbones frozen.")

    def unfreeze_backbones(self):
        """Unfreeze all parameters in both backbones."""
        for param in self.bioclip.parameters():
            param.requires_grad = True
        for param in self.dinov2.parameters():
            param.requires_grad = True
        print("Backbones unfrozen.")
