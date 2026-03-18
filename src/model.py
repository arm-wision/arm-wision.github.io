import torch
import torch.nn as nn
import open_clip 

class PlantBioCLIP(nn.Module):
    def __init__(self, num_classes=7800, checkpoint='hf-hub:imageomics/bioclip'):
        super(PlantBioCLIP, self).__init__()

        # Load BioCLIP ViT-L/14
        self.model, _, _ = open_clip.create_model_and_transforms(checkpoint)
        # extract the visual backbone from bioclip 
        self.backbone = self.model.visual 
        self.feature_dim = 768 

        # classification head 
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # extract features
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits 
