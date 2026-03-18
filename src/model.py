import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip 

class PlantBioCLIP(nn.Module):
    def __init__(self, num_classes=7800, checkpoint='hf-hub:imageomics/bioclip', input_res=448):
        super(PlantBioCLIP, self).__init__()

        # Load BioCLIP ViT-L/14
        self.model, _, _ = open_clip.create_model_and_transforms(checkpoint)
        self.backbone = self.model.visual 
        self.feature_dim = 768 
        self.input_res = input_res

        # Classification head 
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(), 
            nn.Dropout(0.2), 
            nn.Linear(1024, num_classes)
        )

        # Handle Positional Embedding Interpolation for 448px
        self._interpolate_pos_embeddings(input_res)

    def _interpolate_pos_embeddings(self, input_res):
        """
        Interpolate positional embeddings to match a new input resolution.
        BioCLIP ViT-L/14 is natively trained on 224x224 (patch_size=14).
        """
        patch_size = 14
        new_grid_size = input_res // patch_size
        old_pos_embed = self.backbone.positional_embedding # [L, D]
        
        # ViT-L/14 has 257 tokens (1 class token + 16x16 grid)
        class_token_embed = old_pos_embed[:1]
        patch_embeds = old_pos_embed[1:]
        
        old_grid_size = int(patch_embeds.shape[0]**0.5) # Should be 16
        
        if old_grid_size != new_grid_size:
            patch_embeds = patch_embeds.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
            patch_embeds = F.interpolate(patch_embeds, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False)
            patch_embeds = patch_embeds.permute(0, 2, 3, 1).reshape(-1, self.feature_dim)
            
            new_pos_embed = torch.cat([class_token_embed, patch_embeds], dim=0)
            self.backbone.positional_embedding = nn.Parameter(new_pos_embed)
            print(f"Interpolated positional embeddings from {old_grid_size}x{old_grid_size} to {new_grid_size}x{new_grid_size}")

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits 
