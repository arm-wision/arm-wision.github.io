import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip 

class PlantBioCLIP(nn.Module):
    def __init__(self, num_classes=7800, checkpoint='hf-hub:imageomics/bioclip', input_res=448):
        super(PlantBioCLIP, self).__init__()

        # Load BioCLIP
        self.model, _, _ = open_clip.create_model_and_transforms(checkpoint)
        self.backbone = self.model.visual 
        
        # Dynamically determine feature dimension and patch size
        if hasattr(self.backbone, 'output_dim'):
            self.feature_dim = self.backbone.output_dim
        elif hasattr(self.backbone, 'proj') and self.backbone.proj is not None:
            self.feature_dim = self.backbone.proj.shape[1]
        else:
            # Fallback for ViT-B/16
            self.feature_dim = 768

        self.input_res = input_res

        # Classification head 
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(), 
            nn.Dropout(0.2), 
            nn.Linear(1024, num_classes)
        )

        # Handle Positional Embedding Interpolation for new resolution
        self._interpolate_pos_embeddings(input_res)

    def _interpolate_pos_embeddings(self, input_res):
        """
        Interpolate positional embeddings to match a new input resolution.
        """
        # Determine patch size and old grid size
        pos_embed = self.backbone.positional_embedding
        # Some versions have [L, D], others [1, L, D]
        if pos_embed.dim() == 3:
            pos_embed = pos_embed.squeeze(0)
            
        old_pos_embed = pos_embed # [L, D]
        D = old_pos_embed.shape[-1]
        
        # Identify class tokens (usually 1)
        # For ViT, L = grid_size^2 + num_extra_tokens
        # num_extra_tokens is 1 (class token)
        num_extra_tokens = 1 
        class_token_embed = old_pos_embed[:num_extra_tokens]
        patch_embeds = old_pos_embed[num_extra_tokens:]
        
        L = patch_embeds.shape[0]
        old_grid_size = int(L**0.5)
        
        # Determine patch size from original BioCLIP (224px / old_grid_size)
        # BioCLIP is typically 224px.
        original_res = 224
        patch_size = original_res // old_grid_size
        
        new_grid_size = input_res // patch_size
        
        if old_grid_size != new_grid_size:
            print(f"Interpolating positional embeddings from {old_grid_size}x{old_grid_size} to {new_grid_size}x{new_grid_size}")
            patch_embeds = patch_embeds.reshape(1, old_grid_size, old_grid_size, D).permute(0, 3, 1, 2)
            patch_embeds = F.interpolate(patch_embeds, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False)
            patch_embeds = patch_embeds.permute(0, 2, 3, 1).reshape(-1, D)
            
            new_pos_embed = torch.cat([class_token_embed, patch_embeds], dim=0)
            
            # If original was [1, L, D], restore that shape
            if self.backbone.positional_embedding.dim() == 3:
                new_pos_embed = new_pos_embed.unsqueeze(0)
                
            self.backbone.positional_embedding = nn.Parameter(new_pos_embed)

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        # OpenCLIP visual backbone returns pooled features or class token
        logits = self.classifier(features)
        return logits 
