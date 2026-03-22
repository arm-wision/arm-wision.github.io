import torch
import torch.nn as nn
import torch.nn.functional as F
from .bioclip import PlantBioCLIP
from .dinov2 import PlantDINOv2
from .convnext import PlantConvNeXt


class PlantEnsemble(nn.Module):
    def __init__(self, num_classes=7800, input_res=448,
                 bioclip_name='hf-hub:imageomics/bioclip',
                 dinov2_name='vit_giant_patch14_dinov2.lvd142m',
                 convnext_name='convnextv2_huge.fcmae_ft_in22k_in1k_384'):
        """
        Triple Ensemble for PlantCLEF 2026.
        1. BioCLIP    (Taxonomic Foundation)     -- feature_dim: 512
        2. DINOv2     (Geometric/Structural)      -- feature_dim: 1024
        3. ConvNeXt-V2 (Convolutional/Local)      -- feature_dim: 1536

        Optimisations:
        - Projection heads include LayerNorm to stabilise features before L2 fusion
        - Frozen state cached as a flag to avoid per-forward parameter inspection
        """
        super(PlantEnsemble, self).__init__()

        print(f"Initializing Triple Ensemble ({input_res}px)...")

        # --- Backbones ---
        self.bioclip  = PlantBioCLIP(checkpoint=bioclip_name, input_res=input_res)
        self.dinov2   = PlantDINOv2(model_name=dinov2_name,   input_res=input_res)
        self.convnext = PlantConvNeXt(model_name=convnext_name, input_res=input_res)

        # --- Per-backbone projection heads ---
        # Linear + LayerNorm stabilises each backbone's output distribution
        # before L2 normalisation and fusion. Using LayerNorm here is more
        # robust than BatchNorm since backbone outputs have different scales.
        PROJ_DIM = 512
        self.proj_bio = nn.Sequential(
            nn.Linear(self.bioclip.feature_dim,  PROJ_DIM),
            nn.LayerNorm(PROJ_DIM)
        )
        self.proj_dino = nn.Sequential(
            nn.Linear(self.dinov2.feature_dim,   PROJ_DIM),
            nn.LayerNorm(PROJ_DIM)
        )
        self.proj_conv = nn.Sequential(
            nn.Linear(self.convnext.feature_dim, PROJ_DIM),
            nn.LayerNorm(PROJ_DIM)
        )

        self.fusion_dim = PROJ_DIM * 3  # 1536
        print(f"Backbone dims: BioCLIP={self.bioclip.feature_dim}, "
              f"DINOv2={self.dinov2.feature_dim}, "
              f"ConvNeXt={self.convnext.feature_dim}")
        print(f"Projected + Fused Feature Dimension: {self.fusion_dim}")

        # --- Deep Fusion Classifier ---
        # Compiled with torch.compile for kernel fusion across the MLP ops.
        # The classifier is a simple static graph -- ideal for compile.
        # LayerNorm > BatchNorm for ViT outputs (batch stats unreliable at small B)
        # torch.compile removed -- it can cause mid-training recompilation hangs
        # with gradient checkpointing + autocast. Re-evaluate after training is stable.
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

        # Cached frozen state flag -- avoids calling next(backbone.parameters())
        # on every forward pass, which adds overhead at scale.
        self._backbones_frozen = False

        # Note: CUDA streams removed -- they deadlock with autocast + gradient
        # checkpointing in certain PyTorch versions. Sequential execution is safe.

    def forward(self, x):
        """
        Forward pass with sequential backbone execution.

        CUDA streams were removed as they deadlock with autocast +
        gradient checkpointing in certain PyTorch versions. Sequential
        execution is stable and still fully GPU-utilised.
        """
        ctx = torch.no_grad() if self._backbones_frozen else torch.enable_grad()

        with ctx:
            feat_bio  = self.bioclip(x)
            feat_dino = self.dinov2(x)
            feat_conv = self.convnext(x)

        # Project each backbone to a common 512-d space with LayerNorm
        feat_bio  = self.proj_bio(feat_bio)
        feat_dino = self.proj_dino(feat_dino)
        feat_conv = self.proj_conv(feat_conv)

        # L2-normalise to equalise feature scales before fusion
        feat_bio  = F.normalize(feat_bio,  dim=1)
        feat_dino = F.normalize(feat_dino, dim=1)
        feat_conv = F.normalize(feat_conv, dim=1)

        # Fuse and classify
        combined = torch.cat([feat_bio, feat_dino, feat_conv], dim=1)
        return self.classifier(combined)

    def set_grad_checkpointing(self, enable=True):
        """Enable gradient checkpointing on all backbones to trade compute for VRAM."""
        for backbone in [self.bioclip, self.dinov2, self.convnext]:
            if hasattr(backbone, 'set_grad_checkpointing'):
                backbone.set_grad_checkpointing(enable)
            elif hasattr(backbone, 'model'):
                if hasattr(backbone.model, 'set_grad_checkpointing'):
                    backbone.model.set_grad_checkpointing(enable)
        print(f"Gradient checkpointing {'enabled' if enable else 'disabled'} on all backbones.")

    def freeze_backbones(self):
        """Freeze all three backbones to warm up the fusion head
        without disrupting pre-trained features."""
        for backbone in [self.bioclip, self.dinov2, self.convnext]:
            for param in backbone.parameters():
                param.requires_grad = False
        self._backbones_frozen = True
        print("All 3 backbones frozen.")

    def unfreeze_backbones(self):
        """Unfreeze all three backbones for full fine-tuning."""
        for backbone in [self.bioclip, self.dinov2, self.convnext]:
            for param in backbone.parameters():
                param.requires_grad = True
        self._backbones_frozen = False
        print("All 3 backbones unfrozen.")
