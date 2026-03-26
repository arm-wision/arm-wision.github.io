import os

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_MODE = "plantclef"  # or "kaggle_test"

DATASETS = {
    "plantclef": {
        "raw_csv":     "/workspace/plantclef/raw/PlantCLEF2024_single_plant_training_metadata.csv",
        "img_dir":     "/workspace/plantclef/raw/train/images_max_side_800/",
        "cleaned_csv": "/workspace/plantclef/processed/train_metadata_cleaned_verified.csv"
    },
    "kaggle_test": {
        "raw_csv":     "/workspace/plantclef/processed/kaggle_test_metadata.csv",
        "img_dir":     "/workspace/plantclef/raw/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/images/",
        "cleaned_csv": "/workspace/plantclef/processed/kaggle_test_metadata_cleaned.csv"
    }
}

RAW_CSV     = DATASETS[DATASET_MODE]["raw_csv"]
IMG_DIR     = DATASETS[DATASET_MODE]["img_dir"]
CLEANED_CSV = DATASETS[DATASET_MODE]["cleaned_csv"]

# ---------------------------------------------------------------------------
# Resolution & batch sizes
# Timing targets:
#   Feature extraction : 1-2 hours
#   Phase 1            : 1-2 hours
#   Phase 2 (LoRA)     : < 24 hours
# ---------------------------------------------------------------------------
RESOLUTION    = 384   # ConvNeXt native resolution; 26% cheaper than 448px (scales as res²)
BATCH_SIZE    = 384   # Phase 1 extraction + head training
P2_BATCH_SIZE = 128   # Reduced from 256 to fit on 32GB GPU with LoRA+Gradients

# ---------------------------------------------------------------------------
# Chunked forward chunk sizes
# EXTRACT_CHUNK_SIZE : feature extraction  (no_grad, safe to be large -> faster)
# CHUNK_SIZE         : Phase 1 validation  (no_grad)
# P2_CHUNK_SIZE      : Phase 2 training    (LoRA backward is small, moderate size)
# ---------------------------------------------------------------------------
EXTRACT_CHUNK_SIZE = 384  # no_grad during extraction -- 5090 can handle full batch
CHUNK_SIZE         = 32
P2_CHUNK_SIZE      = 32 # Balanced for speed and VRAM; 128 caused OOM on 32GB GPU

# ---------------------------------------------------------------------------
# Training schedule
# ---------------------------------------------------------------------------
ACCUMULATION_STEPS = 2    # Increased from 1 to keep effective batch size at 256

EPOCHS_PHASE1      = 10
EPOCHS_PHASE2      = 5    # Increased for deeper fine-tuning
P2_SAMPLES_PER_EPOCH = 1000000 # 4x more data per epoch for 80% accuracy push

VAL_EVERY_N_EPOCHS = 1    # Validate every epoch in Phase 2 (short run, tight feedback)
MAX_VAL_BATCHES    = 150  # More validation batches for better statistical confidence
PATIENCE           = 3    # More patience for the longer run

# ---------------------------------------------------------------------------
# LoRA (Phase 2)
# Applied to DINOv2 + ConvNeXt.  BioCLIP stays fully frozen.
# Reduces trainable params from ~800M -> ~5M; enables batch=256 + fast convergence.
# ---------------------------------------------------------------------------
LORA_R       = 64    # Quadrupled brain capacity (from 16)
LORA_ALPHA   = 128   # alpha = 2 * r
LORA_DROPOUT = 0.05

# ---------------------------------------------------------------------------
# Backbone selection
# ---------------------------------------------------------------------------
MODE = "5090"   # Blackwell-optimized: switch to "A100" for giant/huge, "4090" for standard

if MODE == "5090":
    # 5090 has 32GB VRAM + massive FP8 throughput
    BIOCLIP_NAME  = "hf-hub:imageomics/bioclip"
    DINOV2_NAME   = "vit_large_patch14_dinov2"
    CONVNEXT_NAME = "convnextv2_large.fcmae_ft_in22k_in1k_384"
    # Blackwell specific speedups
    USE_FP8       = True
    USE_COMPILE   = True
    LOAD_CACHE_TO_GPU = True 
elif MODE == "4090":
    BIOCLIP_NAME  = "hf-hub:imageomics/bioclip"
    DINOV2_NAME   = "vit_large_patch14_dinov2"
    CONVNEXT_NAME = "convnextv2_large.fcmae_ft_in22k_in1k_384"
    USE_FP8       = False
    USE_COMPILE   = False
    LOAD_CACHE_TO_GPU = False
else:
    BIOCLIP_NAME  = "hf-hub:imageomics/bioclip"
    DINOV2_NAME   = "vit_giant_patch14_dinov2.lvd142m"
    CONVNEXT_NAME = "convnextv2_huge.fcmae_ft_in22k_in1k_384"
    USE_FP8       = False
    USE_COMPILE   = False
    LOAD_CACHE_TO_GPU = False

# ---------------------------------------------------------------------------
# Region features + PCA
# Region features: extract full image + center crop per backbone.
# Doubles raw feature richness before PCA with zero extra training cost.
# PCA compresses [bio_full+bio_crop+dino_full+dino_crop+conv_full+conv_crop]
# (512+512+1024+1024+1536+1536 = 6144-d) -> PCA_COMPONENTS-d.
# Phase 1 then trains a tiny linear head on PCA_COMPONENTS-d features.
# ---------------------------------------------------------------------------
USE_REGION_FEATURES = True
PCA_COMPONENTS      = 512   # target dim after PCA (< raw 6144-d -> much faster Phase 1)
# ---------------------------------------------------------------------------
# Checkpoint paths (centralised so all modules agree)
# ---------------------------------------------------------------------------
FEATURE_CACHE_PATH = "models/legacy_v1/phase1_feature_cache.pt" # Reuse existing cache
P1_CKPT_PATH       = "models/blackwell_v2/phase1_checkpoint.pth"
P2_CKPT_DIR        = "models/blackwell_v2/phase2_checkpoint"
P2_EPOCH_CKPT      = "models/blackwell_v2/phase2_epoch_checkpoint.pth"
PCA_TRANSFORM_PATH  = "models/legacy_v1/pca_transform.pkl" # Reuse existing PCA

