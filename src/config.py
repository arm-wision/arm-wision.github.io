import os

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_MODE = "plantclef"  # or "kaggle_test"

DATASETS = {
    "plantclef": {
        "raw_csv":     "/workspace/plantclef/raw/PlantCLEF2024_single_plant_training_metadata.csv",
        "img_dir":     "/workspace/plantclef/raw/train/images_max_side_800/",
        "cleaned_csv": "/workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv"
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
P2_BATCH_SIZE = 96    # Reduced from 128 to accommodate Parallel Streams + torch.compile

# ---------------------------------------------------------------------------
# Chunked forward chunk sizes
# EXTRACT_CHUNK_SIZE : feature extraction  (no_grad, safe to be large -> faster)
# CHUNK_SIZE         : Phase 1 validation  (no_grad)
# P2_CHUNK_SIZE      : Phase 2 training    (LoRA backward is small, moderate size)
# ---------------------------------------------------------------------------
EXTRACT_CHUNK_SIZE = 64   
CHUNK_SIZE         = 32
P2_CHUNK_SIZE      = 8    # Reduced from 16 to cap peak VRAM during parallel backbone execution

# ---------------------------------------------------------------------------
# Training schedule
# ---------------------------------------------------------------------------
ACCUMULATION_STEPS = 6    # Increased from 4 to maintain effective batch size (~576)

EPOCHS_PHASE1      = 20
EPOCHS_PHASE2      = 5    
P2_SAMPLES_PER_EPOCH = 1000000 

VAL_EVERY_N_EPOCHS = 1    
MAX_VAL_BATCHES    = 150  
PATIENCE           = 3    

# ---------------------------------------------------------------------------
# LoRA (Phase 2)
# ---------------------------------------------------------------------------
LORA_R       = 64    
LORA_ALPHA   = 128   
LORA_DROPOUT = 0.05

# ---------------------------------------------------------------------------
# Backbone selection
# ---------------------------------------------------------------------------
MODE = "5090"   

if MODE == "5090":
    BIOCLIP_NAME  = "hf-hub:imageomics/bioclip"
    DINOV2_NAME   = "vit_large_patch14_dinov2"
    CONVNEXT_NAME = "convnextv2_large.fcmae_ft_in22k_in1k_384"
    USE_FP8       = False  
    USE_COMPILE   = True
    LOAD_CACHE_TO_GPU = False 
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
# ---------------------------------------------------------------------------
USE_REGION_FEATURES = True
PCA_COMPONENTS      = 1024  
# ---------------------------------------------------------------------------
# Checkpoint paths 
# ---------------------------------------------------------------------------
FEATURE_CACHE_PATH = "models/blackwell_v2/phase1_feature_cache.pt" 
P1_CKPT_PATH       = "models/blackwell_v2/phase1_checkpoint.pth"
P2_CKPT_DIR        = "models/blackwell_v2/phase2_checkpoint"
P2_EPOCH_CKPT      = "models/blackwell_v2/phase2_epoch_checkpoint.pth"
PCA_TRANSFORM_PATH  = "models/blackwell_v2/pca_transform.pkl" 
