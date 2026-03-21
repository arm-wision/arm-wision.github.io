import os

# --- Toggle Dataset ---
DATASET_MODE = "kaggle_test" # or "plantclef"

# --- Common Config ---
RESOLUTION = 448
BATCH_SIZE = 48 # Optimized for RTX 4090 (24GB VRAM)

# --- Dataset Paths ---
DATASETS = {
    "plantclef": {
        "raw_csv": "data/plantclef-2026/PlantCLEF2024_single_plant_training_metadata.csv",
        "img_dir": "data/train/", # Assuming unzipped to data/train/
        "cleaned_csv": "data/train_metadata_cleaned.csv"
    },
    "kaggle_test": {
        "raw_csv": "data/kaggle_test_metadata.csv",
        "img_dir": "data/kaggle_test/split_ttv_dataset_type_of_plants/Train_Set_Folder/",
        "cleaned_csv": "data/kaggle_test_metadata_cleaned.csv"
    }
}

# --- Current Active Paths ---
RAW_CSV = DATASETS[DATASET_MODE]["raw_csv"]
IMG_DIR = DATASETS[DATASET_MODE]["img_dir"]
CLEANED_CSV = DATASETS[DATASET_MODE]["cleaned_csv"]

# --- Hardware Config ---
MODE = "4090" # Optimized for RTX 4090
if MODE == "4090":
    BIOCLIP_NAME = "hf-hub:imageomics/bioclip"
    DINOV2_NAME = "vit_large_patch14_dinov2"
    CONVNEXT_NAME = "convnextv2_large.fcmae_ft_in22k_in1k_384"
else: # A100 / High-VRAM Mode
    BIOCLIP_NAME = "hf-hub:imageomics/bioclip"
    DINOV2_NAME = "vit_giant_patch14_dinov2.lvd142m"
    CONVNEXT_NAME = "convnextv2_huge.fcmae_ft_in22k_in1k_384"
