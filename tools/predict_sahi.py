import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd

# Add workspace root to path to find local packages
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import config
from models.ensemble import PlantEnsemble
from models.sahi import CUDASahiEngine

def main():
    parser = argparse.ArgumentParser(description="PlantCLEF 2026 - Quick Image Predictor")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("--topk", type=int, default=5, help="Number of top species to show")
    args = parser.parse_args()

    # 1. Setup
    TRAIN_CSV = "/workspace/plantclef/processed/train_metadata_cleaned_verified.csv"
    df_train = pd.read_csv(TRAIN_CSV, sep=';')
    species_ids = sorted(df_train['species_id'].unique())
    idx_to_species = {i: s for i, s in enumerate(species_ids)}

    # 2. Model
    MODEL_PATH = "models/best_calibrated/mp_rank_00_model_states.pt"
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = "models/final/mp_rank_00_model_states.pt"

    model = PlantEnsemble(num_classes=len(species_ids), input_res=config.RESOLUTION)
    model.apply_lora(r=config.LORA_R, lora_alpha=config.LORA_ALPHA)
    
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location='cuda')
        state_dict = ckpt['module'] if isinstance(ckpt, dict) and 'module' in ckpt else ckpt
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded weights from {MODEL_PATH}")

    # 3. Engine
    engine = CUDASahiEngine(model, resolution=config.RESOLUTION)
    
    # 4. Predict
    print(f"Running inference on: {args.image}...")
    probs = engine.predict_image(args.image)
    
    if probs is not None:
        top_indices = np.argsort(probs)[-args.topk:][::-1]
        print("\n" + "="*40)
        print(f"      TOP {args.topk} SPECIES PREDICTIONS")
        print("="*40)
        for i, idx in enumerate(top_indices):
            s_id = idx_to_species[idx]
            conf = probs[idx]
            print(f" {i+1}. ID: {s_id:<10} | Confidence: {conf:.4f}")
        print("="*40 + "\n")

if __name__ == "__main__":
    main()
