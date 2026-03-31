import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import glob

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import config
from models.ensemble import PlantEnsemble
from models.sahi import CUDASahiEngine

def load_competition_model(model_path, num_classes, device='cuda'):
    """Helper to initialize and load the ensemble."""
    model = PlantEnsemble(num_classes=num_classes, input_res=config.RESOLUTION)
    model.apply_lora(r=config.LORA_R, lora_alpha=config.LORA_ALPHA)
    
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        state_dict = ckpt['module'] if isinstance(ckpt, dict) and 'module' in ckpt else ckpt
        # Clean state dict keys
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded weights from {model_path}")
    
    return model

def main():
    BASE_DIR = "/workspace/PlantCLEF2026"
    TRAIN_CSV = "/workspace/plantclef/processed/train_metadata_cleaned_verified.csv"
    TEST_DIR = "/workspace/plantclef/raw/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/images/"
    GT_CSV = "/workspace/plantclef/raw/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/PlantCLEF2025_test_labels.csv"
    OUTPUT_CSV = os.path.join(BASE_DIR, "submission_plantclef2026.csv")

    # 1. Setup Species Mapping
    df_train = pd.read_csv(TRAIN_CSV, sep=';')
    species_ids = sorted(df_train['species_id'].unique())
    idx_to_species = {i: s for i, s in enumerate(species_ids)}
    
    # 2. Initialize Model and Engine
    MODEL_PATH = os.path.join(BASE_DIR, "models/best_calibrated/mp_rank_00_model_states.pt")
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = os.path.join(BASE_DIR, "models/final/mp_rank_00_model_states.pt")
        
    model = load_competition_model(MODEL_PATH, len(species_ids))
    engine = CUDASahiEngine(model, resolution=config.RESOLUTION)
    
    # 3. Process Images
    test_images = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
    results = []
    print(f"Starting CUDA-Accelerated SAHI on {len(test_images)} images...")
    
    for img_path in tqdm(test_images):
        quadrat_id = os.path.splitext(os.path.basename(img_path))[0]
        probs = engine.predict_image(img_path)
        
        if probs is not None:
            # Predict top species (threshold 0.05)
            top_indices = np.argsort(probs)[::-1]
            preds = [top_indices[0]] # Always keep Top-1
            for i in top_indices[1:15]:
                if probs[i] > 0.05: preds.append(i)
            
            species = [idx_to_species[p] for p in preds]
            results.append({"quadrat_id": quadrat_id, "species_ids": species})

    # 4. Save and Report
    df = pd.DataFrame(results)
    df_csv = df.copy()
    df_csv['species_ids'] = df_csv['species_ids'].apply(lambda x: "[" + ", ".join(map(str, x)) + "]")
    df_csv.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"Submission saved to {OUTPUT_CSV}")

    # Accuracy check
    if os.path.exists(GT_CSV):
        df_gt = pd.read_csv(GT_CSV, sep=';')
        gt_dict = df_gt.groupby('quadrat_id')['species_id'].apply(set).to_dict()
        f1s = []
        for res in results:
            qid = res['quadrat_id']; true_set = gt_dict.get(qid, set())
            pred_set = set(res['species_ids'])
            tp = len(true_set & pred_set); fp = len(pred_set - true_set); fn = len(true_set - pred_set)
            f1s.append((2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0)
        print(f"\n ► Final Macro-F1 Score: {np.mean(f1s):.4f}")

if __name__ == "__main__":
    main()
