import os
import sys

# Automatically add the 'src' directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import glob
from collections import Counter

# Local imports
import config
from models.ensemble import PlantEnsemble

try:
    import plantclef_ext
    HAS_EXT = True
    print("SUCCESS: CUDA extensions (plantclef_ext) loaded for fast inference.")
except ImportError:
    HAS_EXT = False
    print("WARNING: CUDA extensions not found. Falling back to slow CPU tiling.")

class TestInferenceWrapper:
    def __init__(self, model_path, csv_path, device='cuda'):
        self.device = device
        self.resolution = config.RESOLUTION
        
        # 1. Load species mapping
        df_train = pd.read_csv(csv_path, sep=';')
        species_ids = sorted(df_train['species_id'].unique())
        self.idx_to_species = {i: s for i, s in enumerate(species_ids)}
        self.num_classes = len(species_ids)

        # 2. Initialize model
        self.model = PlantEnsemble(
            num_classes=self.num_classes, 
            input_res=self.resolution,
            bioclip_name=config.BIOCLIP_NAME,
            dinov2_name=config.DINOV2_NAME,
            convnext_name=config.CONVNEXT_NAME
        )
        self.model.apply_lora(r=config.LORA_R, lora_alpha=config.LORA_ALPHA)
        
        # 3. Load weights
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=device)
            state_dict = ckpt['module'] if isinstance(ckpt, dict) and 'module' in ckpt else ckpt
            new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict, strict=False)
            
        self.model.to(device)
        self.model.eval()
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        self.std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    def predict_batch(self, tiles):
        """Runs inference on a batch of tiles (GPU tensor)."""
        if tiles.shape[2] != self.resolution:
            tiles = F.interpolate(tiles, size=(self.resolution, self.resolution), mode='bilinear')
        tiles = (tiles - self.mean) / self.std
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = self.model(tiles)
            probs = torch.softmax(logits, dim=1)
        return probs

    def run_sahi_cuda(self, image_path, slice_h=512, slice_w=512, overlap=0.2):
        if not os.path.exists(image_path): return None
        
        # Load image to GPU once
        img_pil = Image.open(image_path).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float().to(self.device) / 255.0
        
        # 1. Fast CUDA Tiling
        if HAS_EXT:
            tiles = plantclef_ext.extract_tiles(img_tensor, slice_h, slice_w, overlap)
        else:
            # Slow fallback
            tiles = img_tensor.unsqueeze(0)

        # 2. Batch Inference (cap VRAM)
        CHUNK_SIZE = 16
        all_probs = []
        for i in range(0, len(tiles), CHUNK_SIZE):
            all_probs.append(self.predict_batch(tiles[i:i+CHUNK_SIZE]))
        
        tile_probs = torch.cat(all_probs, dim=0)

        # 3. Fused Max Pooling
        if HAS_EXT:
            final_probs = plantclef_ext.fused_max_pool(tile_probs)
        else:
            final_probs = torch.max(tile_probs, dim=0)[0]
            
        return final_probs.cpu().numpy()

def main():
    BASE_DIR = "/workspace/PlantCLEF2026"
    MODEL_PATH = os.path.join(BASE_DIR, "models/best_calibrated/mp_rank_00_model_states.pt")
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = os.path.join(BASE_DIR, "models/final/mp_rank_00_model_states.pt")
    
    TRAIN_CSV = "/workspace/plantclef/processed/train_metadata_cleaned_verified.csv"
    TEST_DIR = "/workspace/plantclef/raw/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/images/"
    # Try to find ground truth labels
    GT_CSV = "/workspace/plantclef/raw/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/PlantCLEF2025_test_labels.csv"
    OUTPUT_CSV = os.path.join(BASE_DIR, "submission_plantclef2026.csv")

    wrapper = TestInferenceWrapper(MODEL_PATH, TRAIN_CSV)
    test_images = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
    
    results = []
    print(f"Starting Inference on {len(test_images)} images...")
    
    for img_path in tqdm(test_images):
        quadrat_id = os.path.splitext(os.path.basename(img_path))[0]
        probs = wrapper.run_sahi_cuda(img_path)
        
        if probs is not None:
            idx = np.argmax(probs)
            # Simple thresholding
            preds = [idx]
            top_indices = np.argsort(probs)[::-1]
            for i in top_indices[1:10]:
                if probs[i] > 0.05: preds.append(i)
            
            species = [wrapper.idx_to_species[p] for p in preds]
            results.append({"quadrat_id": quadrat_id, "species_ids": species})

    # Save CSV
    df = pd.DataFrame(results)
    df_csv = df.copy()
    df_csv['species_ids'] = df_csv['species_ids'].apply(lambda x: "[" + ", ".join(map(str, x)) + "]")
    df_csv.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"Submission saved to {OUTPUT_CSV}")

    # ACCURACY CALCULATION
    if os.path.exists(GT_CSV):
        print("\n" + "="*40)
        print("      GROUND TRUTH EVALUATION")
        print("="*40)
        df_gt = pd.read_csv(GT_CSV, sep=';')
        gt_dict = df_gt.groupby('quadrat_id')['species_id'].apply(set).to_dict()
        
        f1s = []
        for res in results:
            qid = res['quadrat_id']
            if qid not in gt_dict: continue
            true_set = gt_dict[qid]
            pred_set = set(res['species_ids'])
            
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            f1s.append(f1)
        
        print(f" ► Macro-Averaged F1 Score: {np.mean(f1s):.4f}")
        print("="*40)
    else:
        print("\n[Note] Ground truth file not found at expected path. Accuracy calculation skipped.")

if __name__ == "__main__":
    main()
