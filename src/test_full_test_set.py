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
from sklearn.metrics import precision_score, recall_score, f1_score

# Local imports
import config
from models.ensemble import PlantEnsemble

class TestInferenceWrapper:
    def __init__(self, model_path, csv_path, device='cuda'):
        self.device = device
        self.resolution = config.RESOLUTION
        
        # 1. Load species mapping from training metadata
        print(f"Loading species mapping from {csv_path}...")
        df_train = pd.read_csv(csv_path, sep=';')
        species_ids = sorted(df_train['species_id'].unique())
        self.idx_to_species = {i: s for i, s in enumerate(species_ids)}
        self.num_classes = len(species_ids)
        print(f"Found {self.num_classes} unique species.")

        # 2. Initialize model
        self.model = PlantEnsemble(
            num_classes=self.num_classes, 
            input_res=self.resolution,
            bioclip_name=config.BIOCLIP_NAME,
            dinov2_name=config.DINOV2_NAME,
            convnext_name=config.CONVNEXT_NAME
        )
        
        # 3. Apply LoRA (must match training)
        self.model.apply_lora(
            r=config.LORA_R, 
            lora_alpha=config.LORA_ALPHA, 
            lora_dropout=config.LORA_DROPOUT
        )
        
        # 4. Load weights
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=device)
            state_dict = ckpt['module'] if isinstance(ckpt, dict) and 'module' in ckpt else ckpt
            
            # Strip 'module.' prefix and filter out phase1_head (not used for inference)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                if not name.startswith('phase1_head.'):
                    new_state_dict[name] = v
                
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded weights from {model_path}")
        else:
            raise FileNotFoundError(f"Model path {model_path} not found!")
            
        self.model.to(device)
        self.model.eval()
        
        # Normalization constants
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        self.std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    def predict_tile(self, image_tile):
        img = image_tile.resize((self.resolution, self.resolution), Image.Resampling.BILINEAR)
        img_np = np.array(img)
        if len(img_np.shape) == 2:
            img_np = np.stack([img_np]*3, axis=-1)
        elif img_np.shape[2] == 4:
            img_np = img_np[:,:,:3]
            
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        img_tensor = (img_tensor - self.mean) / self.std
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = self.model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            
        return probs.cpu().float().numpy()

    def run_sahi_inference(self, image_path, slice_height=512, slice_width=512, overlap_ratio=0.2):
        if not os.path.exists(image_path):
            return None

        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        
        all_probs = []
        all_probs.append(self.predict_tile(image))

        if height > slice_height or width > slice_width:
            y_step = int(slice_height * (1 - overlap_ratio))
            x_step = int(slice_width * (1 - overlap_ratio))
            for y in range(0, height - slice_height + 1, y_step):
                for x in range(0, width - slice_width + 1, x_step):
                    tile = image.crop((x, y, x + slice_width, y + slice_height))
                    all_probs.append(self.predict_tile(tile))
                
        final_probs = np.max(np.vstack(all_probs), axis=0)
        return final_probs

def showcase_metrics(results, all_probs_captured):
    """
    Detailed showcase of prediction metrics when GT is missing.
    Analyzes model behavior, confidence, and detection patterns.
    """
    print("\n" + "="*60)
    print("      PLANTCLEF 2026 - INFERENCE METRICS SHOWCASE")
    print("="*60)
    
    total_samples = len(results)
    if total_samples == 0:
        print("No results to analyze.")
        return

    # 1. Detection Statistics
    all_predicted_species = [s for r in results for s in r['species_ids']]
    species_counts = Counter(all_predicted_species)
    unique_species = set(all_predicted_species)
    detections_per_image = [len(r['species_ids']) for r in results]
    
    print(f" ► Total Images Processed:     {total_samples}")
    print(f" ► Unique Species Identified:  {len(unique_species)}")
    print(f" ► Total Species Detections:   {len(all_predicted_species)}")
    print(f" ► Avg Species per Image:      {np.mean(detections_per_image):.2f}")
    print(f" ► Max Species in one Image:   {np.max(detections_per_image)}")
    
    # 2. Confidence Calibration
    # Filter out near-zero probabilities for meaningful stats
    flat_probs = np.concatenate(all_probs_captured)
    sig_probs = flat_probs[flat_probs > 0.001]
    
    print("\n--- Confidence Statistics ---")
    print(f" ► Mean Top-1 Probability:     {np.mean([np.max(p) for p in all_probs_captured]):.4f}")
    print(f" ► Overall Mean (Prob > 0.001): {np.mean(sig_probs):.4f}")
    print(f" ► 90th Percentile Confidence: {np.percentile(sig_probs, 90):.4f}")
    
    # 3. Species Richness & Distribution
    print("\n--- Top 10 Most Frequent Species ---")
    print(f"{'Species ID':<15} | {'Occurrences':<12} | {'Frequency'}")
    print("-" * 45)
    for s_id, count in species_counts.most_common(10):
        print(f"{s_id:<15} | {count:<12} | {count/total_samples*100:6.1f}%")

    # 4. Long-Tail Analysis
    rare_species_count = sum(1 for c in species_counts.values() if c == 1)
    print(f"\n ► Rare Species (1 detection): {rare_species_count} ({rare_species_count/len(unique_species)*100:.1f}% of richness)")
    print("="*60 + "\n")

def evaluate_predictions(results, ground_truth_df):
    """
    Calculates metrics by comparing predicted species lists to ground truth.
    Used only if GROUND_TRUTH_CSV is provided.
    """
    gt_dict = ground_truth_df.groupby('quadrat_id')['species_id'].apply(set).to_dict()
    sample_f1s = []
    all_tp, all_fp, all_fn = 0, 0, 0

    for res in results:
        qid = res['quadrat_id']
        if qid not in gt_dict: continue

        true_set = gt_dict[qid]
        pred_set = set(res['species_ids'])

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        sample_f1s.append(f1)
        all_tp += tp; all_fp += fp; all_fn += fn

    return {
        "Sample-Avg F1": np.mean(sample_f1s) if sample_f1s else 0,
        "Global Precision": all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0,
        "Global Recall": all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    }

def main():
    BASE_DIR = "/workspace/PlantCLEF2026"
    MODEL_PATH = os.path.join(BASE_DIR, "models/best_calibrated/mp_rank_00_model_states.pt")
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = os.path.join(BASE_DIR, "models/final/mp_rank_00_model_states.pt")
    
    TRAIN_CSV = "/workspace/plantclef/processed/train_metadata_cleaned_verified.csv"
    # User specified directory
    TEST_DIR = "/workspace/plantclef/raw/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/images/"
    GROUND_TRUTH_CSV = None # Set to your GT CSV if available for benchmarking
    OUTPUT_CSV = os.path.join(BASE_DIR, "submission_plantclef2025_test.csv")

    has_gt = GROUND_TRUTH_CSV is not None and os.path.exists(GROUND_TRUTH_CSV)
    
    THRESHOLD = 0.05 
    TOP_K = 15 

    wrapper = TestInferenceWrapper(MODEL_PATH, TRAIN_CSV)
    
    test_images = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
    print(f"Found {len(test_images)} test images in 2025 directory.")
    
    results = []
    all_probs_captured = []
    
    for img_path in tqdm(test_images, desc="Inference"):
        quadrat_id = os.path.splitext(os.path.basename(img_path))[0]
        probs = wrapper.run_sahi_inference(img_path)
        
        if probs is not None:
            all_probs_captured.append(probs)
            sorted_indices = np.argsort(probs)[::-1]
            
            predicted_indices = [sorted_indices[0]]
            for idx in sorted_indices[1:TOP_K]:
                if probs[idx] > THRESHOLD:
                    predicted_indices.append(idx)
                else:
                    break
            
            predicted_species = [wrapper.idx_to_species[idx] for idx in predicted_indices]
            results.append({"quadrat_id": quadrat_id, "species_ids": predicted_species})
            
    # Save results
    df_results = pd.DataFrame(results)
    df_csv = df_results.copy()
    df_csv['species_ids'] = df_csv['species_ids'].apply(lambda x: "[" + ", ".join(map(str, x)) + "]")
    df_csv.to_csv(OUTPUT_CSV, sep=',', index=False, quoting=csv.QUOTE_ALL)
    print(f"Submission saved to {OUTPUT_CSV}")

    # Metrics Showcase
    showcase_metrics(results, all_probs_captured)

    if has_gt:
        print("--- Ground Truth Evaluation ---")
        df_gt = pd.read_csv(GROUND_TRUTH_CSV, sep=';')
        metrics = evaluate_predictions(results, df_gt)
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

if __name__ == "__main__":
    main()
