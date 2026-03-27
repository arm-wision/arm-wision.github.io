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
import os
import pandas as pd
from tqdm import tqdm
import csv
import glob

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
            
            # Strip 'module.' prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
                
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded weights from {model_path}")
            if msg.missing_keys:
                print(f"Warning: Missing keys: {len(msg.missing_keys)}")
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
            # Use Softmax for single-label trained model candidates
            probs = torch.softmax(logits, dim=1)
            
        return probs.cpu().float().numpy()

    def run_sahi_inference(self, image_path, slice_height=512, slice_width=512, overlap_ratio=0.2):
        if not os.path.exists(image_path):
            return None

        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        
        all_probs = []
        # Global view
        all_probs.append(self.predict_tile(image))

        # Tiled views
        if height > slice_height or width > slice_width:
            y_step = int(slice_height * (1 - overlap_ratio))
            x_step = int(slice_width * (1 - overlap_ratio))
            for y in range(0, height - slice_height + 1, y_step):
                for x in range(0, width - slice_width + 1, x_step):
                    tile = image.crop((x, y, x + slice_width, y + slice_height))
                    all_probs.append(self.predict_tile(tile))
                
        # Max-pooling across views
        final_probs = np.max(np.vstack(all_probs), axis=0)
        return final_probs

def main():
    BASE_DIR = "/workspace/PlantCLEF2026"
    MODEL_PATH = os.path.join(BASE_DIR, "models/best_calibrated/mp_rank_00_model_states.pt")
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = os.path.join(BASE_DIR, "models/final/mp_rank_00_model_states.pt")
    
    TRAIN_CSV = "/workspace/plantclef/processed/train_metadata_cleaned_verified.csv"
    TEST_DIR = "/workspace/plantclef/raw/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/images/"
    OUTPUT_CSV = os.path.join(BASE_DIR, "submission_plantclef2026.csv")
    
    # Threshold for predicting a species
    # Since it's softmax but multiple species can be present, 
    # we can take top K or anything above a small threshold.
    THRESHOLD = 0.05 
    TOP_K = 15 # Maximum species to predict per quadrat if threshold is met

    wrapper = TestInferenceWrapper(MODEL_PATH, TRAIN_CSV)
    
    test_images = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
    print(f"Found {len(test_images)} test images.")
    
    results = []
    
    for img_path in tqdm(test_images, desc="Inference"):
        quadrat_id = os.path.splitext(os.path.basename(img_path))[0]
        probs = wrapper.run_sahi_inference(img_path)
        
        if probs is not None:
            # Sort indices by probability
            sorted_indices = np.argsort(probs)[::-1]
            
            # Selection logic: top 1 ALWAYS, then anything above THRESHOLD up to TOP_K
            predicted_indices = [sorted_indices[0]]
            for idx in sorted_indices[1:TOP_K]:
                if probs[idx] > THRESHOLD:
                    predicted_indices.append(idx)
                else:
                    break
            
            # Map indices to species IDs
            predicted_species = [wrapper.idx_to_species[idx] for idx in predicted_indices]
            
            # Format: "[ID1, ID2, ...]"
            species_str = "[" + ", ".join(map(str, predicted_species)) + "]"
            results.append({
                "quadrat_id": quadrat_id,
                "species_ids": species_str
            })
            
    # Save to CSV with quotes
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, sep=',', index=False, quoting=csv.QUOTE_ALL)
    print(f"Submission saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
