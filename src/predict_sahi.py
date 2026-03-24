import torch
import torch.nn.functional as F
from models.ensemble import PlantEnsemble
from PIL import Image
import numpy as np
import os
import config

class EnsembleClassifierWrapper:
    """
    High-level wrapper for the Triple Ensemble (BioCLIP + DINOv2 + ConvNeXt).
    Provides a simplified interface for inference on individual image tiles,
    handling all necessary resizing, normalization, and tensor conversions.
    """
    def __init__(self, model_path, device='cuda'):
        """
        Initialize the ensemble and load pre-trained weights.

        Args:
            model_path (str): Path to the saved .pt or .pth state dict.
            device (str): Computation device ('cuda' or 'cpu').
        """
        self.device = device
        self.resolution = config.RESOLUTION
        
        # 1. Initialize the Triple architecture
        self.model = PlantEnsemble(
            num_classes=7800, 
            input_res=self.resolution,
            bioclip_name=config.BIOCLIP_NAME,
            dinov2_name=config.DINOV2_NAME,
            convnext_name=config.CONVNEXT_NAME
        )
        
        # 2. Apply LoRA (Must match training rank/alpha)
        # Check if we should apply LoRA based on config
        self.model.apply_lora(
            r=config.LORA_R, 
            lora_alpha=config.LORA_ALPHA, 
            lora_dropout=config.LORA_DROPOUT
        )
        
        # 3. Load weights
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=device)
            # Handle DeepSpeed/Full-ckpt vs state-dict-only
            if isinstance(ckpt, dict) and 'module' in ckpt:
                state_dict = ckpt['module']
            else:
                state_dict = ckpt
                
            # Strip 'module.' prefix if present (from DistributedDataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
                
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"[Inference] Loaded ensemble model from {model_path}")
            if msg.missing_keys:
                print(f"[Warning] Missing keys: {len(msg.missing_keys)}")
        else:
            print(f"[Error] Model path {model_path} not found!")
            
        self.model.to(device)
        self.model.eval()
        
        # Normalization constants (DALI matching)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        self.std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    def predict_tile(self, image_tile):
        """
        Runs a single forward pass on an image tile.
        """
        # 1. Resize and Convert to Tensor [C, H, W]
        img = image_tile.resize((self.resolution, self.resolution), Image.Resampling.BILINEAR)
        img_np = np.array(img)
        
        # Handle grayscale or RGBA
        if len(img_np.shape) == 2:
            img_np = np.stack([img_np]*3, axis=-1)
        elif img_np.shape[2] == 4:
            img_np = img_np[:,:,:3]
            
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        
        # 2. Normalization
        img_tensor = (img_tensor - self.mean) / self.std
        
        # 3. Forward Pass
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = self.model(img_tensor)
            # Use Softmax for probability distribution
            probs = torch.softmax(logits, dim=1)
            
        return probs.cpu().float().numpy()

def run_sahi_inference(image_path, model_wrapper, slice_height=512, slice_width=512, overlap_ratio=0.2):
    """
    Implements Slicing Aided Hyper Inference (SAHI)
    """
    if not os.path.exists(image_path):
        return None

    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    # Grid calculation
    y_step = int(slice_height * (1 - overlap_ratio))
    x_step = int(slice_width * (1 - overlap_ratio))
    
    all_probs = []
    
    # 1. Always include the global resized view
    global_probs = model_wrapper.predict_tile(image)
    all_probs.append(global_probs)

    # 2. Add tiled views if image is large enough
    if height > slice_height or width > slice_width:
        for y in range(0, height - slice_height + 1, y_step):
            for x in range(0, width - slice_width + 1, x_step):
                tile = image.crop((x, y, x + slice_width, y + slice_height))
                all_probs.append(model_wrapper.predict_tile(tile))
                
    # Max-pooling across views
    final_probs = np.max(np.vstack(all_probs), axis=0)
    return final_probs

if __name__ == "__main__":
    # Point to the DeepSpeed model states file
    MODEL_PATH = "models/best_calibrated/mp_rank_00_model_states.pt"
    
    if not os.path.exists(MODEL_PATH):
        # Fallback to final if best doesn't exist
        MODEL_PATH = "models/final/mp_rank_00_model_states.pt"

    print(f"Initializing model with LoRA R={config.LORA_R}...")
    wrapper = EnsembleClassifierWrapper(MODEL_PATH)
    
    # Path to a test image
    TEST_IMAGE = "/workspace/PlantCLEF2026/verify_images.py" # Placeholder check
    # Let's find an actual image
    import glob
    images = glob.glob(os.path.join(config.IMG_DIR, "**/*.jpg"), recursive=True)
    if images:
        TEST_IMAGE = images[0]
        results = run_sahi_inference(TEST_IMAGE, wrapper)
        if results is not None:
            top_indices = np.argsort(results)[-5:][::-1]
            print(f"\n[Results] Top Species in {os.path.basename(TEST_IMAGE)}:")
            for idx in top_indices:
                print(f"  Index: {idx:4d} | Confidence: {results[idx]:.4f}")
    else:
        print("No images found in IMG_DIR to test.")
