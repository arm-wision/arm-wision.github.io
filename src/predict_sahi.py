import torch
import torch.nn.functional as F
from models.ensemble import PlantEnsemble
from PIL import Image
import numpy as np
import os
import config

try:
    import plantclef_ext
except ImportError:
    plantclef_ext = None

class EnsembleClassifierWrapper:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.resolution = config.RESOLUTION
        self.model = PlantEnsemble(
            num_classes=7800, 
            input_res=self.resolution,
            bioclip_name=config.BIOCLIP_NAME,
            dinov2_name=config.DINOV2_NAME,
            convnext_name=config.CONVNEXT_NAME
        )
        self.model.apply_lora(r=config.LORA_R, lora_alpha=config.LORA_ALPHA)
        
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=device)
            state_dict = ckpt['module'] if isinstance(ckpt, dict) and 'module' in ckpt else ckpt
            new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict, strict=False)
            
        self.model.to(device)
        self.model.eval()
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        self.std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    def predict_batch(self, batched_tiles):
        """ Runs inference on a batch of tiles (already on GPU). """
        # batched_tiles: (N, C, H, W)
        # 1. Resize if needed (DALI-style or custom kernel preferred)
        if batched_tiles.shape[2] != self.resolution:
            batched_tiles = F.interpolate(batched_tiles, size=(self.resolution, self.resolution), mode='bilinear')
            
        # 2. Normalize
        batched_tiles = (batched_tiles - self.mean) / self.std
        
        # 3. Forward
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = self.model(batched_tiles)
            probs = torch.softmax(logits, dim=1)
        return probs

def run_sahi_inference(image_path, model_wrapper, slice_height=512, slice_width=512, 
                       overlap_ratio=0.2, tax_filter=None):
    if not os.path.exists(image_path): return None

    # Load image once to GPU
    image_pil = Image.open(image_path).convert('RGB')
    img_tensor = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float().to('cuda') / 255.0

    # 1. CUDA Tiling
    if plantclef_ext is not None:
        tiles = plantclef_ext.extract_tiles(img_tensor, slice_height, slice_width, overlap_ratio)
        # tiles: (N, C, H, W)
    else:
        # Fallback to slow Python slicing
        # (Simplified for brevity)
        tiles = img_tensor.unsqueeze(0) 

    # 2. Batch Inference
    # Split into sub-batches if N is too large for VRAM
    CHUNK_SIZE = 16
    tile_probs = []
    for i in range(0, len(tiles), CHUNK_SIZE):
        batch = tiles[i:i+CHUNK_SIZE]
        tile_probs.append(model_wrapper.predict_batch(batch))
    
    tile_probs = torch.cat(tile_probs, dim=0)

    # 3. Fused Max Pooling
    if plantclef_ext is not None:
        final_probs = plantclef_ext.fused_max_pool(tile_probs)
    else:
        final_probs = torch.max(tile_probs, dim=0)[0]

    # 4. Taxonomic Filtering
    if tax_filter is not None:
        final_probs = tax_filter.filter_predictions(final_probs)

    return final_probs.cpu().numpy()

if __name__ == "__main__":
    MODEL_PATH = "models/best_calibrated/mp_rank_00_model_states.pt"
    wrapper = EnsembleClassifierWrapper(MODEL_PATH)
    
    # Initialize taxonomic filter (dummy neighbors for example)
    if plantclef_ext is not None:
        dummy_neighbors = [list(range(100)) for _ in range(7800)]
        filter_obj = plantclef_ext.TaxonomicFilter(dummy_neighbors)
    else:
        filter_obj = None

    import glob
    images = glob.glob(os.path.join(config.IMG_DIR, "**/*.jpg"), recursive=True)
    if images:
        results = run_sahi_inference(images[0], wrapper, tax_filter=filter_obj)
        if results is not None:
            top_indices = np.argsort(results)[-5:][::-1]
            print(f"\n[Results] Top Species: {top_indices}")
