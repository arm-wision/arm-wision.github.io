import torch
import torch.nn.functional as F
from models.ensemble import PlantEnsemble
from PIL import Image
import numpy as np
import os

class EnsembleClassifierWrapper:
    """
    High-level wrapper for the Triple Ensemble (BioCLIP + DINOv2 + ConvNeXt).
    Provides a simplified interface for inference on individual image tiles,
    handling all necessary resizing, normalization, and tensor conversions.
    """
    def __init__(self, model_path, num_classes=7800, device='cuda'):
        """
        Initialize the ensemble and load pre-trained weights.

        Args:
            model_path (str): Path to the saved .pth state dict.
            num_classes (int): Number of plant species (default 7,800).
            device (str): Computation device ('cuda' or 'cpu').
        """
        # Initialize the Triple architecture (448px input resolution)
        self.model = PlantEnsemble(num_classes=num_classes, input_res=448)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"[Inference] Loaded ensemble model from {model_path}")
        else:
            print(f"[Warning] Model path {model_path} not found. Using untrained ensemble for dry run.")
            
        self.model.to(device)
        self.model.eval() # Set to evaluation mode
        self.device = device

    def predict_tile(self, image_tile):
        """
        Runs a single forward pass on a 448x448 image tile.

        Args:
            image_tile (PIL.Image): A cropped image patch from a larger plot.

        Returns:
            numpy.ndarray: Sigmoid probabilities for all 7,800 classes.
        """
        # 1. Resize and Convert to Tensor [C, H, W]
        img = image_tile.resize((448, 448))
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        
        # 2. Normalization (BioCLIP/DINOv2 standard means/stds)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # 3. Forward Pass with AMP (Automatic Mixed Precision) for speed
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                logits = self.model(img_tensor.to(self.device))
            # Use Sigmoid for multi-label support (each species treated independently)
            probs = torch.sigmoid(logits) 
            
        return probs.cpu().float().numpy()

def run_sahi_inference(image_path, model_wrapper, slice_height=512, slice_width=512, overlap_height_ratio=0.2, overlap_width_ratio=0.2):
    """
    Implements Slicing Aided Hyper Inference (SAHI) to handle high-resolution 
    vegetation quadrats where small plants may be lost if resized directly.

    Args:
        image_path (str): Path to the input high-res JPG.
        model_wrapper (EnsembleClassifierWrapper): The initialized ensemble.
        slice_height (int): Height of each tile in pixels.
        slice_width (int): Width of each tile in pixels.
        overlap_height_ratio (float): Vertical overlap between tiles (0.0 to 1.0).
        overlap_width_ratio (float): Horizontal overlap between tiles (0.0 to 1.0).

    Returns:
        numpy.ndarray: Final probability vector (max-pooled across all tiles).
    """
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return None

    # Load original high-res image
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    # If image is smaller than slice size, process directly
    if height <= slice_height and width <= slice_width:
        return model_wrapper.predict_tile(image).squeeze()

    all_probs = []
    
    # Calculate step size based on overlap (e.g., 512px slice with 20% overlap = 409px step)
    y_step = int(slice_height * (1 - overlap_height_ratio))
    x_step = int(slice_width * (1 - overlap_width_ratio))
    
    # Iterate across the image grid
    print(f"[SAHI] Slicing {os.path.basename(image_path)} ({width}x{height}) into {slice_width}x{slice_height} tiles...")
    for y in range(0, max(1, height - slice_height + 1), y_step):
        for x in range(0, max(1, width - slice_width + 1), x_step):
            # Crop tile and predict
            tile = image.crop((x, y, min(x + slice_width, width), min(y + slice_height, height)))
            probs = model_wrapper.predict_tile(tile)
            all_probs.append(probs)
            
    if not all_probs:
        return None
        
    # Final Aggregation Strategy: Max-Pooling
    # If a species is found in ANY tile with high confidence, we report it.
    final_probs = np.max(np.vstack(all_probs), axis=0)
    return final_probs

if __name__ == "__main__":
    from config import RESOLUTION
    
    MODEL_PATH = "models/ensemble_best.pth"
    wrapper = EnsembleClassifierWrapper(MODEL_PATH)
    
    # Example path - can be updated to point to any test image
    TEST_IMAGE = "/workspace/plantclef/raw/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/images/GUARDEN-CBNMed-44-7-12-03-20240629.jpg"
    
    if os.path.exists(TEST_IMAGE):
        results = run_sahi_inference(TEST_IMAGE, wrapper)
        if results is not None:
            top_indices = np.argsort(results)[-5:][::-1]
            print(f"Top Species Found in {os.path.basename(TEST_IMAGE)} (Ensemble):")
            for idx in top_indices:
                print(f"Species Index: {idx} | Confidence: {results[idx]:.4f}")
    else:
        print(f"Test image {TEST_IMAGE} not found.")
