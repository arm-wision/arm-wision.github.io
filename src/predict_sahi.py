import torch
import torch.nn.functional as F
from model import PlantBioCLIP
from PIL import Image
import numpy as np
import os

class BioCLIPClassifierWrapper:
    """
    Wrapper for SAHI to use PlantBioCLIP for tiled classification.
    """
    def __init__(self, model_path, num_classes=7800, device='cuda'):
        self.model = PlantBioCLIP(num_classes=num_classes, input_res=448)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"Warning: Model path {model_path} not found. Using untrained model.")
        self.model.to(device)
        self.model.eval()
        self.device = device

    def predict_tile(self, image_tile):
        # Convert PIL image to tensor
        img = image_tile.resize((448, 448))
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        # Normalize (BioCLIP specific)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        with torch.no_grad():
            logits = self.model(img_tensor.to(self.device))
            probs = torch.sigmoid(logits) # Multi-label/Multi-class probabilities
        return probs.cpu().numpy()

def run_sahi_inference(image_path, model_wrapper, slice_height=512, slice_width=512, overlap_height_ratio=0.2, overlap_width_ratio=0.2):
    """
    Perform manual slicing and aggregate results.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return None

    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    # Process the whole image if it's smaller than a tile
    if height <= slice_height and width <= slice_width:
        return model_wrapper.predict_tile(image).squeeze()

    all_probs = []
    
    y_step = int(slice_height * (1 - overlap_height_ratio))
    x_step = int(slice_width * (1 - overlap_width_ratio))
    
    # Ensure we cover the whole image
    for y in range(0, max(1, height - slice_height + 1), y_step):
        for x in range(0, max(1, width - slice_width + 1), x_step):
            tile = image.crop((x, y, min(x + slice_width, width), min(y + slice_height, height)))
            probs = model_wrapper.predict_tile(tile)
            all_probs.append(probs)
            
    # Aggregate probabilities (max pooling across tiles)
    if not all_probs:
        return None
        
    final_probs = np.max(np.vstack(all_probs), axis=0)
    return final_probs

if __name__ == "__main__":
    # Example Usage
    MODEL_PATH = "models/bioclip_best.pth"
    wrapper = BioCLIPClassifierWrapper(MODEL_PATH)
    
    # Use a real image if available
    TEST_IMAGE = "data/PlantCLEF2025_test_images/PlantCLEF2025_test_images/2024-CEV3-20240602.jpg"
    
    if os.path.exists(TEST_IMAGE):
        results = run_sahi_inference(TEST_IMAGE, wrapper)
        if results is not None:
            # Print top 5 species found
            top_indices = np.argsort(results)[-5:][::-1]
            print(f"Top Species Found in {os.path.basename(TEST_IMAGE)}:")
            for idx in top_indices:
                print(f"Species Index: {idx} | Confidence: {results[idx]:.4f}")
    else:
        print(f"Test image {TEST_IMAGE} not found for demonstration.")
