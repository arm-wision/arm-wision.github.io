import torch
import torch.nn.functional as F
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from model import PlantBioCLIP
from PIL import Image
import numpy as np

class BioCLIPClassifierWrapper:
    """
    Wrapper for SAHI to use PlantBioCLIP for tiled classification.
    Since SAHI is primarily for detection, we adapt it for multi-label classification 
    over high-res quadrats (tiles).
    """
    def __init__(self, model_path, num_classes=7800, device='cuda'):
        self.model = PlantBioCLIP(num_classes=num_classes, input_res=448)
        self.model.load_state_dict(torch.load(model_path))
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
            probs = torch.sigmoid(logits) # Multi-label probabilities
        return probs.cpu().numpy()

def run_sahi_inference(image_path, model_wrapper, slice_height=512, slice_width=512, overlap_height_ratio=0.2, overlap_width_ratio=0.2):
    """
    Manually perform SAHI-style slicing and aggregate classification results.
    """
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    # Simple manual tiling (SAHI alternative for classification)
    all_probs = []
    
    y_step = int(slice_height * (1 - overlap_height_ratio))
    x_step = int(slice_width * (1 - overlap_width_ratio))
    
    for y in range(0, height - slice_height + 1, y_step):
        for x in range(0, width - slice_width + 1, x_step):
            tile = image.crop((x, y, x + slice_width, y + slice_height))
            probs = model_wrapper.predict_tile(tile)
            all_probs.append(probs)
            
    # Aggregate probabilities (max pooling across tiles to find all species present)
    final_probs = np.max(np.vstack(all_probs), axis=0)
    return final_probs

if __name__ == "__main__":
    # Example Usage
    wrapper = BioCLIPClassifierWrapper("bioclip_epoch_0.pth")
    # Replace with an actual quadrat image path
    results = run_sahi_inference("data/PlantCLEF2025_test_images/PlantCLEF2025_test_images/2024-CEV3-20240602.jpg", wrapper)
    
    # Print top 5 species found
    top_indices = np.argsort(results)[-5:][::-1]
    print("Top Species Found in Quadrat:")
    for idx in top_indices:
        print(f"Species Index: {idx} | Confidence: {results[idx]:.4f}")
