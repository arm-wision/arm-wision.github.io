import os
import cudf
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class BlurDataset(Dataset):
    """
    Performance-optimized Dataset for fast image loading during blur auditing.
    Uses OpenCV for grayscale reading (faster than color) and standardizes
    image sizes to 512px to enable high-throughput GPU batch processing.
    """
    def __init__(self, paths, target_size=512):
        self.paths = paths
        self.target_size = target_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            # Grayscale is sufficient for Laplacian variance (edge detection)
            img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
            if img is None:
                return torch.zeros((1, self.target_size, self.target_size)), idx
            
            # Standardize size for batching on the GPU
            if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
                img = cv2.resize(img, (self.target_size, self.target_size))
            
            return torch.from_numpy(img).float().unsqueeze(0), idx
        except Exception:
            return torch.zeros((1, self.target_size, self.target_size)), idx

def gpu_blur_audit(paths, batch_size=256, threshold=100):
    """
    Parallelized Blur Detection on the GPU using the Laplacian Variance method.
    
    This function slides a 3x3 Laplacian kernel over batches of images to detect edges.
    Variance is calculated across the resulting edge map; lower variance indicates
    a lack of sharp edges (blur).

    Args:
        paths (list): List of absolute file paths to images.
        batch_size (int): Number of images to process in parallel on the GPU.
        threshold (int): Minimum variance score to be considered 'in-focus'.

    Returns:
        np.ndarray: Array of blur scores corresponding to the input paths.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = BlurDataset(paths)
    # Use multi-threaded loading to keep the GPU saturated
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
    
    # Laplacian Kernel: High-pass filter for edge detection
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device)
    kernel = kernel.view(1, 1, 3, 3)
    
    scores = np.zeros(len(paths))
    
    print(f"[Blur Audit] Running on {device} ({len(paths):,} images)...")
    with torch.no_grad():
        for imgs, indices in tqdm(loader, desc="GPU Processing"):
            imgs = imgs.to(device)
            # Perform batch convolution: [Batch, 1, H, W] -> [Batch, 1, H-2, W-2]
            laplacian = F.conv2d(imgs, kernel, padding=1)
            # Focus Score = Variance of the Laplacian
            batch_scores = torch.var(laplacian, dim=(1, 2, 3))
            scores[indices.numpy()] = batch_scores.cpu().numpy()
            
    return scores

def audit_data(csv_path, img_dir, output_path, max_images_per_species=500, blur_threshold=100):
    """
    Main entry point for data hygiene. Performs metadata cleaning, file verification,
    quality (blur) filtering, and long-tail balancing using GPU acceleration.

    Args:
        csv_path (str): Path to the raw PlantCLEF metadata CSV.
        img_dir (str): Root directory containing species-id folders.
        output_path (str): Where to save the 'gold-standard' cleaned metadata.
        max_images_per_species (int): Hard cap for common species (downsampling).
        blur_threshold (int): Minimum focus score (Laplacian Variance).
    """
    print(f"[Audit] Loading metadata from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Use NVIDIA cuDF for near-instant CSV loading/processing
    df = cudf.read_csv(csv_path, sep=';')
    initial_count = len(df)
    print(f"Initial records: {initial_count:,}")

    # 1. Deduplication
    df = df.drop_duplicates(subset=['image_name'])
    print(f"After removing duplicate image names: {len(df):,}")

    # 2. Physical File Verification
    # Scans the disk to ensure metadata rows have matching image files
    print(f"Scanning {img_dir} for existing images...")
    if not os.path.exists(img_dir):
        print(f"Error: Image directory {img_dir} not found.")
        return

    existing_files = set()
    species_folders = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
    
    for species_id in tqdm(species_folders, desc="Scanning species folders"):
        species_path = os.path.join(img_dir, species_id)
        files = os.listdir(species_path)
        for f in files:
            existing_files.add(f"{species_id}/{f}")

    # Switch to Pandas for more complex string-masking logic
    pdf = df.to_pandas()
    pdf['relative_path'] = pdf['species_id'].astype(str) + "/" + pdf['image_name']
    pdf['full_path'] = img_dir + "/" + pdf['relative_path']
    
    print("Verifying files against metadata...")
    mask = pdf['relative_path'].isin(existing_files)
    pdf = pdf[mask]
    print(f"After removing missing files: {len(pdf):,}")

    # 3. Quality Filtering: The GPU Blur Audit
    if blur_threshold > 0:
        pdf['blur_score'] = gpu_blur_audit(pdf['full_path'].tolist(), threshold=blur_threshold)
        pdf = pdf[pdf['blur_score'] >= blur_threshold]
        print(f"After removing blurry images: {len(pdf):,}")

    # 4. Long-Tail Balancing (Phase 1 Capping)
    # Reduces common classes to ensure they don't dominate early training epochs
    if max_images_per_species > 0:
        print(f"Capping species at {max_images_per_species} images...")
        pdf = pdf.groupby('species_id').head(max_images_per_species)
        print(f"Final records after balancing: {len(pdf):,}")

    # 5. Clean and Save
    final_df = pdf.drop(columns=['relative_path', 'full_path'])
    if 'blur_score' in final_df.columns:
        final_df = final_df.drop(columns=['blur_score'])
        
    final_df.to_csv(output_path, sep=';', index=False)
    print(f"[Audit Complete] Cleaned metadata saved to: {output_path}")
    
    unique_species = pdf['species_id'].nunique()
    print(f"Final Unique Species: {unique_species:,}")
    if initial_count > 0:
        print(f"Total Data Reduction: {100 * (1 - len(pdf)/initial_count):.2f}%")

if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Add the 'src' directory to sys.path so we can import config
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    
    from config import RAW_CSV, IMG_DIR, CLEANED_CSV
    
    # Run the audit with default parameters
    audit_data(RAW_CSV, IMG_DIR, CLEANED_CSV, max_images_per_species=500, blur_threshold=100)
