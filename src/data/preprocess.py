import os
import subprocess
import cudf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np


class BlurDataset(Dataset):
    """
    Optimized Dataset for high-speed image loading.
    Uses PIL + Torchvision transforms which are generally faster for
    high-throughput pipelines when paired with multiple workers.
    """
    def __init__(self, paths, target_size=512):
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            # Loading via PIL + Transform is highly optimized for DataLoader workers
            img = Image.open(self.paths[idx])
            return self.transform(img), idx
        except Exception:
            # Return zero tensor for corrupted images to keep batch size consistent
            return torch.zeros((1, 512, 512)), idx


def gpu_blur_audit(paths, batch_size=512, threshold=100):
    """
    Optimized GPU Blur Detection.
    Uses a massive batch size to saturate high-VRAM GPUs (RTX 5090).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = BlurDataset(paths)

    # Increase num_workers and use pin_memory to feed the GPU faster
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=12,
        pin_memory=True,
        prefetch_factor=4
    )

    # 3x3 Laplacian Kernel for edge detection
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                          dtype=torch.float32, device=device).view(1, 1, 3, 3)

    scores = np.zeros(len(paths))
    print(f"[Blur Audit] Running on {device} ({len(paths):,} images)...")

    with torch.no_grad():
        for imgs, indices in tqdm(loader, desc="GPU Processing", unit="batch"):
            imgs = imgs.to(device, non_blocking=True)
            # Batch convolution: Edge detection
            laplacian = F.conv2d(imgs, kernel, padding=1)
            # Focus Score = Variance (Higher is sharper)
            batch_scores = torch.var(laplacian, dim=(2, 3)).squeeze()
            scores[indices.numpy()] = batch_scores.cpu().numpy()

    return scores


def audit_data(csv_path, img_dir, output_path, max_images_per_species=500, blur_threshold=100):
    """
    Full GPU-Accelerated Preprocessing Pipeline.
    """
    print(f"[Audit] Loading metadata from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # 1. GPU-Accelerated CSV Loading
    df = cudf.read_csv(csv_path, sep=';')
    print(f"Initial records: {len(df):,}")

    # 2. Fast File Discovery (Shell-assisted)
    # Python's os.walk is too slow for 1.4M files.
    print(f"Scanning {img_dir} for existing images (using shell find)...")
    # This command finds all files in the directory and returns 'species_id/image_name'
    find_cmd = f"find {img_dir} -maxdepth 2 -type f -name '*.jpg' | sed 's|{img_dir}||'"
    try:
        existing_files_str = subprocess.check_output(find_cmd, shell=True).decode('utf-8')
        existing_files = cudf.Series(existing_files_str.splitlines()).str.lstrip('/')
    except Exception as e:
        print(f"Error scanning files: {e}")
        return

    # 3. GPU-Accelerated Join/Filter
    # Create the relative path column in cuDF (STAYS ON GPU)
    df['relative_path'] = df['species_id'].astype(str) + "/" + df['image_name']

    # Filter metadata to only include files that physically exist
    df = df[df['relative_path'].isin(existing_files)]
    print(f"After physical file verification: {len(df):,}")

    # 4. GPU Blur Audit
    if blur_threshold > 0:
        # Construct full paths for the loader
        full_paths = (img_dir + df['relative_path']).to_arrow().to_pylist()
        blur_scores = gpu_blur_audit(full_paths, threshold=blur_threshold)

        # Add scores back to cuDF and filter
        df['blur_score'] = cudf.Series(blur_scores)
        df = df[df['blur_score'] >= blur_threshold]
        print(f"After removing blurry images: {len(df):,}")

    # 5. Long-Tail Balancing (GPU-Accelerated)
    if max_images_per_species > 0:
        print(f"Capping species at {max_images_per_species} images...")
        # cuDF groupby.head() is significantly faster than Pandas
        df = df.groupby('species_id').head(max_images_per_species)
        print(f"Final records after balancing: {len(df):,}")

    # 6. Cleanup and Save
    # Drop temp columns and save to CSV
    cols_to_drop = [c for c in ['relative_path', 'blur_score'] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    df.to_csv(output_path, sep=';', index=False)

    print(f"[Audit Complete] Final Unique Species: {df['species_id'].nunique():,}")
    print(f"Cleaned metadata saved to: {output_path}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config import RAW_CSV, IMG_DIR, CLEANED_CSV

    audit_data(RAW_CSV, IMG_DIR, CLEANED_CSV, max_images_per_species=500, blur_threshold=100)
