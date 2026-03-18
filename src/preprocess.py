import os
import cudf
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def audit_data(csv_path, img_dir, output_path, max_images_per_species=500):
    """
    GPU-accelerated audit and cleaning of PlantCLEF metadata.
    1. Verifies file existence.
    2. Removes duplicates.
    3. Balances the long-tail distribution.
    """
    print(f"Loading metadata from {csv_path}...")
    # Use cuDF for massive CSV speedup
    df = cudf.read_csv(csv_path, sep=';')
    initial_count = len(df)
    print(f"Initial records: {initial_count:,}")

    # 1. Basic Cleaning
    df = df.drop_duplicates(subset=['image_name'])
    print(f"After removing duplicate image names: {len(df):,}")

    # 2. Verify File Existence (The Bottleneck)
    # Instead of 1.4M os.path.exists() calls, we list all files once.
    print("Scanning disk for existing images (this may take a few minutes)...")
    existing_files = set()
    
    # We use a fast walk or list directory for species-specific folders
    species_folders = os.listdir(img_dir)
    for species_id in tqdm(species_folders, desc="Scanning species folders"):
        species_path = os.path.join(img_dir, species_id)
        if os.path.isdir(species_path):
            files = os.listdir(species_path)
            for f in files:
                existing_files.add(f"{species_id}/{f}")

    # Convert cuDF to pandas temporarily for the set intersection (faster for this specific operation)
    pdf = df.to_pandas()
    # Construct expected relative paths: "species_id/image_name"
    pdf['relative_path'] = pdf['species_id'].astype(str) + "/" + pdf['image_name']
    
    print("Verifying files against metadata...")
    mask = pdf['relative_path'].isin(existing_files)
    pdf = pdf[mask]
    print(f"After removing missing files: {len(pdf):,}")

    # 3. Address the Long-Tail (Downsampling)
    if max_images_per_species > 0:
        print(f"Capping species at {max_images_per_species} images to balance long-tail...")
        # Group by species and take the first N
        pdf = pdf.groupby('species_id').head(max_images_per_species)
        print(f"Final records after balancing: {len(pdf):,}")

    # 4. Save Cleaned Metadata
    # We save back with the original separator for consistency
    pdf.drop(columns=['relative_path']).to_csv(output_path, sep=';', index=False)
    print(f"Cleaned metadata saved to: {output_path}")
    
    # Quick Stats
    unique_species = pdf['species_id'].nunique()
    print(f"Final Unique Species: {unique_species:,}")
    print(f"Data Reduction: {100 * (1 - len(pdf)/initial_count):.2f}%")

if __name__ == "__main__":
    # Configure paths
    RAW_CSV = "data/PlantCLEF2024_single_plant_training_metadata.csv"
    IMG_DIR = "data/train/"
    CLEANED_CSV = "data/train_metadata_cleaned.csv"

    if not os.path.exists(IMG_DIR):
        print(f"Error: Image directory {IMG_DIR} not found. Ensure data is downloaded.")
    elif not os.path.exists(RAW_CSV):
        print(f"Error: Metadata {RAW_CSV} not found.")
    else:
        audit_data(RAW_CSV, IMG_DIR, CLEANED_CSV)
