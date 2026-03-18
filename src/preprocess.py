import os
import cudf
import pandas as pd
import numpy as np
from tqdm import tqdm

def audit_data(csv_path, img_dir, output_path, max_images_per_species=500):
    """
    GPU-accelerated audit and cleaning of PlantCLEF metadata.
    """
    print(f"Loading metadata from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Use cuDF for massive CSV speedup
    df = cudf.read_csv(csv_path, sep=';')
    initial_count = len(df)
    print(f"Initial records: {initial_count:,}")

    # 1. Basic Cleaning
    df = df.drop_duplicates(subset=['image_name'])
    print(f"After removing duplicate image names: {len(df):,}")

    # 2. Verify File Existence
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
            # Store as "species_id/image_name"
            existing_files.add(f"{species_id}/{f}")

    # Convert to pandas for easier string manipulation and set operations
    pdf = df.to_pandas()
    pdf['relative_path'] = pdf['species_id'].astype(str) + "/" + pdf['image_name']
    
    print("Verifying files against metadata...")
    mask = pdf['relative_path'].isin(existing_files)
    pdf = pdf[mask]
    print(f"After removing missing files: {len(pdf):,}")

    # 3. Address the Long-Tail (Downsampling)
    if max_images_per_species > 0:
        print(f"Capping species at {max_images_per_species} images to balance long-tail...")
        pdf = pdf.groupby('species_id').head(max_images_per_species)
        print(f"Final records after balancing: {len(pdf):,}")

    # 4. Save Cleaned Metadata
    pdf.drop(columns=['relative_path']).to_csv(output_path, sep=';', index=False)
    print(f"Cleaned metadata saved to: {output_path}")
    
    unique_species = pdf['species_id'].nunique()
    print(f"Final Unique Species: {unique_species:,}")
    if initial_count > 0:
        print(f"Data Reduction: {100 * (1 - len(pdf)/initial_count):.2f}%")

if __name__ == "__main__":
    RAW_CSV = "data/PlantCLEF2024_single_plant_training_metadata.csv"
    IMG_DIR = "data/train/"
    CLEANED_CSV = "data/train_metadata_cleaned.csv"

    audit_data(RAW_CSV, IMG_DIR, CLEANED_CSV)
