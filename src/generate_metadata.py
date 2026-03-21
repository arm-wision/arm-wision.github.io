import os
import pandas as pd
from tqdm import tqdm

img_dir = "/workspace/plantclef/raw/train/images_max_side_800"
output_csv = "/workspace/plantclef/raw/PlantCLEF2024_single_plant_training_metadata.csv"

data = []
species_folders = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]

print(f"Scanning {len(species_folders)} species folders...")
for species_id in tqdm(species_folders):
    species_path = os.path.join(img_dir, species_id)
    images = [f for f in os.listdir(species_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img in images:
        data.append({
            'image_name': img,
            'species_id': int(species_id)
        })

df = pd.DataFrame(data)
print(f"Total images found: {len(df)}")
df.to_csv(output_csv, sep=';', index=False)
print(f"Metadata saved to {output_csv}")
