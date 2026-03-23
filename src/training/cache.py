import os
import torch
from torch.amp import autocast
from tqdm import tqdm

from config import EXTRACT_CHUNK_SIZE


def chunked_backbone_forward(backbone, x, chunk_size):
    """
    Run a backbone on x in sub-batches of chunk_size to cap peak VRAM.
    Only chunk_size images worth of activations exist at any moment.
    """
    return torch.cat([backbone(chunk) for chunk in x.split(chunk_size)])


def extract_and_cache_features(model, loader, device, cache_path):
    """
    One-time frozen-backbone feature extraction for all training images.
    Saves to disk so Phase 1 head training never re-runs the backbones.

    Uses EXTRACT_CHUNK_SIZE (64) which is larger than training chunk sizes
    since no gradient graph is built -- maximises GPU throughput.

    Timing estimate: ~1-1.5 hrs at 384px, batch=384, chunk=64.
    """
    print("[Feature Cache] Extracting backbone features (one-time pass)...")
    model.eval()
    all_bio, all_dino, all_conv, all_labels = [], [], [], []

    with torch.no_grad():
        for data in tqdm(loader, desc="Extracting features"):
            images = data[0]['data'].to(memory_format=torch.channels_last)
            labels = data[0]['label'].squeeze().long()
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                feat_bio  = chunked_backbone_forward(model.bioclip,  images, EXTRACT_CHUNK_SIZE)
                feat_dino = chunked_backbone_forward(model.dinov2,   images, EXTRACT_CHUNK_SIZE)
                feat_conv = chunked_backbone_forward(model.convnext, images, EXTRACT_CHUNK_SIZE)
            all_bio.append(feat_bio.cpu())
            all_dino.append(feat_dino.cpu())
            all_conv.append(feat_conv.cpu())
            all_labels.append(labels.cpu())

    loader.reset()
    cache = {
        'bio':    torch.cat(all_bio),
        'dino':   torch.cat(all_dino),
        'conv':   torch.cat(all_conv),
        'labels': torch.cat(all_labels),
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(cache, cache_path)
    print(f"[Feature Cache] Saved {len(cache['labels']):,} samples to {cache_path}")
    return cache


class CachedFeatureDataset(torch.utils.data.Dataset):
    """Wraps cached backbone features for fast Phase 1 head training."""
    def __init__(self, cache):
        self.bio    = cache['bio']
        self.dino   = cache['dino']
        self.conv   = cache['conv']
        self.labels = cache['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.bio[idx], self.dino[idx], self.conv[idx], self.labels[idx]
