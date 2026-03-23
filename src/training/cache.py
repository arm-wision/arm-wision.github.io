import os
import torch
from torch.amp import autocast
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from config import EXTRACT_CHUNK_SIZE


def chunked_backbone_forward(backbone, x, chunk_size):
    """
    Run a backbone on x in sub-batches of chunk_size to cap peak VRAM.
    Only chunk_size images worth of activations exist at any moment.
    """
    return torch.cat([backbone(chunk) for chunk in x.split(chunk_size)])


def extract_and_cache_features(model, loader, device, cache_path,
                                shard_size=200):
    """
    One-time frozen-backbone feature extraction for all training images.

    Optimisations for speed:
    - EXTRACT_CHUNK_SIZE=128: larger chunks, fewer kernel launches (no_grad is safe)
    - Async shard saving: GPU never waits for disk -- background thread writes
      each shard while the GPU processes the next batch immediately
    - shard_size=200 batches: ~77k images per shard, ~1.5GB RAM peak

    Timing target: < 1 hour at 384px, batch=384, chunk=128.
    """
    print("[Feature Cache] Extracting backbone features (async shard saving)...")
    model.eval()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    shard_dir = cache_path + "_shards"
    os.makedirs(shard_dir, exist_ok=True)

    batch_bio, batch_dino, batch_conv, batch_labels = [], [], [], []
    shard_idx   = 0
    shard_paths = []
    executor    = ThreadPoolExecutor(max_workers=1)  # single background save thread
    pending_future = None

    def _save_shard(data, path):
        """Runs in background thread -- GPU continues while this writes to disk."""
        torch.save(data, path)

    def flush_shard_async():
        nonlocal shard_idx, pending_future
        path = os.path.join(shard_dir, f"shard_{shard_idx:04d}.pt")
        shard_data = {
            'bio':    torch.cat(batch_bio),
            'dino':   torch.cat(batch_dino),
            'conv':   torch.cat(batch_conv),
            'labels': torch.cat(batch_labels),
        }
        shard_paths.append(path)
        batch_bio.clear(); batch_dino.clear()
        batch_conv.clear(); batch_labels.clear()
        shard_idx += 1
        # Wait for previous save to finish before submitting next
        # (avoids piling up shard data in RAM while writes queue up)
        if pending_future is not None:
            pending_future.result()
        pending_future = executor.submit(_save_shard, shard_data, path)

    with torch.no_grad():
        for idx, data in enumerate(tqdm(loader, desc="Extracting features")):
            images = data[0]['data'].to(memory_format=torch.channels_last)
            labels = data[0]['label'].squeeze().long()
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                feat_bio  = chunked_backbone_forward(model.bioclip,  images, EXTRACT_CHUNK_SIZE)
                feat_dino = chunked_backbone_forward(model.dinov2,   images, EXTRACT_CHUNK_SIZE)
                feat_conv = chunked_backbone_forward(model.convnext, images, EXTRACT_CHUNK_SIZE)
            batch_bio.append(feat_bio.cpu())
            batch_dino.append(feat_dino.cpu())
            batch_conv.append(feat_conv.cpu())
            batch_labels.append(labels.cpu())

            if (idx + 1) % shard_size == 0:
                flush_shard_async()

    if batch_bio:
        flush_shard_async()

    # Wait for the last background save to complete
    if pending_future is not None:
        pending_future.result()
    executor.shutdown(wait=False)
    loader.reset()

    # Merge all shards into a single cache file
    print(f"[Feature Cache] Merging {shard_idx} shards...")
    all_bio, all_dino, all_conv, all_labels = [], [], [], []
    for path in shard_paths:
        shard = torch.load(path, weights_only=False)
        all_bio.append(shard['bio'])
        all_dino.append(shard['dino'])
        all_conv.append(shard['conv'])
        all_labels.append(shard['labels'])
        del shard

    cache = {
        'bio':    torch.cat(all_bio),
        'dino':   torch.cat(all_dino),
        'conv':   torch.cat(all_conv),
        'labels': torch.cat(all_labels),
    }
    torch.save(cache, cache_path)
    print(f"[Feature Cache] Saved {len(cache['labels']):,} samples to {cache_path}")

    import shutil
    shutil.rmtree(shard_dir)
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
