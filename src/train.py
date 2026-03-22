import os
# ---------------------------------------------------------------------------
# Memory allocator -- set BEFORE importing torch so the allocator is
# initialised correctly. expandable_segments reduces fragmentation by ~2-3GB,
# directly freeing headroom for larger batches.
# ---------------------------------------------------------------------------
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256"
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassAccuracy
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
import wandb
import cudf
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from models.ensemble import PlantEnsemble
from data.dataloader import get_dali_loaders
from config import (RAW_CSV, IMG_DIR, CLEANED_CSV, BATCH_SIZE, RESOLUTION,
                    MODE, BIOCLIP_NAME, DINOV2_NAME, CONVNEXT_NAME)

# ---------------------------------------------------------------------------
# Hardware flags
# ---------------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32  = True   # ~2x matmul throughput on Blackwell
torch.backends.cudnn.allow_tf32        = True
torch.backends.cuda.enable_flash_sdp(True)       # FlashAttention 2 for BioCLIP + DINOv2
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# Gradient accumulation: effective batch = BATCH_SIZE * ACCUMULATION_STEPS
ACCUMULATION_STEPS = 2

# Chunked forward: each backbone processes CHUNK_SIZE images at a time,
# keeping peak activation memory low while the logical batch stays large.
# Increase BATCH_SIZE and tune CHUNK_SIZE to find the best GPU utilisation.
CHUNK_SIZE = 32  # increased from 8 -- halves loop iterations per batch on 5090

# DeepSpeed config: ZeRO Stage 1 offloads optimizer states to CPU RAM,
# freeing ~4-8GB of VRAM that would otherwise hold Adam moment tensors.
DS_CONFIG = {
    "zero_optimization": {
        "stage": 1,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True      # pinned CPU RAM for faster PCIe transfers
        }
    },
    "bf16": {"enabled": True},      # consistent with autocast dtype
    "gradient_accumulation_steps": ACCUMULATION_STEPS,
    "train_micro_batch_size_per_gpu": BATCH_SIZE,
    "steps_per_print": 50,
    "wall_clock_breakdown": False,
    "distributed_backend": "nccl",  # skip MPI entirely -- not needed on single GPU
}

FEATURE_CACHE_PATH = "models/phase1_feature_cache.pt"


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class LogitAdjustmentLoss(nn.Module):
    """
    Logit Adjustment: shifts logits by log(prior) to demand larger margins
    for common classes and ease requirements for rare ones.
    """
    def __init__(self, class_counts, tau=1.0, base_criterion=None):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        priors = counts / counts.sum()
        self.adjustment = (tau * torch.log(priors + 1e-12)).to('cuda')
        self.criterion  = base_criterion if base_criterion else nn.BCEWithLogitsLoss()

    def forward(self, x, y):
        return self.criterion(x + self.adjustment, y)


class AsymmetricLoss(nn.Module):
    """
    ASL: aggressively down-weights easy negatives across 7,800 classes
    using separate gamma values for positive and negative samples.
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps  = eps

    def forward(self, x, y):
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = y       * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        with torch.no_grad():
            pt      = xs_pos * y + xs_neg * (1 - y)
            weights = (1 - pt).pow(self.gamma_pos * y + self.gamma_neg * (1 - y))
        loss *= weights
        return -loss.mean()


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.counter = 0


# ---------------------------------------------------------------------------
# Chunked backbone forward
# ---------------------------------------------------------------------------

def chunked_backbone_forward(backbone, x, chunk_size=CHUNK_SIZE):
    """
    Run a backbone on x in chunks of chunk_size to cap peak activation memory.

    With batch=64 and chunk_size=8, only 8 images worth of activations exist
    in VRAM at any moment, while the gradient graph still covers the full batch.
    This allows much larger logical batch sizes without OOM.
    """
    return torch.cat([backbone(chunk) for chunk in x.split(chunk_size)])


# ---------------------------------------------------------------------------
# Phase 1 feature caching
# ---------------------------------------------------------------------------

def extract_and_cache_features(model, loader, device, cache_path):
    """
    One-time extraction of frozen backbone features for all 1.4M images.
    Cached features eliminate 10x redundant backbone compute across Phase 1 epochs.
    """
    print("[Feature Cache] Extracting backbone features (one-time pass)...")
    model.eval()
    all_bio, all_dino, all_conv, all_labels = [], [], [], []

    with torch.no_grad():
        for data in tqdm(loader, desc="Extracting features"):
            images = data[0]['data'].to(memory_format=torch.channels_last)
            labels = data[0]['label'].squeeze().long()
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # Use chunked forward during extraction to keep VRAM flat
                feat_bio  = chunked_backbone_forward(model.bioclip,  images)
                feat_dino = chunked_backbone_forward(model.dinov2,   images)
                feat_conv = chunked_backbone_forward(model.convnext, images)
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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model, loader, criterion, num_classes, device, use_cache=False):
    """
    GPU-accelerated validation. All torchmetrics state stays on GPU;
    single CPU transfer happens only at .compute().
    """
    model.eval()
    val_loss = 0.0
    f1_metric        = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    recall_metric    = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
    acc_metric       = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)

    with torch.no_grad():
        for i, data in enumerate(loader):
            if use_cache:
                feat_bio, feat_dino, feat_conv, labels = [t.to(device) for t in data]
                labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    fb = F.normalize(model.proj_bio(feat_bio),   dim=1)
                    fd = F.normalize(model.proj_dino(feat_dino), dim=1)
                    fc = F.normalize(model.proj_conv(feat_conv), dim=1)
                    outputs = model.classifier(torch.cat([fb, fd, fc], dim=1))
                    loss    = criterion(outputs, labels_one_hot)
            else:
                images = data[0]['data'].to(memory_format=torch.channels_last)
                labels = data[0]['label'].squeeze().long()
                labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Use chunked forward in validation too -- same OOM risk as training
                    feat_bio  = chunked_backbone_forward(model.bioclip,  images)
                    feat_dino = chunked_backbone_forward(model.dinov2,   images)
                    feat_conv = chunked_backbone_forward(model.convnext, images)
                    feat_bio  = F.normalize(model.proj_bio(feat_bio),   dim=1)
                    feat_dino = F.normalize(model.proj_dino(feat_dino), dim=1)
                    feat_conv = F.normalize(model.proj_conv(feat_conv), dim=1)
                    outputs   = model.classifier(torch.cat([feat_bio, feat_dino, feat_conv], dim=1))
                    loss      = criterion(outputs, labels_one_hot)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            f1_metric.update(predicted, labels)
            precision_metric.update(predicted, labels)
            recall_metric.update(predicted, labels)
            acc_metric.update(predicted, labels)

    f1        = f1_metric.compute().item()
    precision = precision_metric.compute().item()
    recall    = recall_metric.compute().item()
    acc       = acc_metric.compute().item() * 100.0
    for m in [f1_metric, precision_metric, recall_metric, acc_metric]:
        m.reset()
    if not use_cache:
        loader.reset()
    return {"loss": val_loss / (i + 1), "acc": acc, "f1": f1,
            "precision": precision, "recall": recall}


# ---------------------------------------------------------------------------
# Phase 1: cached head training
# ---------------------------------------------------------------------------

def run_phase1_cached(model, cache, optimizer, scheduler, criterion,
                      epoch, num_classes, device):
    """
    Phase 1 trains only projection heads + classifier on cached features.
    No backbone compute -- only lightweight MLP ops each step.
    DeepSpeed engine handles optimizer step and gradient accumulation.
    """
    model.train()
    train_acc_metric = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)

    dataset    = CachedFeatureDataset(cache)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE * 4,
        shuffle=True, num_workers=4, pin_memory=True
    )

    running_loss = 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Epoch {epoch} [Phase1-Cached]")

    for i, (feat_bio, feat_dino, feat_conv, labels) in pbar:
        feat_bio  = feat_bio.to(device,  non_blocking=True)
        feat_dino = feat_dino.to(device, non_blocking=True)
        feat_conv = feat_conv.to(device, non_blocking=True)
        labels    = labels.to(device,    non_blocking=True)
        labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            fb      = F.normalize(model.proj_bio(feat_bio),   dim=1)
            fd      = F.normalize(model.proj_dino(feat_dino), dim=1)
            fc      = F.normalize(model.proj_conv(feat_conv), dim=1)
            outputs = model.classifier(torch.cat([fb, fd, fc], dim=1))
            loss    = criterion(outputs, labels_one_hot)

        # DeepSpeed handles loss scaling, gradient accumulation, and optimizer step
        model.backward(loss)
        model.step()

        _, predicted = outputs.max(1)
        train_acc_metric.update(predicted, labels)
        running_loss += loss.item()

        if i % 10 == 0:
            acc = train_acc_metric.compute().item() * 100.0
            pbar.set_postfix({"Loss": f"{running_loss/(i+1):.4f}", "Acc": f"{acc:.2f}%"})

    train_acc = train_acc_metric.compute().item() * 100.0
    train_acc_metric.reset()
    return train_acc


# ---------------------------------------------------------------------------
# Phase 2: full fine-tuning
# ---------------------------------------------------------------------------

def run_epoch(model, train_loader, criterion, epoch, num_classes, device):
    """
    Phase 2 full fine-tuning with chunked backbone forward passes.

    Validation is handled by the caller every VAL_EVERY_N_EPOCHS epochs
    rather than every epoch, saving ~2-3 hours across a 30-epoch run.
    """
    model.train()
    train_acc_metric = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)

    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"Epoch {epoch} [Train]")
    running_loss = 0.0

    for i, data in pbar:
        images = data[0]['data'].to(memory_format=torch.channels_last)
        labels = data[0]['label'].squeeze().long()
        labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            # Chunked forward: only CHUNK_SIZE activations live in VRAM at once
            feat_bio  = chunked_backbone_forward(model.module.bioclip,  images)
            feat_dino = chunked_backbone_forward(model.module.dinov2,   images)
            feat_conv = chunked_backbone_forward(model.module.convnext, images)

            feat_bio  = F.normalize(model.module.proj_bio(feat_bio),   dim=1)
            feat_dino = F.normalize(model.module.proj_dino(feat_dino), dim=1)
            feat_conv = F.normalize(model.module.proj_conv(feat_conv), dim=1)

            outputs = model.module.classifier(
                torch.cat([feat_bio, feat_dino, feat_conv], dim=1)
            )
            loss = criterion(outputs, labels_one_hot)

        model.backward(loss)
        model.step()

        _, predicted = outputs.max(1)
        train_acc_metric.update(predicted, labels)
        running_loss += loss.item()

        if i % 10 == 0:
            acc = train_acc_metric.compute().item() * 100.0
            pbar.set_postfix({"Loss": f"{running_loss/(i+1):.4f}", "Acc": f"{acc:.2f}%"})

    train_loader.reset()

    train_acc_final = train_acc_metric.compute().item() * 100.0
    train_acc_metric.reset()

    wandb.log({"epoch": epoch, "train_acc": train_acc_final, "lr": model.get_lr()[0]})
    print(f"[Epoch {epoch}] Train Acc: {train_acc_final:.2f}%")
    return train_acc_final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train():
    """
    Progressive 2-Stage Training:

    Phase 1 — Feature Caching + Head Warmup (DeepSpeed ZeRO Stage 1)
      Backbones run once; features cached. Head trains on tensors only.

    Phase 2 — Full Fine-Tuning (DeepSpeed ZeRO Stage 1 + Chunked Forward)
      All weights active. Chunked backbone forward caps peak VRAM, allowing
      batch sizes far beyond what fits naively. ZeRO offloads optimizer states.
    """
    wandb.init(
        project="plantclef-2026",
        name=f"ensemble-optimised-{MODE.lower()}",
        config={
            "lr_phase1":             1e-3,
            "lr_phase2_backbone":    5e-6,
            "lr_phase2_head":        2e-4,
            "architecture":          "Triple-Threat(Bio+Dino+Conv)",
            "precision":             "bfloat16",
            "resolution":            RESOLUTION,
            "batch_size":            BATCH_SIZE,
            "chunk_size":            CHUNK_SIZE,
            "accumulation_steps":    ACCUMULATION_STEPS,
            "effective_batch_size":  BATCH_SIZE * ACCUMULATION_STEPS,
            "epochs_phase1":         10,
            "epochs_phase2":         30,
            "scheduler":             "OneCycleLR",
            "zero_stage":            1,
            "asl_gamma_neg":         4,
            "asl_gamma_pos":         1,
            "bioclip_backbone":      BIOCLIP_NAME,
            "dinov2_backbone":       DINOV2_NAME,
            "convnext_backbone":     CONVNEXT_NAME,
            "feature_caching":       True,
            "tf32_matmul":           True,
            "flash_attention":       True,
        }
    )
    config = wandb.config
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = CLEANED_CSV if os.path.exists(CLEANED_CSV) else RAW_CSV

    df     = cudf.read_csv(csv_path, sep=';')
    counts = df['species_id'].value_counts().sort_index().to_arrow().to_pylist()
    del df
    torch.cuda.empty_cache()

    asl       = AsymmetricLoss(gamma_neg=config.asl_gamma_neg, gamma_pos=config.asl_gamma_pos)
    criterion = LogitAdjustmentLoss(class_counts=counts, base_criterion=asl)

    os.makedirs("models", exist_ok=True)

    # -----------------------------------------------------------------------
    # PHASE 1: Feature Caching + Head Warmup
    # -----------------------------------------------------------------------
    print("\n--- PHASE 1: Feature Caching + Head Warmup ---")

    train_loader, val_loader, num_classes = get_dali_loaders(
        csv_path, IMG_DIR, batch_size=config.batch_size, sampling_mode='natural'
    )

    model = PlantEnsemble(
        num_classes=num_classes,
        input_res=config.resolution,
        bioclip_name=config.bioclip_backbone,
        dinov2_name=config.dinov2_backbone,
        convnext_name=config.convnext_backbone
    ).to(DEVICE).to(memory_format=torch.channels_last)

    model.set_grad_checkpointing(True)
    model.freeze_backbones()

    if os.path.exists(FEATURE_CACHE_PATH):
        print(f"[Feature Cache] Loading from {FEATURE_CACHE_PATH}...")
        cache = torch.load(FEATURE_CACHE_PATH)
    else:
        cache = extract_and_cache_features(model, train_loader, DEVICE, FEATURE_CACHE_PATH)

    # Phase 1: only head params -- ZeRO Stage 1 for optimizer state offload
    phase1_params = (
        list(model.proj_bio.parameters())  +
        list(model.proj_dino.parameters()) +
        list(model.proj_conv.parameters()) +
        list(model.classifier.parameters())
    )
    # DeepSpeedCPUAdam is required for ZeRO offload -- it runs the optimizer
    # step directly on CPU RAM where the states live, avoiding PCIe round-trips.
    optimizer_p1 = DeepSpeedCPUAdam(phase1_params, lr=config.lr_phase1,
                                    weight_decay=0.05)

    cache_dataset      = CachedFeatureDataset(cache)
    steps_per_epoch_p1 = len(cache_dataset) // (BATCH_SIZE * 4 * ACCUMULATION_STEPS)
    total_steps_p1     = steps_per_epoch_p1 * config.epochs_phase1

    scheduler_p1 = OneCycleLR(optimizer_p1, max_lr=config.lr_phase1,
                               total_steps=total_steps_p1, pct_start=0.3,
                               anneal_strategy='cos')

    # Wrap with DeepSpeed for ZeRO optimizer state offload
    model_engine_p1, optimizer_p1, _, scheduler_p1 = deepspeed.initialize(
        model=model,
        optimizer=optimizer_p1,
        lr_scheduler=scheduler_p1,
        config=DS_CONFIG,
    )

    best_val_acc = 0.0

    for epoch in range(config.epochs_phase1):
        train_acc = run_phase1_cached(
            model_engine_p1, cache, optimizer_p1, scheduler_p1,
            criterion, epoch, num_classes, DEVICE
        )
        metrics = validate(model_engine_p1.module, val_loader, criterion,
                           num_classes, DEVICE, use_cache=False)
        print(f"\n[Phase1 Epoch {epoch}] Train Acc: {train_acc:.2f}%  "
              f"Val Acc: {metrics['acc']:.2f}%  F1: {metrics['f1']:.4f}")
        wandb.log({"epoch": epoch, "train_acc": train_acc,
                   **{f"val_{k}": v for k, v in metrics.items()}})

    # -----------------------------------------------------------------------
    # PHASE 2: Full Fine-Tuning with Differential LR + Chunked Forward
    # -----------------------------------------------------------------------
    print("\n--- PHASE 2: Full Fine-Tuning (Long-Tail Calibration) ---")

    train_loader, val_loader, _ = get_dali_loaders(
        csv_path, IMG_DIR, batch_size=config.batch_size, sampling_mode='sqrt'
    )

    # Unwrap from Phase 1 DeepSpeed engine before re-wrapping for Phase 2
    raw_model = model_engine_p1.module
    raw_model.unfreeze_backbones()

    # Differential LR: backbones at 5e-6, heads at 2e-4
    optimizer_p2 = DeepSpeedCPUAdam([
        {'params': raw_model.bioclip.parameters(),    'lr': config.lr_phase2_backbone},
        {'params': raw_model.dinov2.parameters(),     'lr': config.lr_phase2_backbone},
        {'params': raw_model.convnext.parameters(),   'lr': config.lr_phase2_backbone},
        {'params': raw_model.proj_bio.parameters(),   'lr': config.lr_phase2_head},
        {'params': raw_model.proj_dino.parameters(),  'lr': config.lr_phase2_head},
        {'params': raw_model.proj_conv.parameters(),  'lr': config.lr_phase2_head},
        {'params': raw_model.classifier.parameters(), 'lr': config.lr_phase2_head},
    ], weight_decay=0.01)

    steps_per_epoch_p2 = len(train_loader)
    total_steps_p2     = steps_per_epoch_p2 * config.epochs_phase2

    scheduler_p2 = OneCycleLR(
        optimizer_p2,
        max_lr=[config.lr_phase2_backbone] * 3 + [config.lr_phase2_head] * 4,
        total_steps=total_steps_p2,
        pct_start=0.1,
        anneal_strategy='cos',
    )

    model_engine_p2, optimizer_p2, _, scheduler_p2 = deepspeed.initialize(
        model=raw_model,
        optimizer=optimizer_p2,
        lr_scheduler=scheduler_p2,
        config=DS_CONFIG,
    )

    # Validate every N epochs rather than every epoch.
    # Each val pass runs all 1.4M images through the full model -- skipping
    # 2 out of every 3 saves ~2-3 hours across a 30-epoch run.
    VAL_EVERY_N_EPOCHS = 3

    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    start_epoch = config.epochs_phase1
    final_epoch = start_epoch + config.epochs_phase2 - 1

    for epoch in range(start_epoch, start_epoch + config.epochs_phase2):
        run_epoch(
            model_engine_p2, train_loader,
            criterion, epoch, num_classes, DEVICE
        )

        # Validate on scheduled epochs and always on the final epoch
        if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == final_epoch:
            metrics = validate(
                model_engine_p2.module, val_loader, criterion, num_classes, DEVICE
            )
            val_acc = metrics['acc']

            print(f"\n[Epoch {epoch}] Validation Results:")
            print(f" - Validation Loss:      {metrics['loss']:.4f}")
            print(f" - Validation Accuracy:  {metrics['acc']:.2f}%")
            print(f" - Macro-F1 Score:       {metrics['f1']:.4f}")
            print(f" - Macro-Precision:      {metrics['precision']:.4f}")
            print(f" - Macro-Recall:         {metrics['recall']:.4f}")

            wandb.log({
                "epoch":         epoch,
                "val_acc":       metrics['acc'],
                "val_loss":      metrics['loss'],
                "val_f1":        metrics['f1'],
                "val_precision": metrics['precision'],
                "val_recall":    metrics['recall'],
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_engine_p2.save_checkpoint("models/", tag="best_calibrated")
                print(f"--> Saved New Best Ensemble (Acc: {val_acc:.2f}%)")

            early_stopping(val_acc)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}. "
                      f"Best val acc: {best_val_acc:.2f}%")
                break

    model_engine_p2.save_checkpoint("models/", tag="final")
    wandb.finish()


if __name__ == "__main__":
    train()
