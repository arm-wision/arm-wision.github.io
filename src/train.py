import os
import sys
# Ensure src/ is on the path so local packages (training/, models/, data/)
# take precedence over any same-named packages installed globally
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256"
)

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
import wandb
import cudf
from dotenv import load_dotenv
import gc

load_dotenv()

from models.ensemble import PlantEnsemble
from data.dataloader import get_dali_loaders
from training import (
    LogitAdjustmentLoss, AsymmetricLoss, EarlyStopping,
    extract_and_cache_features, CachedFeatureDataset,
    validate, run_phase1_cached, run_epoch,
    load_phase1_checkpoint, save_phase1_checkpoint,
    load_phase1_heads_for_phase2,
    load_phase2_checkpoint, save_epoch_checkpoint, save_deepspeed_checkpoint,
    phase1_is_complete, fit_and_save, load_pca, apply_pca,
)
from config import (
    RAW_CSV, IMG_DIR, CLEANED_CSV, BATCH_SIZE, P2_BATCH_SIZE, P2_CHUNK_SIZE,
    RESOLUTION, MODE, BIOCLIP_NAME, DINOV2_NAME, CONVNEXT_NAME,
    ACCUMULATION_STEPS, EPOCHS_PHASE1, EPOCHS_PHASE2, P2_SAMPLES_PER_EPOCH,
    VAL_EVERY_N_EPOCHS, PATIENCE,
    LORA_R, LORA_ALPHA, LORA_DROPOUT,
    FEATURE_CACHE_PATH, P1_CKPT_PATH, P2_CKPT_DIR, P2_EPOCH_CKPT,
    PCA_TRANSFORM_PATH,
)

# ---------------------------------------------------------------------------
# Hardware flags
# ---------------------------------------------------------------------------
torch.backends.cudnn.benchmark           = True
torch.backends.cuda.matmul.allow_tf32    = True
torch.backends.cudnn.allow_tf32          = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# ---------------------------------------------------------------------------
# DeepSpeed configs
# ---------------------------------------------------------------------------
# Blackwell Optimisation: Disable all CPU offloading to leverage 32GB VRAM
DS_CONFIG_P1 = {
    "zero_optimization": {
        "stage": 1,
        # Offload disabled for 5090 Blackwell efficiency
    },
    "bf16": {"enabled": True},
    "gradient_accumulation_steps": ACCUMULATION_STEPS,
    "train_micro_batch_size_per_gpu": BATCH_SIZE,
    "steps_per_print": 50,
    "wall_clock_breakdown": False,
    "distributed_backend": "nccl",
}

# Phase 2: no CPU offload -- LoRA makes optimizer states tiny, keep on GPU
DS_CONFIG_P2 = {
    "zero_optimization": {"stage": 1},
    "bf16": {"enabled": True},
    "gradient_accumulation_steps": ACCUMULATION_STEPS,
    "train_micro_batch_size_per_gpu": P2_BATCH_SIZE,
    "steps_per_print": 50,
    "wall_clock_breakdown": False,
    "distributed_backend": "nccl",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train():
    """
    Progressive 2-Stage Training:

    Phase 1 — Feature Caching + Head Warmup
      Auto-skipped if a complete Phase 1 checkpoint already exists.
      Backbones frozen; features extracted once (~1-1.5 hrs at 384px).
      Only projection heads + classifier train (~30-40 min for 10 epochs).

    Phase 2 — LoRA Fine-Tuning
      LoRA adapters on DINOv2 + ConvNeXt (~5M trainable params vs ~800M).
      BioCLIP fully frozen throughout.
      batch=256, 3 epochs -- target < 24 hours total.
    """
    wandb.init(
        project="plantclef-2026",
        name=f"ensemble-lora-{MODE.lower()}",
        config={
            "resolution":          RESOLUTION,
            "batch_size":          BATCH_SIZE,
            "p2_batch_size":       P2_BATCH_SIZE,
            "epochs_phase1":       EPOCHS_PHASE1,
            "epochs_phase2":       EPOCHS_PHASE2,
            "lora_r":              LORA_R,
            "lora_alpha":          LORA_ALPHA,
            "lora_dropout":        LORA_DROPOUT,
            "lr_phase1":           1e-3,
            "lr_phase2_lora":      1e-4,
            "lr_phase2_head":      2e-4,
            "patience":            PATIENCE,
            "bioclip_backbone":    BIOCLIP_NAME,
            "dinov2_backbone":     DINOV2_NAME,
            "convnext_backbone":   CONVNEXT_NAME,
            "phase2_strategy":     "LoRA(DINOv2+ConvNeXt)+FrozenBioCLIP",
        }
    )
    config = wandb.config
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = CLEANED_CSV if os.path.exists(CLEANED_CSV) else RAW_CSV
    os.makedirs("models", exist_ok=True)

    # Load class counts for Logit Adjustment then free GPU RAM immediately
    df     = cudf.read_csv(csv_path, sep=';')
    counts = df['species_id'].value_counts().sort_index().to_arrow().to_pylist()
    del df
    torch.cuda.empty_cache()

    # Logit Adjustment for 7,800-class long-tail
    num_classes = len(counts)
    criterion   = LogitAdjustmentLoss(class_counts=counts)

    # -----------------------------------------------------------------------
    # PHASE 1: Feature Caching + Head Warmup
    # Auto-skipped if Phase 1 already completed (checkpoint covers all epochs)
    # -----------------------------------------------------------------------
    skip_phase1 = phase1_is_complete()
    
    # Load model and apply Blackwell-native compilation
    model = PlantEnsemble(
        num_classes=num_classes, 
        input_res=RESOLUTION,
        bioclip_name=BIOCLIP_NAME,
        dinov2_name=DINOV2_NAME,
        convnext_name=CONVNEXT_NAME,
    ).to(DEVICE).to(memory_format=torch.channels_last)
    
    # Blackwell Optimisation: max-autotune uses Triton for specialized Blackwell kernels
    if getattr(config, 'USE_COMPILE', False):
        print("[Blackwell] Applying torch.compile(model, mode='max-autotune')...")
        model = torch.compile(model, mode="max-autotune")

    if skip_phase1:
        print(f"\n[Phase1] Complete checkpoint found -- skipping Phase 1.")
        # Need to initialize loaders to get total samples etc, though we skip P1
        _t_loader, _v_loader, _ = get_dali_loaders(csv_path, IMG_DIR, batch_size=BATCH_SIZE)
        model.set_grad_checkpointing(True)
        model.freeze_backbones()
        del _t_loader, _v_loader
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("\n--- PHASE 1: Feature Caching + Head Warmup ---")
        train_loader, val_loader, _ = get_dali_loaders(
            csv_path, IMG_DIR, batch_size=BATCH_SIZE,
            resolution=RESOLUTION, sampling_mode='natural'
        )
        
        model.set_grad_checkpointing(True)
        model.freeze_backbones()

        # Load or extract feature cache
        if os.path.exists(FEATURE_CACHE_PATH):
            print(f"[Feature Cache] Loading from {FEATURE_CACHE_PATH}...")
            from tqdm import tqdm
            with tqdm(total=1, desc="Loading feature cache", unit="file") as pbar:
                cache = torch.load(FEATURE_CACHE_PATH, weights_only=False)
                pbar.update(1)
            
            # Blackwell: Load entire 22.5GB cache to GPU to eliminate PCIe bottlenecks
            # The 5090 has 32GB VRAM, plenty for the 22.5GB cache + model weights.
            if getattr(config, 'LOAD_CACHE_TO_GPU', False):
                print("[Blackwell] Moving feature cache to GPU memory...")
                for k in cache:
                    if torch.is_tensor(cache[k]):
                        cache[k] = cache[k].to(DEVICE, non_blocking=True)
        else:
            cache = extract_and_cache_features(model, train_loader, DEVICE, FEATURE_CACHE_PATH)

        # Fit PCA on cached features (or load existing transform)
        if 'features_pca' not in cache:
            if os.path.exists(PCA_TRANSFORM_PATH):
                print("[PCA] Loading existing transform...")
                pca   = load_pca(PCA_TRANSFORM_PATH)
                cache = apply_pca(cache, pca)
            else:
                print("[PCA] Fitting PCA on cached features...")
                cache, _ = fit_and_save(cache)
            torch.save(cache, FEATURE_CACHE_PATH)
            print("[PCA] Cache updated with PCA features.")

        # Phase 1 trains only the phase1_head (linear probe on PCA features).
        # Much faster than training proj heads on raw features (~3x smaller input).
        # Phase 2 still uses full proj heads + ensemble (unaffected by this).
        phase1_params  = list(model.phase1_head.parameters())
        optimizer_p1   = optim.AdamW(phase1_params, lr=config.lr_phase1,
                                     weight_decay=0.05, fused=True)

        from training import CachedPCADataset
        cache_dataset      = CachedPCADataset(cache) if 'features_pca' in cache else CachedFeatureDataset(cache)
        steps_per_epoch_p1 = len(cache_dataset) // (BATCH_SIZE * 4 * ACCUMULATION_STEPS)
        total_steps_p1     = (steps_per_epoch_p1 + 5) * EPOCHS_PHASE1

        scheduler_p1 = OneCycleLR(optimizer_p1, max_lr=config.lr_phase1,
                                   total_steps=total_steps_p1, pct_start=0.3,
                                   anneal_strategy='cos')

        model_engine_p1, optimizer_p1, _, scheduler_p1 = deepspeed.initialize(
            model=model, optimizer=optimizer_p1,
            lr_scheduler=scheduler_p1, config=DS_CONFIG_P1,
        )

        best_val_acc   = 0.0
        start_epoch_p1, best_val_acc = load_phase1_checkpoint(model_engine_p1, DEVICE)

        for epoch in range(start_epoch_p1, EPOCHS_PHASE1):
            train_acc = run_phase1_cached(
                model_engine_p1, cache, criterion, epoch, num_classes, DEVICE
            )
            metrics = validate(model_engine_p1.module, val_loader, criterion,
                               num_classes, DEVICE)
            print(f"\n[Phase1 Epoch {epoch}] Train: {train_acc:.2f}%  "
                  f"Val: {metrics['acc']:.2f}%  F1: {metrics['f1']:.4f}")
            wandb.log({"epoch": epoch, "train_acc": train_acc,
                       **{f"val_{k}": v for k, v in metrics.items()}})

            if metrics['acc'] > best_val_acc:
                best_val_acc = metrics['acc']
            save_phase1_checkpoint(model_engine_p1, epoch, best_val_acc)

        # Free Phase 1 engine + data before Phase 2
        model = model_engine_p1.module
        del model_engine_p1, optimizer_p1, scheduler_p1
        del cache_dataset, cache, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # PHASE 2: LoRA Fine-Tuning
    # -----------------------------------------------------------------------
    print("\n--- PHASE 2: LoRA Fine-Tuning (Long-Tail Calibration) ---")

    train_loader, val_loader, _ = get_dali_loaders(
        csv_path, IMG_DIR, batch_size=P2_BATCH_SIZE,
        resolution=RESOLUTION, sampling_mode='natural',
        samples_per_epoch=P2_SAMPLES_PER_EPOCH
    )

    # Load best Phase 1 head weights (if any match the current structure)
    load_phase1_heads_for_phase2(model, DEVICE)

    # -----------------------------------------------------------------------
    # PHASE 2 - Part A: Head Warmup (Frozen Backbones)
    # -----------------------------------------------------------------------
    # Resume check BEFORE warmup
    resumed_epoch, _, resumed_step = load_phase2_checkpoint(model, DEVICE, tag="checkpoint_latest")
    
    if resumed_epoch is None or resumed_epoch < EPOCHS_PHASE1:
        print("\n--- PHASE 2A: Head Warmup (Frozen Backbones) ---")
        model.freeze_backbones()
        
        warmup_optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=2e-4, weight_decay=0.01
        )
        
        # One warmup epoch (Epoch -1)
        start_step = resumed_step if (resumed_epoch == -1 or resumed_epoch == EPOCHS_PHASE1 - 1) else 0
        run_epoch(model, train_loader, criterion, -1, num_classes, DEVICE, 
                  optimizer=warmup_optimizer, start_step=start_step, 
                  p2_chunk_size=P2_BATCH_SIZE)
        print("Warmup complete. Classifier is now initialized.")
    else:
        print(f"\n[Phase2A] Skipping warmup (Already at epoch {resumed_epoch})")

    # -----------------------------------------------------------------------
    # PHASE 2 - Part B: LoRA Fine-Tuning
    # -----------------------------------------------------------------------
    print("\n--- PHASE 2B: LoRA Fine-Tuning (Long-Tail Calibration) ---")
    
    # Apply LoRA to DINOv2 + ConvNeXt; BioCLIP stays frozen
    model.apply_lora(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)

    # Separate optimizer param groups: lower LR for LoRA, higher for heads
    lora_params = [p for name, p in model.named_parameters()
                   if 'lora_' in name and p.requires_grad]
    head_params = [p for name, p in model.named_parameters()
                   if 'lora_' not in name and p.requires_grad]

    # Freeze BioCLIP explicitly (LoRA was not applied to it)
    for param in model.bioclip.parameters():
        param.requires_grad = False

    # Phase 2 optimizer
    optimizer_p2 = optim.AdamW([
        {'params': lora_params, 'lr': config.lr_phase2_lora},
        {'params': head_params, 'lr': config.lr_phase2_head},
    ], weight_decay=0.01, fused=True)

    # Wrap with Lookahead to find flatter minima
    from timm.optim import Lookahead
    optimizer_p2 = Lookahead(optimizer_p2, alpha=0.5, k=6)

    steps_per_epoch_p2 = len(train_loader)
    total_steps_p2     = (steps_per_epoch_p2 + 5) * EPOCHS_PHASE2

    scheduler_p2 = OneCycleLR(
        optimizer_p2,
        max_lr=[config.lr_phase2_lora, config.lr_phase2_head],
        total_steps=total_steps_p2,
        pct_start=0.1,
        anneal_strategy='cos',
    )

    model_engine_p2, optimizer_p2, _, scheduler_p2 = deepspeed.initialize(
        model=model, optimizer=optimizer_p2,
        lr_scheduler=scheduler_p2, config=DS_CONFIG_P2,
    )

    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.001)
    best_val_acc   = 0.0
    start_epoch    = EPOCHS_PHASE1
    final_epoch    = EPOCHS_PHASE1 + EPOCHS_PHASE2 - 1

    # Resume from checkpoint if one exists
    resumed_epoch, resumed_acc, resumed_step = load_phase2_checkpoint(model_engine_p2, DEVICE, tag="checkpoint_latest")
    if resumed_epoch is not None:
        # If the checkpoint is from the warmup (e.g. -1), we start at the 
        # actual Phase 2 start epoch (EPOCHS_PHASE1). 
        # If it's a real Phase 2 epoch (e.g. 10+), we resume there.
        start_epoch  = max(EPOCHS_PHASE1, resumed_epoch)
        best_val_acc = resumed_acc

    for epoch in range(start_epoch, EPOCHS_PHASE1 + EPOCHS_PHASE2):
        # Only use resumed_step for the very first epoch we resume into
        current_start_step = resumed_step if epoch == resumed_epoch else 0
        run_epoch(model_engine_p2, train_loader, criterion, epoch, num_classes, DEVICE, 
                  start_step=current_start_step)

        # Lightweight per-epoch checkpoint every epoch
        save_epoch_checkpoint(model_engine_p2, epoch, best_val_acc)

        if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == final_epoch:
            metrics = validate(
                model_engine_p2.module, val_loader, criterion, num_classes, DEVICE
            )
            val_acc = metrics['acc']

            print(f"\n[Epoch {epoch}] Val Loss: {metrics['loss']:.4f}  "
                  f"Acc: {metrics['acc']:.2f}%  F1: {metrics['f1']:.4f}  "
                  f"Prec: {metrics['precision']:.4f}  Rec: {metrics['recall']:.4f}")

            wandb.log({"epoch": epoch, **{f"val_{k}": v for k, v in metrics.items()}})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_deepspeed_checkpoint(
                    model_engine_p2, "models/", "best_calibrated", epoch, best_val_acc
                )
                print(f"--> New Best (Acc: {val_acc:.2f}%)")

            save_deepspeed_checkpoint(
                model_engine_p2, P2_CKPT_DIR, "checkpoint_latest", epoch, best_val_acc
            )

            early_stopping(val_acc)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}. Best: {best_val_acc:.2f}%")
                break

    save_deepspeed_checkpoint(
        model_engine_p2, "models/", "final", epoch, best_val_acc
    )
    wandb.finish()


if __name__ == "__main__":
    train()
