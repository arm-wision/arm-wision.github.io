import os
import torch
import config as _cfg
P1_CKPT_PATH  = getattr(_cfg, "P1_CKPT_PATH",  "models/phase1_checkpoint.pth")
P2_CKPT_DIR   = getattr(_cfg, "P2_CKPT_DIR",   "models/phase2_checkpoint")
P2_EPOCH_CKPT = getattr(_cfg, "P2_EPOCH_CKPT", "models/phase2_epoch_checkpoint.pth")
EPOCHS_PHASE1 = getattr(_cfg, "EPOCHS_PHASE1", 10)


def phase1_is_complete():
    """
    Returns True if a Phase 1 checkpoint exists covering all epochs.
    Used to auto-skip Phase 1 on restarts when the head is already trained.
    """
    if not os.path.exists(P1_CKPT_PATH):
        return False
    try:
        meta = torch.load(P1_CKPT_PATH, map_location='cpu', weights_only=False)
        return meta.get('epoch', -1) >= EPOCHS_PHASE1 - 1
    except Exception:
        return False


def load_phase1_checkpoint(model_engine, device):
    """
    Load Phase 1 checkpoint into a DeepSpeed engine.
    Strips _orig_mod. prefix if checkpoint was saved while torch.compile was active.
    Returns (start_epoch, best_val_acc).
    """
    if not os.path.exists(P1_CKPT_PATH):
        return 0, 0.0

    ckpt        = torch.load(P1_CKPT_PATH, map_location=device, weights_only=False)
    saved_state = ckpt['model_state']
    if any('_orig_mod.' in k for k in saved_state.keys()):
        saved_state = {k.replace('._orig_mod.', '.'): v for k, v in saved_state.items()}
    model_engine.module.load_state_dict(saved_state)
    start_epoch  = ckpt['epoch'] + 1
    best_val_acc = ckpt.get('best_val_acc', 0.0)
    print(f"[Phase1] Resumed from epoch {start_epoch} "
          f"(best val acc: {best_val_acc:.2f}%)")
    return start_epoch, best_val_acc


def save_phase1_checkpoint(model_engine, epoch, best_val_acc):
    """Save lightweight Phase 1 checkpoint (model weights + metadata)."""
    raw_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
    torch.save({
        'epoch':        epoch,
        'model_state':  raw_model.state_dict(),
        'best_val_acc': best_val_acc,
    }, P1_CKPT_PATH)
    print(f"[Phase1] Checkpoint saved (epoch {epoch})")


def load_phase1_heads_for_phase2(raw_model, device):
    """
    Load only the projection head + classifier weights from the Phase 1 checkpoint
    into the Phase 2 model.  Backbone keys are intentionally skipped because Phase 2
    applies LoRA which changes the backbone key structure.
    Returns best_val_acc from Phase 1 for logging.
    """
    if not os.path.exists(P1_CKPT_PATH):
        print("[Phase2] No Phase 1 checkpoint found -- starting from scratch.")
        return 0.0

    ckpt        = torch.load(P1_CKPT_PATH, map_location=device, weights_only=False)
    saved_state = ckpt['model_state']

    # Only load projection heads and classifier -- skip backbones entirely
    # (LoRA changes backbone key structure; backbone weights come from pretrained init)
    head_prefixes = ('proj_bio.', 'proj_dino.', 'proj_conv.', 'classifier.')
    head_state    = {k: v for k, v in saved_state.items()
                     if any(k.startswith(p) for p in head_prefixes)}

    raw_model.load_state_dict(head_state, strict=False)
    best_val_acc = ckpt.get('best_val_acc', 0.0)
    print(f"[Phase2] Loaded Phase 1 head weights "
          f"(epoch {ckpt['epoch']}, val acc: {best_val_acc:.2f}%)")
    return best_val_acc


def load_phase2_checkpoint(model_engine, device, tag=None):
    """
    Load Phase 2 checkpoint.  Tries full DeepSpeed checkpoint first (optimizer +
    scheduler states intact), then falls back to lightweight epoch checkpoint.
    Returns (start_epoch, best_val_acc).
    """
    if os.path.exists(P2_CKPT_DIR):
        _, client_state = model_engine.load_checkpoint(P2_CKPT_DIR, tag=tag)
        if client_state is not None:
            start_epoch  = client_state['epoch'] + 1
            best_val_acc = client_state.get('best_val_acc', 0.0)
            print(f"[Phase2] Resumed from DeepSpeed checkpoint epoch {start_epoch} "
                  f"(best val acc: {best_val_acc:.2f}%)")
            return start_epoch, best_val_acc

    if os.path.exists(P2_EPOCH_CKPT):
        ckpt = torch.load(P2_EPOCH_CKPT, map_location=device, weights_only=False)
        model_engine.module.load_state_dict(ckpt['model_state'])
        start_epoch  = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        print(f"[Phase2] Resumed from epoch checkpoint {start_epoch} "
              f"(optimizer reset, best val acc: {best_val_acc:.2f}%)")
        return start_epoch, best_val_acc

    # Fallback to mid-epoch progress
    progress_path = "models/phase2_progress.pth"
    if os.path.exists(progress_path):
        ckpt = torch.load(progress_path, map_location=device, weights_only=False)
        raw_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
        raw_model.load_state_dict(ckpt['model_state'])
        print(f"[Phase2] Found mid-epoch progress (Epoch {ckpt['epoch']}, Step {ckpt['step']})")
        return ckpt['epoch'], 0.0

    return None, 0.0  # None signals no checkpoint found


def save_epoch_checkpoint(model_engine, epoch, best_val_acc):
    """Lightweight per-epoch checkpoint -- model weights only, saves every epoch."""
    raw_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
    torch.save({
        'epoch':        epoch,
        'model_state':  raw_model.state_dict(),
        'best_val_acc': best_val_acc,
    }, P2_EPOCH_CKPT)


def save_progress_checkpoint(model_engine, epoch, step, total_steps):
    """
    Lightweight mid-epoch progress checkpoint. Saves to a fixed path,
    overwriting previous progress within the same epoch.
    """
    path = "models/phase2_progress.pth"
    raw_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
    torch.save({
        'epoch':        epoch,
        'step':         step,
        'total_steps':  total_steps,
        'model_state':  raw_model.state_dict(),
    }, path)
    print(f"\n[Progress] 5% checkpoint saved (Epoch {epoch}, Step {step}/{total_steps})")


def save_deepspeed_checkpoint(model_engine, directory, tag, epoch, best_val_acc):
    """Full DeepSpeed checkpoint -- model + optimizer + scheduler states."""
    model_engine.save_checkpoint(
        directory, tag=tag,
        client_state={'epoch': epoch, 'best_val_acc': best_val_acc}
    )
