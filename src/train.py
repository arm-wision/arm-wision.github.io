import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassAccuracy
import wandb
import os
import cudf
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file (e.g., HF_TOKEN)
load_dotenv()

from models.ensemble import PlantEnsemble
from data.dataloader import get_dali_loaders
from config import RAW_CSV, IMG_DIR, CLEANED_CSV, BATCH_SIZE, RESOLUTION, MODE, BIOCLIP_NAME, DINOV2_NAME, CONVNEXT_NAME

# Enable CuDNN benchmark for speed
torch.backends.cudnn.benchmark = True

# Gradient accumulation steps -- effective batch = BATCH_SIZE * ACCUMULATION_STEPS
ACCUMULATION_STEPS = 4


class LogitAdjustmentLoss(nn.Module):
    """
    Implements Logit Adjustment (LA) to mitigate class prior bias.

    In long-tailed datasets, models naturally bias towards frequent classes. LA
    shifts the logits by a factor of log(prior) during training, effectively
    demanding a larger margin for common classes and easing the requirement
    for rare ones.
    """
    def __init__(self, class_counts, tau=1.0, base_criterion=None):
        super(LogitAdjustmentLoss, self).__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        priors = counts / counts.sum()
        self.adjustment = (tau * torch.log(priors + 1e-12)).to('cuda')
        self.criterion = base_criterion if base_criterion else nn.BCEWithLogitsLoss()

    def forward(self, x, y):
        return self.criterion(x + self.adjustment, y)


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-Label / Imbalanced Classification.

    ASL aggressively down-weights easy negatives, which dominate in
    large-scale classification (7,800+ classes). It uses different
    focusing parameters (gamma_neg, gamma_pos) to balance the positive
    and negative signals.
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos

        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        with torch.no_grad():
            pt = (xs_pos * y + xs_neg * (1 - y))
            weights = (1 - pt).pow(self.gamma_pos * y + self.gamma_neg * (1 - y))

        loss *= weights
        return -loss.mean()


class EarlyStopping:
    """
    Monitors validation accuracy and stops training if no improvement is seen.
    """
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
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


def validate(model, loader, criterion, num_classes, device):
    """
    Comprehensive validation loop.
    Computes Loss, Top-1 Accuracy, Macro-F1, Precision, and Recall.

    All metrics are accumulated on GPU via torchmetrics to avoid per-batch
    CPU transfers. A single .cpu() call happens at the end for val_loss only.
    """
    model.eval()
    val_loss = 0.0

    # torchmetrics keeps all state on GPU -- no CPU transfers until .compute()
    f1_metric        = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    recall_metric    = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
    acc_metric       = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)

    with torch.no_grad():
        for i, data in enumerate(loader):
            images = data[0]['data']
            labels = data[0]['label'].squeeze().long()
            labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()

            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels_one_hot)

            val_loss += loss.item()
            _, predicted = outputs.max(1)

            # Update metrics on GPU -- no .cpu() / .numpy() calls here
            f1_metric.update(predicted, labels)
            precision_metric.update(predicted, labels)
            recall_metric.update(predicted, labels)
            acc_metric.update(predicted, labels)

    # Single compute() call at end -- one GPU->CPU transfer total
    f1        = f1_metric.compute().item()
    precision = precision_metric.compute().item()
    recall    = recall_metric.compute().item()
    acc       = acc_metric.compute().item() * 100.0

    f1_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    acc_metric.reset()

    loader.reset()
    return {
        "loss":      val_loss / (i + 1),
        "acc":       acc,
        "f1":        f1,
        "precision": precision,
        "recall":    recall
    }


def run_epoch(model, train_loader, val_loader, optimizer, criterion, scaler,
              epoch, num_classes, device, phase=1):
    """
    Executes a single training epoch with gradient accumulation.

    Metrics are accumulated on GPU via torchmetrics and only transferred
    to CPU once per epoch.
    """
    model.train()

    # torchmetrics accumulator for train accuracy -- stays on GPU
    train_acc_metric = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} [Train]")
    optimizer.zero_grad()
    running_loss = 0.0

    for i, data in pbar:
        images = data[0]['data']
        labels = data[0]['label'].squeeze().long()

        # labels arrives from DALI on GPU; one_hot stays on GPU too
        labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()

        with autocast(device_type='cuda'):
            # ensemble.forward() applies no_grad internally for frozen backbones --
            # never wrap the full forward here or classifier gradients will be lost.
            outputs = model(images)

            # Scale loss by accumulation steps so gradients average correctly
            loss = criterion(outputs, labels_one_hot) / ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        # Only step the optimiser every ACCUMULATION_STEPS batches
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        _, predicted = outputs.max(1)
        train_acc_metric.update(predicted, labels)
        running_loss += loss.item() * ACCUMULATION_STEPS

        # .item() syncs GPU->CPU but only every 10 steps -- acceptable overhead
        if i % 10 == 0:
            acc = train_acc_metric.compute().item() * 100.0
            pbar.set_postfix({
                "Loss": f"{running_loss / (i + 1):.4f}",
                "Acc":  f"{acc:.2f}%"
            })

    # Handle any remaining gradients in the last partial accumulation window
    if (i + 1) % ACCUMULATION_STEPS != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    # Final train accuracy -- single GPU->CPU transfer
    train_acc = train_acc_metric.compute().item() * 100.0
    train_acc_metric.reset()
    train_loader.reset()

    metrics = validate(model, val_loader, criterion, num_classes, device)

    print(f"\n[Epoch {epoch}] Final Results:")
    print(f" - Train Accuracy:       {train_acc:.2f}%")
    print(f" - Validation Loss:      {metrics['loss']:.4f}")
    print(f" - Validation Accuracy:  {metrics['acc']:.2f}%")
    print(f" - Macro-F1 Score:       {metrics['f1']:.4f}")
    print(f" - Macro-Precision:      {metrics['precision']:.4f}")
    print(f" - Macro-Recall:         {metrics['recall']:.4f}")

    wandb.log({
        "epoch":        epoch,
        "train_acc":    train_acc,
        "val_acc":      metrics['acc'],
        "val_loss":     metrics['loss'],
        "val_f1":       metrics['f1'],
        "val_precision": metrics['precision'],
        "val_recall":   metrics['recall']
    })
    return metrics['acc']


def train():
    """
    Main training orchestrator. Implements Progressive 2-Stage Training:

    Phase 1: Representation Learning (Frozen Backbones)
    - Goal: Warm up the fusion head without disrupting pre-trained features.
    - Data: Natural distribution.

    Phase 2: Calibration (Full Fine-Tuning)
    - Goal: Recalibrate the model for long-tail performance.
    - Data: Square-root resampling (boosts rare species).
    - Params: All weights unfrozen, lower learning rate.
    """
    wandb.init(
        project="plantclef-2026",
        name=f"ensemble-progressive-balancing-{MODE.lower()}",
        config={
            "lr_phase1":            1e-4,
            "lr_phase2":            2e-5,
            "architecture":         "Triple-Threat(Bio+Dino+Conv)",
            "resolution":           RESOLUTION,
            "batch_size":           BATCH_SIZE,
            "accumulation_steps":   ACCUMULATION_STEPS,
            "effective_batch_size": BATCH_SIZE * ACCUMULATION_STEPS,
            "epochs_phase1":        10,
            "epochs_phase2":        30,
            "asl_gamma_neg":        4,
            "asl_gamma_pos":        1,
            "bioclip_backbone":     BIOCLIP_NAME,
            "dinov2_backbone":      DINOV2_NAME,
            "convnext_backbone":    CONVNEXT_NAME,
        }
    )
    config = wandb.config
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = CLEANED_CSV if os.path.exists(CLEANED_CSV) else RAW_CSV

    # Load class counts for Logit Adjustment then immediately free GPU memory
    df = cudf.read_csv(csv_path, sep=';')
    counts = df['species_id'].value_counts().sort_index().to_arrow().to_pylist()
    del df
    torch.cuda.empty_cache()

    # --- PHASE 1: Representation Learning (Backbones Frozen) ---
    print("\n--- PHASE 1: Representation Learning ---")
    train_loader, val_loader, num_classes = get_dali_loaders(
        csv_path, IMG_DIR, batch_size=config.batch_size, sampling_mode='natural'
    )

    model = PlantEnsemble(
        num_classes=num_classes,
        input_res=config.resolution,
        bioclip_name=config.bioclip_backbone,
        dinov2_name=config.dinov2_backbone,
        convnext_name=config.convnext_backbone
    ).to(DEVICE)

    # Trade compute for VRAM on all backbones
    model.set_grad_checkpointing(True)
    model.freeze_backbones()

    asl       = AsymmetricLoss(gamma_neg=config.asl_gamma_neg, gamma_pos=config.asl_gamma_pos)
    criterion = LogitAdjustmentLoss(class_counts=counts, base_criterion=asl)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr_phase1, weight_decay=0.05)
    scaler    = GradScaler('cuda')

    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(config.epochs_phase1):
        run_epoch(model, train_loader, val_loader, optimizer, criterion,
                  scaler, epoch, num_classes, DEVICE, phase=1)

    # --- PHASE 2: Calibration (Full Fine-Tuning + Sqrt Resampling) ---
    print("\n--- PHASE 2: Calibration (Long-Tail Recovery) ---")
    train_loader, val_loader, _ = get_dali_loaders(
        csv_path, IMG_DIR, batch_size=config.batch_size, sampling_mode='sqrt'
    )

    model.unfreeze_backbones()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr_phase2, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs_phase2)

    for epoch in range(config.epochs_phase1, config.epochs_phase1 + config.epochs_phase2):
        val_acc = run_epoch(model, train_loader, val_loader, optimizer, criterion,
                            scaler, epoch, num_classes, DEVICE, phase=2)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/ensemble_best_calibrated.pth")
            print(f"--> Saved New Best Ensemble (Acc: {val_acc:.2f}%)")

        scheduler.step()

    torch.save(model.state_dict(), "models/ensemble_final.pth")
    wandb.finish()


if __name__ == "__main__":
    train()
