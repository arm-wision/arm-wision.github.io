import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import os
import cudf
from dotenv import load_dotenv

# Load environment variables from .env file (e.g., HF_TOKEN)
load_dotenv()

from models.ensemble import PlantEnsemble
from data.dataloader import get_dali_loaders 
from config import RAW_CSV, IMG_DIR, CLEANED_CSV, BATCH_SIZE, RESOLUTION, MODE, BIOCLIP_NAME, DINOV2_NAME, CONVNEXT_NAME

# Enable CuDNN benchmark for speed
torch.backends.cudnn.benchmark = True

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
        # Calculate class priors [n_i / total]
        counts = torch.tensor(class_counts, dtype=torch.float32)
        priors = counts / counts.sum()
        # Pre-calculate the log-prior adjustment term
        self.adjustment = (tau * torch.log(priors + 1e-12)).to('cuda')
        self.criterion = base_criterion if base_criterion else nn.BCEWithLogitsLoss()

    def forward(self, x, y):
        # Apply the additive shift to the logits
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

        # Asymmetric clipping of negatives to ignore 'easy' samples
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Calculate weighting factors based on confidence
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

from sklearn.metrics import f1_score, precision_score, recall_score

def validate(model, loader, criterion, num_classes, device):
    """
    Comprehensive validation loop.
    Computes Loss, Top-1 Accuracy, Macro-F1, Precision, and Recall.
    """
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
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
            
            # Collect for sklearn metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate comprehensive metrics
    # average='macro' is critical for long-tailed datasets as it treats all classes equally
    acc = 100. * (np.array(all_preds) == np.array(all_labels)).mean()
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    loader.reset() 
    return {
        "loss": val_loss / (i + 1),
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

from tqdm import tqdm

def run_epoch(model, train_loader, val_loader, optimizer, criterion, scaler, epoch, num_classes, device):
    """
    Executes a single training epoch and evaluates performance.
    """
    model.train()
    correct = total = 0
    
    # Wrap train_loader with tqdm for a visual progress bar
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} [Train]")
    
    for i, data in pbar:
        images = data[0]['data']
        labels = data[0]['label'].squeeze().long()
        labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels_one_hot)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update tqdm description with current metrics
        if i % 10 == 0:
            acc = 100.*correct/total
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{acc:.2f}%"})
            
    train_loader.reset()
    # Post-epoch validation
    metrics = validate(model, val_loader, criterion, num_classes, device)
    
    # Comprehensive Reporting
    print(f"\n[Epoch {epoch}] Final Results:")
    print(f" - Validation Loss:      {metrics['loss']:.4f}")
    print(f" - Validation Accuracy:  {metrics['acc']:.2f}%")
    print(f" - Macro-F1 Score:      {metrics['f1']:.4f}")
    print(f" - Macro-Precision:     {metrics['precision']:.4f}")
    print(f" - Macro-Recall:        {metrics['recall']:.4f}")
    
    wandb.log({
        "epoch": epoch, 
        "train_acc": 100.*correct/total, 
        "val_acc": metrics['acc'], 
        "val_loss": metrics['loss'],
        "val_f1": metrics['f1'],
        "val_precision": metrics['precision'],
        "val_recall": metrics['recall']
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
            "lr_phase1": 1e-4,
            "lr_phase2": 2e-5,
            "architecture": "Triple-Threat(Bio+Dino+Conv)",
            "resolution": RESOLUTION,
            "batch_size": BATCH_SIZE,
            "epochs_phase1": 10,
            "epochs_phase2": 30,
            "asl_gamma_neg": 4,
            "asl_gamma_pos": 1,
            "bioclip_backbone": BIOCLIP_NAME,
            "dinov2_backbone": DINOV2_NAME,
            "convnext_backbone": CONVNEXT_NAME,
        }
    )
    config = wandb.config
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the cleaned CSV if it exists, otherwise use raw
    csv_path = CLEANED_CSV if os.path.exists(CLEANED_CSV) else RAW_CSV

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
    
    model.freeze_backbones()
    
    # Setup Logit Adjustment using class priors from the metadata
    df = cudf.read_csv(csv_path, sep=';')
    counts = df['species_id'].value_counts().sort_index().to_arrow().to_pylist()
    
    asl = AsymmetricLoss(gamma_neg=config.asl_gamma_neg, gamma_pos=config.asl_gamma_pos)
    criterion = LogitAdjustmentLoss(class_counts=counts, base_criterion=asl)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.lr_phase1, weight_decay=0.05)
    scaler = GradScaler('cuda')
    
    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(config.epochs_phase1):
        run_epoch(model, train_loader, val_loader, optimizer, criterion, scaler, epoch, num_classes, DEVICE)
        
    # --- PHASE 2: Calibration (Full Fine-Tuning + Sqrt Resampling) ---
    print("\n--- PHASE 2: Calibration (Long-Tail Recovery) ---")
    # Switch to Square-Root Sampling to boost rare species visibility
    train_loader, val_loader, _ = get_dali_loaders(
        csv_path, IMG_DIR, batch_size=config.batch_size, sampling_mode='sqrt'
    )
    
    model.unfreeze_backbones()
    # Aggressive but careful fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=config.lr_phase2, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs_phase2)
    
    for epoch in range(config.epochs_phase1, config.epochs_phase1 + config.epochs_phase2):
        val_acc = run_epoch(model, train_loader, val_loader, optimizer, criterion, scaler, epoch, num_classes, DEVICE)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/ensemble_best_calibrated.pth")
            print(f"--> Saved New Best Ensemble (Acc: {val_acc:.2f}%)")
        
        scheduler.step()

    torch.save(model.state_dict(), "models/ensemble_final.pth")
    wandb.finish()

if __name__ == "__main__":
    train()
