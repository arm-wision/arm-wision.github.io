import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import os
from models.ensemble import PlantEnsemble
from data.dataloader import get_dali_loaders 

# Enable CuDNN benchmark for RTX 4090 speed
torch.backends.cudnn.benchmark = True

class AsymmetricLoss(nn.Module):
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
        return -loss.sum()

class EarlyStopping:
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
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
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
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    loader.reset()
    return val_loss / (i + 1), 100. * correct / total

def train():
    wandb.init(
        project="plantclef-2026",
        name="ensemble-bioclip-dinov2-4090",
        config={
            "lr": 5e-5,
            "architecture": "Ensemble(BioCLIP+DINOv2)",
            "resolution": 448,
            "batch_size": 128, # Reduced for ensemble VRAM
            "num_threads": 16,
            "val_split": 0.1,
            "asl_gamma_neg": 4,
            "asl_gamma_pos": 1,
            "epochs": 50,
            "patience": 5,
        }
    )
    config = wandb.config
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = "data/train_metadata_cleaned.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/PlantCLEF2024_single_plant_training_metadata.csv"

    train_loader, val_loader, num_classes = get_dali_loaders(
        csv_path, 
        "data/train/", 
        batch_size=config.batch_size,
        val_split=config.val_split,
        num_threads=config.num_threads
    )
    
    print(f"Initializing Ensemble for {num_classes} classes...")
    model = PlantEnsemble(num_classes=num_classes, input_res=config.resolution).to(DEVICE)
    
    # Optional: Freeze backbones for the first 2 epochs to stabilize the head
    # model.freeze_backbones() 
    
    criterion = AsymmetricLoss(gamma_neg=config.asl_gamma_neg, gamma_pos=config.asl_gamma_pos)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.05)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    scaler = GradScaler('cuda')
    early_stopping = EarlyStopping(patience=config.patience)
    
    best_val_acc = 0.0
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    for epoch in range(config.epochs): 
        model.train()
        train_loss = correct = total = 0
        num_batches = len(train_loader)

        for i, data in enumerate(train_loader):
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

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if i % 100 == 0:
                acc = 100. * correct / total
                wandb.log({"batch/loss": loss.item() / labels.size(0), "batch/accuracy": acc})
                print(f"Epoch {epoch} | Batch {i}/{num_batches} | Loss: {loss.item():.4f} | Acc: {acc:.2f}%")

        train_loader.reset()

        val_loss, val_acc = validate(model, val_loader, criterion, num_classes, DEVICE)
        print(f"Epoch {epoch} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        wandb.log({
            "epoch": epoch, 
            "epoch/train_acc": 100.*correct/total,
            "epoch/val_acc": val_acc,
            "epoch/val_loss": val_loss,
            "epoch/lr": scheduler.get_last_lr()[0]
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(models_dir, "ensemble_best.pth"))
            print(f"--> Saved New Best Ensemble (Acc: {val_acc:.2f}%)")

        scheduler.step()
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    torch.save(model.state_dict(), os.path.join(models_dir, "ensemble_final.pth"))
    wandb.finish()

if __name__ == "__main__":
    train()
