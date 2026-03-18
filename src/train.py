import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.cuda.amp import GradScaler, autocast
import wandb
from model import PlantBioCLIP 
from dataloader import get_dali_loader 

def train():
    # Init wandb 
    wandb.init(
        project="plantclef-2026",
        name="bioclip-baseline",
        config={
            "learning_rate": 3e-5,
            "architecture": "BioCLIP-ViT-L/14",
            "dataset": "PlantCLEF-1.4M",
            "batch_size": 128,
            "resolution": 448,
            "optimizer": "AdamW",
            "loss": "CrossEntropy",
            "epochs": 20,
        }
    )
    config = wandb.config

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PlantBioCLIP(num_classes=7800).to(DEVICE)
    loader = get_dali_loader("train_metadata.csv", "images/", batch_size=config.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.05)
    scaler = GradScaler()
    
    print(f"Star training on {DEVICE}...")

    model.train()

    for epoch in range(config.epochs): 
        total_loss, correct, total = 0 
        for i, data in enumerate(loader):
            images = data[0]['data']
            labels = data[0]['label'].squeeze().long()

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # --- Metrics Calculation --- 
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # log batch metrics to wandb
            if i % 50 == 0:
                acc = 100. * correct / total
                wandb.log({
                    "batch/loss": loss.item(),
                    "batch/accuracy": acc,
                    "batch/learning_rate": optimizer.param_groups[0]['lr'],
                    "batch/progress": i / len(loader)
                })
                print(f"Epoch {epoch} | Batch {i} | Loss: {loss.item():.4f} | Acc: {acc:.2f}%")
        
        # Log Epoch Metrics
        epoch_acc = 100. * correct / total
        avg_loss = total_loss / len(loader)
        wandb.log({
            "epoch/loss": avg_loss,
            "epoch/accuracy": epoch_acc,
            "epoch": epoch
        })

        # Save Checkpoint and Log to WandB Artifacts
        checkpoint_path = f"checkpoint_epoch_{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        wandb.save(checkpoint_path)

    wandb.finish()

if __name__ == "__main__":
    train()
