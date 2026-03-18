import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.cuda.amp import GradScaler, autocast
from model import PlantBioCLIP 
from dataloader import get_dali_loader 

def train():
    # Hyperparams
    BATCH_SIZE, LR, EPOCHS = 128, 3e-5, 20 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PlantBioCLIP(num_classes=7800).to(DEVICE)
    loader = get_loaders("train_metadata.csv", "images/", batch_size=BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scaler = GradScaler()

    model.train()
    for epoch in range(EPOCHS): 
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}], Step [{i}], Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
