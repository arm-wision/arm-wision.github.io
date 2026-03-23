import torch
import torch.nn.functional as F
from torch.amp import autocast
from torchmetrics.classification import (MulticlassF1Score, MulticlassPrecision,
                                         MulticlassRecall, MulticlassAccuracy)
from tqdm import tqdm
import wandb

from config import BATCH_SIZE, CHUNK_SIZE, P2_CHUNK_SIZE, MAX_VAL_BATCHES
from .cache import chunked_backbone_forward, CachedFeatureDataset


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model, loader, criterion, num_classes, device):
    """
    GPU-accelerated validation via torchmetrics.
    All state accumulates on GPU; single CPU transfer at .compute().
    Stops at MAX_VAL_BATCHES for speed (~25,600 images, within 0.5% of full val).
    """
    model.eval()
    val_loss = 0.0
    f1_m   = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    prec_m = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    rec_m  = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
    acc_m  = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)

    val_chunk = CHUNK_SIZE * 2   # no_grad -> 2x chunk is safe
    i = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader, desc="Validating", unit="batch", leave=False)):
            if i >= MAX_VAL_BATCHES:
                break
            images = data[0]['data'].to(memory_format=torch.channels_last)
            labels = data[0]['label'].squeeze().long()
            labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                feat_bio  = chunked_backbone_forward(model.bioclip,  images, val_chunk)
                feat_dino = chunked_backbone_forward(model.dinov2,   images, val_chunk)
                feat_conv = chunked_backbone_forward(model.convnext, images, val_chunk)
                feat_bio  = F.normalize(model.proj_bio(feat_bio),   dim=1)
                feat_dino = F.normalize(model.proj_dino(feat_dino), dim=1)
                feat_conv = F.normalize(model.proj_conv(feat_conv), dim=1)
                outputs   = model.classifier(torch.cat([feat_bio, feat_dino, feat_conv], dim=1))
                loss      = criterion(outputs, labels_one_hot)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            f1_m.update(predicted, labels)
            prec_m.update(predicted, labels)
            rec_m.update(predicted, labels)
            acc_m.update(predicted, labels)

    metrics = {
        "loss":      val_loss / (i + 1),
        "acc":       acc_m.compute().item()  * 100.0,
        "f1":        f1_m.compute().item(),
        "precision": prec_m.compute().item(),
        "recall":    rec_m.compute().item(),
    }
    for m in [f1_m, prec_m, rec_m, acc_m]:
        m.reset()
    loader.reset()
    return metrics


# ---------------------------------------------------------------------------
# Phase 1: cached head training
# ---------------------------------------------------------------------------

def run_phase1_cached(model, cache, criterion, epoch, num_classes, device):
    """
    Train only projection heads + classifier on pre-extracted backbone features.
    No backbone compute -- pure MLP ops, very fast (~30-40 min for 10 epochs).
    """
    model.train()
    acc_m = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)

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

        model.backward(loss)
        model.step()

        _, predicted = outputs.max(1)
        acc_m.update(predicted, labels)
        running_loss += loss.item()

        if i % 10 == 0:
            pbar.set_postfix({
                "Loss": f"{running_loss/(i+1):.4f}",
                "Acc":  f"{acc_m.compute().item()*100:.2f}%"
            })

    train_acc = acc_m.compute().item() * 100.0
    acc_m.reset()
    return train_acc


# ---------------------------------------------------------------------------
# Phase 2: LoRA fine-tuning epoch
# ---------------------------------------------------------------------------

def run_epoch(model, train_loader, criterion, epoch, num_classes, device):
    """
    Phase 2 training with LoRA adapters on DINOv2 + ConvNeXt.

    BioCLIP: fully frozen, explicit no_grad skips its activation storage.
    DINOv2 + ConvNeXt: LoRA adapters are trainable; base weights frozen.
    Only ~5M params update vs ~800M for full fine-tuning.
    """
    model.train()
    acc_m = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)

    pbar         = tqdm(enumerate(train_loader), total=len(train_loader),
                        desc=f"Epoch {epoch} [Train]")
    running_loss = 0.0

    for i, data in pbar:
        images = data[0]['data'].to(memory_format=torch.channels_last)
        labels = data[0]['label'].squeeze().long()
        labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            # BioCLIP: fully frozen -- skip gradient graph entirely
            with torch.no_grad():
                feat_bio = chunked_backbone_forward(
                    model.module.bioclip, images, P2_CHUNK_SIZE)

            # DINOv2 + ConvNeXt: LoRA params need gradients
            feat_dino = chunked_backbone_forward(
                model.module.dinov2, images, P2_CHUNK_SIZE)
            feat_conv = chunked_backbone_forward(
                model.module.convnext, images, P2_CHUNK_SIZE)

            feat_bio  = F.normalize(model.module.proj_bio(feat_bio),   dim=1)
            feat_dino = F.normalize(model.module.proj_dino(feat_dino), dim=1)
            feat_conv = F.normalize(model.module.proj_conv(feat_conv), dim=1)

            outputs = model.module.classifier(
                torch.cat([feat_bio, feat_dino, feat_conv], dim=1))
            loss = criterion(outputs, labels_one_hot)

        model.backward(loss)
        model.step()

        _, predicted = outputs.max(1)
        acc_m.update(predicted, labels)
        running_loss += loss.item()

        if i % 10 == 0:
            pbar.set_postfix({
                "Loss": f"{running_loss/(i+1):.4f}",
                "Acc":  f"{acc_m.compute().item()*100:.2f}%"
            })

    train_loader.reset()
    train_acc = acc_m.compute().item() * 100.0
    acc_m.reset()

    wandb.log({"epoch": epoch, "train_acc": train_acc, "lr": model.get_lr()[0]})
    print(f"[Epoch {epoch}] Train Acc: {train_acc:.2f}%")
    return train_acc
