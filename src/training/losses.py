import torch
import torch.nn as nn


class LogitAdjustmentLoss(nn.Module):
    """
    Shifts logits by log(prior) to demand larger margins for common classes
    and ease requirements for rare ones -- critical for 7,800-class long-tail.
    """
    def __init__(self, class_counts, tau=1.0, label_smoothing=0.1):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        priors = counts / counts.sum()
        self.adjustment = (tau * torch.log(priors + 1e-12)).to('cuda')
        # Use CrossEntropyLoss for single-label 7,800-class classification
        self.criterion  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, x, y):
        # y should be class indices (LongTensor)
        return self.criterion(x + self.adjustment, y)


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss: aggressively down-weights easy negatives across 7,800 classes
    using separate gamma parameters for positive and negative samples.
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
        loss    = los_pos + los_neg
        with torch.no_grad():
            pt      = xs_pos * y + xs_neg * (1 - y)
            weights = (1 - pt).pow(self.gamma_pos * y + self.gamma_neg * (1 - y))
        loss *= weights
        
        # Sum across classes (C=7800), then mean across batch (B=128/256)
        # Using .mean() across all B*C elements dilutes the positive class loss
        # by a factor of 1/7800. Summing first ensures a strong training signal.
        return -loss.sum(dim=1).mean()


class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.001):
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
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.counter    = 0
