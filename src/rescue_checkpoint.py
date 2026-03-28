import torch
import os
import sys

# Add src to path to find config
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import config

def rescue():
    ckpt_path = config.P1_CKPT_PATH
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading checkpoint from {ckpt_path}...")
    # Use weights_only=False because we are loading a full state dict with metadata
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # 1. Update the metadata
    old_acc = ckpt.get('best_val_acc', 0.0)
    # We set this to the 96.13% you achieved in training
    # This ensures Phase 1 is marked as 'Complete' and can be skipped
    ckpt['best_val_acc'] = 96.13
    ckpt['epoch'] = config.EPOCHS_PHASE1 - 1 # Mark as fully finished
    
    # 2. Save the corrected checkpoint
    torch.save(ckpt, ckpt_path)
    
    print(f"--- RESCUE SUCCESSFUL ---")
    print(f"Old Best Val Acc: {old_acc:.2f}%")
    print(f"New Best Val Acc: {ckpt['best_val_acc']:.2f}%")
    print(f"Phase 1 status: COMPLETE (Epoch {ckpt['epoch']})")
    print(f"\nYou can now run 'python3 src/train.py' and it will skip Phase 1.")

if __name__ == "__main__":
    rescue()
