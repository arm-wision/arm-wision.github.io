# PlantCLEF 2026 - Master TODO & Roadmap

## Phase 1: Foundations & Training
- [x] **Triple Ensemble:** Integrated BioCLIP, DINOv2, and ConvNeXt-V2 backbones.
- [x] **Configurable Backbones:** Added "4090" and stronger GPU hardware modes to `config.py`.
- [x] **Modular Architecture:** Organized `src/` into `models/` and `data/` subdirectories.
- [x] **Mixed Precision (AMP):** Upgraded from FP16 to **BF16** for native 5090 tensor core
      throughput. GradScaler removed (BF16 has sufficient dynamic range).
- [x] **TF32 Matmul:** Enabled `torch.backends.cuda.matmul.allow_tf32` and
      `torch.backends.cudnn.allow_tf32` for ~2x matmul throughput on 5090.
- [x] **FlashAttention 2:** Enabled via `torch.backends.cuda.enable_flash_sdp(True)`.
      BioCLIP and DINOv2 attention layers automatically route through FA2 for
      O(N) memory attention and faster throughput on the 5090.
- [x] **Fused AdamW:** Replaced standard AdamW with `fused=True` for single-kernel
      parameter updates (~10-15% optimizer step speedup).
- [x] **High-Res Support:** Implemented positional embedding interpolation for 448px resolution.
- [x] **Channels-Last Memory Format:** Applied `torch.channels_last` to model and image
      tensors for improved ConvNeXt throughput on NVIDIA tensor cores.
- [x] **Modular Training Loop:** Integrated DALI, WandB, and Early Stopping.
- [x] **GPU-Accelerated Metrics:** Replaced sklearn with torchmetrics -- all F1/Precision/Recall
      accumulation on GPU, single CPU transfer per epoch.
- [x] **Inference Pipeline:** Updated SAHI manual tiling to support the new ensemble.
- [ ] **Environment Setup:** Run `bash Infrastructure/scripts/setup_environment.sh`
      on the target machine.
- [ ] **Initial Preprocessing:** Run `python3 src/data/preprocess.py` to
      generate the cleaned metadata.
    - [x] **GPU-Accelerated Blur Filtering:** Implemented DALI + PyTorch Laplacian Variance
          scoring for 1.4M images entirely on GPU.
    - [x] **cuDF-Optimized Metadata Cleaning:** 100x speedup in CSV processing
          using GPU-accelerated dataframes.
- [x] **Baseline Training:** Run `python3 src/train.py` with the Triple Ensemble.
- [x] **Phase 1 Feature Caching:** Backbone features extracted once and cached to disk.
      Phase 1 head training runs on cached tensors -- ~10x faster than re-running
      frozen backbones each epoch.
- [x] **Differential Learning Rates (Phase 2):** Backbones use 5e-6 LR, projection heads
      and classifier use 2e-4 LR.
- [x] **OneCycleLR Scheduler:** Replaced CosineAnnealingLR with OneCycleLR stepped
      per batch -- converges ~30% faster with fewer total epochs required.

### Batch Size Optimisations
- [x] **Expandable Segments Allocator:** Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,
      max_split_size_mb:256` before torch import, reducing VRAM fragmentation by ~2-3GB.
- [x] **DeepSpeed ZeRO Stage 1:** Integrated DeepSpeed with ZeRO Stage 1 optimizer state
      offloading to pinned CPU RAM. Frees ~4-8GB of VRAM previously consumed by Adam
      moment tensors, directly enabling larger batch sizes.
- [x] **Chunked Backbone Forward:** Each backbone processes `CHUNK_SIZE=8` images at a
      time regardless of logical batch size. Peak activation VRAM is capped at 8 images
      while gradients still cover the full batch -- enables batch=64+ without OOM.
- [x] **Gradient Checkpointing (deep):** Enabled on all backbone transformer blocks
      via `set_grad_checkpointing(True)`, trading ~30% compute for ~40% less activation
      memory during Phase 2 full fine-tuning.

- [ ] **Multi-Scale Feature Fusion:** Implement **Feature Pyramid Networks (FPN)** or
      **Multi-Scale Inference** (DINOv2) to handle tiny species (Challenge 2).

## Phase 2: Synthetic Complexity & The Long-Tail
- [ ] **"Plant Sticker" Factory (SAM):**
    - Use SAM (Segment Anything Model) to automate the extraction of plants
        from the 1.4M images.
    - [ ] Create `src/data/sam_extractor.py`.
- [ ] **Synthetic Collage Generation:**
    - Generate 500k "Synthetic Collages" using the Copy-Paste method (Challenge 1).
    - [ ] Implement **Alpha Blending**, **Z-Ordering**, and **Global Augmentation**
        (color cast/shadow filters) for realism.
    - [ ] Create `src/data/collage_generator.py`.
- [ ] **Long-Tail Strategies (Challenge 3):**
    - [x] **Square-Root (Power) Sampling:** Weighted DALI file-list generator.
    - [x] **Asymmetric Loss (ASL):** Refined with separate gamma for positives/negatives.
    - [x] **Logit Adjustment:** Loss-level logit shifting based on class priors.
    - [x] **Progressive 2-Stage Balancing:** Feature caching + head warmup → full fine-tune.
    - [ ] Explore **Label Propagation** for crowded collage ground truth accuracy.

## Phase 3: Domain Adaptation & High-Res Tiling
- [ ] **Pseudo-Labeling:** Teacher-Student loop on unlabeled LUCAS quadrats.
- [ ] **SAHI Tiling Engine:**
    - [ ] Optimize `src/predict_sahi.py` for GPU batching.
    - [ ] **Overlapping Tiles:** 20% overlap to prevent split-plant misses.
    - [ ] **Hybrid Attention (Zoom-in):** Low-res hotspot pass to save ~70% compute.
- [ ] **Distribution Matching:** LUCAS unlabeled data for species prior estimation.
- [ ] **Quality/Blur Filtering:** Laplacian Variance on expanded dataset.

## Phase 4: Metric Optimization & Ecological Guardrails
- [ ] **Ensemble Weighting:** Differentiable optimization for backbone blend weights.
- [ ] **Metadata Integration:** Structured self-attention over geolocation/altitude/climate.
- [ ] **Threshold Optimization (Challenge 4):**
    - [ ] Brent's Method for per-class F1 thresholds across all 7,800 species.
    - [ ] Frequency-Based Adjustment: scale thresholds by $N_{class}$.
    - [ ] Temperature Scaling: logit calibration via $T$ on validation data.
    - [ ] Empty Plot Logic: predict "nothing" when no species exceed threshold.
- [ ] **Ecological/Botanical Filtering:**
    - [ ] SINR or GIS-based geographic constraint lookup.
    - [ ] Integrate GPS metadata into prediction pipeline.
- [ ] **Taxonomic & Co-occurrence Smoothing:** Co-occurrence matrix for impossible combos.
- [ ] **Final Ensembling:** Model Soup (weight averaging) or multi-model averaging.
- [ ] **Performance Optimization (deferred):**
    - [ ] Re-evaluate `torch.compile(mode='max-autotune')` on full model once stable.
    - [ ] Re-evaluate CUDA streams for parallel backbones once confirmed compatible
          with autocast + gradient checkpointing on target PyTorch version.
    - [ ] Revert DALI to Mixed GPU Decoding once NVML driver issues resolved.
    - [ ] Evaluate FP8 activations via `transformer-engine` for further VRAM reduction.

---
*Last updated: March 22, 2026*
