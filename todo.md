# PlantCLEF 2026 - Master TODO & Roadmap

## Phase 1: Foundations & Training
- [x] **Triple Ensemble:** BioCLIP, DINOv2-L, ConvNeXt-V2-L backbones.
- [x] **Configurable Backbones:** "4090" / "A100" hardware modes in `config.py`.
- [x] **Modular Architecture:** `src/` split into `models/`, `data/`, `training/`.
- [x] **BF16 Mixed Precision:** Upgraded from FP16; GradScaler removed.
- [x] **TF32 Matmul:** ~2x throughput on Blackwell tensor cores.
- [x] **FlashAttention 2:** Enabled via `enable_flash_sdp(True)` for ViT backbones.
- [x] **Fused AdamW:** Single-kernel parameter updates for Phase 2.
- [x] **Resolution 384px:** ConvNeXt native resolution; 26% cheaper than 448px.
- [x] **Channels-Last Memory Format:** `torch.channels_last` for ConvNeXt throughput.
- [x] **GPU-Accelerated Metrics:** torchmetrics replaces sklearn; GPU-resident accumulation.
- [x] **DALI Pipeline:** Full GPU image decode/resize/augment; no CPU bottleneck.
- [x] **Phase 1 Feature Caching:** Backbone features extracted once (~1-1.5 hrs at 384px)
      and cached to disk. Phase 1 head training runs on cached tensors (~30-40 min).
- [x] **Auto-Skip Phase 1:** Training automatically skips Phase 1 if a complete
      checkpoint exists, saving ~4-5 hours on every restart.
- [x] **Differential Learning Rates:** LoRA params at lower LR, heads at higher LR.
- [x] **OneCycleLR Scheduler:** Per-batch stepping; converges ~30% faster.
- [x] **DeepSpeed ZeRO Stage 1:** Phase 1 CPU-offloads optimizer states (head params tiny).
      Phase 2 keeps states on GPU (LoRA params tiny, no RAM pressure).
- [x] **Chunked Backbone Forward:** Caps peak VRAM regardless of batch size.
      EXTRACT_CHUNK_SIZE=64, CHUNK_SIZE=32, P2_CHUNK_SIZE=16.
- [x] **Gradient Checkpointing:** Enabled on all backbones for Phase 1.
      Disabled before applying LoRA to avoid hook conflicts.

### RTX 5090 Blackwell Optimization (Current)
- [ ] **Zero-I/O: Pin Feature Cache to VRAM:** Load 22.5GB `phase1_feature_cache.pt` directly to CUDA to eliminate disk bottlenecks.
- [ ] **FP8 Training:** Enable Blackwell native FP8 via TransformerEngine for Phase 2.
- [ ] **Inductor Max-Autotune:** Use `torch.compile(mode="max-autotune")` for ensemble heads.
- [ ] **Vectorized Ensemble Heads:** Consolidate multiple head forward passes into a single GEMM.
- [ ] **ZeRO-1 / No-Offload Tuning:** Maximize 5090 compute throughput; disable all PCIe/CPU synchronization.

### LoRA Fine-Tuning (Phase 2)
- [x] **LoRA on DINOv2 + ConvNeXt:** PEFT LoRA (r=16) on attention + MLP layers.
      Reduces trainable params from ~800M to ~5M. BioCLIP fully frozen throughout.
- [x] **BioCLIP Frozen in Phase 2:** Tree-of-Life pretraining already provides
      taxonomically discriminative features -- no benefit from adapting further.
- [x] **batch=256 in Phase 2:** LoRA removes full-backbone backward, enabling
      large batches without OOM. Phase 2 target: < 24 hours for 3 epochs.
- [x] **Per-Epoch Checkpointing:** Lightweight epoch checkpoint every epoch +
      full DeepSpeed checkpoint on each validation pass.
- [x] **Early Stopping (patience=2):** Stops Phase 2 if no improvement for
      2 consecutive validations.

### Batch Size Optimisations
- [x] **Expandable Segments Allocator:** Reduces VRAM fragmentation ~2-3GB.
- [x] **Separate Phase configs:** DS_CONFIG_P1 (CPU offload) vs DS_CONFIG_P2 (GPU).
      Phase 2 CPU offload removed after Bus Error from RAM exhaustion.

- [ ] **Multi-Scale Feature Fusion:** FPN or Multi-Scale DINOv2 for tiny species.
- [ ] **Environment Setup:** Run `bash Infrastructure/scripts/setup_environment.sh`.
- [ ] **Initial Preprocessing:** Run `python3 src/data/preprocess.py`.

## Phase 2: Synthetic Complexity & The Long-Tail
- [ ] **"Plant Sticker" Factory (SAM):** Create `src/data/sam_extractor.py`.
- [ ] **Synthetic Collage Generation:**
    - [ ] Alpha Blending, Z-Ordering, Global Augmentation.
    - [ ] Create `src/data/collage_generator.py`.
- [ ] **Long-Tail Strategies (Challenge 3):**
    - [x] Square-Root (Power) Sampling via DALI weighted file-list.
    - [x] Asymmetric Loss (ASL) with separate gamma values.
    - [x] Logit Adjustment based on class priors.
    - [x] Progressive 2-Stage: feature caching + head warmup → LoRA fine-tune.
    - [ ] Label Propagation for crowded collage ground truth.

## Phase 3: Domain Adaptation & High-Res Tiling
- [ ] **Pseudo-Labeling:** Teacher-Student loop on unlabeled LUCAS quadrats.
- [ ] **SAHI Tiling Engine:**
    - [ ] GPU-batched `src/predict_sahi.py`.
    - [ ] 20% overlapping tiles to prevent split-plant misses.
    - [ ] Hybrid Attention hotspot pass (~70% compute saving).
- [ ] **Distribution Matching:** LUCAS unlabeled data for species prior estimation.
- [ ] **Quality/Blur Filtering:** Laplacian Variance on expanded dataset.

## Phase 4: Metric Optimization & Ecological Guardrails
- [ ] **Ensemble Weighting:** Learnable backbone blend weights.
- [ ] **Metadata Integration:** Self-attention over geolocation/altitude/climate.
- [ ] **Threshold Optimization (Challenge 4):**
    - [ ] Brent's Method for per-class F1 thresholds (7,800 species).
    - [ ] Frequency-Based Adjustment: scale by $N_{class}$.
    - [ ] Temperature Scaling: logit calibration via $T$ on validation.
    - [ ] Empty Plot Logic: predict "nothing" when no species exceed threshold.
- [ ] **Ecological Filtering:**
    - [ ] SINR or GIS-based geographic constraint lookup.
    - [ ] GPS metadata integration.
- [ ] **Taxonomic Co-occurrence Smoothing:** Suppress impossible combinations.
- [ ] **Final Ensembling:** Model Soup (weight averaging) or multi-model averaging.
- [ ] **Performance Optimization (deferred):**
    - [ ] FP8 activations via `transformer-engine` for further VRAM reduction.
    - [ ] Re-evaluate `torch.compile` once LoRA + DeepSpeed + chunked forward
          confirmed stable end-to-end.
    - [ ] CUDA streams for parallel backbones (blocked by autocast + grad ckpt).
    - [ ] Revert DALI to Mixed GPU Decoding once NVML driver issues resolved.

---
*Last updated: March 26, 2026*
