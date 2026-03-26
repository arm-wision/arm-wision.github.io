# PlantCLEF 2026 - Master TODO & Roadmap

## Phase 1: Foundations & Training
- [x] **Triple Ensemble:** BioCLIP, DINOv2-L, ConvNeXt-V2-L backbones.
- [x] **Configurable Backbones:** "5090" / "4090" hardware modes.
- [x] **BF16 & TF32 Precision:** High-throughput Blackwell tensor core utilization.
- [x] **DALI Pipeline:** Full GPU image decode/resize/augment; no CPU bottleneck.
- [x] **Boosted Phase 1 Head:** 3-layer MLP (2048-1024-7800) for non-linear feature mapping.
- [x] **High-Fidelity PCA:** 1024 components (up from 512) to preserve fine-grained detail.
- [x] **Auto-Skip Phase 1:** Training skips Phase 1 if a compatible 1024-d checkpoint exists.

### RTX 5090 Blackwell Optimization (Completed)
- [x] **Zero-I/O: Pin Feature Cache to VRAM:** 22.5GB cache loaded to CUDA to eliminate PCIe bottlenecks.
- [x] **FP8 Training:** Enabled native Blackwell FP8 (`float8_e4m3fn`) for adapters and heads.
- [x] **Inductor Max-Autotune:** `torch.compile(mode="max-autotune")` for Triton-optimized kernels.
- [x] **Vectorized Ensemble Projections:** Refactored into a single Grouped GEMM for max throughput.
- [x] **ZeRO-1 / No-Offload Tuning:** Pure GPU training; disabled all CPU offloading.

### LoRA Fine-Tuning (Phase 2 - The 80% Push)
- [x] **High-Rank LoRA ($R=64$):** Increased adaptation capacity for 7,800-class discrimination.
- [x] **1M Samples per Epoch:** Quadrupled data volume to maximize feature learning.
- [x] **Natural Sampling:** Switched from `sqrt` to `natural` to optimize for test distribution.
- [x] **Logit-Adjusted Cross-Entropy:** Softmax-based classification to prevent "loss hacking."
- [x] **Unique Checkpointing:** Step-level resumes with automatic latest-file discovery.
- [ ] **Phase 2C Final Convergence:** Reach 80% Validation Accuracy before Phase 3.

## Phase 2: Synthetic Complexity & The Long-Tail
- [ ] **"Plant Sticker" Factory (SAM):** Create `src/data/sam_extractor.py`.
- [ ] **Synthetic Collage Generation:**
    - [ ] Alpha Blending, Z-Ordering, Global Augmentation.
    - [ ] Create `src/data/collage_generator.py`.
- [ ] **Label Propagation:** For crowded collage ground truth.

## Phase 3: Domain Adaptation & High-Res Tiling
- [ ] **Pseudo-Labeling:** Teacher-Student loop on unlabeled LUCAS quadrats.
- [ ] **SAHI Tiling Engine:**
    - [x] Refactored `src/predict_sahi.py` for Blackwell compatibility.
    - [ ] 20% overlapping tiles to prevent split-plant misses.
    - [ ] Hybrid Attention hotspot pass (~70% compute saving).

## Phase 4: Metric Optimization & Ecological Guardrails
- [ ] **Threshold Optimization (Challenge 4):** Brent's Method for per-class F1 thresholds.
- [ ] **Ecological Filtering:** SINR or GIS-based geographic constraint lookup.
- [ ] **Final Ensembling:** Model Soup (weight averaging) or multi-model averaging.

---
*Last updated: March 26, 2026*
