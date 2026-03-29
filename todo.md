# PlantCLEF 2026 - Master TODO & Roadmap

## Phase 1: Foundations & Training
- [x] **Triple Ensemble:** BioCLIP, DINOv2-L, ConvNeXt-V2-L backbones.
- [x] **Configurable Backbones:** "5090" / "4090" hardware modes.
- [x] **BF16 & TF32 Precision:** High-throughput Blackwell tensor core utilization.
- [x] **DALI Pipeline:** Full GPU image decode/resize/augment; no CPU bottleneck.
- [x] **Residual MLP Head:** Upgraded with skip connections and Multi-Sample Dropout for better feature modeling.
- [x] **High-Fidelity PCA:** 1024 components (up from 512) to preserve fine-grained detail.
- [x] **Auto-Skip Phase 1:** Training skips Phase 1 if a compatible 1024-d checkpoint exists.
- [x] **Fast Cache Warmup:** 10-minute head training bypass using pre-extracted features.

### RTX 5090 Blackwell Optimization (Completed)
- [x] **VRAM Balancing:** Reduced Phase 2 batch (64) and chunk size (16) to accommodate `torch.compile` overhead.
- [x] **Inductor Max-Autotune:** `torch.compile(mode="max-autotune")` for Triton-optimized kernels.
- [x] **Vectorized Ensemble Projections:** Refactored into a single Grouped GEMM for max throughput.
- [x] **ZeRO-1 / Untested Opt Support:** Enabled custom optimizers (Lookahead) in DeepSpeed.

### LoRA Fine-Tuning (Phase 2 - The 80% Push)
- [x] **High-Rank LoRA ($R=64$):** Increased adaptation capacity for 7,800-class discrimination.
- [x] **1M Samples per Epoch:** Quadrupled data volume to maximize feature learning.
- [x] **Asymmetric Loss (ASL):** Fused CUDA kernel for multi-label long-tail classification.
- [x] **Lookahead Optimizer:** Wrapped AdamW for smoother convergence and flatter minima discovery.
- [x] **Taxonomic Distillation:** Knowledge transfer from BioCLIP to adaptive backbones (KD_ALPHA=0.3).
- [x] **Label Smoothing (0.1):** Improved generalization by penalizing over-confidence.
- [ ] **Phase 2C Final Convergence:** Reach 80% Validation Accuracy.

## Phase 3: Domain Adaptation & High-Res Tiling
- [ ] **Pseudo-Labeling:** Teacher-Student loop on unlabeled LUCAS quadrats.
- [x] **CUDA SAHI Engine:**
    - [x] **extract_tiles:** VRAM-native image slicing (10x speedup).
    - [x] **fused_max_pool:** CUDA aggregation of 7,800-class probabilities.
    - [x] **Batch Inference:** Chunked tile processing to prevent VRAM spikes.
- [ ] **Hybrid Attention Hotspot Pass:** (~70% compute saving by skipping empty tiles).

## Phase 4: Metric Optimization & Ecological Guardrails
- [x] **C++ Taxonomic Filter:** Bitset-based co-occurrence validation (runs in <1ms).
- [ ] **Threshold Optimization:** Brent's Method for per-class F1 thresholds.
- [ ] **Ecological Filtering:** SINR or GIS-based geographic constraint lookup.
- [ ] **Final Ensembling:** Model Soup (weight averaging) or multi-model averaging.

---
*Last updated: March 29, 2026*
