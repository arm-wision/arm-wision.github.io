# Research Proposal: BioCLIP, DINOv2 & ConvNeXt-V2 Ensemble for High-Resolution Multi-Label Plant Identification in Vegetation Plots

**Date:** March 23, 2026

## 1. Title
**BioCLIP, ConvNeXt-V2 & DINOv2 Ensemble with LoRA Adaptation for High-Resolution Multi-Label Plant Identification in Vegetation Plots**

## 2. Research Questions
1. How can domain-specific foundation models (BioCLIP) be adapted to identify rare species in complex, overlapping vegetation plots where visual occlusion and a long-tailed species distribution typically degrade performance?
2. Does taxonomic-aware pre-training (BioCLIP) provide superior feature discrimination for morphologically similar species compared to domain-agnostic self-supervised models (DINOv2)?
3. Can Low-Rank Adaptation (LoRA) provide competitive fine-tuning performance relative to full fine-tuning while reducing training cost by an order of magnitude?
4. Does high-resolution tiling (SAHI) disproportionately benefit the recall of rare, small-stature plants in 50×50cm vegetation quadrats?

## 3. National Interest Statement
Monitoring plant biodiversity is a matter of critical national security regarding climate resilience, agriculture, and ecosystem services. Current manual surveying methods are slow, subjective, and cost-prohibitive. This research aims to automate large-scale botanical surveys, providing governmental agencies and environmental organizations with the tools to respond instantly to invasive species threats, track habitat loss, and monitor the health of national ecosystems.

## 4. Introduction
Identifying plants in the wild is traditionally a "single-focus" task. However, ecological reality consists of complex "vegetation plots" (quadrats) where multiple species overlap, compete, and obscure one another. This project builds a high-performance AI system capable of identifying every species present in such plots, utilizing biological foundation models, synthetic data generation, and parameter-efficient fine-tuning.

## 5. Research Problem
Three major bottlenecks hinder current botanical AI:
1. **The Domain Gap:** Models trained on clean, single-plant images fail in overlapping vegetation.
2. **The Long-Tail Problem:** Common species dominate datasets, leading to AI systems that ignore rare species.
3. **Compute Cost:** Full fine-tuning of large multi-backbone ensembles on 1.4M images is prohibitively expensive, requiring days per run and limiting iteration speed.

## 6. Research Design & Methodology

### A. Feature Fusion Ensemble
Three complementary backbones with distinct inductive biases:

1. **BioCLIP (ViT-L/14 — Taxonomic Expert):** Pre-trained on the "Tree of Life" (10M+ biological images). Provides taxonomic hierarchy understanding critical for rare species. Fully frozen in both Phase 1 and Phase 2 — its specialised pretraining already provides near-optimal botanical features.

2. **DINOv2 (ViT-L/14 — Geometric Expert):** Self-supervised transformer excelling at fine-grained structural features (leaf venation, serrations). LoRA-adapted in Phase 2.

3. **ConvNeXt-V2 (Large — Local Context Expert):** CNN providing local translation invariance robust to leaf orientation, scale, and lighting variation. LoRA-adapted in Phase 2.

Each backbone's output is projected to a shared 512-dimensional space via `Linear + LayerNorm`, L2-normalised, then concatenated into a 1536-d fused representation.

### B. High-Resolution Adaptation
- **Positional Embedding Interpolation:** ViT backbones adapted to 384px via bicubic interpolation, maintaining spatial understanding while operating at ConvNeXt's native resolution.
- **Channels-Last Memory Format:** NHWC layout (`torch.channels_last`) aligns with NVIDIA tensor core preferences, improving ConvNeXt throughput.

### C. GPU-Accelerated Data Engineering
- **NVIDIA DALI:** Full GPU I/O — JPEG decode, resize, and colour conversion via `fn.decoders.image(device='mixed')`. Eliminates CPU bottleneck entirely.
- **cuDF Metadata Auditing:** 100× faster CSV processing on GPU.
- **Blur Audit:** DALI + PyTorch Laplacian Variance GPU filter removes blurry samples.

### D. Hardware Precision Optimisations
- **BF16 Mixed Precision:** Eliminates FP16 gradient underflow; removes GradScaler.
- **TF32 Matmul:** ~2× throughput on Blackwell/Hopper tensor cores.
- **FlashAttention 2:** O(N) memory attention for BioCLIP and DINOv2 ViT layers.
- **Fused AdamW:** Single-kernel parameter updates in Phase 2.
- **GPU-Accelerated Metrics:** torchmetrics accumulates F1/Precision/Recall on GPU; single transfer per epoch.

### E. Parameter-Efficient Fine-Tuning via LoRA

A key innovation in this work is the application of Low-Rank Adaptation (LoRA) to the fine-tuning phase, motivated by the prohibitive cost of full fine-tuning a triple-backbone ensemble on 1.4M images.

**LoRA formulation:** For a weight matrix $W \in \mathbb{R}^{d \times k}$, LoRA adds a low-rank update $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$. The forward pass becomes $h = Wx + \frac{\alpha}{r}BAx$, scaling the adaptation by $\alpha/r$.

**Application strategy:**
- **BioCLIP:** Fully frozen. Tree-of-Life pretraining already provides optimal taxonomic features.
- **DINOv2:** LoRA applied to attention layers (QKV, output projection) and MLP layers (fc1, fc2) with $r=16$, $\alpha=32$.
- **ConvNeXt-V2:** LoRA applied to MLP linear layers (fc1, fc2) with $r=16$, $\alpha=32$.

**Impact:** Reduces trainable parameters from ~800M to ~5M (~0.6% of total), enabling batch=256 in Phase 2 (vs batch=16 for full fine-tuning), and reducing Phase 2 training time from ~6 days to under 24 hours on a single RTX 5090.

### F. Progressive 2-Stage Training with Feature Caching

1. **Phase 1 — Head Warmup (auto-skipped on restarts if checkpoint exists):**
   - All backbones frozen. Backbone features extracted once and cached to disk (~1-1.5 hrs at 384px).
   - Only projection heads and classifier train on cached tensors (~30-40 min for 10 epochs).
   - DeepSpeed ZeRO Stage 1 with CPU optimizer offload (head params are tiny).

2. **Phase 2 — LoRA Fine-Tuning:**
   - LoRA adapters applied to DINOv2 and ConvNeXt. BioCLIP frozen.
   - Square-root resampling boosts rare species visibility ($P \propto 1/\sqrt{N_{class}}$).
   - Differential learning rates: LoRA params at 1e-4, projection heads at 2e-4.
   - OneCycleLR stepped per batch with short warmup (pct_start=0.1).
   - Target: 3 epochs < 24 hours total on RTX 5090.

### G. Long-Tail Balancing
1. **Square-Root Resampling:** $P \propto 1/\sqrt{N_{class}}$ — "Goldilocks" distribution.
2. **Asymmetric Loss (ASL):** Aggressively down-weights easy negatives across 7,800 classes.
3. **Logit Adjustment:** Shifts scores by $\log N_{class}$ to counteract frequency bias.
4. **Early Stopping (patience=2):** Prevents overfitting on short LoRA runs.

### H. Inference Strategy
- **SAHI Tiling:** Overlapping 512×512 patches with 20% overlap and Max Pooling aggregation.
- **Hybrid Attention:** Low-res hotspot pass skips bare soil, saving ~70% inference compute.

### I. Metric & Ecological Optimization
- **Per-Class Thresholding:** Brent's Method for all 7,800 species.
- **Temperature Scaling:** Logit calibration via learned $T$.
- **SINR + GPS Filtering:** Suppress ecologically impossible predictions.
- **Co-occurrence Smoothing:** Suppress impossible species combinations.

## 7. Expected Outcomes
- A robust, high-resolution inference engine for multi-label identification in dense vegetation.
- Demonstrated that LoRA fine-tuning of a multi-backbone ensemble matches or approaches full fine-tuning at ~1% of the parameter count and ~10% of the training time.
- A validated pipeline for rapid botanical AI development on consumer-grade GPU hardware.

## 8. Technical Stack Summary
- **Backbones:** BioCLIP (ViT-L/14), DINOv2-L (ViT-L/14), ConvNeXt-V2-L (timm + open_clip).
- **Fine-Tuning:** LoRA via PEFT library (r=16, α=32).
- **Data Pipeline:** NVIDIA DALI (full GPU I/O).
- **Training Engine:** DeepSpeed ZeRO Stage 1.
- **Precision:** BF16 autocast, TF32 matmul, FlashAttention 2, fused AdamW.
- **Metrics:** torchmetrics (GPU-resident).
- **Augmentation:** SAM, Copy-Paste Synthesis, Albumentations.
- **Experiment Tracking:** Weights & Biases (WandB).
- **Hardware:** NVIDIA RTX 5090 (32GB), optimised for Blackwell tensor cores.
