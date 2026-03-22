# Research Proposal: BioCLIP & DINOv2 Ensemble for High-Resolution Multi-Label Plant Identification in Vegetation Plots

**Date:** March 22, 2026

## 1. Title
**BioCLIP, ConvNeXt-V2 & DINOv2 Ensemble for High-Resolution Multi-Label Plant Identification in Vegetation Plots**

## 2. Research Questions
1. How can domain-specific foundation models (BioCLIP) be adapted to identify rare species in complex, overlapping vegetation plots where visual occlusion and a long-tailed species distribution typically degrade performance?
2. Does taxonomic-aware pre-training (BioCLIP) provide superior feature discrimination for morphologically similar species compared to domain-agnostic self-supervised models (DINOv2) in high-density vegetation plots?
3. To what extent do Segment Anything Model (SAM)-derived synthetic multi-species collages reduce the need for pre-labeled real-world data in automated biodiversity monitoring?
4. Does high-resolution tiling (SAHI) disproportionately benefit the recall of rare, small-stature plants in 50x50cm vegetation quadrats?

## 3. National Interest Statement
Monitoring plant biodiversity is a matter of critical national security regarding climate resilience, agriculture, and ecosystem services. Current manual surveying methods are slow, subjective, and cost-prohibitive. This research aims to automate large-scale botanical surveys, providing governmental agencies and environmental organizations with the tools to respond instantly to invasive species threats, track habitat loss, and monitor the health of national ecosystems.

## 4. Introduction
Identifying plants in the wild is traditionally a "single-focus" task. However, ecological reality consists of complex "vegetation plots" (quadrats) where multiple species overlap, compete, and obscure one another. This project seeks to build a high-performance AI system capable of "dissecting" these plots to identify every species present, utilizing the latest advancements in biological foundation models and synthetic data generation.

## 5. Research Problem
Three major bottlenecks hinder current botanical AI:
1. **The Domain Gap:** Models trained on centered, clean, single-plant images often fail in the "messy" real-world context of overlapping vegetation.
2. **The Long-Tail Problem:** Common species are over-represented in datasets, leading to AI systems that ignore endangered or high-priority rare species.
3. **Multi-Label Ambiguity:** Standard classification losses struggle when an image contains 5–15 overlapping species, often failing to detect sub-dominant plants.

## 6. Research Design & Methodology
Our methodology follows a four-phase pipeline designed for the PlantCLEF 2026 challenge:

### A. Feature Fusion Ensemble
To achieve state-of-the-art accuracy, we employ a multi-modal feature fusion ensemble that leverages three distinct architectural inductive biases. Our implementation uses a high-capacity fusion head (LayerNorm, GELU, and Dropout) to combine features from:

1. **BioCLIP (ViT-L/14 — The Taxonomic Expert):**
   - **Rationale:** Pre-trained on the "Tree of Life" (10M+ biological images), BioCLIP provides an intrinsic understanding of taxonomic hierarchies and botanical relationships critical for identifying rare species with limited training data.

2. **DINOv2 (ViT-L/14 — The Geometric Expert):**
   - **Rationale:** A self-supervised transformer excelling at fine-grained structural features (leaf venation, serrations) with robust "objectness" that separates individual plants from cluttered backgrounds.

3. **ConvNeXt-V2 (Large — The Local Context Expert):**
   - **Rationale:** Convolutional inductive biases provide superior local translation invariance, robust to variations in leaf orientation, scale, and lighting — acting as a stabilizer for the Transformer-based backbones.

Each backbone's output is projected to a shared 512-dimensional space via a `Linear + LayerNorm` projection head, L2-normalised, then concatenated into a 1536-dimensional fused representation. This equalises each backbone's contribution regardless of raw feature scale.

### B. High-Resolution Adaptation
- **Positional Embedding Interpolation:** Pre-trained 224px Transformer backbones adapted to **448px** via bicubic interpolation of positional embeddings, capturing 4× more pixel-level detail.
- **Channels-Last Memory Format:** Images and model weights stored in NHWC layout (`torch.channels_last`), aligning with NVIDIA tensor core preferences for improved ConvNeXt throughput.

### C. GPU-Accelerated Data Engineering
- **NVIDIA DALI Pipeline:** Full GPU-resident I/O — JPEG decoding, resizing, and colour conversion via `fn.decoders.image(device='mixed')`, eliminating CPU bottlenecks entirely.
- **cuDF Metadata Auditing:** 100× faster CSV processing and file verification on GPU.
- **Blur Audit:** DALI + PyTorch Laplacian Variance GPU filter removes 15,500 blurry samples (~1.1%) from 1.4M images before training.

### D. Hardware Precision & Throughput Optimisations
We exploit the full capability of the RTX 5090's Blackwell tensor cores:

- **BF16 Mixed Precision:** `bfloat16` autocast throughout. BF16 has the same dynamic range as FP32, eliminating gradient underflow and the need for `GradScaler` while halving activation memory vs FP32. ~15–25% throughput improvement over FP16.
- **TF32 Matmul Acceleration:** `allow_tf32` flags enabled for ~2× matmul throughput with negligible accuracy impact.
- **FlashAttention 2:** Enabled via `torch.backends.cuda.enable_flash_sdp(True)`. BioCLIP and DINOv2 use `scaled_dot_product_attention` internally, automatically benefiting from O(N) memory attention and fused kernel execution on the 5090.
- **Fused AdamW:** `fused=True` executes all parameter updates in a single CUDA kernel (~10–15% optimizer step speedup).
- **GPU-Accelerated Metrics:** All F1, Precision, Recall, and Accuracy use `torchmetrics` with GPU-resident state accumulation and a single CPU transfer per epoch at `.compute()`.

### E. Single-GPU Batch Size Maximisation
Training a triple-backbone ensemble on 1.4M images at 448px on a single GPU presents a fundamental VRAM constraint. We apply a layered strategy to maximise effective batch size:

1. **Expandable Segments Allocator:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set before PyTorch initialises its allocator, reducing memory fragmentation by ~2–3GB and recovering headroom for larger batches at no computational cost.

2. **DeepSpeed ZeRO Stage 1:** Adam optimizer states (mean and variance tensors) consume approximately 2× the model parameter footprint in VRAM. With three large backbones, this amounts to ~8–12GB. DeepSpeed ZeRO Stage 1 partitions and offloads these optimizer states to pinned CPU RAM via PCIe 5.0, freeing ~4–8GB of VRAM during training. This is the single largest batch size lever available on a single GPU.

3. **Chunked Backbone Forward:** Rather than passing the full batch through each backbone at once, images are processed in sub-chunks of size 8. This caps peak activation VRAM at 8-image activations regardless of the logical batch size, while the gradient graph still accumulates over the full batch. Combined with gradient accumulation, effective batch sizes of 64–128 become achievable without OOM.

4. **Gradient Checkpointing:** Enabled on all transformer blocks in BioCLIP and DINOv2 (`set_grad_checkpointing(True)`), trading ~30% additional compute for ~40% reduction in stored activation memory during the backward pass.

Together, these techniques shift the batch size ceiling from ~16 (naive) to 64+ images per step on a single RTX 5090.

### F. Long-Tail Balancing: The Calibration Strategy

1. **Square-Root (Power) Resampling:** Sampling probability $P \propto 1/\sqrt{N_{class}}$ creates a "Goldilocks" distribution — neither dominated by common species nor overfit to rare ones.

2. **Progressive 2-Stage Training with Feature Caching:**
   - **Phase 1 (Head Warmup):** Backbones frozen. Features extracted once via chunked forward and cached to disk. All 10 Phase 1 epochs train only the projection heads and classifier on cached tensors — approximately **10× faster** than naive frozen-backbone training.
   - **Phase 2 (Calibration):** All weights unfrozen. Differential LR: backbones at 5e-6, heads at 2e-4. Square-root resampling active.

3. **OneCycleLR Scheduler:** Stepped per batch with a short warmup and long cosine decay, converging ~30% faster than per-epoch CosineAnnealingLR.

4. **Asymmetric Loss (ASL) with Logit Adjustment:** ASL down-weights easy negatives across 7,800 classes. Logit adjustment shifts scores by $\log N_{class}$ to counteract frequency bias at the loss level.

### G. Synthetic Scene Generation
- SAM-based "Plant Sticker Factory" extracts individual instances from 1.4M training images.
- 500k synthetic multi-species collages generated via Copy-Paste with alpha blending, Z-ordering, and global augmentation (colour cast, shadow) for domain realism.

### H. Inference Strategy
- **SAHI Tiling:** Overlapping 512×512 patches with 20% overlap and Max Pooling aggregation.
- **Hybrid Attention:** Low-res "hotspot" pass to skip bare soil regions and save ~70% compute.

### I. Metric & Ecological Optimization
- **Per-Class Thresholding:** Brent's Method on validation data for all 7,800 species.
- **Temperature Scaling:** Learned temperature parameter $T$ on validation data.
- **SINR + GPS Filtering:** Suppress ecologically impossible predictions geographically.
- **Co-occurrence Smoothing:** Suppress impossible species combinations taxonomically.

## 7. Expected Outcomes
- A robust, high-resolution inference engine for multi-label identification in dense vegetation.
- Improved recall and precision for rare species in the long-tail distribution.
- A validated pipeline for synthetic-to-real domain adaptation in botanical monitoring.
- Demonstrated techniques for large-scale multi-backbone ensemble training on a single consumer GPU, including DeepSpeed ZeRO offloading and chunked backbone forward passes as practical solutions to VRAM constraints.

## 8. Technical Stack Summary
- **Backbones:** BioCLIP (ViT-L/14), DINOv2 (ViT-L/14), ConvNeXt-V2-Large (via `timm` and `open_clip`).
- **Data Pipeline:** NVIDIA DALI (full GPU I/O).
- **Training Engine:** DeepSpeed ZeRO Stage 1 (optimizer state offload).
- **Precision:** BF16 autocast, TF32 matmul, FlashAttention 2, fused AdamW.
- **Metrics:** torchmetrics (GPU-resident accumulation).
- **Augmentation:** SAM, Copy-Paste Synthesis, Albumentations.
- **Experiment Tracking:** Weights & Biases (WandB).
- **Hardware:** NVIDIA RTX 5090 (32GB), optimised for Blackwell tensor cores.
