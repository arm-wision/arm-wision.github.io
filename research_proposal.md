# Research Proposal: BioCLIP & DINOv2 Ensemble for High-Resolution Multi-Label Plant Identification in Vegetation Plots

**Date:** March 18, 2026

## 1. Title
**BioCLIP, ConvNeXt-L & DINOv2 Ensemble for High-Resolution Multi-Label Plant Identification in Vegetation Plots**

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
Two major bottlenecks hinder current botanical AI:
1. **The Domain Gap:** Models trained on centered, clean, single-plant images often fail in the "messy" real-world context of overlapping vegetation.
2. **The Long-Tail Problem:** Common species are over-represented in datasets, leading to AI systems that ignore endangered or high-priority rare species.
3. **Multi-Label Ambiguity:** Standard classification losses struggle when an image contains 5–15 overlapping species, often failing to detect sub-dominant plants.

## 6. Research Design & Methodology
Our methodology follows a four-phase pipeline designed for the PlantCLEF 2026 challenge:

### A. Feature Fusion Ensemble
To achieve state-of-the-art accuracy, we employ a multi-modal feature fusion ensemble that leverages three distinct architectural inductive biases. Our implementation uses a high-capacity fusion head (LayerNorm, GELU, and Dropout) to combine features from:

1. **BioCLIP (ViT-L/14 - The Taxonomic Expert):** 
   - **Rationale:** Unlike standard vision models, BioCLIP is pre-trained on the "Tree of Life" (10M+ biological images). It provides the ensemble with an intrinsic understanding of taxonomic hierarchies and botanical relationships, which is critical for identifying rare species with limited training data.
   
2. **DINOv2 (ViT-G/14 - The Geometric Expert):** 
   - **Rationale:** A self-supervised transformer that excels at extracting high-resolution structural features. It is world-class at identifying fine-grained textures (leaf venation, serrations) and provides robust "objectness" that helps the model separate individual plants from messy, high-entropy backgrounds.

3. **ConvNeXt-V2 (Huge - The Local Context Expert):** 
   - **Rationale:** While Transformers excel at global context, Convolutional Neural Networks (CNNs) possess superior local translation invariance. ConvNeXt-V2 provides a "local" perspective that is highly robust to variations in leaf orientation, scale, and lighting, acting as a stabilizer for the Transformer-based backbones.

### B. High-Resolution Adaptation
Botanical identification often depends on minute features (e.g., trichomes, stamen structure). To support this:
- **Positional Embedding Interpolation:** We have adapted the pre-trained 224px Transformer backbones to process **448px** inputs. This is achieved via **bicubic interpolation** of the positional embeddings, allowing the models to maintain their spatial understanding while capturing 4x more pixel-level detail.

### C. GPU-Accelerated Data Engineering
To handle the scale of 1.4 million images efficiently, we have implemented a pure-GPU preprocessing pipeline:
- **Metadata Auditing:** Utilizing **cuDF** for 100x faster CSV processing and string-based file verification.
- **Blur Audit (Laplacian Variance):** Implementing a **PyTorch-based GPU filter** that scores focus for thousands of images in parallel. This ensures the training set is free from motion blur and out-of-focus samples that could degrade model performance.

### D. Long-Tail Balancing: The Calibration Strategy
Botanical data distributions are intrinsically long-tailed. To ensure our ensemble generalizes to rare and endangered species, we implement:

1. **Square-Root (Power) Resampling:** 
   - **Rationale:** Standard random sampling over-represents common species, while class-balanced sampling over-represents rare species (causing overfitting on limited samples). We utilize a sampling probability $P \propto 1/\sqrt{N_{class}}$, creating a "Goldilocks" distribution that maintains the diversity of common species while significantly increasing the exposure of the model to rare ones.
   
2. **Progressive 2-Stage Training:**
   - **Phase 1 (Representation):** The backbones are frozen, and the model is trained on the natural distribution to learn robust general features.
   - **Phase 2 (Calibration):** The backbones are unfrozen, and the model is fine-tuned using aggressive class-aware sampling. This recalibrates the decision boundaries for the long-tail without degrading the underlying feature quality.

3. **Asymmetric Loss (ASL) with Logit Adjustment:**
   - **Rationale:** We address the massive negative-positive imbalance (7,800 classes) using ASL to down-weight easy negatives. Furthermore, we apply logit adjustment (shifting predicted scores by class priors $\log N_{class}$) to mitigate the inherent bias toward frequent species.

### E. Synthetic Scene Generation

### C. Inference Strategy
To maintain the resolution of tiny botanical features across 50x50cm plots:
- **Slicing Aided Hyper Inference (SAHI):** Implementation of an overlapping tiling pipeline (512x512 patches) with **Max Pooling** aggregation.

### D. Metric & Ecological Optimization
Final refinement focuses on biological reality and the F1-score leaderboard:
- **Per-Class Thresholding:** Utilizing **Brent's Method** on validation data to optimize F1 across all species.
- **Spatially Aware Filtering:** Integrating **Spatial Implicit Neural Representations (SINR)** and GPS metadata to suppress ecologically impossible predictions.

## 7. Expected Outcomes
- A robust, high-resolution inference engine capable of multi-label identification in dense vegetation.
- Improved recall and precision for rare species in the long-tail distribution.
- A validated pipeline for synthetic-to-real domain adaptation in botanical monitoring.

## 8. Technical Stack Summary
- **Backbones:** BioCLIP, DINOv2 (via `timm` and `open_clip`).
- **Data Pipeline:** NVIDIA DALI for GPU-accelerated I/O.
- **Augmentation:** SAM, Copy-Paste Synthesis, Albumentations.
- **Hardware:** Optimization for NVIDIA RTX 4090/A100 (Mixed Precision, Batch-Parallel Inference).
