# Research Proposal: BioCLIP & DINOv2 Ensemble for High-Resolution Multi-Label Plant Identification in Vegetation Plots

**Date:** March 18, 2026

## 1. Title
**BioCLIP & DINOv2 Ensemble for High-Resolution Multi-Label Plant Identification in Vegetation Plots**

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

### A. Feature Extraction & Foundations
We employ a **Feature Fusion Ensemble** consisting of two state-of-the-art backbones:
- **BioCLIP (ViT-L/14):** A foundation model pre-aligned with the "Tree of Life," providing intrinsic understanding of taxonomic hierarchies.
- **DINOv2 (ViT-L/14):** A self-supervised model that captures robust geometric and structural features (textures, serrations) essential for fine-grained discrimination.

### B. Synthetic Scene Generation
To bridge the gap between single-plant training data and multi-label quadrats, we implement a "Quadrat Factory":
- **SAM-based Extraction:** Automated extraction of foreground plant "stickers."
- **Synthetic Collages:** Generating 500k multi-species images using Alpha Blending and **Square-Root (Power) Sampling** to ensure rare species are disproportionately represented.
- **Asymmetric Loss (ASL):** Handling the massive negative-positive imbalance typical of multi-label 7,800+ class distributions.

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
