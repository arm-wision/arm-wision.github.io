# PlantCLEF 2026 - Master TODO & Roadmap

## Phase 1: Foundations & Training
- [x] **Triple Ensemble:** Integrated BioCLIP, DINOv2, and ConvNeXt-V2 backbones.
- [x] **Configurable Backbones:** Added "4090" and stronger GPU hardware modes to `train.py`.
- [x] **Modular Architecture:** Organized `src/` into `models/` and `data/` subdirectories.
- [x] **Mixed Precision (AMP):** Optimized training loop for RTX 4090 using `torch.amp`.
- [x] **High-Res Support:** Implemented positional embedding 
        interpolation for 448px resolution.
- [x] **Modular Training Loop:** Integrated DALI, WandB, and Early Stopping.
- [x] **Inference Pipeline:** Updated SAHI manual tiling to support the new ensemble.
- [ ] **Environment Setup:** Run `bash Infastructure/scripts/setup_environment.sh` 
        on the target machine.
- [ ] **Initial Preprocessing:** Run `python3 src/data/preprocess.py` to 
        generate the cleaned metadata.
    - [x] **GPU-Accelerated Blur Filtering:** Implemented PyTorch-based 
          Laplacian Variance scoring for 1.4M images.
    - [x] **cuDF-Optimized Metadata Cleaning:** 100x speedup in CSV 
          processing using GPU-accelerated dataframes.
- [ ] **Baseline Training:** Run `python3 src/train.py` with the BioCLIP+DINOv2 ensemble.
    - *Goal:* Establish a world-class single-plant classifier using biological 
        foundation models.
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
    - [x] **Square-Root (Power) Sampling:** Implement a weighted file-list generator for DALI to boost rare species visibility ($P \propto 1/\sqrt{N}$).
    - [x] **Asymmetric Loss (ASL) Optimization:** Refine the existing ASL implementation to handle the 7,800-class negative imbalance more effectively.
    - [x] **Logit Adjustment:** Implement post-hoc or loss-level logit shifting based on class priors ($N_{class}$).
    - [x] **Progressive 2-Stage Balancing:** 
        - Phase 1: Backbone-frozen, natural distribution (Representation learning).
        - Phase 2: Backbone-unfrozen, class-aware sampling, lower LR (Calibration).
    - [ ] Explore **Label Propagation** to ensure ground truth accuracy in crowded collages.

## Phase 3: Domain Adaptation & High-Res Tiling
- [ ] **Pseudo-Labeling:**
    - Run Phase 2 model on unlabeled LUCAS quadrats (Teacher-Student loop).
    - Add top 20% high-confidence predictions back into the training set.
- [ ] **SAHI Tiling Engine:**
    - Refine the tiling pipeline for parallel inference (Batch processing of tiles).
    - [ ] Optimize `src/predict_sahi.py` for GPU batching.
    - [ ] **Overlapping Tiles:** Ensure 20% overlap to prevent 
        "split-plant" misses (Challenge 2).
    - [ ] **Hybrid Attention (Zoom-in):** Implement a low-res "hotspot" 
        pass to ignore bare soil and save 70% compute.
- [ ] **Distribution Matching:** Use LUCAS unlabeled data to estimate 
        **species priors** and adjust training weights.
- [ ] **Quality/Blur Filtering:**
    - Use Laplacian Variance to score and remove out-of-focus images 
        from the expanded dataset.

## Phase 4: Metric Optimization & Ecological Guardrails
- [ ] **Ensemble Weighting:**
    - Instead of standard averaging, apply differentiable optimization techniques 
        to learn the exact optimal blending weights.
- [ ] **Metadata Integration:**
    - Leverage structured self-attention to map correlations between visual 
        traits and metadata (geolocation, altitude, climate).
- [ ] **Threshold Optimization (Challenge 4):**
    - Use **Brent's Method** on the validation set to find **Per-Class Thresholds** 
        for all 7,800 species.
    - [ ] **Frequency-Based Adjustment:** Scale thresholds based on 
        training sample counts ($N_{class}$).
    - [ ] **Temperature Scaling:** Calibrate probabilities via logit scaling 
        ($T$) on validation data.
    - [ ] **Empty Plot Logic:** Handle cases where no species exceed 
        threshold by predicting "nothing."
- [ ] **Ecological/Botanical Filtering:**
    - Implement **Spatial Implicit Neural Representations (SINR)** or 
        GIS-based lookups for geographic constraints.
    - [ ] Integrate GPS metadata into the prediction pipeline.
- [ ] **Taxonomic & Co-occurrence Smoothing:**
    - Use a co-occurrence matrix to suppress "impossible" species combinations.
- [ ] **Final Ensembling:**
    - Explore "Model Soup" (weight averaging) or multi-model averaging for final F1 boost.
- [ ] **Performance Optimization:** 
    - Revert `src/data/dataloader.py` to **Mixed GPU Decoding** (CPU header parsing + GPU bitstream decoding) once NVML driver issues are resolved.

---
*Last updated: March 18, 2026*
