# PlantCLEF 2026 - Master TODO & Roadmap

## Phase 1: Foundations & Training
- [x] **Triple Ensemble:** Integrated BioCLIP, DINOv2, and ConvNeXt-V2 backbones.
- [x] **Configurable Backbones:** Added "4090" and stronger GPU hardware modes to `train.py`.
- [x] **Modular Architecture:** Organized `src/` into `models/` and `data/` subdirectories.
- [x] **Mixed Precision (AMP):** Optimized training loop for RTX 4090 using `torch.amp`.
- [x] **High-Res Support:** Implemented positional embedding interpolation for 448px resolution.
- [x] **Modular Training Loop:** Integrated DALI, WandB, and Early Stopping.
- [x] **Inference Pipeline:** Updated SAHI manual tiling to support the new ensemble.
- [ ] **Environment Setup:** Run `bash Infastructure/scripts/setup_environment.sh` on the target machine.
- [ ] **Initial Preprocessing:** Run `python3 src/data/preprocess.py` to generate the cleaned metadata.
- [ ] **Baseline Training:** Run `python3 src/train.py` with the BioCLIP+DINOv2 ensemble.
    - *Goal:* Establish a world-class single-plant classifier using biological foundation models.

## Phase 2: Synthetic Complexity & The Long-Tail
- [ ] **"Plant Sticker" Factory (SAM):**
    - Use SAM (Segment Anything Model) to automate the extraction of plants from the 1.4M images.
    - [ ] Create `src/data/sam_extractor.py`.
- [ ] **Synthetic Collage Generation:**
    - Generate 500k "Synthetic Collages" using the Copy-Paste method.
    - Implement **Alpha Blending** for edge smoothing.
    - [ ] Create `src/data/collage_generator.py`.
- [ ] **Long-Tail Strategies:**
    - Implement **Square-Root (Power) Sampling** to boost rare species visibility.
    - [ ] Integrate **Asymmetric Loss (ASL)** to handle multi-label imbalance (implemented in `train.py`, needs verification on collages).
    - [ ] Explore **Label Propagation** to ensure ground truth accuracy in crowded collages.

## Phase 3: Domain Adaptation & High-Res Tiling
- [ ] **Pseudo-Labeling:**
    - Run Phase 2 model on unlabeled LUCAS quadrats.
    - Add top 20% high-confidence predictions back into the training set.
- [ ] **SAHI Tiling Engine:**
    - Refine the tiling pipeline for parallel inference (Batch processing of tiles).
    - [ ] Optimize `src/predict_sahi.py` for GPU batching.
- [ ] **Quality/Blur Filtering:**
    - Use Laplacian Variance to score and remove out-of-focus images from the expanded dataset.

## Phase 4: Metric Optimization & Ecological Guardrails
- [ ] **Per-Class Thresholding:**
    - Use **Brent's Method** on the validation set to find optimal thresholds for all 7,800 species.
- [ ] **Ecological/Botanical Filtering:**
    - Implement **Spatial Implicit Neural Representations (SINR)** or GIS-based lookups for geographic constraints.
    - [ ] Integrate GPS metadata into the prediction pipeline.
- [ ] **Taxonomic & Co-occurrence Smoothing:**
    - Use a co-occurrence matrix to suppress "impossible" species combinations.
- [ ] **Final Ensembling:**
    - Explore "Model Soup" (weight averaging) or multi-model averaging for final F1 boost.

---
*Last updated: March 18, 2026*
