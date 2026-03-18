# PlantCLEF 2026 - Project TODO

## 🚀 Active/Immediate Tasks
- [ ] **Environment Setup:** Run `bash Infastructure/scripts/setup_environment.sh` on the target machine.
- [ ] **Initial Preprocessing:** Run `python3 src/preprocess.py` to generate the cleaned metadata.
- [ ] **Baseline Training:** Run `python3 src/train.py` with the BioCLIP+DINOv2 ensemble.

## 🧪 Advanced Preprocessing (Future Exploration)
- [ ] **Quality/Blur Filtering:**
    - Use Laplacian Variance to score and remove out-of-focus images.
    - Prevents noisy gradients from low-quality data.
- [ ] **Smart Balancing (Long-Tail):**
    - Implement Weighted Oversampling for rare species instead of just capping common ones.
    - Ensure the model sees the "tail" of the distribution more frequently.
- [ ] **Botanical-Specific Augmentations:**
    - Integrate `albumentations` for `RandomSunFlare`, `RandomFog`, and `FancyPCA`.
    - Simulate varied field conditions (forest moisture, harsh sun).
- [ ] **"Plant Sticker" Factory (Background Removal):**
    - Use SAM (Segment Anything Model) to mask backgrounds.
    - Focus features strictly on botanical structures.

## ✅ Completed Tasks
- [x] **Modular Architecture:** Organized `src/` into `models/` and `data/` subdirectories.
- [x] **Model Components:** Created `bioclip.py`, `dinov2.py`, and `ensemble.py`.
- [x] **Feature Fusion Ensemble:** Created a dual-backbone head (BioCLIP + DINOv2).
- [x] **Mixed Precision (AMP):** Optimized training loop for RTX 4090 using `torch.amp`.
- [x] **High-Res Support:** Implemented positional embedding interpolation for 448px resolution.
- [x] **Modular Training Loop:** Integrated DALI, WandB, and Early Stopping.
- [x] **Inference Pipeline:** Updated SAHI manual tiling to support the new ensemble.

---
*Last updated: March 18, 2026*
