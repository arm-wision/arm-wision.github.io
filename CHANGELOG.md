# Changelog

All notable changes to the PlantCLEF 2026 codebase after the paper was
submitted. Dates are in `YYYY-MM-DD`.

## [Unreleased] - 2026-05-29

Audit pass to bring the repository state into agreement with
`report/main.tex` and make the codebase navigable for the course
assessment. No paper numbers change; this is an alignment + cleanup
release.

### Paper-code alignment

- **Rewrite `src/inf_script_phen.py`** to implement the full
  seasonal-phenology pipeline described in Appendix C: 4x4 grid tiling
  at scales 1.0 *and* 0.8, ExG vegetation filter (drop tiles below
  15% green pixels), entropy-weighted Bayesian aggregation
  `w_t proportional to exp(-H_t) * ExG_t`, and a circular Gaussian
  day-of-year prior built from the GBIF month histograms
  (sigma = 18 d, epsilon = 0.05) combined multiplicatively in log
  space at beta = 1.0. The previous version had only a simplified
  month-bin multiplicative prior and would crash before producing
  output (`amp_autocast` arg-count bug, missing 336-px positional
  embedding resampling).
- **Replace `src/train.py`** with a thin `runpy` shim that forwards
  the command line to the actual paper training script at
  `experiments/i002_bioclip25_cap_image/train.py`. The previous
  `src/train.py` was an unrelated DALI + DeepSpeed + LoRA ensemble
  pipeline that the paper does *not* describe.
- **Resolve hardcoded `/workspace/scratch_space_arjun/...` paths**
  in `src/inf_script.py` and `src/inf_script_phen.py` to a path
  computed relative to `__file__`, so the scripts find the i002
  sources on any host that preserves the repository layout.
- **Reconstruct the missing `data/metadata_utils.py`** for the i002
  and i003 experiments. The i002 / i003 `train.py`, `dataset.py`,
  `verify_dataset_cap.py`, and `prepare_combined_manifest.py` all
  import from `data.metadata_utils`, but the file had never been
  committed; the paper training script crashed at import time. The
  reconstruction (`apply_max_images_per_species_cap`,
  `apply_max_train_rows_cap`, `load_metadata_csv`,
  `print_species_distribution`, `build_weighted_sampler`,
  `_pick_col`, `_sniff_delimiter`, and the `_*_CANDIDATES`
  constants) is driven by the import surface of the call sites and
  matches the bin-histogram format used in the existing experiment
  reports.

### Repository layout

- **New `legacy/`** directory at the repo root, with a `README.md`
  explaining what's in it and that none of it is used by the paper
  code. Contents:
  - `legacy/{config.py,data,models,training,extensions}` - the
    abandoned triple-backbone (BioCLIP + DINOv2-Large +
    ConvNeXt-V2-Large) LoRA-ensemble pipeline. Moved with `git mv`
    so commit history follows the files.
  - `legacy/{research_proposal.md,todo.md,background.txt}` - the
    initial-plan documents that describe the ensemble approach the
    paper does not use.
- **New top-level `engines/rust/`** with seven Rust crates
  (`train_resizer`, `test_resizer`, `train_packer`, `val_packer`,
  `fused_resizer_packer`, `train_indexer`, `val_indexer`) that
  implement the SIMD-parallel preprocessing layer the project
  actually used to resize and shard the 2.65 M-image manifest. Each
  crate is built on Rayon + `fast_image_resize` + the `tar` crate;
  the train resizer also exposes a C-FFI library variant. `target/`
  build artefacts are excluded; `Cargo.lock` files are kept where
  they existed.
- **New top-level `coordinator/`** with the multi-node training
  control plane that the project used for the parallel-pod
  experiments behind the paper's appendix (within-backbone
  saturation diagnostics, the i003 GBIF tail-augmentation arm,
  the cross-checkpoint ensembling sweep). Contents:
  - `coordinator/python/cluster/` - typed cluster-manifest module
    (`Manifest`, `HostEntry`, `TrainingTask`, `load_manifest`); one
    third-party dep (`pyyaml`).
  - `coordinator/go/` - Go control plane (no external deps): TCP
    master/worker NK/GO/HB handshake (`main.go`), UDP heartbeat
    receiver + telemetry HTTP API (`telemetry.go`), and standalone
    `expert/` and `inf/` launchers. All three Go packages build
    clean under `go 1.22`.
  - `coordinator/configs/cluster.example.yaml` - multi-host
    topology example.
  - Environment variables use a `CLUSTER_*` prefix and the
    project-specific CLI path is lifted to a `CLUSTER_CLI_PATH`
    env var. The original launcher script and Python CLI are not
    ported because they route through phase modules that do not
    exist in this repo; the coordinator's README documents a
    `torchrun` drop-in.
- **New `CHANGELOG.md`** (this file).
- **New `BACKLOG.md`** documenting the future-work directions
  (saturated / high-priority / frontier) that would have shipped
  with more time.
- **Trim `requirements.txt`** to the dependencies actually needed by
  the i002 paper pipeline (`torch`, `torchvision`, `open_clip_torch`,
  `pandas`, `numpy`, `Pillow`, `tqdm`, `python-dotenv`, `wandb`).
  Dropped the legacy stack (`deepspeed`, `peft`, `nvidia-dali-cuda120`,
  `cudf-cu12`, `cupy-cuda12x`, `albumentations`, `opencv-python`,
  `torchmetrics`, `accelerate`, `timm`, `scipy`, `scikit-learn`,
  `segment-anything`, `sahi`); they live in `legacy/` now.
- **Rewrite top-level `README.md`** to describe the i002 paper
  system: layout map, final-system spec (backbone, heads, schedule,
  optimiser, augmentations, inference recipe), two-stage `torchrun`
  invocation, anchor and phenology inference commands, and notes for
  assessors on which directories carry paper code vs. development
  trace.
- **Delete junk files**: empty `0.5` placeholder, scratch `PROGRESS.md`.

### Verified, no changes

The following pieces of the paper code were audited and confirmed to
already match the report. No edits required.

- `experiments/i002_bioclip25_cap_image/{model,dataset,train,
  transforms,utils}.py`: per-head MLPs (LayerNorm -> Linear(1024 ->
  1024) -> GELU -> Dropout 0.2), last-N transformer blocks plus
  `ln_post` + `proj` unfreeze, auxiliary loss weights 0.30 (genus) +
  0.15 (family), bf16 AdamW with head LR 1e-4 and backbone LR 1e-6,
  one-epoch linear warmup then cosine decay to 1% of peak, stratified
  10% validation split (seed 42; species with fewer than five images
  stay in train), augmentation pipeline.
- `experiments/i001_data_download/final_data.ipynb`: confirmed
  the 1,408,033 + 1,245,748 = 2,653,781 row, 7,806-species manifest
  counts reported in Table 1.
- `experiments/i002_*/scores_*.csv`: confirmed every public-Macro-F1
  number quoted in Tables 4-6 (tau x T sweep, fixed top-k, gap /
  relative-threshold selection rules) traces directly to a Kaggle
  submission archived in this repo.
- `src/inf_script.py` recipe constants (4x4 grid, 448-px tiles,
  224 + 336 dual resolution, softmax-mean, logit adjustment
  tau = 0.25, adaptive threshold T = 0.03, k in [2, 10]).
