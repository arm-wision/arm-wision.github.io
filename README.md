# PlantCLEF 2026 - ANU Submission

Code accompanying the working note *"Fine-Tuning of BioCLIP 2.5 with
Taxonomic Heads for Multi-Species Plant Identification"* (CLEF 2026).
Public Macro F1 of the headline submission: **0.41826**; corresponding
private score **0.40283**. Best private of the project: **0.40600**
(logit-adjusted single-resolution variant). Final standing: 7th place
private.

The report lives in [`report/main.tex`](report/main.tex).

---

## Repository layout

```
.
├── report/                            LaTeX source of the working note
│   ├── main.tex
│   └── sections/                      One file per section + appendices
├── src/
│   ├── train.py                       Thin shim -> i002 paper training
│   ├── inf_script.py                  Headline inference (0.41826 recipe)
│   └── inf_script_phen.py             Pivot 3: seasonal phenology prior
├── experiments/
│   ├── i001_data_download/            Data rebuild (PlantCLEF24 + iNat)
│   ├── i002_bioclip25_cap_image/      Paper model: per-head MLPs, no cap
│   ├── i003_bioclip25_cap_image_extra500/   Per-species cap study
│   ├── 001-015_*                      Earlier experiments + baselines
│   └── 010_outputs/                   Intermediate fusion outputs
├── engines/
│   └── rust/                          SIMD-parallel resize + WebDataset
│                                      pack + DALI index crates
├── coordinator/                       Multi-node training control plane
│   ├── python/cluster/                Cluster-manifest module + TrainingTask
│   ├── go/                            Go hub: TCP handshake + telemetry
│   └── configs/                       cluster.example.yaml topology
├── legacy/                            Abandoned ensemble-pipeline code
│                                      + early planning docs
├── eda/                               Exploratory data analysis
├── docs/                              Submission archive (Kaggle API)
├── scripts/                           Submission helpers
└── tools/                             Misc tooling
```

The numbered directories under `experiments/` document every
direction explored over the project (zero-shot baselines, the early
DINOv3 / ConvNeXt lines, the SimSiam SSL experiment, etc.). The paper
appendix reproduces the same map with exploration-depth labels.

---

## Final system (i002)

The paper's headline configuration is a single BioCLIP 2.5 ViT-H/14
backbone with per-head MLPs for species, genus and family.

* **Backbone**: BioCLIP 2.5 ViT-H/14 (`hf-hub:imageomics/bioclip-2.5-vith14`).
  The lower 28 transformer blocks stay frozen to preserve the
  Tree-of-Life prior; only the last 4 blocks plus the final layer norm
  and projection are unfrozen.
* **Heads**: three independent MLPs of the form
  `LayerNorm -> Linear(1024 -> 1024) -> GELU -> Dropout(0.2)`,
  feeding linear classifiers of sizes 7,806 / 1,446 / 181 for species /
  genus / family.
* **Loss**: weighted joint cross-entropy with label smoothing 0.1,
  weights `1.0 * L_species + 0.30 * L_genus + 0.15 * L_family`;
  missing taxonomy labels encoded as `-1` and masked.
* **Training data**: the i001 manifest, 2,653,781 single-plant images
  across 7,806 species (PlantCLEF 2024 + a research-grade iNaturalist
  pull, deduplicated, genus / family pre-filled). No per-species cap.
* **Schedule**: two stages of ten epochs each. Stage 1 trains the head
  MLPs and classifiers with the backbone fully frozen; Stage 2 resumes
  the weights and additionally unfreezes the last 4 transformer blocks
  + `ln_post` + `proj`.
* **Optimiser**: AdamW (head LR `1e-4`, backbone LR `1e-6`, weight decay
  `1e-4`), one epoch of linear warmup then cosine decay to 1% of peak,
  global-norm gradient clip 1.0.
* **Precision**: bfloat16 AMP (no GradScaler), DDP across 2x RTX 5090
  via `torchrun --nproc_per_node=2`.
* **Inference**: each quadrat is partitioned into a 4x4 grid of 16
  tiles; every tile is forwarded through the encoder at both 224 and
  336 pixels (the ViT-H/14 pos-embed is bicubically resampled for the
  336 px pass). Per-tile softmax probabilities are averaged across
  tiles and across the two resolutions, class-prior logit adjustment
  with `tau = 0.25` is applied against the Laplace-smoothed training
  prior, and every species with post-adjustment probability above
  `T = 0.03` is emitted, clamped to `[k_min = 2, k_max = 10]`.

Full hyperparameter, augmentation, and split specification:
[report Appendix B (Table 7)](report/sections/appendix_development_trace.tex).

---

## Quick start

### Environment

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `open_clip_torch` (BioCLIP 2.5 weights),
`pandas`, `Pillow`, `torchvision`.

### Train the paper model

`src/train.py` is a thin shim that forwards to the i002 training
script. The two stages from the paper:

```bash
# Stage 1: head only, 10 epochs, backbone frozen
torchrun --nproc_per_node=2 src/train.py \
    --metadata-csv  path/to/metadata_filled_genus_family.csv \
    --train-image-root path/to/images \
    --epochs 10 --batch-size 512 --grad-accum-steps 2 \
    --precision bf16 --freeze-backbone \
    --head-lr 1e-4 --weight-decay 1e-4 \
    --output-dir outputs/stage1_head_only

# Stage 2: resume + unfreeze last 4 blocks, 10 epochs
torchrun --nproc_per_node=2 src/train.py \
    --metadata-csv  path/to/metadata_filled_genus_family.csv \
    --train-image-root path/to/images \
    --epochs 10 --batch-size 128 --grad-accum-steps 4 \
    --precision bf16 --unfreeze-last-n-blocks 4 \
    --backbone-lr 1e-6 --head-lr 1e-4 --weight-decay 1e-4 \
    --resume outputs/stage1_head_only/checkpoints/best.pt \
    --resume-weights-only \
    --output-dir outputs/stage2_last4_blocks
```

Pre-baked launcher scripts for each stage live under
`experiments/i002_bioclip25_cap_image/scripts/`.

### Reproduce the headline submission

```bash
python src/inf_script.py \
    --checkpoint    outputs/stage2_last4_blocks/checkpoints/best.pt \
    --image-dir     path/to/plantclef/test \
    --metadata-csv  path/to/metadata_filled_genus_family.csv \
    --output        submission.csv
```

The script is single-purpose: it implements the fixed 4x4 grid +
224 + 336 + logit adjustment + adaptive-threshold recipe that scored
0.41826 public on the leaderboard.

### Phenology pivot (paper Pivot 3)

```bash
python src/inf_script_phen.py \
    --checkpoint    outputs/stage2_last4_blocks/checkpoints/best.pt \
    --image-dir     path/to/plantclef/test \
    --metadata-csv  path/to/metadata_filled_genus_family.csv \
    --phenology-csv path/to/gbif_month_histograms.csv \
    --output        submission_phen.csv
```

This adds the four phenology-specific stages from Appendix C: a 4x4
grid taken at scales 1.0 and 0.8, an ExG vegetation filter that drops
tiles with under 15% green pixels, entropy-weighted Bayesian
aggregation `w_t proportional to exp(-H_t) * ExG_t`, and a multiplicative
circular-Gaussian day-of-year prior built from the GBIF month
histograms (`sigma = 18` days, `epsilon = 0.05`, `beta = 1.0`).

---

## Notes for assessors

* `src/train.py`, `src/inf_script.py`, and `src/inf_script_phen.py`
  are the three entry points that correspond to the paper. They all
  delegate (directly or by import) to
  `experiments/i002_bioclip25_cap_image/`.
* The numbered `experiments/0XX_*` directories contain the older
  directions documented in the appendix (zero-shot baselines, DINOv3,
  SimSiam, etc.). They are kept for traceability of the development
  trace, not because they are part of the final pipeline.
* `outputs/` and the various `scores_*.csv` files under
  `experiments/i002_*` contain the raw Kaggle leaderboard scores
  underlying every number quoted in the paper.

---

## Citation

```bibtex
@inproceedings{anu-plantclef2026,
  title  = {Fine-Tuning of BioCLIP 2.5 with Taxonomic Heads for Multi-Species Plant Identification},
  author = {Raj, Arjun and de Mel, Manindra and Wasif, Razeen and Brake, William},
  booktitle = {CLEF 2026 Working Notes},
  year   = {2026},
}
```
