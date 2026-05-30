# Backlog: future novelty directions

This file is a working-note record of the research directions we
scoped, prototyped, or deferred for a future iteration of the
PlantCLEF system. Nothing here is implemented in the codebase that
backs `report/main.tex`; the paper's "Considered Approaches Not
Implemented" appendix and Section 6 (Discussion) discuss most of these
at a higher level.

The three buckets below:

* **Saturated** - in the paper as a shipped feature.
* **Future work (high priority)** - directions with a concrete design
  or partial prototype that would have shipped with more time.
* **Unsaturated frontier** - directions we have a hypothesis for but
  no implementation or evidence yet.

---

## Saturated: in the paper's final system

The following appear in `src/inf_script.py`, `src/inf_script_phen.py`,
and the i002 training code, and are documented in the paper:

* **Partial-unfreeze BioCLIP 2.5 with taxonomic heads.** Last four
  transformer blocks plus `ln_post`/`proj` unfrozen on the i001
  manifest, with per-head MLPs for species / genus / family
  (Section 3 of the report).
* **Tiled inference with adaptive selection.** 4x4 grid, 224 + 336
  dual-resolution single-checkpoint ensemble, softmax-mean
  aggregation, class-prior logit adjustment (tau = 0.25), adaptive
  probability threshold (T = 0.03, k in [2, 10]) (Section 3.4-3.7).
* **Seasonal phenology prior (pivot 3).** Circular-Gaussian DOY pdf
  built from GBIF month histograms, multiplied into the visual
  posterior in log space at beta = 1.0; ExG vegetation filter and
  entropy-weighted Bayesian aggregation on a multi-scale (1.0 + 0.8)
  tiling (Appendix C).
* **Triple-backbone classifier retraining (cRT, pivot 1).** Frozen
  3328-d concatenation of BioCLIP 2.5 + DINOv3 + ConvNeXt-V2-L
  features with two MLP heads; reported as an unsuccessful pivot.
* **Genus/family co-occurrence reranking (pivot 2).** Sibling-prior
  built from the anchor posterior's own aggregated genus/family
  supports; reported as a near-neutral pivot.

---

## Future work (high priority)

Directions with a concrete design that we would carry into the next
iteration first.

### 1. Asymmetric Dual-Teacher Distillation (AD-TD)

Reach triple-backbone Macro-F1 without the inference-time cost of
running three encoders.

* **Teachers**: the i002 partially-unfrozen BioCLIP 2.5 and a similarly
  fine-tuned DINOv3 (the experiment 008 line in the appendix).
* **Student**: a single DeiT (Data-efficient Image Transformer) with
  two dedicated distillation tokens - one supervised by the BioCLIP
  posterior (biological / taxonomic reasoning), one supervised by the
  DINOv3 dense features (spatial precision and segmentation cues).
* **Why now**: our within-backbone saturation diagnostic
  (Section 5.3, Jaccard 0.923 between intra-BioCLIP variants) shows
  headroom does not live inside a single backbone family; AD-TD is a
  concrete way to combine two backbones at training time only.

### 2. Spectral Vision: Global Filter Networks (GFNet)

Replace `O(N^2)` self-attention in the encoder with `O(N log N)`
2-D FFT-based token mixing. The diagnostic case for this:

* Quadrats are mostly low-frequency context (soil, litter, shadow)
  with a few high-frequency diagnostic features (leaf serrations,
  petal venation). Frequency-domain filters can learn to suppress
  the former and amplify the latter directly.
* Memory linear in N enables training at 448-px or larger without
  the gradient-checkpointing burden of ViT-H/14.

### 3. Agentic LLM arbiter (Visual Chain-of-Thought)

For high-entropy quadrats where the visual posterior is uncertain
between morphologically similar species, route a small set of
candidate species + image patches to a small local LLM
(Gemma 3 / Nemotron class) for a tie-break step.

* **Trigger**: predictive entropy above a calibrated threshold, or
  Jaccard between the visual top-K and the phenology-reweighted
  top-K below a threshold.
* **Inputs**: the image, the per-tile attention maps, the
  candidate-species shortlist with their GBIF habitat / pH / GDD
  ranges, and the observation date.
* **Output**: a veto vote on each candidate (eliminate as
  ecologically implausible) and an optional reranking.

### 4. Ecological co-occurrence prior (Bayesian engine)

The genus / family rerank we shipped (pivot 2) used the model's *own*
posterior to build the co-occurrence prior, which makes the prior
statistically dependent on the visual posterior. A genuine
co-occurrence prior built from external community-survey data
(e.g. LUCAS, EVA, GBIF Plot summaries) would be information-theoretically
orthogonal in a stronger sense.

### 5. Threshold optimisation per class

Instead of the single `T = 0.03` we sweep, fit a per-class threshold
T_s via Brent's method on a held-out validation slice. This is
expected to help mid-frequency species where the long-tail logit
adjustment over-corrects.

---

## Unsaturated frontier

Directions with a hypothesis but no implementation yet.

### Ecological physics and logic priors

* **Ecological Pauli exclusion**. Add a repulsion term to a GCN head
  on the predicted species set, penalising sets that contain two
  species occupying the same exact ecological niche in a 1 m^2
  quadrat. Trained from Ellenberg indicator values + Grime CSR
  strategy labels.
* **Ising / spin-glass refinement**. Model the per-quadrat species
  set as a spin system whose interactions encode pairwise
  co-occurrence statistics; refine the visual posterior by
  simulated annealing toward the ground-state composition.
* **Allelopathic phenology**. Use known chemical-warfare relations
  between species (e.g. juglone from Juglans suppressing many
  understorey species) as a multiplicative suppression term on
  detection probability.

### Environmental and edaphic intelligence

* **SoilGrids-driven masking**. Veto species whose pH / cation
  exchange capacity / clay-content tolerance ranges (per the global
  SoilGrids product) do not include the quadrat's coordinates.
* **Ellenberg indicator refinement**. Apply the standardised
  Ellenberg L / T / F / N / R values as soft posteriors over the
  shortlist before threshold selection.
* **PageRank on ecological networks**. Identify "keystone" species
  in a quadrat by running PageRank on a co-occurrence graph, then
  propagate confidence to species that frequently associate with
  those keystones.

### Mathematical optimisation and acceleration

* **Square-root-space logic evaluation.** Inspired by R. Ryan
  Williams' arXiv:2502.17779 (TIME with `O(sqrt(t log t))` SPACE).
  Partition the rule graph used by the ecological constraints into
  blocks (generic / family constraints first, then spatial), map
  inter-block dependencies into a tree-evaluation problem, and
  evaluate gradients with a Cook-Mertz-style space-efficient
  walker. The motivation is to let the neuro-symbolic constraint
  refinement run inside a 24 GB consumer GPU's working set even on
  large rule graphs.
* **Entropy-gated agentic triggering.** Replace the fixed-threshold
  trigger for the LLM arbiter (Future Work #3) with a multi-sample
  predictive variance trigger, so the LLM is invoked exactly when
  the visual ensemble is internally inconsistent.
* **TensorRT + INT8 quantisation.** Move beyond bf16 to INT8
  compilation of the BioCLIP backbone for high-throughput scanning
  on Blackwell hardware. Unblocks long-quadrat-streaming scenarios
  (1000+ FPS).
* **Loopy belief propagation.** Generalise the binary AC-3 style
  pruning we describe in the paper's Appendix D (considered, not
  implemented) into a full probabilistic BP solver over the
  per-tile species posteriors and the ecological constraint graph.

### Test-time and self-supervised techniques

* **Test-time training (TTT) via SSL rotation/jigsaw**. At inference,
  briefly adapt the encoder per-quadrat using a rotation-prediction
  or jigsaw SSL head, exploiting that the quadrat itself is many
  augmentable views of the same patch of ground.
* **Conformal prediction for set-valued output**. Calibrate per-quadrat
  prediction sets with a guaranteed coverage level, which is the
  natural fit for the Macro-F1 scoring rule.
* **Retrieval-augmented classification for the extreme tail**. Maintain
  a FAISS index over per-species prototype embeddings; for the
  rarest species, route the visual embedding to nearest-neighbour
  retrieval against the prototype bank instead of relying on the
  softmax head.

---

## Where to find more

* `report/sections/appendix_development_trace.tex`, "Considered
  Approaches Not Implemented" - the paper's own list of directions we
  scoped and did not run.
* `report/sections/discussion.tex` - shorter prose discussion of why
  the visual encoder is the right place to look for the next gains.
