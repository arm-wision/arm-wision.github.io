Title: 

Possible Research questions:
1. How can domain-specific foundation models (BioCLIP) be adapted to identify rare species in complex, overlapping vegetation plots where visual occlusion and a long-tailed species distribution typically degrade performance?
2. Does taxonomic-aware pre-training (BioCLIP) provide superior feature discrimination for morphologically similar species compared to domain-agnostic self-supervised models (DINOv2) in high-density vegetation plots?
3. To what extent do SAM-derived synthetic multi-species collages reduce the need for pre labeled data in automated biodiversity monitoring?
4. Does high-resolution tiling disproportionately benefit the recall of rare, small-stature plants in 50x50cm quadrats?

# Research Proposal
1. National Interest Statement
Monitoring plant biodiversity is a matter of critical national security regarding climate resilience, agriculture, and ecosystem services. Current manual surveying methods are slow and cost-prohibitive. This research aims to automate large-scale botanical surveys, providing the government and environmental agencies with the tools to respond instantly to invasive species threats and habitat loss.

2. Introduction
Identifying plants in the wild is traditionally a "one-at-a-time" task. However, ecological reality consists of complex "vegetation plots" (quadrats) where multiple species overlap. This project seeks to build an AI system capable of "dissecting" these plots to identify every species present.

3. Research Problem
Two major bottlenecks exist in botanical AI:
The Domain Gap: Models trained on centered, clean images fail in the "messy" wild.
The Long-Tail: Common weeds are over-represented in data, while endangered, high-priority species are often ignored by AI.

4. Research Design
Our design follows a three-stage pipeline:
Knowledge Transfer: Utilizing BioCLIP, a foundation model pre-aligned with the Tree of Life.
Synthetic Complexity: Creating "Synthetic Collages" to simulate multi-species plots, using Asymmetric Loss to manage the imbalance.
Inference Strategy: Implementing Slicing Aided Hyper Inference (SAHI) to maintain high resolution for tiny plant features.
To eliminate CPU-side bottlenecks during the training of our high-resolution BioCLIP backbone, we implemented a fully GPU-accelerated data pipeline. We utilized NVIDIA DALI for hardware-accelerated JPEG decoding (NVDEC) and spatial augmentations, alongside cuDF for rapid metadata manipulation. This configuration ensures 100% GPU utilization and significantly reduces epoch time for the 1.4 million image corpus.

5. Evaluation Criteria
Success will be measured by the Micro-F1 Score across 7,800 species. Specifically, we will track "Recall for Rare Species" to ensure the model does not achieve a high score simply by guessing common plants.

6. Conclusion
This research will provide a state-of-the-art framework for fine-grained biodiversity monitoring. By bridging the gap between foundation models and real-world ecological complexity, we can transform global conservation efforts into a data-driven science.
