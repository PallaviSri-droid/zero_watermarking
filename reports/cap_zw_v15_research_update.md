# CAP-ZW-v15 Research Update

## Why the previous candidates failed

The 200-image locked benchmark revealed a consistent Pareto trade-off:

- V11 preserved robustness but retained substantial binary-code collisions.
- V12 sharply reduced exact collisions and improved AUC/EER, but mean robustness and bit entropy remained below the desired target.
- V13 pushed collision counts lower again, but the robustness tail expanded substantially.
- V14 improved entropy and the near-collision tail, yet robustness remained too weak.

The central failure is therefore not a missing scalar loss term. The model uses one compact binary head to simultaneously satisfy invariance, discrimination, collision avoidance and bit-quality constraints. This creates competing gradients and allows one objective to dominate at different stages.

## Literature-driven design decisions

### 1. Explicit false-positive/discriminability objective
Rad-Mark (Neurocomputing, 2025) identifies high similarity between feature images from different hosts as a source of high false-positive rates. A 2025 medical-image study likewise notes that discriminability has received less attention than robustness. Therefore CAP-ZW keeps collision-tail diagnostics as a first-class research target.

### 2. Two-space representation
ConZWNet (Journal of Information Security and Applications, 2025) uses a two-stage framework: robust feature learning first, followed by watermark generation/identification feedback. CAP-ZW-v15 adopts the same general lesson—separate representation learning from binary code formation—but implements a different architecture and objective.

### 3. Relational stability
Rel-Zero (CVPR 2026) reports that relations between patch pairs can remain more stable under AI editing than absolute patch appearance. CAP-ZW-v15 therefore includes a lightweight spatial relation descriptor trained for clean/attacked consistency. This is not claimed as novel by itself; the research novelty remains the explicit collision-aware Pareto optimization and benchmark protocol.

### 4. Global hash geometry
Central Similarity Quantization (CVPR 2020) shows that global organization of binary hash centers can improve separation compared with purely local pairwise objectives. CAP-ZW-v15 consequently uses a continuous global embedding loss before the binary head instead of relying only on local binary-pair penalties.

### 5. Binary code quality
Deep supervised hashing and subsequent hashing literature motivate discrete-valued outputs, balance and reduced quantization ambiguity. V15 retains entropy, balance, decorrelation and confidence controls.

## CAP-ZW-v15 architecture

1. Multi-scale residual grayscale encoder with GroupNorm.
2. Global unit-norm embedding for continuous-space separation.
3. 4x4 spatial patch relation matrix for structural consistency.
4. Fusion layer producing the binary representation.
5. Learned per-bit thresholds for the final binary code.
6. Explicit collision-tail and collision-mass objectives.
7. Memory-based negative coverage.
8. Robust clean/attacked consistency in both embedding and binary spaces.

## What is intentionally not claimed

- The relational branch is not claimed to be a new primitive; Rel-Zero already establishes relational patch invariance.
- Contrastive learning, entropy regularization, hard-negative mining, memory banks, residual CNNs and binary hashing are established techniques.
- V15 is a candidate and must be selected by the locked multi-seed benchmark, not by a single favorable seed.

## Required evaluation

Every final claim must use the same manifest fingerprint, attack-grid fingerprint, image split, bit length and evaluator across V11, V12, V13, V14, V15 and reproduced competitors.

Minimum publication analysis:

- 5 independent seeds.
- Full held-out test split after model selection.
- NC, BER, intra-image Hamming statistics.
- Inter-image mean/min Hamming distance.
- ROC-AUC, EER, FAR/FRR.
- Exact, ultra-near and near collision rates.
- q01/q05/q10 inter-image tails and q90/q95 intra-image tails.
- Bit entropy, balance error and mean absolute inter-bit correlation.
- Bootstrap or Student-t confidence intervals where appropriate.

## References

- Rad-Mark: Reliable adversarial zero-watermarking. Neurocomputing, 2025. DOI: 10.1016/j.neucom.2025.129970.
- ConZWNet: A contrastive learning-based zero-watermarking network for high robustness and distinguishability. Journal of Information Security and Applications, 2025. DOI: 10.1016/j.jisa.2025.104139.
- Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing. CVPR, 2026.
- Central Similarity Quantization for Efficient Image and Video Retrieval. CVPR, 2020. DOI: 10.1109/CVPR42600.2020.00315.
- Deep Supervised Hashing for Fast Image Retrieval. CVPR, 2016.
