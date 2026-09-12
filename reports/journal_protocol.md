# Journal-Grade Experimental Protocol

## Research hypothesis
CAP-ZW is designed to improve the robustness/discriminability trade-off of medical-image zero-watermarking by explicitly optimizing attacked-view consistency, hard-negative separation, bit balance/entropy, and bit decorrelation under a multi-objective training procedure.

## Primary comparison
Every method is evaluated on the same held-out patient/study groups, image size, hash length, attacks, attack strengths, watermark length, and verification rule.

## Baselines
1. DCT-Mean
2. DCT-Balanced
3. KAZE-DCT-Mean
4. KAZE-DCT-Balanced
5. ResNet50-DCT (pretrained deep feature baseline)
6. AlexNet feature + DCT proxy baseline
7. Additional published methods only when their algorithms can be reproduced from the paper/code without ambiguous assumptions.

## Proposed method
CAP-ZW with BEMQ/STE binary hashing, attack-paired training, hard-negative collision loss, balance/entropy regularization, bit-decorrelation regularization, and MGDA-style task weighting.

## Primary metrics
- mean NC (higher is better)
- mean BER (lower is better)
- maximum intra-image normalized Hamming distance (lower is better)
- minimum inter-image normalized Hamming distance (higher is better)
- collision gap = minimum inter-image distance - maximum intra-image distance (higher is better)
- ROC-AUC (higher is better)
- EER (lower is better)
- false acceptance / false rejection at a stated operating threshold

## Hash-quality metrics
- mean bit entropy
- absolute balance error from p=0.5
- mean absolute off-diagonal bit correlation

## Robustness protocol
Use a fixed attack grid covering additive noise, blur, JPEG compression, rotation, crop-resize and compound attacks. Report curves against attack strength, not only one aggregate number.

## Statistical protocol
Run at least 5 independent seeds for model-training experiments. Report mean, standard deviation and bootstrap 95% confidence intervals. Use paired tests across identical image/attack pairs when comparing methods. Do not tune the final test threshold on the test set.

## Leakage control
For volumetric, longitudinal or multi-view data, split by patient/study group. Never allow a patient/study identifier to occur in more than one of train/validation/test.

## Reproducibility
Record Python/package versions, dataset version, manifest hash, random seeds, checkpoint hash, configuration and attack parameters for every experiment.

## Novelty discipline
Standard components (DCT, KAZE, CNNs, contrastive learning, balanced hashing, STE/BEMQ, BCH, chaotic encryption and Pareto optimization) are not claimed as novel individually. Novelty must be framed around the exact collision-aware formulation, its zero-watermarking integration, attack-stratified protocol and empirically validated robustness/discriminability improvement.

## Publication claim rule
No claimed percentage improvement is written into the paper until it is computed from the held-out experimental results. Published numbers from other papers remain in a separate literature table and are never mixed with reproduced results.
