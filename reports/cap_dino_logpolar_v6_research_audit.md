# CAP-DINO-LogPolar v6 research audit

## Evidence from v4 and v5

The locked 200-image test protocol produced the following measured observations.

- v4: mean NC 0.787473, BER 0.106438, AUC 0.973053, EER 0.088238, exact collisions 0, bit entropy 0.881909, q05 tail gap -0.0625, q10 tail gap 0.015625.
- v5: mean NC 0.916920, BER 0.041669, AUC 0.980213, EER 0.066721, exact collisions 8, bit entropy 0.775552, q05 tail gap -0.03125, q10 tail gap 0.046875.
- v5 had max intra-image Hamming distance 0.59375 and 100% CAP-dominant clean-image routing.

The principal v5 failure is therefore not average representation quality. It is tail robustness and expert collapse: a small number of attacked samples are catastrophic, while the gate does not use DINO/Log-Polar as genuine alternatives.

## Literature-grounded design decisions

Recent medical zero-watermarking work emphasizes discriminability and false-positive control rather than robustness alone. Rad-Mark explicitly targets the high similarity of feature images from different hosts and uses adversarial feature optimization to improve discrimination (NeuCom 2025, DOI 10.1016/j.neucom.2025.129970). A 2025 IEEE JBHI medical zero-watermarking paper similarly argues that discriminability has been under-discussed. CVPR 2026 Rel-Zero shows that patch-pair relational invariance is a useful route to robust zero-watermarking under strong editing. A 2026 deep-hashing survey frames efficient binary representation learning around compact codes, Hamming distance, and loss-function design.

DINOv2 remains a strong generic visual feature baseline. Domain-specific CXR encoders such as RAD-DINO motivate a later controlled backbone substitution experiment; this is not mixed into v6 so that v6 remains interpretable as a fusion/optimization ablation.

## v6 changes

1. Use the clean-image gate as the single routing decision for the clean/attacked pair.
2. Apply shared branch dropout during training so the CAP branch cannot dominate every sample.
3. Add both lower and upper population gate-usage constraints and a minimum cross-sample gate variance target.
4. Optimize robustness on the actual STE binary code, not only soft code drift.
5. Penalize a robustness quantile and the worst tail fraction, targeting the observed v5 failure mode.
6. Preserve v12-v14 collision controls: lower-tail separation, hard entropy/balance, binary collision mass, and bit decorrelation.
7. Train with all 11 benchmark attack families for the publication configuration.

## Controlled-experiment rule

v6 is a research candidate, not a superiority claim. V5, V12, and V14 remain retained baselines. A model may be called the selected candidate only after the same locked 200-image test protocol, full 967-image test set, and multi-seed evaluation have been completed.

## Data preprocessing note

The current common loader resizes grayscale images directly to a square using bilinear interpolation. This can introduce aspect-ratio distortion. A future preprocessing ablation should compare direct resize with aspect-ratio-preserving pad/letterbox, and any paper-level comparison must rerun all baselines under the same preprocessing choice.
