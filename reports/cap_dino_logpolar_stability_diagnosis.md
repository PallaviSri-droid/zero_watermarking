# CAP-DINO-LogPolar stability diagnosis

## Decision
Do not continue the v7 optimization as-is. The corrected v7 evaluation is poor enough that a long run would not be justified. However, the failure is informative and does not justify a random new architecture.

## Observed v7 failure
Corrected 200-image development evaluation:

- NC 0.240198
- BER 0.348089
- max intra-HD 0.507812
- mean inter-HD 0.379180
- AUC 0.689006
- EER 0.359636
- bit entropy 0.799783
- balance error 0.225664
- mean absolute bit correlation 0.063568
- exact collisions 0
- <=0.05 pairs 0
- <=0.10 pairs 0
- q05 inter 0.304688
- q95 intra 0.421875

The code space itself is not collapsed: entropy and decorrelation are materially better than v6, and exact/near collisions are reduced to zero. The dominant failure is therefore robustness: attacked instances move too far from the clean code.

## Why this happens

1. v7 made the hard binary robustness tail trainable. This aligns optimization with the binary metric, but STE gradients are coarse around bit thresholds. The model can improve separation while destabilizing bits across attacks.
2. v7 added several simultaneous pressures: hard robustness, branch anti-dominance, stronger collision-tail constraints, branch dropout, and a residual mixer. This is too many optimization changes to apply to a stable v6 representation from scratch.
3. The gate became nearly uniform (0.346/0.312/0.342), while CAP remained the argmax for every example. Therefore the gate regularizers did not produce useful adaptive specialization; they mostly constrained the gate without improving the final code.
4. The v7 residual mixer was added without warm-starting from the known-good v6 checkpoint. Even with a reduced residual scale, a new randomly initialized interaction path can perturb the hash boundary before the existing representation is protected.

## Research evidence guiding the repair

Recent zero-watermarking literature increasingly separates robustness from distinguishability rather than optimizing robustness alone. ConZWNet explicitly uses contrastive learning for attack-invariant features and a two-stage training strategy to improve stability. This motivates a conservative representation-alignment stage before stronger collision pressure.

Deep hashing literature commonly combines similarity-preserving pairwise objectives with explicit quantization control. This motivates adding a small quantization term so the continuous representation is compatible with the final binary code instead of relying only on STE.

Rel-Zero (CVPR 2026) shows that relational patch-pair structure can remain invariant under editing, motivating a later relational-branch experiment. It is intentionally excluded from the current repair because it would confound the current optimization diagnosis.

CXR foundation-model work motivates a later RAD-DINO/RayDINO ablation, but backbone replacement is also deferred until the optimization path is stable.

## Repair selected

The stable repair keeps the proven v6 three-branch fusion path and introduces only mechanisms with a direct explanation for the observed failure:

- warm-start common weights from the measured v6 checkpoint;
- remove branch dropout during the repair;
- use a smooth Bernoulli bit-disagreement probability for robustness, which directly approximates expected Hamming disagreement while retaining useful gradients;
- retain q90 robustness as a soft constraint rather than a hard STE objective;
- use fused-feature contrastive alignment for clean/attacked pairs;
- add a modest quantization penalty to reduce the continuous-to-binary mismatch;
- retain collision-tail, entropy, balance, and decorrelation objectives, but activate them strongly only after a representation-alignment warmup;
- keep gate regularization weak and non-prescriptive.

## Why this has a higher prior probability of helping

The repair starts from the representation that already demonstrated substantially better robustness than v7. It does not require a randomly initialized mixer or a new backbone to discover a good region of the objective landscape. The new robustness term is smoother than direct hard-code distance, and the two-stage schedule prevents early collision constraints from destroying attack invariance before it is established.

No performance gain is assumed in advance. The repair will be accepted only if a matched development run improves the robustness/discriminability trade-off relative to v6 and does not trade all gains for a new collision failure.

## Acceptance gate

A repair run must be evaluated on the same 200-image development protocol and checked attack-by-attack. It should show a material improvement in at least two robustness-tail metrics (BER, q95 intra-HD, max intra-HD) while maintaining or improving AUC/EER and without materially worsening near-collision rates. Only then should it proceed to multi-epoch/full-train experiments.
