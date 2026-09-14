# CAP-DINO-LogPolar v4 Audit and v5 Research Design

## Evidence baseline

The 200-image locked development evaluation of CAP-DINO-LP-v4 produced:

- NC 0.787473
- BER 0.106438
- max intra-image Hamming distance 0.453125
- mean inter-image Hamming distance 0.428151
- AUC 0.973053
- EER 0.088238
- exact collisions 0 / 19,900
- <=0.05 collisions 21 / 19,900
- <=0.10 collisions 179 / 19,900
- bit entropy 0.881909
- mean absolute bit correlation 0.242190
- q05 tail gap -0.0625
- q10 tail gap +0.015625

The five-epoch training trajectory improved robustness and entropy, but the learned gate remained close to uniform: epoch 5 mean gates were CAP=0.3470, DINO=0.3276, LogPolar=0.3253. This means v4 behaved more like a fixed three-branch mixture than a genuinely adaptive router.

For comparison, previously measured CAP candidates showed a different trade-off: V12 had NC 0.9696, BER 0.0153, AUC 0.9868, EER 0.0583 and 5 exact collisions; V14 had NC 0.9469, BER 0.0267, AUC 0.9858, EER 0.0613, 4 exact collisions and entropy 0.7267. These figures are development evidence on the same 200-image protocol, not final publication claims.

## Audit: where v4 loses signal

### 1. Attack coverage is under-sampled during training

The v4 trainer exposes 11 attack types but fixes `attack_views=4`. Consequently each source image sees only four deterministic attack views per epoch rather than the full benchmark grid. The evaluator tests 11 attacks. This is a train/evaluation distribution mismatch.

### 2. Negative diversity is limited by the small batch

With batch size 8, the in-batch collision objective sees only a small number of different-image negatives. V12/V14 already demonstrated that memory-bank negatives and hard-negative mining improve the failure mode. v5 therefore carries a memory bank and top-k hard negative selection into the fusion model.

### 3. The v4 gate is explicitly encouraged to stay uniform

The v4 loss contains a negative coefficient on gate entropy. Maximizing entropy for a three-way softmax pushes routing toward the uniform solution. This explains the near-maximum entropy and approximately 1/3,1/3,1/3 routing observed during training.

### 4. Clean and attacked images are routed independently

v4 predicts one gate for the clean image and another for the attacked image. The attack can therefore change which branch controls the final code. v5 computes the clean-image routing once and reuses that gate for the attacked representation during training/evaluation. A separate gate-consistency term penalizes unstable routing.

### 5. Training and evaluation must operate on the same binary representation

Collision risk is ultimately measured after thresholding. v5 keeps a soft code for smooth robustness optimization but also computes a straight-through hard code for collision, entropy and balance constraints. This makes the forward value identical to the evaluated binary representation while retaining gradients.

### 6. DINOv2 is a general visual prior, not a CXR-specific encoder

Generic DINOv2 has demonstrated useful transfer to radiology, but medical-domain foundation models such as RAD-DINO and newer CXR-specialized encoders are explicitly trained on large chest-X-ray corpora. v5 therefore adds a trainable nonlinear adapter around frozen DINOv2 instead of relying on a single linear projection. A future controlled experiment should compare this with RAD-DINO/XRay-DINO rather than assuming generic DINOv2 is optimal.

### 7. Log-polar is appropriate for geometric robustness but is not a complete solution

Log-polar/Fourier magnitude representations are established tools for rotation/scale robustness, while translation needs separate handling. v5 keeps Log-Polar as a complementary branch and adds attack-consistency training instead of treating it as universally invariant.

## v5 design

The v5 candidate combines the strongest measured lessons from CAP V12/V14 with the fusion architecture:

1. Clean-image adaptive gate is shared with attacked views.
2. The gate is not rewarded for maximum entropy; only a weak branch-usage floor remains.
3. Six attack views are used by default, with the CLI allowing the full 11-view benchmark distribution.
4. A memory bank stores clean hard codes for cross-batch collision pressure.
5. Top-k hard-negative collision loss controls both nearest-negative distance and collision mass.
6. Lower-tail q05/q10 separation is explicitly optimized.
7. Hard-forward entropy and balance constraints discourage code collapse.
8. Branch-level cosine consistency encourages CAP, DINO and Log-Polar features to survive attacks.
9. A trainable nonlinear adapter is added to each branch, while the DINOv2 backbone stays frozen.
10. Gate-consistency and weak late-stage anti-uniformity regularization make adaptive routing measurable rather than assumed.

## Research discipline

No superiority claim should be made until v5 is evaluated against V11-V14 under the same manifest, hash length, image count, attack grid, and evaluator. Final publication evidence should use the complete 5,000-image locked subset and multiple independent seeds.

## Relevant literature positioning

Recent zero-watermarking work continues to identify discriminability/false positives as a central weakness, including medical-image work that explicitly evaluates distinguishability. Recent methods also introduce global-local fusion and adversarial optimization. Rel-Zero (CVPR 2026) further motivates relational invariance rather than relying only on absolute global features. These works support treating robustness and false-positive resistance as jointly necessary objectives, but none makes the specific v5 combination a proven novelty claim.
