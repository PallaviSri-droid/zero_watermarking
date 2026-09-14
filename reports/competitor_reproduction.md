# LogPolar + DINOv2 + MRELBP Reproduction Record

## Scope
This repository includes an independently implemented hybrid benchmark combining three established descriptor families. It is a reproducible comparison baseline, not a claim that the exact three-way fusion is a previously published method.

## Components and provenance

### DINOv2
Use the official Meta AI `facebookresearch/dinov2` implementation and the `dinov2_vits14` pretrained backbone. The official repository documents the model loading path and identifies ViT-S/14 as a 21M-parameter distilled backbone.

Reference: Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023.
Official implementation: https://github.com/facebookresearch/dinov2

### Log-polar Fourier descriptor
The benchmark maps the grayscale image to log-polar coordinates with OpenCV `warpPolar`, then uses a Fourier magnitude representation. The descriptor is motivated by the established property that log-polar mappings can convert rotation/scale variation into shifts before a shift-insensitive spectral representation.

The implementation is intentionally compact and deterministic so it can be evaluated under exactly the same attack/metric protocol as CAP-ZW.

### MRELBP
The implementation follows the defining MRELBP principle of regional-median comparisons at multiple spatial scales and forms normalized local-pattern histograms. It is labeled `MRELBP-style` in code because this repository does not claim a line-by-line reproduction of every implementation detail or parameter choice from the original publication.

Reference: Liu et al., *Median Robust Extended Local Binary Pattern for Texture Classification*, IEEE Transactions on Image Processing, 2016, DOI 10.1109/TIP.2016.2522378.

## Fusion protocol

1. Resize all benchmark images to the fixed common resolution.
2. Compute DINOv2, log-polar Fourier and MRELBP-style blocks.
3. L2-normalize each block independently.
4. Apply declared block weights.
5. Concatenate blocks.
6. Fit `StandardScaler` only on the train/validation fitting split.
7. Fit PCA only on the train/validation fitting split.
8. Fit one fixed random projection matrix using the declared seed.
9. Threshold projected scores at zero to form the binary hash.
10. Evaluate only on the locked test split.

## Fairness controls

The CAP-ZW and hybrid benchmark must share:

- identical image IDs;
- patient/group-disjoint split;
- preprocessing and resolution;
- hash length;
- attack grid;
- attack seeds;
- negative-pair definition;
- metric implementation;
- verification thresholding rule;
- reporting denominator for collision rates.

The benchmark refuses to combine summary files whose immutable protocol identifiers disagree.

## Recommended comparison tiers

Tier 1: LogPolar only, DINOv2 only, MRELBP-style only.

Tier 2: pairwise fusions: LogPolar+DINOv2, LogPolar+MRELBP, DINOv2+MRELBP.

Tier 3: full LogPolar+DINOv2+MRELBP.

Tier 4: CAP-ZW-v9, CAP-ZW-v11, locked CAP-ZW candidate.

This hierarchy distinguishes the value of each component from the value of the full fusion.

## Claim discipline
Do not use ImageNet classification numbers, feature quality claims, or literature-reported accuracy as substitutes for zero-watermark verification results. The only primary comparison is the common end-to-end hash/verification benchmark described above.
