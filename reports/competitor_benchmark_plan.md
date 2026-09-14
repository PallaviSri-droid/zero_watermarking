# CAP-ZW Competitor Benchmark Plan

## Objective

Establish a fair, reproducible head-to-head comparison between CAP-ZW and a LogPolar+DINOv2+MRELBP hybrid baseline.

This comparison is a benchmark experiment, not a claim that the fusion baseline is a published method.

## Why the competitor is relevant

The three components provide complementary inductive biases:

- Log-polar representation targets geometric variation such as rotation/scale-related changes.
- DINOv2 provides a frozen pretrained global visual representation.
- MRELBP-style multiscale median local patterns capture local texture structure with robustness-oriented behavior.

CAP-ZW instead learns the hash representation with explicit robustness, cross-image separation and collision-risk objectives.

## Fairness contract

All main comparisons must hold constant:

1. Dataset and patient-level split.
2. Image resolution and grayscale preprocessing.
3. Hash length.
4. Watermark length and verification rule.
5. Locked attack grid from `protocol.DEFAULT_ATTACK_GRID`.
6. Same evaluated image set and negative-pair denominator.
7. Same metric implementation.
8. Same final thresholding rule.
9. Independent seeds where stochastic fitting/training is present.

The hybrid fits StandardScaler/PCA only on the training/validation reference set. CAP-ZW model selection also uses training/validation data only. The locked test set must not influence fitting or hyperparameter selection.

## Methods to compare

### Classical

- DCT-Mean
- DCT-Balanced
- Edge-DCT

KAZE-DCT is excluded from independent evidence unless a true KAZE implementation is available in the benchmark environment.

### Learned / feature-fusion

- CAP-ZW-v9 reference
- CAP-ZW-v11 reference
- CAP-ZW-final-candidate
- LogPolar+DINOv2+MRELBP

## Primary outcome families

### Robustness

- mean NC
- mean BER
- mean intra-image HD
- maximum intra-image HD

### Discriminability

- mean inter-image HD
- minimum inter-image HD
- ROC-AUC
- EER

### Collision risk

- exact collision rate per 10,000 negative pairs
- <=0.05 collision rate per 10,000
- <=0.10 collision rate per 10,000
- negative-pair denominator

### Tail behavior

- inter q01/q05/q10
- intra q90/q95
- q05 tail gap
- q10 tail gap

### Hash quality

- balance error
- mean bit entropy
- mean absolute inter-bit correlation

### Efficiency (secondary)

- fitting time
- evaluation/inference time
- peak memory where reproducibly measurable
- number of trainable parameters for learned methods

## Interpretation rule

No single scalar leaderboard score should replace the metric families. A method should be considered preferable only when the relevant robustness, discriminability and collision-risk evidence supports the conclusion.

A useful summary visualization is a Pareto frontier with:

- x = collision risk (lower is better);
- y = robustness loss / BER (lower is better);
- point shape = method;
- uncertainty = seed-level 95% CI when available.

## Required experiment stages

1. One-seed smoke comparison to validate code and contract.
2. Two-seed screening for candidate configurations.
3. Five-seed runs for finalists.
4. Locked test evaluation after candidate selection.
5. Statistical summary and publication figures.

## What would count as strong evidence for CAP-ZW

A defensible result would be one of the following patterns:

- lower collision rates/tail risk at comparable robustness;
- higher robustness at comparable collision rates;
- statistically supported Pareto improvement across the main operating region;
- consistent gains across attack types and seeds.

A mixed result is also scientifically valuable. For example, if LogPolar+DINOv2+MRELBP has better raw robustness but CAP-ZW has substantially lower collision-tail risk, the paper can position the methods as different operating points and investigate hybridization rather than hiding the trade-off.

## Reproducibility

Every result must record:

`commit → configuration → seed → manifest → checkpoint/model fit state → evaluator → attack grid → summary`
