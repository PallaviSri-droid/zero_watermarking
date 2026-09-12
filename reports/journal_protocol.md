# Journal-Grade Experimental Protocol

## 1. Research question

CAP-ZW tests whether a **collision-aware multi-objective zero-watermarking formulation** can improve the robustness/discriminability trade-off for medical images while reducing cross-image hash collisions under a common evaluation protocol.

The scientific contribution is the formulation and evidence around collision-aware optimization/evaluation. Individual ingredients such as STE/BEMQ, balanced hashing, contrastive objectives, BCH-style coding, CNN features, memory-bank mining or MGDA-style optimization are not claimed as novel by themselves.

## 2. Controlled comparison

All methods compared in the main table must use the same:

- patient/group-level split;
- image preprocessing and resolution;
- hash length;
- watermark length and verification rule;
- attack types and strengths;
- negative-pair sampling/evaluation set;
- metric implementation and thresholding rule.

Literature-reported values and independently reproduced values must remain in separate tables.

## 3. Baseline set

The minimum safe classical comparison is:

1. DCT-Mean
2. DCT-Balanced
3. Edge-DCT

Learning-based baselines may be added only when their implementation, weights and preprocessing can be reproduced without hidden assumptions. KAZE variants must not be reported as independent baselines when the runtime does not provide a true KAZE implementation; a fallback to another feature extractor is not equivalent to KAZE.

## 4. Proposed method

The locked development candidate is `CAP-ZW-final-candidate`, combining the empirically useful collision-focused mechanisms from the v9 reference with the selective robustness guard from v11. The candidate is a starting point for controlled selection, not a presumed winner.

The core objective family includes:

- multi-view attack consistency;
- hard-negative separation with memory-bank mining;
- collision-tail and collision-mass pressure;
- binary/thresholded-code collision pressure;
- balance and entropy regularization;
- bit decorrelation;
- MGDA-style task balancing;
- selective worst-fraction robustness protection.

## 5. Primary metrics

### Robustness

- mean NC ↑
- mean BER ↓
- mean intra-image normalized Hamming distance ↓
- maximum intra-image normalized Hamming distance ↓

### Verification/discriminability

- mean inter-image normalized Hamming distance ↑
- minimum inter-image normalized Hamming distance ↑
- ROC-AUC ↑
- EER ↓
- FAR/FRR at a pre-specified operating threshold

### Collision risk

Report both raw counts and rates normalized per 10,000 negative pairs:

- exact collision (`HD = 0`);
- ultra-near collision (`HD <= 0.05`);
- near collision (`HD <= 0.10`).

Always print the negative-pair denominator.

### Tail separation

- inter-image q01, q05, q10;
- intra-image q90, q95;
- q05 tail gap = inter q05 - intra q95;
- q10 tail gap = inter q10 - intra q90.

`collision_gap = min(inter-image HD) - max(intra-image HD)` is retained as a diagnostic only; it is highly sensitive to evaluation-set size and individual pathological pairs.

### Hash quality

- balance error ↓
- mean bit entropy ↑
- mean absolute off-diagonal bit correlation ↓

## 6. NIH ChestX-ray14 protocol

Use patient identifiers as the grouping variable. No patient/study may occur in more than one of train, validation or test. The test split remains locked during model selection.

For every reported run, record:

- number of images;
- number of unique patients/groups;
- image resolution;
- hash length;
- manifest checksum;
- dataset source/version;
- preprocessing version;
- software environment;
- random seed;
- checkpoint identifier/hash;
- attack-grid identifier.

## 7. Fixed attack benchmark

The main benchmark must use one fixed attack grid across candidates. The repository protocol currently includes Gaussian noise, Gaussian blur, JPEG compression, small rotations and crop-resize; translation and deterministic compound attacks are included when the locked experiment configuration specifies them.

Attack-wise results must be reported in addition to aggregate robustness. The attack grid must not be changed after inspecting test results without labeling the run exploratory.

## 8. Training/selection statistics

Use independent seeds. The publication target is **5 seeds** for shortlisted configurations; 2 seeds are acceptable only for early smoke screening.

For each metric, retain all seed-level values. Report:

- mean;
- standard deviation;
- 95% Student-t confidence interval across seeds.

For paired method comparisons, use paired seed comparisons where the design permits and report an effect size. Correct for multiple comparisons when many configurations are tested. Never fabricate a confidence interval from a single run.

## 9. Controlled candidate selection

Starting from the v11 regime, vary one factor at a time before considering a larger factorial search:

- selective-guard batch fraction;
- guard multiplier/growth;
- binary-collision target;
- hard-negative top-k.

Keep architecture, optimizer family, attack grid, data split, evaluator and reporting code fixed while these factors are changed.

The selection decision must be made using training/validation data. The test set is evaluated only after the configuration is locked.

## 10. Reproducibility artifacts

Every reported number must map to:

`repository commit → configuration → seed → manifest → checkpoint → evaluator → attack grid → metric table`

Generated datasets, checkpoints and large experiment outputs remain outside version control. Source code, configuration, notebooks, evaluation scripts and research documentation are version-controlled.

## 11. Publication figures/tables

The minimum paper-ready set is:

1. Robustness-vs-collision Pareto scatter.
2. EER vs exact/near-collision rate per 10k negative pairs.
3. Attack-wise NC/BER distributions.
4. Inter-image vs intra-image Hamming-distance distributions with tail markers.
5. Hash balance, entropy and decorrelation comparison.
6. Controlled ablation table for selective robustness guard, binary collision pressure, hard-negative mining and MGDA.
7. Multi-seed mean ± 95% CI table.

## 12. Claim discipline

Do not state that CAP-ZW is superior until the common reproduced benchmark supports the claim. Do not copy percentages from literature into the proposed-method results. Report negative or non-significant findings when they occur.

The defensible novelty statement is: **a collision-aware multi-objective formulation and evaluation protocol for medical-image zero-watermarking that explicitly couples attacked-view robustness, different-image separation and collision-tail risk.**
