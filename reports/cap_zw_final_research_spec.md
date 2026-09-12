# CAP-ZW Final Research Specification

## Decision

Stop uncontrolled version chasing. Treat **v9 and v11 as reference checkpoints** and select a final CAP-ZW configuration only after controlled multi-seed evaluation. The proposed contribution is not any individual use of STE, balanced hashing, contrastive learning, BCH, or MGDA. The defensible contribution is a **collision-aware multi-objective zero-watermarking framework and benchmark protocol** that explicitly couples attacked-image robustness with different-image separation and collision-tail risk.

## What the pilot results actually show

The existing pilots are useful for hypothesis generation, not final claims. v9 produced the strongest pilot collision/discrimination profile (AUC 0.997479, EER 0.022720, one exact collision), while v11 recovered robustness with a moderate discrimination cost (NC 0.963909, AUC 0.996883, EER 0.026095, four exact collisions). v10 demonstrates that overly strong global robustness pressure can damage entropy/balance and discrimination. These observations motivate controlled selection rather than another hand-tuned version.

## Core evaluation axes

1. **Robustness:** mean NC, mean BER, mean/max intra-image Hamming distance.
2. **Verification/discriminability:** AUC, EER, mean/min inter-image Hamming distance.
3. **Collision risk:** exact, <=0.05, and <=0.10 Hamming collision rates normalized per 10,000 negative pairs.
4. **Tail separation:** inter-image 1%, 5%, 10% quantiles versus intra-image 90%, 95% quantiles.
5. **Hash quality:** balance error, bit entropy, mean absolute inter-bit correlation.

The raw minimum inter-distance minus maximum intra-distance is retained as a diagnostic, but should not be the sole collision criterion because it is extremely sensitive to sample count and one pathological pair.

## Final training strategy

Use v9/v11 mechanisms as the controlled reference family. Preserve selective robustness protection from v11, collision-tail and hard-negative pressure from v9, and MGDA task balancing. Do not add a new architectural component unless an ablation proves that it independently contributes.

Recommended first candidate family:

- selective robustness guard focused on the worst robustness fraction;
- gradual guard multiplier, starting below the v10 regime;
- binary collision target around the v11 regime rather than v9's stronger setting;
- memory-bank hard-negative mining retained;
- bit balance/entropy/decorrelation retained but monitored for conflict;
- identical encoder, bit length, train/test split, attack grid and evaluator across candidates.

The first controlled sweep should vary one factor at a time around v11: guard fraction, guard multiplier/growth, binary-collision target, and hard-negative top-k. Use 2 seeds for smoke validation and 5 seeds for shortlisted configurations.

## Statistical protocol

For every final configuration, retain seed-level CSVs. Report mean and 95% Student-t confidence intervals across seeds. Never manufacture confidence intervals from a single run. Where pairwise comparisons are made, prefer paired seed comparisons and correct for multiple comparisons when many configurations are tested. Report effect sizes alongside p-values when feasible.

For collision counts, always report the denominator (number of negative pairs) and the normalized rate per 10,000. This prevents misleading comparisons between evaluation-set sizes.

## Dataset protocol

Use the NIH ChestX-ray14 manifest with patient/group identifiers to prevent patient leakage. Keep the test split untouched during model selection. Hyperparameter decisions must use training/validation data; the final test set is for one locked evaluation. Record image count, patient count, image size, hash length, attacks, seeds, software versions, and checkpoint metadata.

## Attack protocol

Keep the common attack grid fixed for the main comparison: Gaussian noise at two strengths, Gaussian blur at two strengths, JPEG at two qualities, small rotations, crop-resize, translation, and a deterministic compound attack. Report both aggregate robustness and attack-wise performance. Do not change the attack grid after seeing test results without clearly labeling the experiment as exploratory.

## Publication figures

1. Robustness-versus-collision trade-off scatter with Pareto-optimal candidates.
2. EER versus exact/near collision rate per 10k negative pairs.
3. Attack-wise NC/BER distribution.
4. Inter-image and intra-image Hamming-distance distributions with 5%/95% tails.
5. Bit entropy and correlation comparison.
6. Ablation table for selective guard, binary collision pressure, hard-negative mining and MGDA.

## Reproducibility rule

Every reported result must map to a checkpoint, seed, manifest version, evaluator command, attack grid and repository commit. Generated datasets, checkpoints and experiment outputs remain local/ignored; source code, configs, notebooks and research documentation are version-controlled.

## Claim discipline

The project should claim: **collision-aware multi-objective formulation/evaluation for medical zero-watermarking**, supported by controlled experiments. It should not claim that contrastive learning, balanced hashing, STE, BCH, MGDA, or stable features are individually novel. Any superiority claim must come from the common reproduced benchmark and statistical analysis, not from literature-reported numbers under different protocols.
