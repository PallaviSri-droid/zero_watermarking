# Publication Execution Plan

This plan is the controlled path from the current CAP-ZW pilot to a defensible journal result.

## Gate 0 — Environment and code health

- use one reproducible Python environment;
- run `python -m pytest -q tests`;
- record `pip freeze` or equivalent environment lock;
- do not alter source code based on a single benchmark result.

## Gate 1 — Data lock

- generate the NIH ChestX-ray14 manifest once;
- preserve patient/group identifiers;
- verify no group leakage across splits;
- compute and record the manifest SHA-256 identifier;
- lock the test split before model selection.

## Gate 2 — Baseline lock

Run the safe classical methods: DCT-Mean, DCT-Balanced and Edge-DCT. KAZE is included only when the environment has a true KAZE implementation. The LogPolar+DINOv2+MRELBP hybrid is a separately documented reproduction baseline.

For the hybrid, report the component tiers before the full fusion:

1. DINOv2 only
2. LogPolar only
3. MRELBP-style only
4. DINOv2 + LogPolar
5. DINOv2 + MRELBP-style
6. LogPolar + MRELBP-style
7. DINOv2 + LogPolar + MRELBP-style

## Gate 3 — CAP-ZW ablation lock

Run the pre-registered CAP-ZW ablation suite on the training/validation split:

- full CAP-ZW;
- no selective robustness guard;
- no binary collision pressure;
- no hard-negative mining;
- no memory bank;
- fixed weighted sum instead of MGDA;
- robustness-only reference.

Use two seeds for screening, then five seeds for shortlisted configurations.

## Gate 4 — Common test benchmark

Evaluate the shortlisted CAP-ZW variants and the hybrid baseline on the locked test split using exactly the same evaluator, attack grid and negative-pair design.

Required outputs:

- NC, BER, mean/max intra-image HD;
- mean/min inter-image HD;
- AUC, EER;
- exact, <=0.05 and <=0.10 collision rates per 10,000 negative pairs;
- inter q01/q05/q10;
- intra q90/q95;
- q05/q10 tail gaps;
- balance error, entropy, bit correlation;
- fit/inference time and hardware metadata.

## Gate 5 — Statistical analysis

- aggregate seed-level results;
- report mean, standard deviation and 95% Student-t CI across five seeds;
- for paired comparisons, preserve identical image/attack pairing;
- report effect sizes;
- correct for multiple comparisons where applicable;
- never manufacture uncertainty from one run.

## Gate 6 — Decision rule

The model-selection decision is made on train/validation only. The test set is not used to choose hyperparameters.

Primary scientific evidence:

1. collision-tail risk;
2. exact/near collision rates;
3. robustness under the common attack grid;
4. verification discrimination.

Hash quality and computational cost are secondary decision criteria and tie-breakers.

## Gate 7 — Paper artifacts

Generate all tables and figures from retained experiment files. No manual number entry.

Minimum set:

- overall method comparison;
- attack-wise table/curves;
- collision-tail table;
- CAP-ZW ablation table;
- component-fusion table for LogPolar/DINOv2/MRELBP;
- five-seed CI table;
- robustness-vs-collision Pareto scatter;
- ROC/EER figure;
- intra/inter Hamming distribution;
- computational-cost comparison.

## Final interpretation rule

If another method wins, report it. The purpose of the protocol is to determine whether the collision-aware CAP-ZW hypothesis is supported, not to guarantee that CAP-ZW wins every metric.
