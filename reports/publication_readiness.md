# Publication Readiness Checklist

This checklist separates **software/reproducibility readiness** from **scientific evidence readiness**. A green repository does not by itself establish a valid scientific claim.

## A. Reproducibility

- [ ] Clean environment installs from `requirements.txt` and `pyproject.toml`.
- [ ] `python -m pytest -q` passes.
- [ ] Dataset manifest is versioned by checksum, but raw medical images are kept outside Git.
- [ ] Patient/group leakage check passes before training and evaluation.
- [ ] Every training run records seed, configuration, checkpoint path and software environment.
- [ ] Every evaluation records checkpoint, manifest, attack-grid identifier and metric version.
- [ ] No credentials, API keys or dataset secrets are committed.

## B. Main scientific benchmark

- [ ] Development/selection uses only training/validation groups.
- [ ] The final test set is locked before the final configuration is selected.
- [ ] At least the safe classical baselines are reproduced under the same protocol: DCT-Mean, DCT-Balanced and Edge-DCT.
- [ ] Any learning baseline has documented weights, preprocessing and provenance.
- [ ] All methods use identical image resolution, hash length, attack grid and evaluator.
- [ ] Attack-wise results are retained, not only aggregate averages.

## C. Collision-aware evidence

- [ ] Exact collision count and rate per 10,000 negative pairs are reported.
- [ ] `HD <= 0.05` and `HD <= 0.10` collision rates are reported with denominators.
- [ ] Inter-image q01/q05/q10 and intra-image q90/q95 are reported.
- [ ] q05/q10 tail gaps are reported.
- [ ] AUC and EER are reported with the exact verification construction.
- [ ] Raw `collision_gap` is treated as a diagnostic rather than the only collision statistic.

## D. Multi-seed statistics

- [ ] Two-seed smoke runs are used only to eliminate obviously broken configurations.
- [ ] Shortlisted configurations use five independent seeds.
- [ ] Mean, standard deviation and 95% Student-t confidence intervals are computed from seed-level outputs.
- [ ] Pairwise comparisons use paired seed results when design permits.
- [ ] Effect sizes are reported for important comparisons.
- [ ] Multiple-comparison correction is applied when many configurations are compared.

## E. Ablation

At minimum, isolate:

1. selective robustness guard;
2. binary collision pressure;
3. hard-negative memory mining;
4. MGDA-style multi-objective weighting.

Architecture, data split, attacks and evaluator remain fixed while individual components are removed/changed.

## F. Figures and tables

- [ ] Robustness-vs-collision Pareto plot.
- [ ] EER vs exact/near-collision rate per 10k negative pairs.
- [ ] Attack-wise NC/BER figure.
- [ ] Inter/intra Hamming-distance distribution with tail markers.
- [ ] Balance/entropy/decorrelation comparison.
- [ ] Ablation table.
- [ ] Five-seed mean ± 95% CI main table.

## G. Claim audit

- [ ] No literature-reported value is mixed with reproduced experimental values.
- [ ] No percentage improvement is hard-coded before the common benchmark is run.
- [ ] Standard components are not presented as individually novel.
- [ ] The paper's novelty statement matches the actual implemented formulation.
- [ ] Negative, non-significant or trade-off results are retained in the analysis.

## Release gate

A result can be called **publication-ready evidence** only when A–G are satisfied for the main claim. Until then, describe results as pilot, development, or exploratory findings.
