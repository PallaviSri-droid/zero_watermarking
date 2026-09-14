# Collision-Aware Pareto Zero-Watermarking for Medical Images

A reproducible research codebase for **collision-aware medical-image zero-watermarking**, with explicit robustness, discriminability and collision-tail evaluation.

> **Research status:** the repository contains a locked candidate, reference checkpoints, a reproducible LogPolar+DINOv2+MRELBP comparison baseline, strict benchmark identity checks, and pre-registered ablation runners. Scientific superiority claims remain blocked until the common multi-seed benchmark is completed.

## Research question

Does explicitly coupling attacked-view robustness, different-image separation and cross-image collision-tail risk produce a better zero-watermarking operating point than conventional robustness-only objectives?

The working research label is **CAP-ZW (Collision-Aware Pareto Zero-Watermarking)**. It is a project label, not a claim of prior publication.

## Scientific contribution

The intended contribution is the **collision-aware multi-objective formulation and evaluation protocol**. Components such as STE/BEMQ, balanced hashing, contrastive learning, CNN features, hard-negative mining and MGDA-style optimization are treated as established building blocks rather than individually novel contributions.

## Benchmark contract

Every main-table comparison must use the same patient/group split, image preprocessing, hash length, attack grid, negative-pair evaluation design, metric implementation and decision rule. Each new summary records a manifest fingerprint and attack-grid fingerprint; the comparison tool rejects incompatible summaries instead of silently mixing protocols.

### Robustness

- mean NC ↑
- mean BER ↓
- mean intra-image normalized Hamming distance ↓
- maximum intra-image normalized Hamming distance ↓

### Discriminability

- mean inter-image normalized Hamming distance ↑
- minimum inter-image normalized Hamming distance ↑
- ROC-AUC ↑
- EER ↓
- FAR/FRR at a stated operating threshold

### Collision risk

Report both counts and rates per 10,000 negative pairs:

- exact collision (`HD = 0`);
- ultra-near collision (`HD <= 0.05`);
- near collision (`HD <= 0.10`).

### Tail separation and hash quality

Report inter q01/q05/q10, intra q90/q95, q05/q10 tail gaps, balance error, mean bit entropy and mean absolute off-diagonal bit correlation.

`collision_gap = min(inter-image HD) - max(intra-image HD)` is a diagnostic and is not used as the sole collision criterion.

## Current development checkpoints

- **CAP-ZW-v9:** strongest current collision/discrimination pilot.
- **CAP-ZW-v10:** robustness-guard experiment showing that overly strong global robustness pressure can damage hash diversity and collision separation.
- **CAP-ZW-v11:** selective robustness refinement.
- **CAP-ZW-final-candidate:** locked candidate combining collision-focused v9 mechanisms with the selective robustness guard from v11. It must still be selected/validated through controlled multi-seed experiments.

The repository deliberately avoids uncontrolled hand-tuned version chasing. See `reports/cap_zw_final_research_spec.md`, `reports/journal_protocol.md`, and `reports/publication_execution_plan.md`.

## LogPolar + DINOv2 + MRELBP comparison baseline

To directly benchmark the competing design, the repository includes an independently reproducible **LogPolar+DINOv2+MRELBP** fusion baseline. The implementation uses official DINOv2 ViT-S/14 weights, a deterministic log-polar Fourier descriptor, and an explicitly labeled MRELBP-style multiscale median-local-pattern descriptor. The exact fusion is treated as a benchmark construction, not a claim of prior publication.

### Fusion pipeline

```text
Medical image
     │
     ├── DINOv2 ViT-S/14 global feature
     ├── Log-polar → Fourier magnitude feature
     └── Multiscale median local-pattern feature (MRELBP-style)
                 │
                 ▼
          block normalization
                 │
                 ▼
          feature concatenation
                 │
                 ▼
       train-split StandardScaler
                 │
                 ▼
          train-split PCA
                 │
                 ▼
       fixed random projection
                 │
                 ▼
          binary zero-hash
```

The scaler/PCA/projection are fitted only on the fitting split and never on the locked test split.

Run the full hybrid benchmark with the common attack grid:

```bash
python scripts/benchmark_logpolar_dino_mrelbp.py \
  --manifest data/manifests/medical_manifest.csv \
  --fit-split train_val \
  --split test \
  --fit-limit 1000 \
  --limit 200 \
  --size 128 \
  --bits 256 \
  --seed 42 \
  --device auto
```

### Hybrid component ablation

Run DINOv2-only, LogPolar-only, MRELBP-style-only, pairwise fusions, and the full three-way fusion:

```bash
python scripts/run_hybrid_component_ablation.py \
  --manifest data/manifests/medical_manifest.csv \
  --fit-split train_val \
  --split test \
  --fit-limit 1000 \
  --limit 200 \
  --bits 256 \
  --seed 42 \
  --device auto
```

The reproduction provenance and limitations are recorded in `reports/competitor_reproduction.md`.

## CAP-ZW ablations

The pre-registered CAP-ZW suite varies one mechanism at a time around the locked candidate:

```text
full CAP-ZW
no selective robustness guard
no binary collision pressure
no hard-negative mining
no memory bank
fixed weighted sum instead of MGDA
robustness-only reference
```

Run the screening suite:

```bash
python scripts/run_cap_zw_ablations.py \
  --manifest data/manifests/medical_manifest.csv \
  --split train_val \
  --limit 512 \
  --size 128 \
  --bits 128 \
  --epochs 3 \
  --batch-size 8 \
  --seed 42 \
  --device auto
```

Evaluate retained ablation checkpoints on the locked test split only after selection:

```bash
python scripts/evaluate_cap_zw_ablation_suite.py \
  --manifest data/manifests/medical_manifest.csv \
  --split test \
  --limit 200 \
  --size 128 \
  --bits 128 \
  --device auto
```

## Reproducible experiment sequence

### 1. Environment

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q tests
```

### 2. Dataset

The primary real-data benchmark is **NIH ChestX-ray14**. Keep raw images outside Git and construct a manifest containing image paths plus patient/group identifiers. The test split must remain locked during model selection.

```bash
python scripts/prepare_nih_chestxray14.py --max-images 5000
```

### 3. Train the locked candidate

```bash
python scripts/train_cap_zw_final.py \
  --manifest data/manifests/medical_manifest.csv \
  --split train_val \
  --limit 5000 \
  --size 128 \
  --bits 256 \
  --epochs 10 \
  --batch-size 16 \
  --seed 42 \
  --device auto
```

### 4. Common classical/deep benchmark

```bash
python scripts/run_full_benchmark.py \
  --manifest data/manifests/medical_manifest.csv \
  --split test \
  --limit 200 \
  --size 128 \
  --bits 256 \
  --seed 42
```

### 5. Compare retained summaries

```bash
python scripts/compare_cap_zw_vs_hybrid.py \
  --input "LogPolar+DINOv2+MRELBP=experiments/results/logpolar_dino_mrelbp/summary_seed_42.csv" \
  --input "CAP-ZW=experiments/results/cap_zw_test/summary.csv"
```

The comparison tool verifies immutable protocol identity before reporting metric winners.

### 6. Five-seed CAP-ZW selection benchmark

```bash
python scripts/run_cap_zw_multiseed.py \
  --manifest data/manifests/medical_manifest.csv \
  --split train_val \
  --seeds 13 23 42 73 97 \
  --limit 5000 \
  --size 128 \
  --bits 256 \
  --epochs 10 \
  --batch-size 16 \
  --device auto \
  --skip-existing
```

Aggregate seed-level results with protocol checks:

```bash
python scripts/aggregate_multi_seed.py \
  experiments/results/final_multiseed/seed_13 \
  experiments/results/final_multiseed/seed_23 \
  experiments/results/final_multiseed/seed_42 \
  experiments/results/final_multiseed/seed_73 \
  experiments/results/final_multiseed/seed_97 \
  --require-seeds 5
```

## Publication decision rule

Selection uses training/validation data. The test split is evaluated only after the candidate is locked. The primary scientific evidence is collision-tail risk and exact/near collision rate, followed by robustness and verification discrimination. Hash quality and computational cost are secondary criteria.

If another method wins, report it. The benchmark is designed to determine whether the CAP-ZW hypothesis is supported, not to guarantee a win.

See `reports/publication_execution_plan.md` for the full experimental gates.

## Reproducibility

Every reported result should be traceable to:

`commit → configuration → seed → manifest → checkpoint/model fit state → evaluator → attack grid → metrics`

The project intentionally does not version raw datasets, checkpoints or large experiment outputs.

## External validation

After the NIH pipeline is stable, add CheXpert and/or MIMIC-CXR-JPG using their own access/licensing procedures. These datasets are external validation, not a substitute for a reproducible NIH main benchmark.

## Citation

A machine-readable citation record is provided in `CITATION.cff`. Please cite the repository version/commit used for reproducibility.
