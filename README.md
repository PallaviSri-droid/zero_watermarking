# Collision-Aware Pareto Zero-Watermarking for Medical Images

A reproducible research codebase for **collision-aware medical-image zero-watermarking**, with explicit robustness, discriminability and collision-tail evaluation.

> **Research status:** the repository contains a locked candidate, reference checkpoints and the full evaluation protocol, plus a reproducible LogPolar+DINOv2+MRELBP comparison baseline. Scientific superiority claims remain blocked until the common multi-seed benchmark is completed.

## Research question

Does explicitly coupling attacked-view robustness, different-image separation and cross-image collision-tail risk produce a better zero-watermarking operating point than conventional robustness-only objectives?

The working research label is **CAP-ZW (Collision-Aware Pareto Zero-Watermarking)**. It is a project label, not a claim of prior publication.

## Scientific contribution

The intended contribution is the **collision-aware multi-objective formulation and evaluation protocol**. Components such as STE/BEMQ, balanced hashing, contrastive learning, CNN features, hard-negative mining and MGDA-style optimization are treated as established building blocks rather than individually novel contributions.

## Repository structure

```text
src/zero_watermarking/      Core algorithms, attacks, metrics, training and protocol
scripts/                    Reproducible training, evaluation and comparison runners
configs/                    Locked experimental configurations
notebooks/                  Analysis and publication-figure notebooks
reports/                    Research protocol, literature matrix and publication checklist
tests/                      Regression/unit tests
```

## Benchmark contract

Every main-table comparison must use the same patient/group split, image preprocessing, hash length, attack grid, negative-pair evaluation design, metric implementation and decision rule.

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

The repository deliberately avoids uncontrolled hand-tuned version chasing. See `reports/cap_zw_final_research_spec.md` and `reports/journal_protocol.md`.

## LogPolar + DINOv2 + MRELBP comparison baseline

To directly benchmark the approach used by your peer group, the repository now includes an independently reproducible **LogPolar+DINOv2+MRELBP** fusion baseline.

### Why these components

**DINO/DINOv2:** self-supervised Vision Transformer representations provide strong generic visual features without task-specific labels. The DINOv2 paper reports robust general-purpose visual features from large-scale self-supervised training. [DINO, Caron et al., 2021](https://arxiv.org/abs/2104.14294); [DINOv2, Oquab et al., 2023](https://arxiv.org/abs/2304.07193).

**Log-polar representation:** log-polar coordinates convert rotation and scale changes into shifts, enabling Fourier-magnitude or other shift-insensitive descriptors to obtain rotation/scale robustness. See the log-polar invariant recognition literature, including the Log-Polar Magnitude descriptor and later invariant pattern-recognition formulations. [Log-Polar Magnitude](https://pmc.ncbi.nlm.nih.gov/articles/PMC5708636/).

**MRELBP:** Median Robust Extended Local Binary Pattern replaces raw-pixel comparisons with regional-median comparisons and uses multiscale local patterns to capture microstructure and macrostructure while improving robustness to noise and rotation. [Liu et al., IEEE TIP 2016, DOI:10.1109/TIP.2016.2522378](https://pubmed.ncbi.nlm.nih.gov/26829791/).

The fusion implementation in `src/zero_watermarking/logpolar_dino_mrelbp.py` is a **benchmark implementation of the combined idea**, not copied code and not a claim that the exact fusion is a published method.

### Fusion pipeline

```text
Medical image
     │
     ├── DINOv2 ViT-S/14 global feature
     │
     ├── Log-polar → Fourier magnitude feature
     │
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

The fitted scaler/PCA/projection are learned only on the fitting split; the held-out test split is never used to fit them.

### Run the comparison

```bash
python scripts/benchmark_logpolar_dino_mrelbp.py \
  --manifest data/manifests/medical_manifest.csv \
  --fit-split train_val \
  --split test \
  --fit-limit 1000 \
  --limit 200 \
  --size 128 \
  --bits 128 \
  --seed 42 \
  --device auto
```

The script evaluates the hybrid using the same CAP-ZW metric implementation, including NC/BER, AUC/EER, inter/intra Hamming distances, collision rates and tail separation.

**Reproducibility note:** the first run downloads DINOv2 ViT-S/14 weights through PyTorch Hub. Cache the model weights locally and record the exact environment/checkpoint metadata with the experiment. DINOv2's official implementation is available from Meta's `facebookresearch/dinov2` repository.

## Reproducible experiment sequence

### 1. Environment

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

### 2. Dataset

The primary real-data benchmark is **NIH ChestX-ray14**. Keep raw images outside Git and construct a manifest containing image paths plus patient/group identifiers. The test split must remain locked during model selection.

The preparation utility is:

```bash
python scripts/prepare_nih_chestxray14.py --max-images 5000
```

For Kaggle authentication, use the credential mechanism supported by the installed Kaggle CLI; never commit credentials or paste API keys into source files.

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

### 4. Run the five-seed selection benchmark

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

Shortlisted configurations require five independent seeds for publication statistics; two seeds are reserved for early smoke screening.

### 5. Evaluate the locked test set

```bash
python scripts/evaluate_cap_zw.py \
  --manifest data/manifests/medical_manifest.csv \
  --split test \
  --checkpoint experiments/checkpoints/cap_zw_final_seed42.pt \
  --limit 200 \
  --size 128 \
  --bits 256
```

Do not tune the final test threshold after seeing the test results.

## Publication analysis

Use the analysis notebooks to generate the paper tables and figures only from retained seed-level outputs. Report mean, standard deviation and 95% Student-t confidence intervals across seeds. Always retain the negative-pair denominator for collision statistics.

For the external feature-fusion baseline, do not compare literature classification accuracy with CAP-ZW watermarking metrics. The proper comparison is an end-to-end hash/verification benchmark under the same images, attacks, hash length and evaluator.

The paper-ready minimum figure set is documented in `reports/journal_protocol.md`.

## Reproducibility

Every reported result should be traceable to:

`commit → configuration → seed → manifest → checkpoint → evaluator → attack grid → metrics`

The project intentionally does not version raw datasets, checkpoints or large experiment outputs.

## External validation

After the NIH pipeline is stable, add CheXpert and/or MIMIC-CXR-JPG using their own access/licensing procedures. These datasets are external validation, not a substitute for a reproducible NIH main benchmark.

## Citation

A machine-readable citation record is provided in `CITATION.cff`. Please cite the repository version/commit used for reproducibility.
