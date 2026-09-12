# Collision-Aware Pareto Zero-Watermarking for Medical Images

A reproducible research codebase for medical-image zero-watermarking with a benchmark-first protocol.

## Research goal

The project tests whether jointly optimizing same-image robustness, different-image separation, collision-tail risk, bit balance, entropy and decorrelation can improve the robustness/discriminability trade-off compared with classical and learning-based baselines.

Working research label: **CAP-ZW (Collision-Aware Pareto Zero-Watermarking)**. This is a research label, not a claim of prior publication.

## Current learned formulation

The active learned formulation is resolved from `zero_watermarking.training.CAP_ZW_VERSION` and stored in each checkpoint. The current implementation is **CAP-ZW-v8**.

CAP-ZW v8 adds discrete collision-aware optimization directly on the thresholded BEMQ code while retaining:

- cross-batch memory mining;
- top-k hard-negative separation;
- continuous collision-tail and diversity objectives;
- multi-view attack training;
- robustness-budget / floor regularization;
- clean/attacked consistency;
- clean + attacked bit balance and entropy regularization;
- hard-code-aware decorrelation;
- MGDA-style multi-objective gradient weighting.

The central hypothesis is not that any individual ingredient is new. The research question is whether this **joint collision-aware formulation and evaluation protocol** produces a measurable improvement in the robustness/discriminability trade-off and reduces cross-image hash collisions under a common medical-image benchmark.

## Metrics

Core diagnostic:

`collision_gap = min(inter-image Hamming distance) - max(intra-image Hamming distance)`

Tail diagnostics include:

- `inter_q05`, `inter_q10`
- `intra_q90`, `intra_q95`
- `q05_tail_gap`, `q10_tail_gap`
- near-collision, ultra-near-collision and exact-collision counts

Primary performance metrics are NC, BER, intra-image Hamming distance, inter-image Hamming distance, ROC-AUC, EER, balance error, bit entropy and mean absolute inter-bit correlation.

## Workflow

```text
Dataset / preprocessing
        ↓
Classical baselines
        ↓
Attack engine
        ↓
Robustness evaluation
        ↓
Discriminability + collision-tail analysis
        ↓
CAP-ZW training
        ↓
Ablation study
        ↓
Multi-seed statistics / confidence intervals
        ↓
Publication figures / tables
```

## Benchmark principles

All compared methods use the same image split, image size, hash length, attack set and evaluator. Synthetic medical-like phantoms are included only as deterministic smoke-test data; scientific claims must use an approved real medical dataset and explicitly report its split and preprocessing.

Reproduced results are kept separate from values quoted from the literature. No improvement percentage is hard-coded or fabricated.

DCT, DTCWT, KAZE, contrastive learning, balanced hashing, STE/BEMQ, BCH, chaos and multi-objective optimization are individually established ideas. Novelty must therefore be demonstrated by the specific formulation, integration, protocol and evidence.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/run_all.py
```

## CAP-ZW training

The training CLI exposes the active formulation and its parameters:

```powershell
python scripts\train_cap_zw.py --help
```

A small controlled pilot can be run with:

```powershell
python scripts\train_cap_zw.py `
  --manifest data\manifests\medical_manifest.csv `
  --split train_val `
  --limit 512 `
  --size 128 `
  --bits 128 `
  --epochs 2 `
  --batch-size 8 `
  --memory-size 256 `
  --memory-warmup 128 `
  --topk-negatives 8 `
  --robust-target 0.022 `
  --robust-softness 0.008 `
  --robustness-quantile 0.80 `
  --attack-views 3 `
  --tail-target 0.26 `
  --diversity-target 0.36 `
  --binary-collision-target 0.125 `
  --temperature 0.08 `
  --mgda-steps 15
```

## Evaluation

```powershell
python scripts\evaluate_cap_zw.py `
  --manifest data\manifests\medical_manifest.csv `
  --split test `
  --checkpoint experiments\checkpoints\cap_zw.pt `
  --limit 200 `
  --size 128 `
  --bits 128
```

The evaluator resolves the model version from checkpoint metadata rather than hard-coding a version string.

## Recommended real dataset: NIH ChestX-ray14

For the first full experiment we use **NIH ChestX-ray14**. It contains 112,120 frontal chest X-rays from 30,805 patients and includes 14 thoracic pathology labels. The Kaggle mirror is convenient for local download, while the patient identifier can be used directly as `group_id` to prevent patient-level leakage.

Kaggle authentication:

```powershell
kaggle auth login
```

Prepare a reproducible manifest:

```powershell
python scripts\prepare_nih_chestxray14.py --max-images 5000
```

The resulting manifest is:

```text
data/manifests/medical_manifest.csv
```

For publication experiments, keep the exact manifest, split, preprocessing, seed, hash length, attack grid and configuration used for every reported experiment.

## External validation plan

After the NIH experiment is stable, add **CheXpert** and/or **MIMIC-CXR-JPG** for external validation. These datasets have different access procedures and should not block the main NIH pipeline.

## Research status

The software pipeline and CAP-ZW training path are execution-tested locally. Scientific conclusions remain pending real-dataset experiments, multiple seeds, confidence intervals, and the complete ablation/benchmark matrix.
