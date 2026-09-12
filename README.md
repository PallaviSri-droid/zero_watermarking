# Collision-Aware Pareto Zero-Watermarking for Medical Images

A reproducible research codebase for medical-image zero-watermarking with a benchmark-first protocol.

## Research goal

The project tests whether jointly optimizing same-image robustness, different-image separation, collision-tail risk, bit balance, entropy and decorrelation can improve the robustness/discriminability trade-off compared with classical and learning-based baselines.

Working research label: **CAP-ZW (Collision-Aware Pareto Zero-Watermarking)**. This is a research label, not a claim of prior publication.

## Current learned formulation: CAP-ZW v5

CAP-ZW v5 adds a **collision-tail objective** to explicitly penalize the closest different-image hashes, together with:

- cross-batch memory mining;
- top-k hard-negative separation;
- a global hash-space uniformity penalty;
- balance + entropy + decorrelation regularization computed on clean and attacked views;
- tail-sensitive robustness weighting so difficult attacks are not hidden by the mean;
- curriculum scheduling that increases separation pressure during training;
- MGDA-style multi-objective gradient weighting.

The central hypothesis is not that any individual ingredient is new. The research question is whether this **joint collision-aware formulation and evaluation protocol** produces a measurable improvement in the robustness/discriminability trade-off and reduces cross-image hash collisions under a common medical-image benchmark.

Core diagnostic:

`collision_gap = min(inter-image Hamming distance) - max(intra-image Hamming distance)`

Additional tail diagnostics are reported with inter-image lower quantiles and intra-image upper quantiles, including `q05_tail_gap` and `q10_tail_gap`.

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
CAP-ZW v5 training
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

## Main metrics

- NC ↑, BER ↓
- mean / maximum intra-image Hamming distance ↓
- mean / minimum inter-image Hamming distance ↑
- inter-image lower-tail quantiles ↑
- intra-image upper-tail quantiles ↓
- ROC-AUC ↑, FAR ↓, FRR ↓, EER ↓
- collision gap ↑
- near/exact collision counts ↓
- bit balance error ↓
- mean bit entropy ↑
- mean absolute inter-bit correlation ↓

## Candidate contribution axes

1. Explicit **collision-tail aware optimization** for zero-watermark representations, rather than relying only on average pair separation.
2. **Pareto/multi-objective optimization** across robustness, discrimination and hash-quality objectives.
3. **Cross-batch collision memory + top-k hard-negative mining** for global rather than batch-local separation.
4. A reproducible **statistical benchmark + ablation framework** suitable for journal experiments.

DCT, DTCWT, KAZE, contrastive learning, balanced hashing, STE/BEMQ, BCH, chaos and multi-objective optimization are individually established ideas. Novelty must therefore be demonstrated by the specific formulation, integration, protocol and evidence.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/run_all.py
```

## CAP-ZW v5 pilot

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
  --tail-target 0.34 `
  --diversity-target 0.45 `
  --temperature 0.08 `
  --mgda-steps 15
```

Evaluation:

```powershell
python scripts\evaluate_cap_zw.py `
  --manifest data\manifests\medical_manifest.csv `
  --split test `
  --checkpoint experiments\checkpoints\cap_zw.pt `
  --limit 200 `
  --size 128 `
  --bits 128
```

## Recommended real dataset: NIH ChestX-ray14

For the first full experiment we use **NIH ChestX-ray14**. It contains 112,120 frontal chest X-rays from 30,805 patients and includes 14 thoracic pathology labels. The Kaggle mirror is convenient for local download, while the patient identifier can be used directly as `group_id` to prevent patient-level leakage.

Kaggle authentication:

```powershell
kaggle auth login
```

Download, recursively extract nested image archives, read the NIH metadata, match image files, and build a verified manifest automatically:

```powershell
python scripts\prepare_nih_chestxray14.py
```

For a smaller pilot:

```powershell
python scripts\prepare_nih_chestxray14.py --max-images 5000
```

The resulting manifest is:

```text
data/manifests/medical_manifest.csv
```

The NIH-specific preparation script prefers the dataset's standard `train_val_list.txt` and `test_list.txt` files when present and records the actual `Patient ID` as `group_id`. This is preferable to the generic manifest generator, which can only infer grouping when metadata are unavailable.

## External validation plan

After the NIH experiment is stable, add **CheXpert** and/or **MIMIC-CXR-JPG** for external validation. These datasets have different access procedures and should be treated as separate validation cohorts.

## Structure

```text
zero_watermarking/
├── configs/
├── data/images/
├── data/raw/
├── experiments/results/
├── figures/
├── notebooks/
├── reports/
├── scripts/
│   ├── prepare_nih_chestxray14.py
│   └── ...
├── src/zero_watermarking/
└── tests/
```

## Research status

The CAP-ZW v5 formulation is implemented in the repository. v4 pilot results motivated the tail-collision upgrade, but no claim of overall superiority should be made until v5 is evaluated against the same baselines on the same held-out images across multiple seeds and attacks, with confidence intervals and a complete ablation matrix.
