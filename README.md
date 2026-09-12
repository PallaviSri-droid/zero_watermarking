# Collision-Aware Pareto Zero-Watermarking for Medical Images

A reproducible research codebase for medical-image zero-watermarking with a benchmark-first protocol.

## Research goal

The project tests whether jointly optimizing same-image robustness, different-image separation, bit balance, and bit decorrelation can improve the robustness/discriminability trade-off compared with classical and learning-based baselines.

Working research label: **CAP-ZW (Collision-Aware Pareto Zero-Watermarking)**. This is a research label, not a claim of prior publication.

Core diagnostic:

`collision_gap = min(inter-image Hamming distance) - max(intra-image Hamming distance)`

A larger positive gap means attacked views of the same image remain closer than hashes from different images.

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
Discriminability + collision analysis
        ↓
Learned CAP-ZW prototype
        ↓
Ablation study
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
- ROC-AUC ↑, FAR ↓, FRR ↓, EER ↓
- collision gap ↑
- bit balance error ↓
- mean bit entropy ↑
- mean absolute inter-bit correlation ↓

## Candidate contribution axes

1. Explicit **collision-aware hard-negative optimization** for zero-watermark representations.
2. **Pareto/multi-objective optimization** across robustness, discrimination and hash-quality objectives.
3. **Attack-stratified and compound-attack** evaluation with common protocols.
4. A reproducible **statistical benchmark + ablation framework** suitable for journal experiments.

DCT, DTCWT, KAZE, contrastive learning, balanced hashing, STE/BEMQ, BCH, chaos and multi-objective optimization are individually established ideas. Novelty must therefore be demonstrated by the specific formulation, integration, protocol and evidence.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/run_all.py
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

For a smaller pilot before downloading/processing the entire collection:

```powershell
python scripts\prepare_nih_chestxray14.py --max-images 5000
```

The resulting manifest is:

```text
data/manifests/medical_manifest.csv
```

The NIH-specific preparation script prefers the dataset's standard `train_val_list.txt` and `test_list.txt` files when present and records the actual `Patient ID` as `group_id`. This is preferable to the generic manifest generator, which can only infer grouping when metadata are unavailable.

After preparation:

```powershell
python scripts\run_all.py --real --limit 200
```

For publication experiments, do not cap the dataset arbitrarily unless the paper explicitly defines a reproducible subset. Keep the exact manifest, split, preprocessing, seed, hash length, attack grid and configuration used for every reported experiment.

## External validation plan

After the NIH experiment is stable, add **CheXpert** and/or **MIMIC-CXR-JPG** for external validation. These datasets are larger but have different access procedures, so they should not block the main NIH pipeline.

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

The software pipeline and offline CAP-ZW training path are execution-tested locally. Scientific conclusions remain pending real-dataset experiments, multiple seeds, confidence intervals, and the complete ablation/benchmark matrix.
