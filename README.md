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
python scripts/run_benchmark.py
```

Open the notebooks in order from `notebooks/00_research_map.ipynb` through `notebooks/06_ablation_and_publication_figures.ipynb`.

## Structure

```text
zero_watermarking/
├── configs/
├── data/images/
├── experiments/results/
├── figures/
├── notebooks/
├── reports/
├── scripts/
├── src/zero_watermarking/
└── tests/
```

## Research status

This release establishes the reproducible benchmark and learned-method hand-off. The next scientific step is to run the protocol on the chosen real medical dataset(s), across multiple seeds, with confidence intervals and complete ablation tables.
