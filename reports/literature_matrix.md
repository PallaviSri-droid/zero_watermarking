# Literature matrix for benchmark design

| Method / paper family | Domain | Main idea | Use in benchmark | Novelty caution |
|---|---|---|---|---|
| KAZE + DCT zero-watermarking | Medical | local feature + frequency transform | Strong classical baseline | Not novel by itself |
| ResNet50 + DCT | Medical | pretrained deep features + DCT + perceptual hash | Deep feature baseline | Published already |
| Improved NASNet-Mobile + DCT | Medical | transfer-learned feature + DCT | Lightweight deep baseline | Published already |
| DTCWT + entropy / QR / DCT families | Medical | multiscale transform and stable frequency features | Transform baselines | Published already |
| Multi-scale SURF / texture medical ZW | Medical | selected feature regions + multiscale representation | Strong classical robustness/discriminability baseline | Published in 2025 |
| DCNN + hyperchaotic medical ZW | Medical | deep feature maps + stable Gram features + encryption | Modern robustness/discriminability baseline | Discriminability already studied |
| SCSA + FocalNet medical ZW | Medical | attention + FocalNet feature extraction | Recent deep baseline | Published in 2026 |
| Dual-branch CNN-Transformer medical ZW | Medical | CNN + Transformer branches for robust feature learning | Recent deep baseline | Published/online before this project |
| Rel-Zero | General image | patch-pair relational invariance | External stress-test reference | CVPR 2026; not medical |
| CAP-ZW (this repo) | Medical target | collision-aware robustness/discrimination + Pareto optimization | Proposed candidate | Novelty must be verified experimentally and by systematic review |

## Candidate real-data tracks

### Primary radiography track

A large chest X-ray dataset is recommended because chest X-ray images are already used by published medical zero-watermarking studies, making it easier to construct a fair reproduction track. Do not copy the dataset into Git; keep it external and store only the manifest and protocol.

### Cross-modality track

Add at least one MRI or OCT dataset after the primary radiography benchmark. The purpose is to test whether the learned representation generalizes beyond one acquisition modality rather than merely optimizing for chest X-ray statistics.

### Development track

Use a small fixed subset of the approved dataset for rapid debugging. Development results must never be presented as final paper results.

## Benchmark protocol

For every method, use the same:

- image preprocessing and resolution;
- train/validation/test split, with patient/study-level grouping where available;
- watermark size and hash length;
- attack operators and attack strengths;
- negative-pair construction;
- random seeds;
- statistical analysis.

Report at minimum:

- NC and BER under every attack strength;
- mean and maximum intra-image Hamming distance;
- mean and minimum inter-image Hamming distance;
- ROC-AUC, FAR, FRR and EER;
- collision gap = minimum inter-image HD − maximum intra-image HD;
- nearest-neighbour false-match rate;
- bit balance error, mean bit entropy and mean absolute inter-bit correlation;
- runtime, memory and parameter count for learned models;
- mean ± confidence interval over multiple seeds.

## Reproduction policy

Separate **literature-reported values** from **results reproduced by this repository**. Published numbers must retain their original dataset and attack protocol. Reproduced values must come from the same implementation/evaluation code where possible. Never manufacture an improvement percentage to complete a table.

## Key references used for benchmark design

- Xiang et al., *A Trusted Medical Image Zero-Watermarking Scheme Based on DCNN and Hyperchaotic System*, IEEE Journal of Biomedical and Health Informatics, 2025, DOI: 10.1109/JBHI.2025.3550324.
- *Robust zero-watermarking algorithm via multi-scale feature analysis for medical images*, Journal of Information Security and Applications, 2025, DOI: 10.1016/j.jisa.2024.103937.
- *Robust medical image zero watermarking algorithm based on Spatial and Channel Synergistic Attention and FocalNet*, Biomedical Signal Processing and Control, 2026, DOI: 10.1016/j.bspc.2025.108243.
- Chen et al., *Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing*, CVPR 2026, pp. 3337–3346.

## Novelty direction

The working research hypothesis is that **explicit collision-aware optimization under a Pareto objective can improve the robustness–discriminability trade-off of medical zero-watermarking**. KAZE, DCT, DTCWT, CNN backbones, contrastive learning, balanced hashing, STE/BEMQ, BCH, chaos and multi-objective optimization are established components and should not be presented as individually novel.
