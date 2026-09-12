# Literature matrix for benchmark design

| Method / paper family | Domain | Main idea | Use in benchmark | Novelty caution |
|---|---|---|---|---|
| KAZE + DCT zero-watermarking | Medical | local feature + frequency transform | Strong classical baseline | Not novel by itself |
| ResNet50 + DCT | Medical | pretrained deep features + DCT + perceptual hash | Deep feature baseline | Published already |
| Improved NasNet-Mobile + DCT | Medical | transfer-learned feature + DCT | Deep feature baseline | Published already |
| DTCWT + QR + DCT | Medical | multiscale transform + QR stability | Transform baseline | Published already |
| DCNN + hyperchaotic medical ZW | Medical | deep features + encryption + robustness/discriminability | Modern medical baseline | Discriminability already studied |
| SCSA + FocalNet medical ZW | Medical | attention + FocalNet | Recent deep baseline | Recent literature already exists |
| Rel-Zero | General image | patch-pair relational invariance | External stress-test reference | Not medical; do not compare as if same protocol |
| CAP-ZW (this repo) | Medical target | collision-aware robustness/discrimination + Pareto optimization | Proposed candidate | Novelty must be verified by systematic review |

## Journal benchmark requirements

Do not stop at NC. Jointly report intra-image distance under attack, inter-image distance across distinct images/patients, ROC-AUC/EER, nearest-neighbour false matches, bit entropy/balance/correlation, compound attacks, and ablations.

The main candidate contribution is the **explicit optimization and evaluation of collision separation**, not any single standard component.
