# CAP-DINO-LogPolar v7 research audit

## 1. Status

This audit is based on the locked 11-attack evaluator and the measured CAP-family
ablation history already present in the repository. The v7 branch is a research
candidate; it is **not** a claimed improvement until a matched train/validation
selection and final locked-test run confirm it.

## 2. What v6 currently loses

The latest valid v6 200-image test run reports:

| Metric | v6 |
|---|---:|
| NC | 0.872471 |
| BER | 0.047901 |
| max intra-image HD | 0.210938 |
| mean inter-image HD | 0.271030 |
| min inter-image HD | 0.000000 |
| AUC | 0.955356 |
| EER | 0.115250 |
| bit balance error | 0.284141 |
| bit entropy | 0.570760 |
| mean absolute bit correlation | 0.363266 |
| exact collisions | 8 |
| q05 inter-image HD | 0.062500 |
| q10 inter-image HD | 0.078125 |
| q90 intra-image HD | 0.085938 |
| q95 intra-image HD | 0.101562 |
| q05 tail gap | -0.039062 |
| q10 tail gap | -0.007812 |
| <=0.05 collisions | 598 |
| <=0.10 collisions | 3237 |

The dominant gate was CAP for 100% of evaluated images, with mean gates
0.4265 / 0.2426 / 0.3309 for CAP / DINO / Log-Polar. This means the fusion is
still not using the complementary branches sufficiently.

### Critical optimization issue

In `cap_dino_logpolar_v6.py`, the hard robustness helper computes the q90 and
worst-case values from detached tensors. The dictionary also exposes detached
hard-tail statistics. Therefore the measured hard tail is useful for monitoring,
but the q90/worst component does not push the representation during backprop.
The mean hard term remains trainable. v7 removes this detach from the optimizer
path while keeping monitoring values detached.

## 3. Lessons from previous CAP candidates

### v11

Excellent clean/attack robustness but high collision risk. The measured behavior
shows that over-penalizing robustness can collapse the code space.

### v12

The strongest overall CAP-only trade-off so far: explicit lower-tail separation,
entropy floor and binary-confidence regularization substantially reduced exact
collisions while retaining useful robustness. Its main lesson is that the lower
inter-image tail must be optimized explicitly rather than relying only on a
nearest-negative objective.

### v14

Best collision-tail/entropy-oriented candidate among the tested controls. Its
hard entropy/balance constraints demonstrate the value of evaluating the actual
binary code rather than only continuous outputs. Its weakness is robustness loss.

### v6

v6 brings the fusion architecture, shared routing, branch dropout and multi-attack
training together, but the measured tail robustness remains poor and CAP routing
still dominates.

## 4. v7 changes

1. **Trainable hard robustness tail.** The hard q90 and worst-tail values remain in
the computation graph. This directly attacks v6's most important implementation
loss.

2. **Residual cross-branch interaction mixer.** The weighted convex mixture is
augmented with a small MLP over the three gated branch embeddings. The path starts
with a small residual scale so it cannot immediately overwhelm the proven v6
representation.

3. **Gate anti-dominance.** In addition to mean branch-usage bounds, v7 penalizes
rows where a single branch receives more than 0.58 probability. This specifically
targets the observed 100% CAP-dominant routing.

4. **Stronger hard-code tail control.** v7 combines lower-tail separation, hard
binary collision pressure, entropy, balance and decorrelation. These are inherited
research controls, not claimed novel components.

5. **Branch dropout increased to 0.25.** Two or more branches are retained per
sample, so the model cannot depend on CAP for every training example.

## 5. Why we are not replacing DINOv2 yet

DINOv2 is a strong generic visual representation and the repository currently
uses the small ViT-S/14 model. A chest-X-ray-specific backbone such as RAD-DINO is
an important future ablation, but swapping it at the same time as the v7 fusion
changes would confound the causal interpretation. Run RAD-DINO only after v7 is
stable and benchmarked.

## 6. Data/preprocessing audit items that remain

The current loader converts the image to grayscale and directly resizes it to a
square using bilinear interpolation. This is reproducible and must remain fixed
for the locked comparison, but it may distort aspect ratio. A publication-grade
ablation should compare the current stretch policy against an aspect-ratio-
preserving pad/crop policy while keeping every model and attack setting identical.
Do not silently change the locked benchmark preprocessing.

The training dataset also enumerates many attack views, but each dataset item
contains one attacked view at a time. A later robustness experiment should add a
multi-attack-per-image training path so the same clean image can be constrained
simultaneously across several corruptions. That is a separate controlled change
from v7 and should not be mixed into the first v7 comparison.

## 7. Publication protocol correction

The 200-image test subset has been used for development diagnostics. Those numbers
must therefore be labelled as development evidence, not a final unbiased test
claim. Final model selection should be performed on a validation partition from
`train_val`; the locked 967-image test partition should be touched once for the
final comparison.

## 8. Next experiments

Run the following in order:

1. v7 CPU smoke test: 128 images, 1 epoch, six attack views or the local smoke
setting; require finite loss/gradients and no skipped batch caused by numerical
failure.
2. v7 200-image development evaluation using the same locked 11-attack evaluator.
3. v7 5-epoch development run on the same 128-image training subset to study the
trajectory of robustness-tail, entropy, gate dominance and collisions.
4. If v7 is competitive, train on the full `train_val` partition and compare seeds
42/1337/2027 using the same configuration.
5. Freeze the selected v7 configuration, then run the locked 967-image test once.
6. Only after this, add RAD-DINO and a relational fourth branch as separate
ablations.

## 9. Success criteria

The goal is not to maximize one metric. A candidate is considered genuinely better
only when it improves the robustness/discriminability/collision trade-off without
creating a new failure in entropy, balance, routing concentration or attack-family
performance. Any reported gain must come from a matched reproduced benchmark, not
from literature values or a changed test protocol.
