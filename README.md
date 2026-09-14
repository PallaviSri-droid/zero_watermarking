# Collision-Aware Zero-Watermarking for Medical Images

A reproducible research codebase for **medical-image zero-watermarking** focused on the joint problem of attack robustness, inter-image discrimination, and binary collision-tail control.

> **Research status:** exploratory. The current reference is CAP-DINO-LogPolar stable-repair. The present experiment branch adds memory-backed hard-negative supervision to address sparse mini-batch negative coverage. No superiority claim is made until matched development evaluation and later locked-test evaluation support it.

## Research question

Can a learned medical-image zero-watermark remain stable under attacks while keeping unrelated medical images well separated in the binary code space?

The project evaluates this as a three-way trade-off:

```text
attack invariance
      +
inter-image discrimination
      +
binary collision control
```

## Current model family

The reference model combines three complementary image representations:

```text
Medical X-ray
   │
   ├── CAP branch
   ├── DINOv2 branch
   └── Log-Polar branch
          │
          ▼
   adaptive three-branch fusion
          │
          ▼
   continuous fused representation
          │
          ▼
   learned 128-bit hash projection
          │
          ▼
   soft / hard binary code
```

The stable-repair objective already contains smooth attack consistency, representation contrastive alignment, pairwise separation, collision pressure, entropy/balance/decorrelation terms, and quantization pressure.

## Current research experiment: hash-tail memory

The current experiment is a controlled training-only intervention:

**memory-backed hard-negative supervision**

The trainer already maintains a detached memory of previously observed clean binary codes. This branch uses that memory during stage-2 optimization so each current clean/attacked query is compared against stored codes from other images, rather than relying only on the current mini-batch.

The intervention does **not** change the feature backbone or model architecture.

The new loss is implemented in:

```text
src/zero_watermarking/hash_tail_memory.py
```

and enabled by:

```text
scripts/train_cap_dino_logpolar_stable.py
```

Default experimental settings:

```text
lambda_memory_hard_negative = 0.20
memory_margin              = 0.32
memory_temperature         = 0.08
memory_topk                = 24
memory_warmup              = 64 stored codes
```

The term is disabled during stage 1 and activates only after the memory has enough entries to provide useful cross-image negatives.

## Development evidence so far

The frozen stable-repair 200-image diagnostic reported:

```text
NC              0.907688
BER             0.040217
AUC             0.937854
EER             0.139276

inter Q05       0.039063
inter Q10       0.054688
intra Q95       0.109375

exact collisions 17
<=0.05 pairs    1598
<=0.10 pairs    4293

bit entropy     0.601988
bit correlation 0.372248
```

These numbers are **development diagnostics**, not final test claims.

A separate continuous/logit/binary forensic analysis identified substantial information loss in the learned hashing pipeline and a large difficult-pair tail that already exists in continuous space. The current experiment therefore targets negative coverage before introducing another feature branch.

## Benchmark contract

All candidate comparisons should keep the following fixed:

- manifest and patient/group split
- preprocessing
- image size
- bit length
- seed
- attack definitions
- evaluation code
- development subset

The current locked development attack grid contains:

```text
Gaussian noise       sigma 0.03 / 0.08
Gaussian blur        sigma 1 / 2
JPEG                  Q70 / Q40
Rotation              5 / 10 degrees
Crop-resize           5%
Translation           3 pixels
Compound              fixed seed 23
```

The 200-image evaluation is for development diagnostics. The 967-image test partition is reserved for the final locked comparison after model selection.

## Metrics

### Robustness

- normalized correlation (NC) ↑
- bit error rate (BER) ↓
- mean intra-image Hamming distance ↓
- intra-image Q90/Q95 ↓
- maximum intra-image Hamming distance ↓

### Discrimination

- mean inter-image Hamming distance ↑
- inter-image Q01/Q05/Q10 ↑
- ROC-AUC ↑
- EER ↓

### Collision risk

- exact collision rate
- pairs with HD <= 0.05
- pairs with HD <= 0.10
- minimum inter-image distance
- q05/q10 tail gaps relative to intra-image attack distances

### Hash quality

- bit entropy
- balance error
- mean absolute bit correlation

## External reference

A recent 2026 Neurocomputing paper combines DINOv3-ViT with multi-scale neighborhood Zernike moments and reports strong robustness across filtering, compression, geometric, noise, and combined attacks. Its reported average NC is 0.975 for combined attacks and 0.964 for Gaussian noise in its experimental protocol.

Those results are not directly comparable to the current development benchmark because the paper evaluates eight selected medical images, uses DINOv3 ViT-B/16 at 512x512, and uses a different attack suite and watermark construction protocol.

The external paper is therefore treated as a **literature reference**, not as an apples-to-apples numerical baseline.

## Reproducibility

Every retained result should be traceable to:

```text
commit
→ configuration
→ seed
→ manifest
→ checkpoint
→ evaluator
→ attack grid
→ metrics
```

Do not tune model selection on the final 967-image test split.

## Research branches

```text
main
 └── research/cap-dino-logpolar-stable-repair
       ├── research/cap-dino-logpolar-relational       (deferred)
       └── research/hash-tail-memory-exp01             (current experiment)
```

The relational branch remains a deferred controlled ablation. It is not promoted until the hash-tail problem is better understood.

## Citation

See `CITATION.cff` for the repository citation record.
