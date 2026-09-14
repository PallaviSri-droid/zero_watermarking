# EXP-01 — Memory-backed hash-tail supervision

## Status

Research experiment. Not a production or final-paper result until the matched development benchmark supports it.

## Motivation

The frozen CAP-DINO-LogPolar stable-repair checkpoint shows a substantial continuous-to-binary degradation and a difficult inter-image tail. The training code already maintains a detached memory of previously seen clean binary codes, but the stable-repair objective does not consume that memory for its batch-local separation/collision terms. This leaves the collision objective with limited negative coverage.

The experiment therefore tests one controlled intervention: compare current clean/attacked soft codes against stored codes from different images during stage 2.

## Hypothesis

Persistent hard-negative coverage should reduce the lower tail of inter-image binary similarity while preserving attack invariance because the memory entries are detached and only the current query codes receive gradients.

## Intervention

Add `memory_hard_negative_loss` with:

- margin = 0.32
- temperature = 0.08
- top-k = 24 nearest stored negatives
- weight = 0.20
- memory warmup = 64 stored codes

The loss is active only during stage 2 and only after the memory reaches the warmup size.

No backbone, fusion architecture, quantizer, preprocessing, attack grid, or test protocol is changed.

## Baseline

Frozen CAP-DINO-LogPolar stable-repair development result (200 test images, diagnostic only):

- NC = 0.907688
- BER = 0.040217
- AUC = 0.937854
- EER = 0.139276
- inter Q05 = 0.039063
- inter Q10 = 0.054688
- intra Q95 = 0.109375
- exact collision pairs = 17
- pairs <= 0.05 = 1598
- pairs <= 0.10 = 4293

## Acceptance criteria

Primary:

- higher inter Q05 and Q10
- fewer exact collisions
- fewer <=0.05 and <=0.10 near-collisions

Guardrails:

- NC must not materially decrease
- BER must not materially increase
- intra Q95 must not materially worsen
- AUC/EER should not materially regress

## Reproducibility

Keep seed, image size, 128-bit hash length, manifest, training split, attack definitions, development evaluation subset, and evaluator unchanged.

The 967-image test partition remains reserved for the final locked comparison after model selection.
