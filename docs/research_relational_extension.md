# Relational Patch Identity Extension

## Motivation

The frozen stable-repair checkpoint separates attacked views from most different-image pairs in continuous space, but binary collision tails remain poor. Recent zero-watermarking research, including Rel-Zero (CVPR 2026), motivates exploiting relationships between local patches rather than relying only on absolute global descriptors.

## Design

The candidate keeps the CAP + DINOv2 + Log-Polar stable-repair path unchanged and adds a small residual branch derived from local DINO patch-token cosine statistics at multiple spatial offsets. The branch is deliberately low-weight and initialized near zero, so the experiment asks whether relational information improves difficult-pair separation without destroying the established robustness.

## Hypothesis

H1: relational patch statistics reduce the lower tail of inter-image similarity and therefore reduce exact/near binary collisions.

H2: explicit relation consistency across clean/attacked views prevents the new branch from becoming attack-sensitive.

## Falsification criteria

Reject the extension if, on the fixed 200-image development evaluation, it materially lowers NC or raises BER without a meaningful improvement in collision-tail metrics. Do not promote it to the final test or mainline until the candidate is compared against the frozen stable baseline under the same attack grid.
