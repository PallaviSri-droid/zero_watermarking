# CAP + DINOv2 + Log-Polar Fusion

The next research candidate will use three complementary branches: a trainable CAP branch for medical-image robustness, frozen DINOv2 for semantic/global structure, and a log-polar Fourier descriptor for rotation/scale-oriented geometry. We will use learned late fusion after per-branch normalization rather than a raw concatenation followed by random projection.

The individual components are established building blocks; the research question is whether their complementary error structure improves the robustness/discriminability/collision Pareto frontier under the locked benchmark.

Planned ablations: CAP-only, DINO-only, Log-Polar-only, CAP+DINO, CAP+Log-Polar, DINO+Log-Polar, and all three. Selection will use identical train/test splits, attacks, hash length, and evaluator, followed by multi-seed confidence intervals.
