# CAP-ZW v12 Research Update

## Why v12 exists

The 200-image ablation exposed a representation-collapse failure mode: the robustness-only reference achieved extremely low BER but produced 2,673 exact collisions and 17,640 near collisions among 19,900 negative pairs. The locked CAP-ZW candidate reduced this collapse but still had 185 exact collisions and a low inter-image lower tail.

The v12 candidate therefore changes the optimization target rather than simply increasing robustness weight.

## Literature-grounded design changes

1. **Explicit lower-tail separation.** Recent zero-watermarking work identifies false positives / poor discriminability as an important weakness, not just robustness. Rad-Mark explicitly targets false positives and adversarial feature optimization. The 2025 JBHI medical-image work likewise emphasizes discriminability as under-studied.
2. **Entropy and bit independence.** Deep hashing literature supports balanced, high-entropy, low-correlation binary codes. v12 adds an explicit entropy floor while retaining balance/decorrelation terms already present in CAP-ZW.
3. **Broader hard-negative coverage.** Hard-negative mining is useful, but relying only on the nearest few negatives can leave the global lower tail poorly controlled. v12 increases top-k coverage and uses a memory bank.
4. **Attack diversity.** v12 trains with noise, salt-and-pepper noise, blur, JPEG, brightness, contrast, rotation, crop-resize, translation, and compound attacks. This is intended to improve generalization across attack families rather than overfit a small training attack subset.
5. **Robustness guard retained but softened.** The 200-image results showed that the original guard can over-constrain separation. v12 keeps a selective guard but lowers its initial influence and relaxes its target.

## What is deliberately not claimed

- v12 is not declared superior to v11 before evaluation.
- The new losses are not claimed as individually novel; they are engineering/research mechanisms assembled into the collision-aware zero-watermarking objective.
- Any improvement claim requires the locked test protocol, multiple seeds, confidence intervals, and competitor reproduction.

## Required evaluation

1. 1-epoch smoke test on a small training subset.
2. 200-image locked test screening against v11 and all registered ablations.
3. 5 seeds on the selected candidate.
4. Common competitor benchmark with identical manifest, attack grid, hash length, and evaluator.
5. Statistical confidence intervals and paired comparisons.
