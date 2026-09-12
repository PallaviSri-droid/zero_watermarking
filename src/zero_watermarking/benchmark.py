from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .attacks import ATTACKS
from .baselines import method_registry
from .metrics import bit_balance, bit_entropy, evaluate_hash_bank, mean_abs_corr
from .synthetic import make_dataset


def run_benchmark(
    n_images: int = 40,
    image_size: int = 224,
    hash_length: int = 256,
    seed: int = 42,
    out_dir: str = "experiments/results",
) -> pd.DataFrame:
    """Run every baseline with an identical attack/evaluation protocol."""
    rng = np.random.default_rng(seed)
    images = make_dataset(n_images, image_size)
    methods = method_registry(hash_length)
    attacks = {
        "gaussian_noise": {"sigma": 0.03},
        "gaussian_blur": {"sigma": 1.0},
        "jpeg": {"quality": 50},
        "rotation": {"degrees": 5},
        "crop_resize": {"fraction": 0.05},
        "compound": {"seed": int(rng.integers(0, 100000))},
    }

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = []

    for method_name, signature_fn in methods.items():
        clean = {i: signature_fn(img) for i, img in images.items()}
        attacked = {i: {} for i in images}
        for i, image in images.items():
            for attack_name, kwargs in attacks.items():
                attacked[i][attack_name] = ATTACKS[attack_name](image, **kwargs)
                attacked[i][attack_name] = signature_fn(attacked[i][attack_name])

        evaluation = evaluate_hash_bank(clean, attacked)
        bit_bank = np.stack([clean[i] for i in sorted(clean)])
        entropy, _ = bit_entropy(bit_bank)

        summary.append(
            {
                "method": method_name,
                **{k: v for k, v in evaluation.items() if k != "details"},
                "balance_error": bit_balance(bit_bank),
                "mean_bit_entropy": entropy,
                "mean_abs_bit_corr": mean_abs_corr(bit_bank),
                "hash_length": hash_length,
                "n_images": n_images,
            }
        )

        details = pd.DataFrame(
            evaluation["details"],
            columns=["image_id", "attack", "hamming", "nc"],
        )
        details.to_csv(output / f"{method_name.replace('/', '_')}_details.csv", index=False)

    frame = pd.DataFrame(summary)
    frame.to_csv(output / "benchmark_summary.csv", index=False)
    (output / "benchmark_config.json").write_text(
        json.dumps(
            {
                "n_images": n_images,
                "image_size": image_size,
                "hash_length": hash_length,
                "seed": seed,
                "attacks": attacks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return frame
