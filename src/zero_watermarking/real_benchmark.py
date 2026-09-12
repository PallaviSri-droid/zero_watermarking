from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .attacks import ATTACKS
from .baselines import method_registry
from .datasets import load_image, load_manifest, validate_manifest
from .metrics import bit_balance, bit_entropy, evaluate_hash_bank, mean_abs_corr


def run_manifest_benchmark(
    manifest_path: str,
    root: str | None = None,
    image_size: int = 224,
    hash_length: int = 256,
    seed: int = 42,
    max_images: int | None = None,
    out_dir: str = "experiments/results/real",
) -> pd.DataFrame:
    """Benchmark classical methods on a user-supplied medical-image manifest.

    The manifest is validated before any image is processed. This runner keeps
    the attack functions and metric implementation identical across methods,
    so comparisons differ only by the representation algorithm.
    """
    records = load_manifest(manifest_path)
    frame = validate_manifest(records, root=root)
    missing = frame.loc[~frame["exists"]]
    if not missing.empty:
        examples = missing["path"].head(5).tolist()
        raise FileNotFoundError(f"Missing {len(missing)} images; examples: {examples}")

    if max_images is not None:
        frame = frame.head(max_images).copy()
    image_map = {
        str(row.image_id): load_image(row.path, size=image_size)
        for row in frame.itertuples(index=False)
    }

    attacks = {
        "gaussian_noise": {"sigma": 0.03},
        "gaussian_blur": {"sigma": 1.0},
        "jpeg": {"quality": 50},
        "rotation": {"degrees": 5},
        "crop_resize": {"fraction": 0.05},
        "compound": {"seed": seed},
    }
    registry = method_registry(hash_length)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for method_name, signature_fn in registry.items():
        clean = {k: signature_fn(v) for k, v in image_map.items()}
        attacked = {k: {} for k in image_map}
        for image_id, image in image_map.items():
            for attack_name, kwargs in attacks.items():
                attacked[image_id][attack_name] = signature_fn(ATTACKS[attack_name](image, **kwargs))

        evaluation = evaluate_hash_bank(clean, attacked)
        bank = np.stack([clean[k] for k in sorted(clean)])
        entropy, _ = bit_entropy(bank)
        rows.append(
            {
                "method": method_name,
                **{k: v for k, v in evaluation.items() if k != "details"},
                "balance_error": bit_balance(bank),
                "mean_bit_entropy": entropy,
                "mean_abs_bit_corr": mean_abs_corr(bank),
                "hash_length": hash_length,
                "n_images": len(image_map),
                "dataset_manifest": str(manifest_path),
                "seed": seed,
            }
        )

        details = pd.DataFrame(
            evaluation["details"],
            columns=["image_id", "attack", "hamming", "nc"],
        )
        details.to_csv(output / f"{method_name.replace('/', '_')}_details.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(output / "benchmark_summary.csv", index=False)
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "root": root,
                "image_size": image_size,
                "hash_length": hash_length,
                "seed": seed,
                "attacks": attacks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary
