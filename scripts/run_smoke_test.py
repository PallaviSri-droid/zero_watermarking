from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from zero_watermarking.benchmark import run_benchmark
from zero_watermarking.learned import HashEncoder
from zero_watermarking.metrics import bit_balance, bit_entropy, hamming
from zero_watermarking.synthetic import make_dataset
from zero_watermarking.transforms import transform_binary_hash
from zero_watermarking.watermark import create_zero_watermark, recover_watermark, normalized_correlation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a complete offline smoke test without a medical dataset.")
    parser.add_argument("--images", type=int, default=12)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=128)
    parser.add_argument("--out", default="experiments/results/smoke")
    args = parser.parse_args()

    images = make_dataset(args.images, args.size)
    assert len(images) == args.images

    # Classical representation + watermark registration/recovery.
    hashes = [transform_binary_hash(images[i], args.bits, "dct") for i in sorted(images)]
    bank = np.stack(hashes)
    entropy, _ = bit_entropy(bank)
    balance = bit_balance(bank)
    assert bank.shape == (args.images, args.bits)
    assert 0.0 <= entropy <= 1.0
    assert balance >= 0.0

    owner = np.resize(np.array([0, 1], dtype=np.uint8), args.bits)
    zw = create_zero_watermark(hashes[0], owner, key="smoke", length=args.bits)
    recovered = recover_watermark(hashes[0], zw, key="smoke", length=args.bits)
    assert normalized_correlation(owner, recovered) > 0.99
    assert hamming(owner, recovered) == 0.0

    # Neural model shape/gradient smoke test.
    import torch
    model = HashEncoder(nbits=args.bits)
    x = torch.from_numpy(images[0][None, None].astype(np.float32))
    logits = model(x, hard=False)
    assert logits.shape == (1, args.bits)
    logits.sum().backward()

    # Full classical benchmark and artifact creation.
    frame = run_benchmark(
        n_images=args.images,
        image_size=args.size,
        hash_length=args.bits,
        seed=42,
        out_dir=args.out,
    )
    assert not frame.empty
    assert Path(args.out, "benchmark_summary.csv").exists()

    print("SMOKE TEST PASSED")
    print(f"methods={len(frame)} images={args.images} bits={args.bits} entropy={entropy:.4f} balance_error={balance:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
