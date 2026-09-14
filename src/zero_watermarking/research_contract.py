from __future__ import annotations

"""Shared reproducibility and fairness contract for publication benchmarks."""

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def manifest_fingerprint(path: str | Path) -> str:
    return sha256_file(path)[:16]


def attack_grid_fingerprint(specs: Iterable[Any]) -> str:
    rows = []
    for spec in specs:
        if hasattr(spec, "__dict__"):
            rows.append(spec.__dict__)
        elif isinstance(spec, dict):
            rows.append(spec)
        else:
            rows.append(str(spec))
    return fingerprint(rows)


def benchmark_identity(
    *,
    manifest: str | Path,
    image_size: int,
    bits: int,
    attacks: Iterable[Any],
    split: str,
    seed: int,
) -> dict[str, str | int]:
    attack_id = attack_grid_fingerprint(attacks)
    return {
        "manifest_id": manifest_fingerprint(manifest),
        "image_size": int(image_size),
        "bits": int(bits),
        "attack_grid_id": attack_id,
        "split": str(split),
        "seed": int(seed),
        "protocol_id": fingerprint(
            {
                "manifest_id": manifest_fingerprint(manifest),
                "image_size": int(image_size),
                "bits": int(bits),
                "attack_grid_id": attack_id,
                "split": str(split),
            }
        ),
    }


def assert_compatible_summaries(left: dict[str, Any], right: dict[str, Any]) -> None:
    """Reject unfair main-table comparisons when protocol identity differs."""
    fields = ("manifest_id", "image_size", "bits", "attack_grid_id", "split")
    mismatches = []
    for field in fields:
        if field in left and field in right and str(left[field]) != str(right[field]):
            mismatches.append(f"{field}: {left[field]!r} != {right[field]!r}")
    if mismatches:
        raise ValueError("Incompatible benchmark summaries:\n" + "\n".join(mismatches))


__all__ = [
    "sha256_file",
    "fingerprint",
    "manifest_fingerprint",
    "attack_grid_fingerprint",
    "benchmark_identity",
    "assert_compatible_summaries",
]
