from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def hamming(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).astype(np.uint8).ravel()
    b = np.asarray(b).astype(np.uint8).ravel()
    if a.shape != b.shape:
        raise ValueError("hashes must have identical shape")
    return float(np.mean(a != b))


def ber(a: np.ndarray, b: np.ndarray) -> float:
    return hamming(a, b)


def nc(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).astype(float).ravel()
    b = np.asarray(b).astype(float).ravel()
    aa, bb = a - a.mean(), b - b.mean()
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float((aa @ bb) / den) if den else 1.0


def bit_balance(bits: np.ndarray) -> float:
    p = np.asarray(bits).astype(float).mean(axis=0)
    return float(np.mean(np.abs(p - 0.5)))


def bit_entropy(bits: np.ndarray):
    p = np.asarray(bits).astype(float).mean(axis=0)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return float(entropy.mean()), entropy


def mean_abs_corr(bits: np.ndarray) -> float:
    """Mean absolute pairwise bit correlation, ignoring constant columns."""
    b = np.asarray(bits).astype(float)
    if b.ndim != 2 or b.shape[0] < 2 or b.shape[1] < 2:
        return 0.0
    keep = np.std(b, axis=0) > 1e-12
    b = b[:, keep]
    if b.shape[1] < 2:
        return 0.0
    centered = b - b.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(centered, axis=0)
    corr = (centered.T @ centered) / np.outer(denom, denom)
    iu = np.triu_indices_from(corr, k=1)
    vals = np.abs(corr[iu])
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else 0.0


def roc_stats(genuine_scores, impostor_scores):
    """Compute ROC metrics where larger scores mean 'same image'."""
    genuine_scores = np.asarray(genuine_scores, dtype=float)
    impostor_scores = np.asarray(impostor_scores, dtype=float)
    if genuine_scores.size == 0 or impostor_scores.size == 0:
        raise ValueError("ROC requires both genuine and impostor scores")
    y = np.r_[np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))]
    scores = np.r_[genuine_scores, impostor_scores]
    auc = float(roc_auc_score(y, scores))
    fpr, tpr, thresholds = roc_curve(y, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[i] + fnr[i]) / 2)
    return {"auc": auc, "eer": eer, "fpr": fpr, "tpr": tpr, "fnr": fnr, "thresholds": thresholds}


def collision_statistics(inter: np.ndarray, intra: np.ndarray, thresholds=(0.05, 0.10)) -> dict[str, float | int]:
    """Report collision risk using pair-normalized rates and tail separation.

    Rates are per 10,000 negative image pairs.  Unlike the raw minimum-distance
    gap, these statistics remain interpretable when the evaluation set changes.
    """
    inter = np.asarray(inter, dtype=np.float64).ravel()
    intra = np.asarray(intra, dtype=np.float64).ravel()
    if inter.size == 0:
        raise ValueError("inter-image distances are required")
    if intra.size == 0:
        raise ValueError("intra-image distances are required")
    total = float(inter.size)
    exact = int(np.count_nonzero(inter == 0.0))
    result: dict[str, float | int] = {
        "negative_pairs": int(inter.size),
        "exact_collision_pairs": exact,
        "exact_collision_rate_per_10k": exact / total * 10000.0,
        "min_inter_hd": float(inter.min()),
        "mean_inter_hd": float(inter.mean()),
        "inter_q01": float(np.quantile(inter, 0.01)),
        "inter_q05": float(np.quantile(inter, 0.05)),
        "inter_q10": float(np.quantile(inter, 0.10)),
        "intra_q90": float(np.quantile(intra, 0.90)),
        "intra_q95": float(np.quantile(intra, 0.95)),
        "collision_gap": float(inter.min() - intra.max()),
        "q05_tail_gap": float(np.quantile(inter, 0.05) - np.quantile(intra, 0.95)),
        "q10_tail_gap": float(np.quantile(inter, 0.10) - np.quantile(intra, 0.90)),
    }
    for threshold in thresholds:
        tag = f"{threshold:.2f}".replace(".", "p")
        count = int(np.count_nonzero(inter <= threshold))
        result[f"collision_pairs_le_{threshold:.2f}"] = count
        result[f"collision_rate_le_{threshold:.2f}_per_10k"] = count / total * 10000.0
    return result


def evaluate_hash_bank(clean_bank, attacked_bank):
    """Evaluate robustness and verification discrimination on a hash bank."""
    ids = sorted(clean_bank)
    if len(ids) < 2:
        raise ValueError("At least two images are required for discrimination metrics")
    intra, inter, genuine_nc, rows = [], [], [], []
    for image_id in ids:
        clean = clean_bank[image_id]
        for attack_name, attacked in attacked_bank[image_id].items():
            d = hamming(clean, attacked)
            score = nc(clean, attacked)
            intra.append(d)
            genuine_nc.append(score)
            rows.append((image_id, attack_name, d, score))
    for pos, i in enumerate(ids):
        for j in ids[pos + 1 :]:
            inter.append(hamming(clean_bank[i], clean_bank[j]))

    genuine_scores = [1.0 - x for x in intra]
    impostor_scores = [1.0 - x for x in inter]
    rs = roc_stats(genuine_scores, impostor_scores)
    stats = collision_statistics(np.asarray(inter), np.asarray(intra))
    return {
        "mean_intra_hd": float(np.mean(intra)),
        "max_intra_hd": float(np.max(intra)),
        "mean_ber": float(np.mean(intra)),
        "mean_nc": float(np.mean(genuine_nc)),
        "min_inter_hd": stats["min_inter_hd"],
        "mean_inter_hd": stats["mean_inter_hd"],
        "collision_gap": stats["collision_gap"],
        "auc": rs["auc"],
        "eer": rs["eer"],
        "collision_statistics": stats,
        "details": rows,
    }
