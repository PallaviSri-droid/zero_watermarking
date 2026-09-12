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
    """Mean absolute pairwise bit correlation, ignoring constant columns.

    Constant hash bits have undefined Pearson correlation. They are excluded
    rather than passing NaNs into publication tables or emitting NumPy warnings.
    """
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
    return {
        "auc": auc,
        "eer": eer,
        "fpr": fpr,
        "tpr": tpr,
        "fnr": fnr,
        "thresholds": thresholds,
    }


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
    collision_gap = min(inter) - max(intra)
    return {
        "mean_intra_hd": float(np.mean(intra)),
        "max_intra_hd": float(np.max(intra)),
        "mean_ber": float(np.mean(intra)),
        "mean_nc": float(np.mean(genuine_nc)),
        "min_inter_hd": float(np.min(inter)),
        "mean_inter_hd": float(np.mean(inter)),
        "collision_gap": float(collision_gap),
        "auc": rs["auc"],
        "eer": rs["eer"],
        "details": rows,
    }
