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
    b = np.asarray(bits).astype(float)
    if b.shape[0] < 3 or b.shape[1] < 2:
        return float("nan")
    c = np.corrcoef(b, rowvar=False)
    iu = np.triu_indices_from(c, k=1)
    vals = np.abs(c[iu])
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else 0.0


def roc_stats(genuine_scores, impostor_scores):
    y = np.r_[np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))]
    scores = np.r_[genuine_scores, impostor_scores]
    auc = float(roc_auc_score(y, scores))
    fpr, tpr, _ = roc_curve(y, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[i] + fnr[i]) / 2)
    return {"auc": auc, "eer": eer, "fpr": fpr, "tpr": tpr}


def evaluate_hash_bank(clean_bank, attacked_bank):
    ids = sorted(clean_bank)
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

    rs = roc_stats([1 - x for x in intra], inter)
    collision_gap = (min(inter) if inter else np.nan) - (max(intra) if intra else np.nan)
    return {
        "mean_intra_hd": float(np.mean(intra)) if intra else np.nan,
        "max_intra_hd": float(np.max(intra)) if intra else np.nan,
        "mean_nc": float(np.mean(genuine_nc)) if genuine_nc else np.nan,
        "min_inter_hd": float(np.min(inter)) if inter else np.nan,
        "mean_inter_hd": float(np.mean(inter)) if inter else np.nan,
        "collision_gap": float(collision_gap),
        "auc": rs["auc"],
        "eer": rs["eer"],
        "details": rows,
    }
