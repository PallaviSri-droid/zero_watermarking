from __future__ import annotations

"""CAP-ZW-v15: dual-space relational contrastive hashing candidate.

Motivation from V11-V14:
- V11 retained robustness but suffered severe code collisions.
- V12/V13/V14 reduced collisions but paid too much in robustness.
- The shared bottleneck was optimizing every property through one small hash
  head. V15 separates representation learning from binary quantization.

V15 therefore learns:
1. a continuous unit-norm semantic embedding for global discrimination,
2. a spatial relational descriptor that is trained to remain stable under
   attacks,
3. a binary hash head trained with explicit collision/bit-quality objectives.

The method is a research candidate, not a superiority claim.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .training import PairAttackDataset, _update_memory

CAP_ZW_V15_VERSION = "CAP-ZW-v15"


class ResidualGNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups1 = max(1, min(8, out_channels))
        groups2 = max(1, min(8, out_channels))
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.GroupNorm(groups1, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(groups2, out_channels)
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class RelationalHashNetV15(nn.Module):
    """Multi-scale residual encoder with a relational branch and binary head."""

    def __init__(self, nbits: int = 128, embedding_dim: int = 128, relation_dim: int = 96) -> None:
        super().__init__()
        self.nbits = int(nbits)
        self.embedding_dim = int(embedding_dim)
        self.relation_dim = int(relation_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.stage1 = ResidualGNBlock(32, 48, stride=2)
        self.stage2 = ResidualGNBlock(48, 80, stride=2)
        self.stage3 = ResidualGNBlock(80, 128, stride=2)
        self.context = nn.Sequential(
            nn.Conv2d(128, 160, 3, 1, 1, bias=False),
            nn.GroupNorm(8, 160),
            nn.GELU(),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(160 + 160 * 4 * 4, 256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, embedding_dim),
        )
        relation_pairs = (16 * 15) // 2
        self.relation_proj = nn.Sequential(
            nn.Linear(relation_pairs, 128),
            nn.GELU(),
            nn.Linear(128, relation_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim + relation_dim, 192),
            nn.GELU(),
            nn.Linear(192, 128),
            nn.GELU(),
        )
        self.hash_head = nn.Linear(128, nbits)
        self.thresholds = nn.Parameter(torch.zeros(nbits))

    def _features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        feat = self.context(x)
        global_avg = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        grid = F.adaptive_avg_pool2d(feat, (4, 4)).flatten(1)
        global_embedding = F.normalize(self.global_proj(torch.cat([global_avg, grid], dim=1)), dim=1)

        patches = F.adaptive_avg_pool2d(feat, (4, 4)).flatten(2).transpose(1, 2)
        patches = F.normalize(patches, dim=-1)
        similarity = torch.bmm(patches, patches.transpose(1, 2))
        idx = torch.triu_indices(16, 16, offset=1, device=x.device)
        relations = similarity[:, idx[0], idx[1]]
        relation_embedding = F.normalize(self.relation_proj(relations), dim=1)
        return global_embedding, relation_embedding

    def forward(self, x: Tensor, hard: bool = True, return_aux: bool = False):
        global_embedding, relation_embedding = self._features(x)
        fused = self.fusion(torch.cat([global_embedding, relation_embedding], dim=1))
        logits = self.hash_head(fused)
        soft = torch.sigmoid(logits - self.thresholds)
        if hard:
            bits = (logits >= self.thresholds).to(soft.dtype)
            code = bits.detach() - soft.detach() + soft
        else:
            code = soft
        if return_aux:
            return code, global_embedding, relation_embedding, logits
        return code


@dataclass
class V15Config:
    epochs: int = 20
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 2e-4
    nbits: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_clip: float = 5.0
    embedding_margin: float = 0.45
    binary_margin: float = 0.30
    relation_consistency_target: float = 0.08
    robust_target: float = 0.022
    entropy_floor: float = 0.78
    balance_target: float = 0.15
    confidence_target: float = 0.16
    lambda_embedding_pos: float = 1.0
    lambda_embedding_neg: float = 0.95
    lambda_relation: float = 0.75
    lambda_robust: float = 1.0
    lambda_binary: float = 0.90
    lambda_collision_tail: float = 1.10
    lambda_collision_mass: float = 0.65
    lambda_entropy: float = 0.60
    lambda_balance: float = 0.40
    lambda_decorrelation: float = 0.35
    lambda_confidence: float = 0.10
    temperature: float = 0.07
    topk_negatives: int = 24
    memory_size: int = 8192
    attack_views: int = 4


def _hamming(a: Tensor, b: Tensor) -> Tensor:
    return torch.cdist(a, b, p=1) / a.shape[1]


def _negative_matrix(codes: Tensor, labels: Tensor, memory_codes: Tensor | None, memory_labels: Tensor | None) -> tuple[Tensor, Tensor | None]:
    dist = _hamming(codes, codes)
    same = labels[:, None].eq(labels[None, :])
    batch_neg = dist.masked_fill(same, float("inf"))
    mem_neg = None
    if memory_codes is not None and memory_codes.numel():
        mem_neg = _hamming(codes, memory_codes.detach())
        if memory_labels is not None:
            mem_neg = mem_neg.masked_fill(labels[:, None].eq(memory_labels.detach()[None, :]), float("inf"))
    return batch_neg, mem_neg


def _topk_mean(values: Tensor, k: int) -> Tensor:
    finite = torch.where(torch.isfinite(values), values, torch.full_like(values, 2.0))
    kk = min(max(int(k), 1), finite.shape[1])
    return torch.topk(finite, kk, largest=False, dim=1).values


def _continuous_negative_loss(embedding: Tensor, labels: Tensor, margin: float, temperature: float) -> Tensor:
    sims = embedding @ embedding.T
    same = labels[:, None].eq(labels[None, :])
    negatives = sims.masked_fill(same, -2.0)
    nearest = negatives.max(dim=1).values
    tau = max(float(temperature), 1e-4)
    return F.softplus((nearest - (1.0 - float(margin))) / tau).mul(tau).mean()


def _binary_collision_terms(clean: Tensor, attacked: Tensor, labels: Tensor, memory_codes: Tensor | None, memory_labels: Tensor | None,
                            margin: float, topk: int) -> tuple[Tensor, Tensor]:
    codes = torch.cat([clean, attacked], dim=0)
    labs = torch.cat([labels, labels], dim=0)
    batch_neg, mem_neg = _negative_matrix(codes, labs, memory_codes, memory_labels)
    sources = [batch_neg] + ([mem_neg] if mem_neg is not None else [])
    tails = []
    masses = []
    for source in sources:
        vals = _topk_mean(source, topk)
        tails.append(F.relu(float(margin) - vals[:, 0]).pow(2).mean())
        masses.append(F.relu(float(margin) - vals).pow(2).mean())
    return torch.stack(tails).mean(), torch.stack(masses).mean()


def _bit_quality(code: Tensor, entropy_floor: float, balance_target: float, confidence_target: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    p = code.mean(0).clamp(1e-5, 1.0 - 1e-5)
    entropy = -(p * torch.log2(p) + (1.0 - p) * torch.log2(1.0 - p)).mean()
    balance = (p - 0.5).abs().mean()
    centered = code - code.mean(0, keepdim=True)
    std = centered.std(0, unbiased=False).clamp_min(1e-3)
    norm = centered / std
    corr = (norm.T @ norm) / max(code.shape[0], 1)
    eye = torch.eye(corr.shape[0], device=code.device, dtype=code.dtype)
    dec = ((corr - eye) * (1.0 - eye)).pow(2).mean()
    hard = (code >= 0.5).to(code.dtype).detach() - code.detach() + code
    hp = hard.mean(0).clamp(1e-5, 1.0 - 1e-5)
    h_entropy = -(hp * torch.log2(hp) + (1.0 - hp) * torch.log2(1.0 - hp)).mean()
    h_balance = (hp - 0.5).abs().mean()
    confidence = F.relu(float(confidence_target) - (2.0 * code - 1.0).abs()).pow(2).mean()
    loss = F.relu(float(entropy_floor) - entropy).pow(2) + 0.5 * F.relu(h_balance - float(balance_target)).pow(2) + 0.5 * F.relu(float(entropy_floor) - h_entropy).pow(2)
    return loss, dec, h_entropy.detach(), confidence


def train_cap_zw_v15(model: nn.Module, loader, config: V15Config, checkpoint: str | None = None, history_path: str | None = None) -> list[dict[str, float]]:
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    memory_embed = torch.empty((0, model.embedding_dim), device=device)
    memory_rel = torch.empty((0, model.relation_dim), device=device)
    memory_codes = torch.empty((0, config.nbits), device=device)
    memory_labels = torch.empty((0,), dtype=torch.long, device=device)
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        totals = {"embedding_pos": 0.0, "embedding_neg": 0.0, "relation": 0.0, "robustness": 0.0,
                  "binary_collision": 0.0, "collision_mass": 0.0, "quality": 0.0, "decorrelation": 0.0,
                  "confidence": 0.0, "total": 0.0, "memory_size": 0.0}
        batches = 0
        progress = min(1.0, (epoch + 1) / max(config.epochs * 0.60, 1.0))
        current_margin = 0.18 + (config.binary_margin - 0.18) * progress
        for batch in loader:
            clean_x, attacked_x, labels = batch[:3]
            clean_x, attacked_x, labels = clean_x.to(device), attacked_x.to(device), labels.to(device)
            clean_bits, clean_emb, clean_rel, _ = model(clean_x, hard=False, return_aux=True)
            attacked_bits, attacked_emb, attacked_rel, _ = model(attacked_x, hard=False, return_aux=True)

            embedding_pos = (1.0 - (clean_emb * attacked_emb).sum(dim=1)).mean()
            embedding_neg = 0.5 * (
                _continuous_negative_loss(clean_emb, labels, config.embedding_margin, config.temperature)
                + _continuous_negative_loss(attacked_emb, labels, config.embedding_margin, config.temperature)
            )
            relation = F.relu((clean_rel - attacked_rel).pow(2).mean(dim=1).sqrt() - config.relation_consistency_target).pow(2).mean()
            robustness = (clean_bits - attacked_bits).abs().mean()
            exact, mass = _binary_collision_terms(clean_bits, attacked_bits, labels, memory_codes if memory_codes.numel() else None,
                                                   memory_labels if memory_labels.numel() else None, current_margin, config.topk_negatives)
            quality, decorrelation, _, confidence = _bit_quality(torch.cat([clean_bits, attacked_bits], dim=0), config.entropy_floor, config.balance_target, config.confidence_target)

            loss = (
                config.lambda_embedding_pos * embedding_pos
                + config.lambda_embedding_neg * embedding_neg
                + config.lambda_relation * relation
                + config.lambda_robust * robustness
                + config.lambda_binary * exact
                + config.lambda_collision_tail * exact
                + config.lambda_collision_mass * mass
                + config.lambda_entropy * quality
                + config.lambda_decorrelation * decorrelation
                + config.lambda_confidence * confidence
                + 0.18 * _hamming(clean_bits, attacked_bits).mean()
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            memory_embed, _ = _update_memory(memory_embed, torch.cat([clean_emb, attacked_emb], dim=0),
                                             torch.zeros(clean_emb.shape[0] * 2, dtype=torch.long, device=device), 4096)
            memory_codes, memory_labels = _update_memory(memory_codes, memory_labels, torch.cat([clean_bits, attacked_bits], dim=0),
                                                         torch.cat([labels, labels], dim=0), config.memory_size)
            totals["embedding_pos"] += float(embedding_pos.detach())
            totals["embedding_neg"] += float(embedding_neg.detach())
            totals["relation"] += float(relation.detach())
            totals["robustness"] += float(robustness.detach())
            totals["binary_collision"] += float(exact.detach())
            totals["collision_mass"] += float(mass.detach())
            totals["quality"] += float(quality.detach())
            totals["decorrelation"] += float(decorrelation.detach())
            totals["confidence"] += float(confidence.detach())
            totals["total"] += float(loss.detach())
            totals["memory_size"] += float(memory_codes.shape[0])
            batches += 1

        denom = max(batches, 1)
        row = {"epoch": float(epoch + 1)}
        row.update({k: v / denom for k, v in totals.items()})
        history.append(row)
        if checkpoint:
            path = Path(checkpoint)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "config": config.__dict__,
                "version": CAP_ZW_V15_VERSION,
                "model_type": "RelationalHashNetV15",
            }, path)
    if history_path:
        path = Path(history_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(path, index=False)
    return history


__all__ = ["CAP_ZW_V15_VERSION", "V15Config", "RelationalHashNetV15", "PairAttackDataset", "train_cap_zw_v15"]
