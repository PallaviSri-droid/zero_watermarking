from __future__ import annotations

"""CAP-DINO-LogPolar + relational patch identity branch."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from .cap_dino_logpolar_stable import CAPDinoLogPolarStable, StableConfig, _finite, _safe_cosine_loss

RELATIONAL_VERSION = "CAP-DINO-LogPolar-relational"

@dataclass
class RelationalConfig(StableConfig):
    relation_dim: int = 24
    relation_hidden: int = 96
    relation_weight: float = 0.18
    relation_scale_init: float = -2.2
    lambda_relation_consistency: float = 0.15

class CAPDinoLogPolarRelational(CAPDinoLogPolarStable):
    """Stable repair plus a conservative relational residual identity path."""
    def __init__(self, config: RelationalConfig | None = None) -> None:
        super().__init__(config or RelationalConfig())
        c = self.config
        hidden = max(int(c.relation_hidden), int(c.fusion_dim // 2))
        self.relation_project = nn.Sequential(
            nn.LayerNorm(c.relation_dim), nn.Linear(c.relation_dim, hidden), nn.GELU(),
            nn.Linear(hidden, c.fusion_dim)
        ).to(self.runtime_device)
        self.relation_scale_logit = nn.Parameter(torch.tensor(float(c.relation_scale_init), device=self.runtime_device))

    @property
    def relation_scale(self) -> Tensor:
        return torch.sigmoid(self.relation_scale_logit) * float(self.config.relation_weight)

    @torch.no_grad()
    def dino_patch_tokens(self, x: Tensor) -> Tensor:
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(self.config.dino_size, self.config.dino_size), mode="bicubic", align_corners=False)
        mean = x.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = x.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        feats: list[Tensor] = []
        for start in range(0, x.shape[0], 16):
            out = self.dino.forward_features((x[start:start + 16] - mean) / std)
            tokens = out.get("x_norm_patchtokens")
            if tokens is None:
                tokens = out.get("x_prenorm")
                if tokens is None:
                    raise RuntimeError("DINO backbone did not expose patch tokens")
                tokens = tokens[:, 1:]
            feats.append(tokens.float())
        return torch.cat(feats, dim=0)

    def relational_features(self, x: Tensor) -> Tensor:
        tokens = self.dino_patch_tokens(x)
        b, n, _ = tokens.shape
        side = int(round(n ** 0.5))
        if side * side != n:
            raise RuntimeError(f"Expected square DINO patch grid, got {n} tokens")
        grid = F.normalize(tokens, dim=-1).reshape(b, side, side, -1)
        offsets = ((0,1),(1,0),(1,1),(1,-1),(0,2),(2,0),(2,2),(2,-2),(0,3),(3,0),(3,3),(3,-3))
        values: list[Tensor] = []
        for dy, dx in offsets:
            y0, y1 = max(0, -dy), min(side, side - dy)
            x0, x1 = max(0, -dx), min(side, side - dx)
            a = grid[:, y0:y1, x0:x1]
            b2 = grid[:, y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            sim = (a * b2).sum(dim=-1)
            values.append(sim.mean(dim=(1,2)))
            values.append(sim.std(dim=(1,2), unbiased=False))
        rel = torch.stack(values, dim=-1)
        return _finite((rel - rel.mean(dim=0, keepdim=True)) / rel.std(dim=0, unbiased=False).clamp_min(1e-3), 10.0)

    def _relational_residual(self, x: Tensor) -> Tensor:
        return self.relation_project(self.relational_features(x)) * self.relation_scale

    def forward_pair(self, clean_x: Tensor, attacked_x: Tensor):
        base = super().forward_pair(clean_x, attacked_x)
        clean_rel = self._relational_residual(clean_x)
        attacked_rel = self._relational_residual(attacked_x)
        clean_fused = _finite(base["clean_fused"] + clean_rel)
        attacked_fused = _finite(base["attacked_fused"] + attacked_rel)
        clean_logits = self.fusion(clean_fused)
        attacked_logits = self.fusion(attacked_fused)
        clean_soft = self.quantizer(clean_logits, hard=False)
        attacked_soft = self.quantizer(attacked_logits, hard=False)
        clean_hard = self.quantizer(clean_logits, hard=True)
        attacked_hard = self.quantizer(attacked_logits, hard=True)
        base.update({"clean_relational": clean_rel, "attacked_relational": attacked_rel,
                     "clean_fused": clean_fused, "attacked_fused": attacked_fused,
                     "clean_logits": clean_logits, "attacked_logits": attacked_logits,
                     "clean_soft": clean_soft, "attacked_soft": attacked_soft,
                     "clean_hard": clean_hard, "attacked_hard": attacked_hard})
        return base

def relational_objective(pair: dict[str, Tensor], labels: Tensor, memory_codes: Tensor | None,
                          memory_labels: Tensor | None, config: RelationalConfig, stage: int = 2):
    from .cap_dino_logpolar_stable import stable_objective
    loss, terms = stable_objective(pair, labels, memory_codes, memory_labels, config, stage=stage)
    relation_consistency = _safe_cosine_loss(pair["clean_relational"], pair["attacked_relational"])
    loss = loss + float(config.lambda_relation_consistency) * relation_consistency
    terms = dict(terms)
    terms["relation_consistency"] = relation_consistency
    terms["relation_scale"] = torch.sigmoid(pair["clean_relational"].new_tensor(0.0)) * float(config.relation_weight)
    return loss, terms

__all__ = ["RELATIONAL_VERSION", "RelationalConfig", "CAPDinoLogPolarRelational", "relational_objective"]
