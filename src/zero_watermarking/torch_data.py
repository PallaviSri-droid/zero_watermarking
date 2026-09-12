from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .attacks import ATTACKS
from .datasets import ImageRecord, load_image


class AttackPairDataset(Dataset):
    def __init__(self, records: list[ImageRecord], size: int = 224, attack_name: str = "gaussian_noise", attack_param: float = 0.03, seed: int = 42):
        self.records = records; self.size = size; self.attack_name = attack_name; self.attack_param = attack_param; self.seed = seed
        self.kw = {"sigma": attack_param}
        if attack_name == "jpeg": self.kw = {"quality": int(attack_param)}
        elif attack_name == "rotation": self.kw = {"degrees": attack_param}
        elif attack_name == "crop_resize": self.kw = {"fraction": attack_param}

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = load_image(rec.path, self.size)
        # Deterministic attack seed per sample makes runs reproducible.
        np.random.seed(self.seed + idx)
        attacked = ATTACKS[self.attack_name](image, **self.kw)
        clean = torch.from_numpy(image).float().unsqueeze(0)
        attacked = torch.from_numpy(np.asarray(attacked, np.float32)).float().unsqueeze(0)
        return clean, attacked, torch.tensor(idx, dtype=torch.long)
