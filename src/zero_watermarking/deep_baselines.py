from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TorchvisionFeatureHash(nn.Module):
    """Frozen pretrained backbone followed by deterministic DCT-binarization.

    The model is intentionally inference-only; training belongs to the proposed CAP-ZW model.
    """
    def __init__(self, backbone: str = "resnet50", hash_length: int = 256, pretrained: bool = True):
        super().__init__()
        from torchvision.models import (
            ResNet50_Weights, AlexNet_Weights, resnet50, alexnet,
        )
        self.hash_length = hash_length
        if backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            net = resnet50(weights=weights)
            self.model = nn.Sequential(*list(net.children())[:-1])
            self.input_size = 224
            self.weights = weights
        elif backbone == "alexnet":
            weights = AlexNet_Weights.DEFAULT if pretrained else None
            net = alexnet(weights=weights)
            self.model = nn.Sequential(net.features, net.avgpool, nn.Flatten(), *list(net.classifier)[:2])
            self.input_size = 224
            self.weights = weights
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        for p in self.parameters(): p.requires_grad = False
        self.eval()

    @torch.inference_mode()
    def encode(self, image: np.ndarray) -> np.ndarray:
        from scipy.fft import dct
        x = torch.from_numpy(np.asarray(image, np.float32))[None, None]
        x = torch.nn.functional.interpolate(x, (self.input_size, self.input_size))
        x = x.repeat(1, 3, 1, 1)
        if self.weights is not None and hasattr(self.weights, "transforms"):
            # Normalize consistently with the pretrained weights.
            x = self.weights.transforms()(x)
        feat = self.model(x).detach().cpu().numpy().ravel()
        feat = dct(feat, norm="ortho")
        feat = np.resize(feat, self.hash_length)
        return (feat >= np.median(feat)).astype(np.uint8)


def build_deep_registry(hash_length: int = 256) -> dict[str, callable]:
    registry = {}
    for name, backbone in [("ResNet50-DCT", "resnet50"), ("AlexNet-DTCWT-DCT-proxy", "alexnet")]:
        try:
            model = TorchvisionFeatureHash(backbone, hash_length)
            registry[name] = model.encode
        except Exception:
            # Keep the benchmark runnable without downloaded model weights.
            registry[name] = None
    return registry
