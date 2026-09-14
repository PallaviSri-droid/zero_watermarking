from __future__ import annotations

import numpy as np

from zero_watermarking.logpolar_dino_mrelbp import logpolar_fourier_feature, mrelbp_feature


def test_logpolar_feature_is_fixed_length_and_finite() -> None:
    image = np.linspace(0.0, 1.0, 128 * 128, dtype=np.float32).reshape(128, 128)
    feature = logpolar_fourier_feature(image)
    assert feature.shape == (32 * 32,)
    assert np.isfinite(feature).all()


def test_mrelbp_feature_is_fixed_length_and_normalized() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((64, 64), dtype=np.float32)
    feature = mrelbp_feature(image, radii=(1, 2), points=8)
    expected = 2 * (2**8) + 2 * 2
    assert feature.shape == (expected,)
    assert np.isfinite(feature).all()
    assert np.all(feature >= 0.0)
