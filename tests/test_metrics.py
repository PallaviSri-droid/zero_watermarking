import numpy as np

from zero_watermarking.metrics import bit_balance, bit_entropy, hamming, nc


def test_hamming():
    assert hamming([0, 0], [0, 1]) == 0.5


def test_nc_identity():
    x = np.array([0, 1, 1, 0])
    assert abs(nc(x, x) - 1.0) < 1e-9


def test_balance():
    bits = np.array([[0, 1], [1, 0]])
    assert abs(bit_balance(bits)) < 1e-9


def test_entropy():
    bits = np.array([[0, 0], [1, 1]])
    entropy, _ = bit_entropy(bits)
    assert entropy > 0.99
