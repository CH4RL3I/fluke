import numpy as np
import pytest


def noise_matrix(seed: int, t: int = 1000, n: int = 100, sigma: float = 0.01) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, sigma, size=(t, n))


def signal_matrix(
    seed: int, t: int = 1500, n: int = 100, sigma: float = 0.01, annual_sr: float = 3.0
) -> np.ndarray:
    """Noise everywhere, but column 0 has genuine drift (annual Sharpe ~ annual_sr, daily data)."""
    x = noise_matrix(seed, t, n, sigma)
    x[:, 0] += annual_sr / np.sqrt(252) * sigma
    return x


@pytest.fixture
def noise():
    return noise_matrix(0)


@pytest.fixture
def signal():
    return signal_matrix(0)
