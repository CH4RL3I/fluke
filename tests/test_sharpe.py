import math
from statistics import NormalDist

import numpy as np
import pytest

from conftest import noise_matrix, signal_matrix
from overfit import (
    deflated_sharpe_best,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    mintrl_from_stats,
    probabilistic_sharpe_ratio,
    psr_from_stats,
    sharpe_ratio,
)
from overfit.sharpe import annualize_sharpe, deannualize_sharpe

N01 = NormalDist()


def test_psr_hand_computed():
    # SR=0.2, T=101 (sqrt(T-1)=10), skew=-0.5, kurt=5, benchmark 0.1
    # variance term = 1 + 0.5*0.2 + (5-1)/4 * 0.04 = 1.14
    # z = 0.1 * 10 / sqrt(1.14)
    assert psr_from_stats(0.2, 101, -0.5, 5.0, 0.1) == pytest.approx(0.825514173, abs=1e-8)


def test_mintrl_hand_computed():
    # 1 + 1.14 * (z_0.95 / 0.1)^2, z_0.95 = 1.6448536...
    assert mintrl_from_stats(0.2, -0.5, 5.0, 0.1, 0.05) == pytest.approx(309.431954, abs=1e-5)
    assert mintrl_from_stats(0.1, sr_benchmark=0.1) == math.inf


def test_expected_max_sharpe_hand_computed():
    # N=100, V[SR]=0.01 -> 0.1 * (0.4228 * z_0.99 + 0.5772 * z_(1-1/(100e)))
    assert expected_max_sharpe(100, 0.01) == pytest.approx(0.25306029, abs=1e-8)
    assert expected_max_sharpe(1, 0.01) == 0.0
    # more trials or wider dispersion raise the bar
    assert expected_max_sharpe(1000, 0.01) > expected_max_sharpe(100, 0.01)
    assert expected_max_sharpe(100, 0.04) == pytest.approx(2 * expected_max_sharpe(100, 0.01))


def test_series_end_to_end_against_hand_values():
    # mean 0.01, sample var 2.5e-4, third central moment 0, m2 = 2e-4, m4 = 6.8e-8 -> kurt 1.7
    r = [0.01, 0.02, -0.01, 0.03, 0.00]
    sr = math.sqrt(0.4)
    assert sharpe_ratio(r) == pytest.approx(sr)
    z = sr * math.sqrt(4) / math.sqrt(1 + (1.7 - 1) / 4 * 0.4)
    assert probabilistic_sharpe_ratio(r) == pytest.approx(N01.cdf(z))
    assert min_track_record_length(r) == pytest.approx(
        1 + (1 + 0.7 / 4 * 0.4) * (N01.inv_cdf(0.95) / sr) ** 2
    )
    sr0 = expected_max_sharpe(50, 0.05)
    z0 = (sr - sr0) * 2 / math.sqrt(1 + 0.7 / 4 * 0.4)
    assert deflated_sharpe_ratio(r, 50, 0.05) == pytest.approx(N01.cdf(z0))


def test_psr_monotone_and_half_at_benchmark():
    r = np.random.default_rng(1).normal(0.001, 0.01, 500)
    sr = sharpe_ratio(r)
    assert probabilistic_sharpe_ratio(r, sr) == pytest.approx(0.5)
    assert probabilistic_sharpe_ratio(r, sr - 0.05) > probabilistic_sharpe_ratio(r, sr + 0.05)


def test_unit_conversion_roundtrip():
    assert deannualize_sharpe(annualize_sharpe(0.1, 252), 252) == pytest.approx(0.1)
    assert annualize_sharpe(0.1, 252) == pytest.approx(0.1 * math.sqrt(252))


def test_constant_returns():
    assert math.isnan(sharpe_ratio(np.full(50, 0.001)))
    with pytest.raises(ValueError, match="zero variance"):
        probabilistic_sharpe_ratio(np.full(50, 0.001))
    with pytest.raises(ValueError, match="zero variance"):
        min_track_record_length(np.full(50, 0.001))


def test_too_short_series():
    with pytest.raises(ValueError):
        sharpe_ratio([0.1, 0.2])


def test_dsr_single_strategy_reduces_to_psr():
    x = np.random.default_rng(3).normal(0.0005, 0.01, (400, 1))
    res = deflated_sharpe_best(x)
    assert res.n_trials == 1
    assert res.sr0 == 0.0
    assert res.dsr == pytest.approx(res.psr)


def test_dsr_noise_not_significant():
    # the luckiest of 100 noise strategies looks great on PSR but must not survive deflation
    for seed in range(6):
        res = deflated_sharpe_best(noise_matrix(seed))
        assert res.dsr < 0.95, seed
    assert deflated_sharpe_best(noise_matrix(0)).psr > deflated_sharpe_best(noise_matrix(0)).dsr


def test_dsr_signal_significant():
    for seed in range(6):
        res = deflated_sharpe_best(signal_matrix(seed))
        assert res.best == 0, seed
        assert res.dsr > 0.95, seed
