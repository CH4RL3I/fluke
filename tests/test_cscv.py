import numpy as np
import pandas as pd
import pytest

from conftest import noise_matrix, signal_matrix
from overfit import cscv_pbo, sharpe_ratio


def test_noise_pbo_is_high():
    pbos = [cscv_pbo(noise_matrix(seed), 16).pbo for seed in range(12)]
    # iid noise: PBO ~ 0.5 on average (40-seed check: 0.50), but noisy for a single draw
    assert 0.4 < np.mean(pbos) < 0.65
    assert all(p > 0.15 for p in pbos)


def test_signal_pbo_is_low():
    for seed in range(3):
        res = cscv_pbo(signal_matrix(seed), 16)
        assert res.pbo < 0.1, seed
        assert res.prob_loss < 0.1
        assert res.oos_sharpe.mean() > 0  # the edge persists out of sample


def test_noise_degradation_slope_not_positive():
    # in-sample winners of pure noise have no persistent edge: OOS Sharpe falls as IS rises
    res = cscv_pbo(noise_matrix(0), 16)
    assert res.degradation_slope < 0.2
    assert res.oos_sharpe.mean() < res.is_sharpe.mean()


def test_result_shapes_and_bounds(noise):
    res = cscv_pbo(noise, 8)
    assert res.n_combinations == 70  # C(8, 4)
    assert res.is_sharpe.shape == res.oos_sharpe.shape == res.logits.shape == (70,)
    assert 0.0 <= res.pbo <= 1.0 and 0.0 <= res.prob_loss <= 1.0
    # lambda is bounded by the rank grid: omega in [1/(N+1), N/(N+1)]
    bound = np.log(100)
    assert np.all(np.abs(res.logits) <= bound + 1e-9)


def test_logit_sign_convention():
    # strategy 0 is the best IS *and* OOS in every split -> omega = N/(N+1) -> lambda = ln(N) > 0
    t, n = 400, 10
    x = np.random.default_rng(0).normal(0, 0.01, (t, n))
    x[:, 0] += 0.05  # overwhelming drift
    res = cscv_pbo(x, 8)
    assert res.pbo == 0.0
    assert np.allclose(res.logits, np.log(n))


def test_is_sharpe_matches_direct_computation():
    x = noise_matrix(4, t=160, n=5)
    res = cscv_pbo(x, 4)
    # 4 blocks of 40 rows; recompute first combination (blocks 0,1 IS) by hand
    is_rows, oos_rows = x[:80], x[80:]
    srs_is = [sharpe_ratio(is_rows[:, j]) for j in range(5)]
    best = int(np.argmax(srs_is))
    # combinations are enumerated lexicographically: (0,1) first
    assert res.is_sharpe[0] == pytest.approx(srs_is[best])
    assert res.oos_sharpe[0] == pytest.approx(sharpe_ratio(oos_rows[:, best]))


def test_accepts_dataframe_and_drops_oldest_remainder():
    x = noise_matrix(5, t=1003, n=6)
    a = cscv_pbo(pd.DataFrame(x), 10)
    b = cscv_pbo(x[3:], 10)
    assert a.n_obs_used == 1000
    assert a.pbo == b.pbo
    np.testing.assert_allclose(a.logits, b.logits)


@pytest.mark.parametrize("s", [3, 7, 15, 1, 0, -2])
def test_bad_block_count_raises(s, noise):
    with pytest.raises(ValueError, match="even"):
        cscv_pbo(noise, s)


def test_single_strategy_raises():
    with pytest.raises(ValueError, match="at least 2 strategies"):
        cscv_pbo(np.random.default_rng(0).normal(size=(200, 1)), 8)


def test_constant_returns_do_not_crash():
    res = cscv_pbo(np.full((160, 4), 0.001), 8)
    assert np.isfinite(res.logits).all()
    assert res.pbo == 1.0  # nothing to choose between -> IS winner is mid-ranked OOS
    assert res.degradation_slope == 0.0


def test_rejects_nan_and_too_short():
    x = noise_matrix(0, t=100, n=3)
    x[5, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cscv_pbo(x, 4)
    with pytest.raises(ValueError, match="rows per block"):
        cscv_pbo(noise_matrix(0, t=20, n=3), 16)


def test_max_combinations_guard(noise):
    with pytest.raises(ValueError, match="max_combinations"):
        cscv_pbo(noise, 32)
