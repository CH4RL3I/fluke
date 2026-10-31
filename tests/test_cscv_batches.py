"""Check batched CSCV against direct return-series calculations on small problems."""

from itertools import combinations

import numpy as np
import pytest
from scipy import stats

from overfit import cscv_pbo


def direct_reference(returns, n_blocks):
    """Enumerate small splits, computing sample deviations from selected observations.

    This deliberately does not use production block sums or its Sharpe helper. Global
    centering retains the public numerical-flatness policy without subtracting squares
    of nearly equal large values in the reference variance calculation.
    """
    x = np.asarray(returns, dtype=float)
    rows_per_block = len(x) // n_blocks
    x = x[len(x) - rows_per_block * n_blocks :]
    shift = x.mean(axis=0)
    scale = np.maximum(np.abs(x).mean(axis=0), 1e-300)
    centered = x - shift
    all_rows = np.arange(len(x))
    block_of_row = all_rows // rows_per_block
    split_sharpes = []
    for selected in combinations(range(n_blocks), n_blocks // 2):
        inside = np.isin(block_of_row, selected)
        pair = []
        for mask in (inside, ~inside):
            observations = centered[mask]
            sd = observations.std(axis=0, ddof=1)
            flat = sd <= 1e-10 * scale
            pair.append(
                np.where(
                    flat,
                    0.0,
                    (observations.mean(axis=0) + shift) / np.where(flat, 1.0, sd),
                )
            )
        split_sharpes.append(pair)

    is_matrix, oos_matrix = np.asarray(split_sharpes).transpose(1, 0, 2)
    winner = is_matrix.argmax(axis=1)
    rows = np.arange(len(winner))
    rank = stats.rankdata(oos_matrix, axis=1, method="average")[rows, winner]
    omega = rank / (x.shape[1] + 1)
    logits = np.log(omega / (1.0 - omega))
    is_best = is_matrix[rows, winner]
    oos_best = oos_matrix[rows, winner]
    if np.ptp(is_best) > 0:
        fit = stats.linregress(is_best, oos_best)
        slope, intercept, r2 = fit.slope, fit.intercept, fit.rvalue**2
    else:
        slope, intercept, r2 = 0.0, oos_best.mean(), 0.0
    return {
        "logits": logits,
        "is_sharpe": is_best,
        "oos_sharpe": oos_best,
        "pbo": np.mean(logits <= 0),
        "prob_loss": np.mean(oos_best < 0),
        "degradation_slope": slope,
        "degradation_intercept": intercept,
        "degradation_r2": r2,
        "n_obs_used": len(x),
        "n_strategies": x.shape[1],
        "n_blocks": n_blocks,
    }


def sample_matrix(kind):
    x = np.random.default_rng(43).normal(0.002, 0.01, (35, 7))
    if kind == "ties":
        x[:, 1] = x[:, 0]
        x[:, 3] = x[:, 2]
        x[:, 4] = 0.0
        x[:, 5] = 0.001
    elif kind == "flat":
        x[:] = 0.001
    elif kind == "shifted":
        x += 1_000.0
    elif kind == "tiny":
        x *= 1e-10
    return x


@pytest.mark.parametrize("kind", ["random", "ties", "flat", "shifted", "tiny"])
@pytest.mark.parametrize("n_blocks", [2, 4, 8])
@pytest.mark.parametrize("batch_size", [1, 3, 64, 1000])
def test_batches_match_direct_return_series_reference(kind, n_blocks, batch_size):
    x = sample_matrix(kind)
    expected = direct_reference(x, n_blocks)
    actual = cscv_pbo(x, n_blocks, batch_size=batch_size)
    for name, want in expected.items():
        # Shifted data makes Sharpe large; regressions amplify last-bit variance changes.
        np.testing.assert_allclose(getattr(actual, name), want, rtol=1e-8, atol=1e-8, err_msg=name)
    assert actual.n_combinations == len(expected["logits"])


def test_exact_is_tie_chooses_first_strategy_and_oos_ties_get_average_rank():
    x = np.array(
        [[0, 0], [1, 1], [0, 0], [1, 1], [1, -1], [2, 0], [1, -1], [2, 0]],
        dtype=float,
    )
    result = cscv_pbo(x, n_blocks=2, batch_size=1)
    # First IS half ties; column zero wins and is strictly first in its OOS half.
    assert result.oos_sharpe[0] == pytest.approx(1.5 / np.std([1, 2, 1, 2], ddof=1))
    assert result.logits[0] == pytest.approx(np.log(2))
    # Reverse split's OOS half ties, so mean rank 1.5 puts omega exactly at 0.5.
    assert result.logits[1] == 0.0


@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, 1.5, "3", None, np.nan])
def test_invalid_batch_sizes_raise_clear_error(value):
    with pytest.raises(ValueError, match="batch_size.*positive integer"):
        cscv_pbo(np.ones((16, 3)), n_blocks=4, batch_size=value)


def test_numpy_integer_batch_size_is_accepted():
    result = cscv_pbo(np.ones((16, 3)), n_blocks=4, batch_size=np.int64(2))
    assert result.n_combinations == 6
