"""Combinatorially Symmetric Cross-Validation (CSCV) and the Probability of Backtest Overfitting.

Reference: Bailey, D. H., Borwein, J. M., Lopez de Prado, M. and Zhu, Q. J. (2016),
"The Probability of Backtest Overfitting", Journal of Computational Finance 20(4), 39-69.

Procedure (Section 2 of the paper):

1. Split the T x N return matrix into S contiguous, equal-length blocks.
2. For each of the C(S, S/2) ways to choose S/2 blocks as in-sample (IS), the other S/2 form
   the out-of-sample (OOS) set.
3. Pick the strategy n* with the best IS Sharpe ratio; find its OOS relative rank
   omega = rank / (N + 1) among all N strategies' OOS Sharpes.
4. logit lambda = ln(omega / (1 - omega)).  PBO = share of combinations with lambda <= 0,
   i.e. the IS winner lands at or below the OOS median.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
from scipy import stats

__all__ = ["PBOResult", "cscv_pbo"]


@dataclass(frozen=True)
class PBOResult:
    """Output of :func:`cscv_pbo`. All Sharpe ratios are per period (not annualised)."""

    pbo: float
    logits: np.ndarray  # lambda, one per combination
    is_sharpe: np.ndarray  # IS Sharpe of the IS-best strategy, per combination
    oos_sharpe: np.ndarray  # OOS Sharpe of that same strategy, per combination
    prob_loss: float  # share of combinations where the IS-best has OOS Sharpe < 0
    degradation_slope: float  # OLS: oos_sharpe = intercept + slope * is_sharpe
    degradation_intercept: float
    degradation_r2: float
    n_blocks: int
    n_strategies: int
    n_obs_used: int

    @property
    def n_combinations(self) -> int:
        return int(self.logits.size)


def _sharpe_from_sums(s1, s2, n, mean_shift, scale):
    """Sharpe from block sums of demeaned returns (s1 = sum, s2 = sum of squares), ddof=1.

    Working on demeaned data avoids catastrophic cancellation in the variance. Columns whose
    volatility is numerically zero get Sharpe 0 (a flat strategy has no measurable skill).
    """
    mean_c = s1 / n
    var = np.maximum((s2 - n * mean_c**2) / (n - 1), 0.0)
    sd = np.sqrt(var)
    flat = sd <= 1e-10 * scale
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(flat, 0.0, (mean_c + mean_shift) / np.where(flat, 1.0, sd))
    return sr


def cscv_pbo(returns, n_blocks: int = 16, max_combinations: int = 500_000) -> PBOResult:
    """Run CSCV on a T x N matrix of per-period returns and compute the PBO.

    Parameters
    ----------
    returns : array-like or DataFrame, shape (T, N)
        One column per strategy configuration, one row per period. No NaNs.
    n_blocks : int
        S, the number of blocks. Must be even and >= 2. C(S, S/2) combinations are evaluated
        (S=16 gives 12,870). If T is not divisible by S, the *oldest* T mod S rows are dropped.
    max_combinations : int
        Safety cap on C(S, S/2); S=32 alone would be ~6e8.
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2:
        raise ValueError("returns must be a 2-D (T x N) matrix")
    t, n = x.shape
    s = n_blocks
    if s < 2 or s % 2:
        raise ValueError("n_blocks must be an even integer >= 2")
    if n < 2:
        raise ValueError("need at least 2 strategies: with N=1 there is no selection to overfit")
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("returns must be finite (no NaN/inf)")
    if comb(s, s // 2) > max_combinations:
        raise ValueError(f"C({s},{s // 2}) exceeds max_combinations={max_combinations}")
    m = t // s  # rows per block
    if m < 2:
        raise ValueError(f"need at least 2 rows per block; got T={t} for {s} blocks")
    x = x[t - m * s :]  # drop oldest remainder

    shift = x.mean(axis=0)
    scale = np.maximum(np.abs(x).mean(axis=0), 1e-300)
    xc = x - shift
    blocks = xc.reshape(s, m, n)
    b1 = blocks.sum(axis=1)  # S x N
    b2 = (blocks**2).sum(axis=1)
    tot1, tot2 = b1.sum(axis=0), b2.sum(axis=0)

    combos = np.array(list(combinations(range(s), s // 2)))  # M x S/2
    n_is = m * (s // 2)
    is1, is2 = b1[combos].sum(axis=1), b2[combos].sum(axis=1)  # M x N
    sr_is = _sharpe_from_sums(is1, is2, n_is, shift, scale)
    sr_oos = _sharpe_from_sums(tot1 - is1, tot2 - is2, n_is, shift, scale)

    best = sr_is.argmax(axis=1)  # ties -> first column
    rows = np.arange(combos.shape[0])
    # relative OOS rank of the IS winner; average rank so ties (e.g. flat strategies) sit mid-way
    ranks = stats.rankdata(sr_oos, axis=1, method="average")[rows, best]
    omega = ranks / (n + 1)
    logits = np.log(omega / (1.0 - omega))

    is_best = sr_is[rows, best]
    oos_best = sr_oos[rows, best]
    if np.ptp(is_best) > 0:
        fit = stats.linregress(is_best, oos_best)
        slope, intercept, r2 = float(fit.slope), float(fit.intercept), float(fit.rvalue**2)
    else:  # degenerate: IS Sharpe never varies (e.g. all strategies flat)
        slope, intercept, r2 = 0.0, float(oos_best.mean()), 0.0

    return PBOResult(
        pbo=float(np.mean(logits <= 0)),
        logits=logits,
        is_sharpe=is_best,
        oos_sharpe=oos_best,
        prob_loss=float(np.mean(oos_best < 0)),
        degradation_slope=slope,
        degradation_intercept=intercept,
        degradation_r2=r2,
        n_blocks=s,
        n_strategies=n,
        n_obs_used=m * s,
    )
