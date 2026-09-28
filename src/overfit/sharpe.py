"""Probabilistic / Deflated Sharpe Ratio and Minimum Track Record Length.

Reference: Bailey, D. H. and Lopez de Prado, M. (2014), "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality", Journal of
Portfolio Management 40(5), 94-107.

Units: every Sharpe ratio in this module is **per period** (mean / std of the
returns as sampled, e.g. daily), never annualised. Annualising multiplies by
sqrt(periods_per_year); if you pass a benchmark that you think of in annual terms,
convert it first with :func:`deannualize_sharpe`. Mixing units silently is the most
common way to get a wrong DSR, so the API never does it for you.

Moments follow the paper: skewness and (non-excess) kurtosis are the population
moments of the return series (normal returns: skew 0, kurtosis 3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329

__all__ = [
    "return_moments",
    "EULER_GAMMA",
    "DSRResult",
    "annualize_sharpe",
    "deannualize_sharpe",
    "deflated_sharpe_ratio",
    "deflated_sharpe_best",
    "expected_max_sharpe",
    "min_track_record_length",
    "probabilistic_sharpe_ratio",
    "psr_from_stats",
    "mintrl_from_stats",
    "sharpe_ratio",
]


def annualize_sharpe(sr: float, periods_per_year: float) -> float:
    """Per-period Sharpe -> annualised Sharpe (i.i.d. scaling by sqrt(periods))."""
    return sr * math.sqrt(periods_per_year)


def deannualize_sharpe(sr_annual: float, periods_per_year: float) -> float:
    """Annualised Sharpe -> per-period Sharpe."""
    return sr_annual / math.sqrt(periods_per_year)


def _clean(returns) -> np.ndarray:
    x = np.asarray(returns, dtype=float).ravel()
    x = x[~np.isnan(x)]
    if x.size < 3:
        raise ValueError("need at least 3 non-NaN observations")
    return x


def sharpe_ratio(returns) -> float:
    """Per-period Sharpe ratio, mean / std (ddof=1). NaN if the series has zero variance."""
    x = _clean(returns)
    sd = x.std(ddof=1)
    if sd <= 1e-12 * max(np.abs(x).mean(), 1e-300):
        return float("nan")
    return float(x.mean() / sd)


def return_moments(x: np.ndarray) -> tuple[float, float]:
    return float(stats.skew(x, bias=True)), float(stats.kurtosis(x, fisher=False, bias=True))


def _sr_variance_term(sr: float, skew: float, kurt: float) -> float:
    """1 - g3*SR + (g4-1)/4 * SR^2: the (T-1)-scaled variance of the Sharpe estimator."""
    v = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if v <= 0:
        raise ValueError("non-positive Sharpe variance: skew/kurtosis are inconsistent with SR")
    return v


def psr_from_stats(
    sr: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0, sr_benchmark: float = 0.0
) -> float:
    r"""Probabilistic Sharpe Ratio from summary statistics (all per-period).

    PSR(SR*) = Phi( (SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2) )
    """
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(_sr_variance_term(sr, skew, kurt))
    return float(stats.norm.cdf(z))


def mintrl_from_stats(
    sr: float,
    skew: float = 0.0,
    kurt: float = 3.0,
    sr_benchmark: float = 0.0,
    alpha: float = 0.05,
) -> float:
    r"""Minimum Track Record Length, in number of observations (same unit as ``sr``).

    MinTRL = 1 + (1 - g3 SR + (g4 - 1)/4 SR^2) (Phi^-1(1 - alpha) / (SR - SR*))^2

    Returns ``inf`` if ``sr <= sr_benchmark`` (no track record is long enough).
    """
    if sr <= sr_benchmark:
        return math.inf
    z = stats.norm.ppf(1.0 - alpha)
    return float(1.0 + _sr_variance_term(sr, skew, kurt) * (z / (sr - sr_benchmark)) ** 2)


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    """PSR of a return series against a per-period benchmark Sharpe ``sr_benchmark``."""
    x = _clean(returns)
    sr = sharpe_ratio(x)
    if math.isnan(sr):
        raise ValueError("returns have zero variance; Sharpe ratio is undefined")
    skew, kurt = return_moments(x)
    return psr_from_stats(sr, x.size, skew, kurt, sr_benchmark)


def min_track_record_length(returns, sr_benchmark: float = 0.0, alpha: float = 0.05) -> float:
    """MinTRL (in observations) for the series' Sharpe to beat ``sr_benchmark`` at 1 - alpha."""
    x = _clean(returns)
    sr = sharpe_ratio(x)
    if math.isnan(sr):
        raise ValueError("returns have zero variance; Sharpe ratio is undefined")
    skew, kurt = return_moments(x)
    return mintrl_from_stats(sr, skew, kurt, sr_benchmark, alpha)


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    r"""Expected maximum per-period Sharpe among ``n_trials`` independent trials with zero skill.

    SR_0 = sqrt(V[SR]) ((1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e))),  g = Euler-Mascheroni.

    ``sr_variance`` is the cross-sectional variance of the trials' per-period Sharpe ratios.
    With a single trial there is no selection, so the benchmark is 0.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if sr_variance < 0:
        raise ValueError("sr_variance must be >= 0")
    if n_trials == 1:
        return 0.0
    g = EULER_GAMMA
    return float(
        math.sqrt(sr_variance)
        * (
            (1 - g) * stats.norm.ppf(1 - 1 / n_trials)
            + g * stats.norm.ppf(1 - 1 / (n_trials * math.e))
        )
    )


def deflated_sharpe_ratio(returns, n_trials: int, sr_variance: float) -> float:
    """DSR = PSR evaluated at the expected-max-Sharpe benchmark SR_0 (per-period units)."""
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe_ratio(returns, sr0)


@dataclass(frozen=True)
class DSRResult:
    """DSR of the best (highest-Sharpe) column of a returns matrix. Sharpes are per period."""

    best: object  # column label (DataFrame) or integer index (ndarray)
    sharpe: float
    sr0: float
    dsr: float
    psr: float  # PSR against 0, i.e. ignoring the selection
    n_trials: int
    n_obs: int


def deflated_sharpe_best(returns) -> DSRResult:
    """Select the best strategy of a T x N matrix by Sharpe and deflate it for the N trials.

    V[SR] is estimated from the cross-section of the N per-period Sharpe ratios. Configurations
    with zero variance are excluded from that estimate but still count as trials.
    """
    df = returns if isinstance(returns, pd.DataFrame) else pd.DataFrame(np.asarray(returns))
    x = df.to_numpy(dtype=float)
    n_trials = x.shape[1]
    srs = np.array([sharpe_ratio(x[:, j]) for j in range(n_trials)])
    if np.all(np.isnan(srs)):
        raise ValueError("all strategies have zero variance")
    j = int(np.nanargmax(srs))
    valid = srs[~np.isnan(srs)]
    var = float(np.var(valid, ddof=1)) if valid.size > 1 else 0.0
    sr0 = expected_max_sharpe(n_trials, var)
    col = x[:, j]
    return DSRResult(
        best=df.columns[j],
        sharpe=float(srs[j]),
        sr0=sr0,
        dsr=probabilistic_sharpe_ratio(col, sr0),
        psr=probabilistic_sharpe_ratio(col, 0.0),
        n_trials=n_trials,
        n_obs=int(np.sum(~np.isnan(col))),
    )
