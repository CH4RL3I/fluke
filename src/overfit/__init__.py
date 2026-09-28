"""overfit: is your best backtest real, or just the luckiest of N tries?"""

from overfit.cscv import PBOResult, cscv_pbo
from overfit.sharpe import (
    DSRResult,
    annualize_sharpe,
    deannualize_sharpe,
    deflated_sharpe_best,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    mintrl_from_stats,
    probabilistic_sharpe_ratio,
    psr_from_stats,
    sharpe_ratio,
)

__version__ = "0.1.0"

__all__ = [
    "DSRResult",
    "PBOResult",
    "annualize_sharpe",
    "cscv_pbo",
    "deannualize_sharpe",
    "deflated_sharpe_best",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "min_track_record_length",
    "mintrl_from_stats",
    "probabilistic_sharpe_ratio",
    "psr_from_stats",
    "sharpe_ratio",
]
