"""Command line interface: ``overfit report returns.csv --blocks 16``."""

from __future__ import annotations

import argparse
import math
import sys

import pandas as pd

from overfit.cscv import cscv_pbo
from overfit.sharpe import annualize_sharpe, deflated_sharpe_best, mintrl_from_stats, return_moments


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.select_dtypes("number")  # drops a date/label index column
    if df.shape[1] < 2:
        raise SystemExit("error: need a CSV with at least 2 numeric strategy columns")
    return df.dropna(how="any")


def _fmt_len(n: float, ppy: float | None) -> str:
    if math.isinf(n):
        return "never (SR <= benchmark)"
    s = f"{math.ceil(n):,} periods"
    return s + (f" (~{n / ppy:.1f} years)" if ppy else "")


def report(path: str, blocks: int, periods_per_year: float | None, alpha: float) -> str:
    df = _load(path)
    pbo = cscv_pbo(df, n_blocks=blocks)
    dsr = deflated_sharpe_best(df)
    x = df[dsr.best].to_numpy()
    skew, kurt = return_moments(x)
    trl0 = mintrl_from_stats(dsr.sharpe, skew, kurt, 0.0, alpha)
    trl_d = mintrl_from_stats(dsr.sharpe, skew, kurt, dsr.sr0, alpha)

    def ann(sr: float) -> str:
        return (
            f" (annualised {annualize_sharpe(sr, periods_per_year):.2f})"
            if periods_per_year
            else ""
        )

    lines = [
        f"Input: {pbo.n_obs_used} periods x {pbo.n_strategies} strategies, "
        f"{pbo.n_blocks} blocks, {pbo.n_combinations:,} CSCV combinations",
        "",
        "Probability of Backtest Overfitting (CSCV)",
        f"  PBO                      {pbo.pbo:.3f}",
        f"  P(OOS loss | IS best)    {pbo.prob_loss:.3f}",
        f"  degradation slope        {pbo.degradation_slope:+.3f}  (R2 {pbo.degradation_r2:.3f})",
        "",
        f"Deflated Sharpe Ratio of best strategy '{dsr.best}'",
        f"  Sharpe (per period)      {dsr.sharpe:.4f}{ann(dsr.sharpe)}",
        f"  expected max SR0         {dsr.sr0:.4f}{ann(dsr.sr0)}  ({dsr.n_trials} trials)",
        f"  PSR (vs 0)               {dsr.psr:.3f}",
        f"  DSR (vs SR0)             {dsr.dsr:.3f}   "
        + ("significant" if dsr.dsr >= 1 - alpha else "not significant")
        + f" at {1 - alpha:.0%}",
        "",
        f"Minimum Track Record Length (confidence {1 - alpha:.0%})",
        f"  vs SR = 0                {_fmt_len(trl0, periods_per_year)}",
        f"  vs SR0                   {_fmt_len(trl_d, periods_per_year)}",
        f"  observed                 {dsr.n_obs:,} periods",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="overfit", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="PBO, DSR and MinTRL for a CSV of strategy returns")
    r.add_argument("csv", help="rows = periods, columns = strategies (non-numeric columns ignored)")
    r.add_argument("--blocks", type=int, default=16, help="CSCV blocks S, even (default 16)")
    r.add_argument(
        "--periods-per-year",
        type=float,
        default=None,
        help="e.g. 252 for daily; only used to display annualised figures",
    )
    r.add_argument("--alpha", type=float, default=0.05, help="significance level (default 0.05)")
    a = p.parse_args(argv)
    try:
        print(report(a.csv, a.blocks, a.periods_per_year, a.alpha))
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
