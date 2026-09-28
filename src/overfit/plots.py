"""Optional plots (requires matplotlib: ``pip install overfit[plot]``)."""

from __future__ import annotations

import numpy as np

from overfit.cscv import PBOResult


def plot_pbo(result: PBOResult, ax_hist, ax_scatter, title: str = "") -> None:
    """Draw the logit histogram and the IS-vs-OOS Sharpe scatter on two given axes."""
    lim = np.log(result.n_strategies) * 1.05  # lambda is bounded by +-ln(N)
    ax_hist.hist(result.logits, bins=np.linspace(-lim, lim, 31), color="#4C72B0", edgecolor="white")
    ax_hist.set_xlim(-lim, lim)
    ax_hist.axvline(0, color="#C44E52", lw=1.5, ls="--")
    ax_hist.set_xlabel(r"logit $\lambda$  (left of 0 = IS winner ranks below OOS median)")
    ax_hist.set_ylabel("combinations")
    ax_hist.set_title(f"{title}  PBO = {result.pbo:.2f}".strip())

    ax_scatter.scatter(result.is_sharpe, result.oos_sharpe, s=6, alpha=0.25, color="#4C72B0")
    xs = np.array([result.is_sharpe.min(), result.is_sharpe.max()])
    ax_scatter.plot(
        xs, result.degradation_intercept + result.degradation_slope * xs, color="#C44E52", lw=1.5
    )
    ax_scatter.axhline(0, color="grey", lw=0.8)
    ax_scatter.set_xlabel("IS Sharpe of IS-best strategy (per period)")
    ax_scatter.set_ylabel("OOS Sharpe of same strategy")
    ax_scatter.set_title(
        f"slope = {result.degradation_slope:.2f},  P(loss) = {result.prob_loss:.2f}"
    )
