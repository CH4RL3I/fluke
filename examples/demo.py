"""Noise vs signal: what PBO and DSR say about 100 strategy configurations.

Run:  uv run python examples/demo.py   (writes docs/pbo_demo.png)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from overfit import annualize_sharpe, cscv_pbo, deflated_sharpe_best
from overfit.plots import plot_pbo

T, N, SIGMA, PPY = 1500, 100, 0.01, 252  # ~6 years of daily data, 100 configurations
DOCS = Path(__file__).resolve().parent.parent / "docs"


def make_matrices(seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, SIGMA, size=(T, N))
    signal = noise.copy()
    signal[:, 0] += 3.0 / np.sqrt(PPY) * SIGMA  # one config with a true annual Sharpe of ~3
    cols = [f"cfg_{i:03d}" for i in range(N)]
    idx = pd.bdate_range("2020-01-01", periods=T)
    return {
        "Pure noise": pd.DataFrame(noise, index=idx, columns=cols),
        "One real edge": pd.DataFrame(signal, index=idx, columns=cols),
    }


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, (name, df) in zip(axes, make_matrices().items(), strict=True):
        pbo = cscv_pbo(df, n_blocks=16)
        dsr = deflated_sharpe_best(df)
        print(
            f"{name:14s} PBO={pbo.pbo:.3f}  best={dsr.best} "
            f"SR_ann={annualize_sharpe(dsr.sharpe, PPY):.2f}  "
            f"SR0_ann={annualize_sharpe(dsr.sr0, PPY):.2f}  PSR={dsr.psr:.3f}  DSR={dsr.dsr:.3f}"
        )
        plot_pbo(pbo, row[0], row[1], title=name + ":")
    fig.tight_layout()
    fig.savefig(DOCS / "pbo_demo.png", dpi=130)
    print(f"saved {DOCS / 'pbo_demo.png'}")


if __name__ == "__main__":
    main()
