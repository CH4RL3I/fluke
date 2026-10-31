"""Opt-in CSCV time/memory benchmark; emits one JSON record and writes no files.

Run each configuration in a fresh process, for example:
    uv run python examples/benchmark_cscv.py --strategies 15000 --blocks 16 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import scipy

from overfit import cscv_pbo


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _peak_rss() -> tuple[int | None, str]:
    try:
        import resource
    except ImportError:
        return None, "unavailable: resource module not supported on this platform"
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports bytes; Linux reports KiB. Avoid guessing for other Unix variants.
    if sys.platform == "darwin":
        return int(raw), "resource.getrusage(RUSAGE_SELF).ru_maxrss (Darwin bytes)"
    if sys.platform.startswith("linux"):
        return int(raw * 1024), "resource.getrusage(RUSAGE_SELF).ru_maxrss (Linux KiB)"
    return None, "unavailable: ru_maxrss unit not defined here for this platform"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=_positive_int, default=512)
    parser.add_argument("--strategies", type=_positive_int, default=15_000)
    parser.add_argument("--blocks", type=_positive_int, default=16)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    returns = rng.normal(0.0, 0.01, size=(args.observations, args.strategies))
    start = time.perf_counter()
    try:
        result = cscv_pbo(returns, n_blocks=args.blocks, batch_size=args.batch_size)
    except ValueError as exc:
        parser.error(str(exc))
    elapsed = time.perf_counter() - start
    peak_bytes, source = _peak_rss()
    record = {
        "parameters": vars(args),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "input_bytes": returns.nbytes,
        "n_observations_used": result.n_obs_used,
        "n_combinations": result.n_combinations,
        "cscv_elapsed_seconds": elapsed,
        "process_peak_rss_bytes": peak_bytes,
        "peak_rss_source": source,
        "pbo": result.pbo,
        "prob_loss": result.prob_loss,
        "memory_scope": (
            "whole process, including imports and input generation; "
            "not incremental CSCV allocation"
        ),
        "timing_scope": "cscv_pbo call only; excludes imports and input generation",
    }
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
