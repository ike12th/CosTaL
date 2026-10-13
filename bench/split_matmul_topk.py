#!/usr/bin/env python3
"""Split exact kNN into its matmul and top-k halves to check docs/01 §1.2:
"once matmul is optimized, naive sorting becomes the new bottleneck".

Both halves use numpy: matmul goes through BLAS (multithreaded), top-k through argpartition
(single-threaded, but already a C introselect). The ratio says where the Rust version should
spend its effort.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from original import clustering as _original  # noqa: E402

pp = _original().pp
from profile_costal import load_levine  # noqa: E402


def main() -> int:
    k = 10
    block = 1024
    print(f"{'N':>8} {'matmul':>10} {'topk':>10} {'topk share':>11}")
    for n in (10_000, 25_000, 50_000, 100_000):
        x = pp(load_levine(n), platform="mass")
        xn = np.ascontiguousarray(
            normalize(x, norm="l2", axis=1).astype(np.float32))
        t_mm = t_tk = 0.0
        for s in range(0, n, block):
            e = min(s + block, n)
            t0 = time.perf_counter()
            sim = xn[s:e] @ xn.T
            t1 = time.perf_counter()
            sim[np.arange(e - s), np.arange(s, e)] = -np.inf
            part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            vals = np.take_along_axis(sim, part, axis=1)
            order = np.argsort(-vals, axis=1)
            np.take_along_axis(part, order, axis=1)
            t2 = time.perf_counter()
            t_mm += t1 - t0
            t_tk += t2 - t1
        print(f"{n:>8} {t_mm:>9.2f}s {t_tk:>9.2f}s {t_tk/(t_mm+t_tk):>10.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
