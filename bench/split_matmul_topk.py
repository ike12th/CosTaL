#!/usr/bin/env python3
"""把精确 kNN 拆成 matmul 和 top-k 两半，核对 docs/01 §1.2：
"matmul 优化完之后，朴素排序会成为新瓶颈"。

两半都用 numpy：matmul 走 BLAS（多线程），top-k 走 argpartition（单线程、
但已经是 C 实现的 introselect）。比值就是 Rust 版该往哪儿投火力的依据。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "reference" / "CosTaL-0.0.2.4"))

from CosTaL.clustering import pp  # noqa: E402
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
