#!/usr/bin/env python3
"""核对 docs/01 §2.2 关于 Tanimoto 与幅度的说法。

论点拆成两句，分别验：
 A) 在 L2 归一化向量上 T = c/(2-c)，是 cosine 的严格单调变换（邻居排序不变）；
 B) 原版 CosTaL 到底在哪个空间上算 Tanimoto —— 归一化后的，还是保留幅度的？

(B) 直接读代码就能定：`find_communities` 把 **归一化后**的 X 写进 .ijv 给
p_knng 选邻居，但 `build_tani_graph(clu, X)` 拿的是**未归一化**的 X，
`cell_norm = norm(X, axis=1)` 也是原始幅度。这里用数字再确认一次。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "reference" / "CosTaL-0.0.2.4"))

from CosTaL.clustering import pp, tani_from_cos  # noqa: E402
from profile_costal import knn_blas_blocked, load_levine  # noqa: E402


def main() -> int:
    x = pp(load_levine(20000), platform="mass")
    idx, cos = knn_blas_blocked(x, 10)

    norms = np.linalg.norm(x, axis=1)
    rows = np.repeat(np.arange(len(x)), idx.shape[1])
    cols = idx.ravel()
    c = cos.ravel().astype(np.float64)

    # 原版 CosTaL 实际算的（保留幅度）
    t_costal = np.array([tani_from_cos(norms[i], norms[j], v)
                         for i, j, v in zip(rows, cols, c)])
    # 若先 L2 归一化再算 Tanimoto 会得到的（退化形式）
    t_degen = c / (2.0 - c)

    print(f"pairs                  : {len(c)}")
    print(f"cos                    : [{c.min():.4f}, {c.max():.4f}]")
    print(f"row norm ‖x‖           : [{norms.min():.3f}, {norms.max():.3f}]  "
          f"(比值 max/min = {norms.max()/norms.min():.1f})")
    print(f"T_CosTaL (带幅度)      : [{t_costal.min():.4f}, {t_costal.max():.4f}]")
    print(f"T = c/(2-c) (归一化后)  : [{t_degen.min():.4f}, {t_degen.max():.4f}]")
    print(f"两者最大绝对差         : {np.abs(t_costal - t_degen).max():.4f}")
    print(f"两者 Pearson r         : {np.corrcoef(t_costal, t_degen)[0,1]:.4f}")
    print(f"Spearman 同序?         : "
          f"{np.array_equal(np.argsort(t_costal), np.argsort(t_degen))}")

    # A) 单调性：c/(2-c) 在 c∈[-1,1) 上严格递增
    cc = np.linspace(-0.99, 0.999, 1000)
    print(f"c/(2-c) 严格单调递增?   : {bool(np.all(np.diff(cc/(2-cc)) > 0))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
