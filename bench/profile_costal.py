#!/usr/bin/env python3
"""对原版 CosTaL 做分阶段 profile。

目的只有一个（docs/01 §1.1 的前置动作）：用真实数字确认或推翻
"瓶颈只有一个地方：kNN 建图"。

口径说明
--------
* 计时的是**原版 CosTaL 自己的函数**（`save_as_ijv` / `build_tani_graph` /
  `la_find_partition`），不是我重写的等价物——否则测的是我的代码。
* kNN 那一格测不了原版：CosTaL 的精确 kNN 是随包分发的 `p_knng_Linux`，
  ELF x86-64 二进制，本机（macOS arm64）跑不了。所以 kNN 一栏给三个数：
    - `bruteforce`：CosTaL 在非 Linux 平台实际走的回退路径
      （`phenograph.bruteforce_nn`，O(n²)）；
    - `blas_blocked`：分块 BLAS matmul + argpartition，代表"一个实现得当的
      精确 kNN"的量级，同时是 Rust 版要打败的下限；
    - `hnsw`：近似 kNN（docs/01 §2 的 ② 路线）。
  三个数把真实 kNN 成本夹在中间，足够判断它是不是"绝大部分时间"。
* `build_tani_graph` 会被再拆一层（parse / tanimoto / igraph），拆解部分是
  复刻同样的代码，已标注 `decomposed`。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "reference" / "CosTaL-0.0.2.4"))

import igraph as ig  # noqa: E402
import leidenalg as la  # noqa: E402
from scipy import sparse as sp  # noqa: E402
from scipy.linalg import norm as dense_norm  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402

from CosTaL.clustering import (  # noqa: E402
    build_tani_graph,
    la_find_partition,
    pp,
    save_as_ijv,
    tani_from_cos,
)

TIMINGS: dict[str, float] = {}


@contextmanager
def stage(name: str, sink: dict | None = None):
    sink = TIMINGS if sink is None else sink
    t0 = time.perf_counter()
    yield
    sink[name] = time.perf_counter() - t0
    print(f"    {name:<28} {sink[name]:8.3f} s", flush=True)


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
# Levine_32dim 的 32 个表面 marker：去掉 Time/Cell_length/DNA1/DNA2/Viability
# 和 file_number/event_number/label/individual 这些非 marker 列。
NON_MARKER = {
    "Time", "Cell_length", "DNA1", "DNA2", "Viability",
    "file_number", "event_number", "label", "individual",
}


def load_levine(n: int | None, seed: int = 0):
    import fcsparser

    path = HERE / "data" / "Levine_32dim_notransform.fcs"
    _, df = fcsparser.parse(str(path), reformat_meta=True)
    markers = [c for c in df.columns if c not in NON_MARKER]
    assert len(markers) == 32, markers
    x = df[markers].to_numpy(dtype=np.float64)
    if n is not None and n < len(x):
        idx = np.random.default_rng(seed).choice(len(x), n, replace=False)
        x = x[np.sort(idx)]
    return np.ascontiguousarray(x)


# --------------------------------------------------------------------------
# kNN 的三种测法
# --------------------------------------------------------------------------
def knn_blas_blocked(x: np.ndarray, k: int, block: int = 256):
    """分块 BLAS matmul + argpartition。返回 (idx, cos) 且不含自身。

    block 取 256 而不是更大：`argpartition` 会分配一个 `block x n` 的 int64
    中间数组，265k 细胞下 block=1024 就是 2.2 GB/块，16 GB 的机器会开始换页，
    测出来的就不是算法时间而是 swap 时间。"""
    xn = normalize(x, norm="l2", axis=1).astype(np.float32, copy=False)
    xn = np.ascontiguousarray(xn)
    n = xn.shape[0]
    idx = np.empty((n, k), dtype=np.int64)
    cos = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, block):
        e = min(s + block, n)
        sim = xn[s:e] @ xn.T                      # (b, n)
        sim[np.arange(e - s), np.arange(s, e)] = -np.inf   # 去掉自身
        np.negative(sim, out=sim)                 # 原地取负，省一份 b x n
        part = np.argpartition(sim, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(vals, axis=1)
        idx[s:e] = np.take_along_axis(part, order, axis=1)
        cos[s:e] = -np.take_along_axis(vals, order, axis=1)
        del sim, part, vals, order
    return idx, cos


def knn_bruteforce(x: np.ndarray, k: int):
    """CosTaL 在 macOS 上实际走的路径。"""
    from phenograph import bruteforce_nn

    d, idx = bruteforce_nn.knnsearch(x, k + 1, "cosine")
    return np.asarray(idx)[:, 1:], np.asarray(d)[:, 1:]


def knn_hnsw(x: np.ndarray, k: int, ef: int = 100, m: int = 16):
    import hnswlib

    xn = normalize(x, norm="l2", axis=1).astype(np.float32)
    n, d = xn.shape
    p = hnswlib.Index(space="cosine", dim=d)
    p.init_index(max_elements=n, ef_construction=ef, M=m)
    p.set_num_threads(os.cpu_count())
    p.add_items(xn, np.arange(n))
    p.set_ef(max(ef, k + 1))
    lab, dist = p.knn_query(xn, k=k + 1)
    return lab[:, 1:].astype(np.int64), (1.0 - dist[:, 1:]).astype(np.float32)


# --------------------------------------------------------------------------
# 把 kNN 结果写成 p_knng 的 .clu 格式，好让 build_tani_graph 原样跑
# --------------------------------------------------------------------------
def write_clu(path: Path, idx: np.ndarray, cos: np.ndarray) -> None:
    """CLUTO 风格：首行 `nrows ncols nnz`，之后每行 k 组 `1-based下标 值`。
    1-based 是因为 build_tani_graph 里会 `i - 1`（原注释写着 "bug in L2knng"）。"""
    n, k = idx.shape
    with open(path, "w", encoding="ascii") as f:
        f.write(f"{n} {n} {n * k}\n")
        for i in range(n):
            f.write(" ".join(f"{j + 1} {v:.6f}" for j, v in zip(idx[i], cos[i])) + "\n")


# --------------------------------------------------------------------------
# build_tani_graph 的内部拆解（复刻同样的代码，标注为 decomposed）
# --------------------------------------------------------------------------
def decompose_build_tani_graph(clu_file: Path, data: np.ndarray) -> dict[str, float]:
    import itertools
    import warnings

    out: dict[str, float] = {}
    with stage("  [d] read+split .clu", out):
        with open(clu_file, "r") as f:
            clu = [line.split() for line in f.readlines()]
    with stage("  [d] parse to py lists", out):
        meta = list(map(int, clu[0]))
        csr = clu[1:]
        shape = (meta[0], int(meta[2] / meta[0]))
        rows = list(itertools.chain.from_iterable(
            itertools.repeat(i, shape[1]) for i in range(shape[0])))
        flat = list(itertools.chain.from_iterable(csr))
        columns = [i - 1 for i in map(int, flat[::2])]
        cos_vals = list(map(float, flat[1::2]))
    with stage("  [d] tanimoto weights", out):
        cell_norm = dense_norm(data, axis=1)
        weights = [tani_from_cos(cell_norm[i], cell_norm[j], v)
                   for i, j, v in zip(rows, columns, cos_vals)]
    with stage("  [d] scipy sym/prune", out):
        warnings.filterwarnings("ignore")
        csr_m = sp.csr_matrix(
            (np.array(weights), (np.array(rows), np.array(columns))),
            shape=(meta[0], meta[1]))
        csr_m = csr_m + csr_m.transpose() - csr_m.multiply(csr_m.transpose())
        csr_m.setdiag(0)
        csr_m = sp.triu(csr_m)
        csr_m.eliminate_zeros()
        warnings.resetwarnings()
    with stage("  [d] igraph build", out):
        sources, targets = csr_m.nonzero()
        edgelist = list(zip(sources.tolist(), targets.tolist()))
        ig.Graph(edges=edgelist, edge_attrs={"weight": csr_m.data.tolist()})
    return out


# --------------------------------------------------------------------------
def run(n: int | None, k: int, knn_methods: list[str], tmp: Path) -> dict:
    tag = "full" if n is None else str(n)
    print(f"\n=== N={tag}  k={k} ===", flush=True)
    TIMINGS.clear()

    with stage("S0a load fcs"):
        x_raw = load_levine(n)
    n_cells, n_dim = x_raw.shape
    print(f"    shape = {x_raw.shape}", flush=True)

    with stage("S0b pp (arcsinh/5)"):
        x = pp(x_raw, platform="mass")

    # --- S1: CosTaL 把矩阵写成 .ijv 文本喂给 p_knng ---
    ijv = tmp / f"prof_{tag}.ijv"
    with stage("S1 save_as_ijv"):
        save_as_ijv(str(ijv), normalize(x, norm="l2", axis=1, copy=True))
    ijv_mb = ijv.stat().st_size / 2**20
    ijv.unlink()

    # --- S2: kNN 的三种测法 ---
    knn_times: dict[str, float] = {}
    idx = cos = None
    for m in knn_methods:
        fn = {"blas_blocked": knn_blas_blocked,
              "bruteforce": knn_bruteforce,
              "hnsw": knn_hnsw}[m]
        with stage(f"S2 knn[{m}]", knn_times):
            r_idx, r_cos = fn(x, k)
        if idx is None:          # 后续阶段统一用第一个方法的结果
            idx, cos = r_idx, r_cos
    TIMINGS.update(knn_times)

    clu = tmp / f"prof_{tag}.clu"
    with stage("S2b write .clu (harness)"):
        write_clu(clu, idx, cos)

    # --- S3: 原版 build_tani_graph，原样跑 ---
    with stage("S3 build_tani_graph"):
        g = build_tani_graph(str(clu), x, local=False, method="tani")
    print(f"    graph = {g.vcount()} nodes / {g.ecount()} edges", flush=True)

    # --- S4: Leiden ---
    with stage("S4 leiden (n_iter=10)"):
        part = la_find_partition(
            graph=g, partition_type=la.RBConfigurationVertexPartition,
            n_iterations=10, resolution_parameter=0.8, seed=42)
    n_clusters = len(set(part.membership))
    print(f"    clusters = {n_clusters}", flush=True)

    print("  -- build_tani_graph 内部拆解 --", flush=True)
    decomposed = decompose_build_tani_graph(clu, x)
    clu.unlink()

    return {
        "n_cells": int(n_cells), "n_dim": int(n_dim), "k": k,
        "ijv_file_mb": round(ijv_mb, 1),
        "n_edges": g.ecount(), "n_clusters": n_clusters,
        "stages": dict(TIMINGS), "decomposed_S3": decomposed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="10000,50000,265627")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--knn", default="blas_blocked,hnsw")
    ap.add_argument("--bruteforce-max", type=int, default=50000,
                    help="bruteforce 是 O(n^2)，超过这个规模就不跑了")
    ap.add_argument("--out", default=str(HERE / "out" / "profile.json"))
    args = ap.parse_args()

    tmp = HERE / "out"
    tmp.mkdir(parents=True, exist_ok=True)

    env = {
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
        "numpy_blas": (np.__config__.show(mode="dicts")  # type: ignore[call-arg]
                       .get("Build Dependencies", {}).get("blas", {}).get("name", "?")
                       if hasattr(np.__config__, "show") else "?"),
    }
    print(json.dumps(env, ensure_ascii=False, indent=2))

    base = [m.strip() for m in args.knn.split(",") if m.strip()]
    results = []
    for s in args.sizes.split(","):
        n = None if s.strip() in ("full", "265627") else int(s)
        methods = list(base)
        eff_n = n if n is not None else 265627
        if eff_n <= args.bruteforce_max and "bruteforce" not in methods:
            methods.append("bruteforce")
        results.append(run(n, args.k, methods, tmp))

    out = Path(args.out)
    out.write_text(json.dumps({"env": env, "results": results},
                              ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
