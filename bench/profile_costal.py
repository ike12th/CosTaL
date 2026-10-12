#!/usr/bin/env python3
"""Stage-by-stage profile of the original CosTaL.

It has one purpose (the prerequisite in docs/01 §1.1): confirm or refute, with real numbers,
"there is only one bottleneck: building the kNN graph".

Methodology
-----------
* What is timed is **the original CosTaL's own functions** (`save_as_ijv` /
  `build_tani_graph` / `la_find_partition`), not a rewritten equivalent — otherwise we would
  be measuring our own code.
* The kNN cell can't be measured on the original: CosTaL's exact kNN is the bundled
  `p_knng_Linux`, an ELF x86-64 binary that can't run on this machine (macOS arm64). So the
  kNN column gives three numbers:
    - `bruteforce`: the fallback CosTaL actually takes on non-Linux platforms
      (`phenograph.bruteforce_nn`, O(n²));
    - `blas_blocked`: blocked BLAS matmul + argpartition, the order of magnitude of "a
      well-implemented exact kNN", and the floor the Rust version has to beat;
    - `hnsw`: approximate kNN (route ② in docs/01 §2).
  The three bracket the real kNN cost, which is enough to tell whether it is "most of the time".
* `build_tani_graph` is broken down one level further (parse / tanimoto / igraph); the
  breakdown replicates the same code and is labeled `decomposed`.
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
# Data
# --------------------------------------------------------------------------
# The 32 surface markers of Levine_32dim: drop Time/Cell_length/DNA1/DNA2/Viability
# and the non-marker columns file_number/event_number/label/individual.
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
# Three ways to measure kNN
# --------------------------------------------------------------------------
def knn_blas_blocked(x: np.ndarray, k: int, block: int = 256):
    """Blocked BLAS matmul + argpartition. Returns (idx, cos), excluding self.

    block is 256 rather than larger: `argpartition` allocates a `block x n` int64 intermediate,
    so with 265k cells block=1024 means 2.2 GB per block and a 16 GB machine starts paging —
    then we'd be timing swap, not the algorithm."""
    xn = normalize(x, norm="l2", axis=1).astype(np.float32, copy=False)
    xn = np.ascontiguousarray(xn)
    n = xn.shape[0]
    idx = np.empty((n, k), dtype=np.int64)
    cos = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, block):
        e = min(s + block, n)
        sim = xn[s:e] @ xn.T                      # (b, n)
        sim[np.arange(e - s), np.arange(s, e)] = -np.inf   # remove self
        np.negative(sim, out=sim)                 # negate in place, saving one b x n copy
        part = np.argpartition(sim, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(vals, axis=1)
        idx[s:e] = np.take_along_axis(part, order, axis=1)
        cos[s:e] = -np.take_along_axis(vals, order, axis=1)
        del sim, part, vals, order
    return idx, cos


def knn_bruteforce(x: np.ndarray, k: int):
    """The path CosTaL actually takes on macOS."""
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
# Write kNN results in p_knng's .clu format so build_tani_graph can run unchanged
# --------------------------------------------------------------------------
def write_clu(path: Path, idx: np.ndarray, cos: np.ndarray) -> None:
    """CLUTO style: first line `nrows ncols nnz`, then per row k pairs of `1-based-index value`.
    1-based because build_tani_graph applies `i - 1` (the original comment says "bug in L2knng")."""
    n, k = idx.shape
    with open(path, "w", encoding="ascii") as f:
        f.write(f"{n} {n} {n * k}\n")
        for i in range(n):
            f.write(" ".join(f"{j + 1} {v:.6f}" for j, v in zip(idx[i], cos[i])) + "\n")


# --------------------------------------------------------------------------
# Breakdown of build_tani_graph (replicates the same code, labeled decomposed)
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

    # --- S1: CosTaL writes the matrix as .ijv text for p_knng ---
    ijv = tmp / f"prof_{tag}.ijv"
    with stage("S1 save_as_ijv"):
        save_as_ijv(str(ijv), normalize(x, norm="l2", axis=1, copy=True))
    ijv_mb = ijv.stat().st_size / 2**20
    ijv.unlink()

    # --- S2: three ways to measure kNN ---
    knn_times: dict[str, float] = {}
    idx = cos = None
    for m in knn_methods:
        fn = {"blas_blocked": knn_blas_blocked,
              "bruteforce": knn_bruteforce,
              "hnsw": knn_hnsw}[m]
        with stage(f"S2 knn[{m}]", knn_times):
            r_idx, r_cos = fn(x, k)
        if idx is None:          # later stages all use the first method's result
            idx, cos = r_idx, r_cos
    TIMINGS.update(knn_times)

    clu = tmp / f"prof_{tag}.clu"
    with stage("S2b write .clu (harness)"):
        write_clu(clu, idx, cos)

    # --- S3: the original build_tani_graph, run unchanged ---
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

    print("  -- build_tani_graph breakdown --", flush=True)
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
                    help="bruteforce is O(n^2); skip it above this size")
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
