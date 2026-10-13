"""End-to-end timings on the paper's small datasets: the original code path vs CosTaL+.

Per dataset (paper preset, k = 10), on an otherwise idle machine:

- original: the `bc()` path (bench/stage_a.py `original_graph`: phenograph's brute-force cosine kNN,
  then the original `map_tani` and `build_graph`) + leidenalg at the paper's settings (resolution
  0.8, 10 iterations). The Linux `p_knng` path can't run on macOS; docs/02 §1.1 has its numbers.
- CosTaL+ `leiden="paper"`: exact kNN + Tanimoto + fuzzy union in Rust, then the same leidenalg call.
- CosTaL+ `leiden="fast"`: the same graph, igraph's native Leiden with 2 iterations (docs/03 §9.1).

Preprocessing (scanpy, identical for both) is reported separately. Small datasets take the best of 3
runs; Levine_13 runs once (the original kNN alone takes minutes).

Usage: python bench/timing_small.py [dataset ...]
Output: bench/out/timing_small.json
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from datasets import NAMES, load, preprocess  # noqa: E402
from stage_a import environment, original_graph, ours_graph  # noqa: E402

OUT = HERE / "out" / "timing_small.json"
K = 10


def leiden_paper(graph) -> float:
    import leidenalg as la

    t0 = time.perf_counter()
    la.find_partition(graph, la.RBConfigurationVertexPartition, weights="weight", n_iterations=10,
                      resolution_parameter=0.8, seed=0)
    return time.perf_counter() - t0


def leiden_fast(graph) -> float:
    random.seed(0)
    t0 = time.perf_counter()
    graph.community_leiden(objective_function="modularity", weights="weight", resolution=0.8, n_iterations=2)
    return time.perf_counter() - t0


def run(name: str) -> dict:
    ds = load(name)
    t0 = time.perf_counter()
    x, _ = preprocess(ds)
    t_pp = time.perf_counter() - t0
    repeat = 1 if x.shape[0] > 100_000 else 3
    best = {}
    for _ in range(repeat):
        _, _, graph_o, t_o = original_graph(x, K)
        t0 = time.perf_counter()
        _, _, graph, _ = ours_graph(x, K)  # timed with the igraph construction, as the original's build_graph
        t_c = {"knn_weights_graph_s": time.perf_counter() - t0}
        sample = {"original_knn_s": t_o["knn_s"], "original_weights_graph_s": t_o["weights_graph_s"],
                  "original_leiden_paper_s": leiden_paper(graph_o),
                  "costal_plus_graph_s": t_c["knn_weights_graph_s"],
                  "costal_plus_leiden_paper_s": leiden_paper(graph),
                  "costal_plus_leiden_fast_s": leiden_fast(graph)}
        best = {k: min(v, best.get(k, v)) for k, v in sample.items()}
    total = {
        "original": best["original_knn_s"] + best["original_weights_graph_s"] + best["original_leiden_paper_s"],
        "costal_plus_paper": best["costal_plus_graph_s"] + best["costal_plus_leiden_paper_s"],
        "costal_plus_fast": best["costal_plus_graph_s"] + best["costal_plus_leiden_fast_s"],
    }
    out = {"cells": int(x.shape[0]), "features": int(x.shape[1]), "repeat": repeat,
           "preprocess_s": t_pp, **best, "end_to_end_s": total,
           "speedup_graph": (best["original_knn_s"] + best["original_weights_graph_s"]) / best["costal_plus_graph_s"],
           "speedup_end_to_end_paper": total["original"] / total["costal_plus_paper"],
           "speedup_end_to_end_fast": total["original"] / total["costal_plus_fast"]}
    print(f"{name:<12} {x.shape[0]:>7} x {x.shape[1]:<4}  original {total['original']:8.2f} s  "
          f"CosTaL+ paper {total['costal_plus_paper']:7.2f} s ({out['speedup_end_to_end_paper']:5.1f}x)  "
          f"fast {total['costal_plus_fast']:6.2f} s ({out['speedup_end_to_end_fast']:6.1f}x)  "
          f"graph {out['speedup_graph']:6.1f}x  [preprocess {t_pp:.1f} s]", flush=True)
    return out


def main(names: list[str]) -> None:
    out = json.loads(OUT.read_text()) if OUT.exists() else {"datasets": {}}
    for name in names:
        out["datasets"][name] = run(name)
        out.update(environment=environment(), k=K)
        OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or NAMES[1:] + NAMES[:1])
