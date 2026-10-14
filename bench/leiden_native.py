"""docs/07 §5.2: CosTaL's native Leiden against igraph's, on the paper's nine datasets.

On each dataset the exact kNN graph (k = 10, Tanimoto weights, fuzzy union) is built once. Then,
at the default resolution 0.8 with 2 iterations:

- ``igraph``: python-igraph's ``community_leiden`` (modularity), 20 seeds through Python's
  ``random``, timed without the graph conversion (reported separately);
- ``native``: ``costal._costal._leiden``, 20 seeds, timed from the edge list (it builds its own
  graph inside);
- ``leidenalg``: the paper's setting (RBConfiguration, 10 iterations), 3 seeds, as a reference
  (not on Mosmann_rare, where one run takes about 10 minutes).

Each partition is scored by its modularity at 0.8 (the objective both optimize) and against the
labels: ARI / NMI on labeled cells and the per-population best-match F1 of
bench/rare_populations.py, summarized as the 5 rarest and the 5 largest populations.

Then ``find_communities_multiscale`` once with igraph's Leiden and once with the native one, and a
check that the native partition is the same at 1 thread and at all threads.

The native Leiden is ``leiden="fast"`` in the API; igraph's is ``leiden="igraph"``.

Usage: python bench/leiden_native.py [dataset ...]      (default: all nine)
Output: bench/out/leiden_native.json (merged per dataset)
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

from local_scaling import data  # noqa: E402
from rare_populations import population_f1  # noqa: E402
from stage_a import environment, label_scores  # noqa: E402

import costal as cp  # noqa: E402
from costal import _costal as ext  # noqa: E402
from costal._api import _igraph  # noqa: E402

OUT = HERE / "out" / "leiden_native.json"
NAMES = ["GSE84133", "GSE74672", "GSE81861", "E-MTAB-3321", "Nilsson_rare", "Samusik_01",
         "Levine_32", "Levine_13", "Mosmann_rare"]
K, RESOLUTION, N_ITERATIONS, SEEDS, PAPER_SEEDS = 10, 0.8, 2, range(20), range(3)


def scores(labels: np.ndarray, m: np.ndarray, quality: float, seconds: float, rare, large) -> dict:
    f1 = population_f1(labels, m, gated_only=True)
    s = label_scores(labels, m)
    return {"seconds": seconds, "modularity": quality, "ari": float(s["ari"]), "nmi": float(s["nmi"]),
            "clusters": int(s["clusters"]),
            "f1_rarest_5": float(np.mean([f1[p] for p in rare])),
            "f1_largest_5": float(np.mean([f1[p] for p in large])),
            "f1_mean": float(np.mean(list(f1.values())))}


def summary(runs: list[dict]) -> dict:
    keys = ("seconds", "modularity", "ari", "nmi", "clusters", "f1_rarest_5", "f1_largest_5", "f1_mean")
    return {k: {"mean": float(np.mean([r[k] for r in runs])), "sd": float(np.std([r[k] for r in runs])),
                "min": float(np.min([r[k] for r in runs])), "max": float(np.max([r[k] for r in runs]))}
            for k in keys}


def show(name: str, impl: str, s: dict) -> None:
    print(f"  {name:<12} {impl:<9} {s['seconds']['mean']:7.3f}s  Q {s['modularity']['mean']:.5f} "
          f"± {s['modularity']['sd']:.5f}  ARI {s['ari']['mean']:.3f} ± {s['ari']['sd']:.3f}  "
          f"clusters {s['clusters']['mean']:5.1f}  F1 rare-5 {s['f1_rarest_5']['mean']:.3f}  "
          f"large-5 {s['f1_largest_5']['mean']:.3f}", flush=True)


def run(name: str) -> dict:
    x, labels, _, _ = data(name)
    lab = labels[labels >= 0]
    abundance = {int(c): float(np.mean(lab == c)) for c in np.unique(lab)}
    order = sorted(abundance, key=abundance.get)
    rare, large = order[:5], order[-5:]
    g = cp.knn(x, K)
    rows, cols, w = ext._fuzzy_union(g.idx, cp.reweight(g, "tanimoto"))
    n = g.n
    del g

    def quality(m) -> float:
        return float(ext._modularity(n, rows, cols, w, np.asarray(m, dtype=np.int32), RESOLUTION))

    t = time.perf_counter()
    graph = _igraph(n, rows, cols, w)
    build_igraph = time.perf_counter() - t
    out = {"n": int(n), "edges": int(len(rows)), "igraph_build_seconds": build_igraph}

    runs = []
    for s in SEEDS:
        random.seed(s)
        t = time.perf_counter()
        m = graph.community_leiden(objective_function="modularity", weights="weight",
                                   resolution=RESOLUTION, n_iterations=N_ITERATIONS).membership
        dt = time.perf_counter() - t
        m = np.asarray(m)
        runs.append(scores(labels, m, quality(m), dt, rare, large))
    out["igraph"] = summary(runs)
    show(name, "igraph", out["igraph"])

    runs = []
    for s in SEEDS:
        t = time.perf_counter()
        m = ext._leiden(n, rows, cols, w, resolution=RESOLUTION, n_iterations=N_ITERATIONS, seed=s)[0]
        dt = time.perf_counter() - t
        runs.append(scores(labels, m, quality(m), dt, rare, large))
    out["native"] = summary(runs)
    show(name, "native", out["native"])
    one = ext._leiden(n, rows, cols, w, resolution=RESOLUTION, n_iterations=N_ITERATIONS, seed=0, n_threads=1)[0]
    every = ext._leiden(n, rows, cols, w, resolution=RESOLUTION, n_iterations=N_ITERATIONS, seed=0)[0]
    out["native_same_at_1_thread_and_all"] = bool(np.array_equal(one, every))

    if name != "Mosmann_rare":
        import leidenalg as la

        runs = []
        for s in PAPER_SEEDS:
            t = time.perf_counter()
            m = la.find_partition(graph, la.RBConfigurationVertexPartition, weights="weight",
                                  n_iterations=10, resolution_parameter=RESOLUTION, seed=s).membership
            dt = time.perf_counter() - t
            m = np.asarray(m)
            runs.append(scores(labels, m, quality(m), dt, rare, large))
        out["leidenalg_paper"] = summary(runs)
        show(name, "leidenalg", out["leidenalg_paper"])
    del graph

    for impl, option in (("igraph", "igraph"), ("native", "fast")):
        t = time.perf_counter()
        m, info = cp.find_communities_multiscale(x, leiden=option, return_info=True)
        dt = time.perf_counter() - t
        sc = scores(labels, m, quality(m), dt, rare, large)
        out[f"multiscale_{impl}"] = {**sc, "carved": len(info["carved"])}
        print(f"  {name:<12} multiscale {impl:<6} {dt:7.1f}s  ARI {sc['ari']:.3f}  clusters {sc['clusters']:3d}  "
              f"F1 rare-5 {sc['f1_rarest_5']:.3f}  large-5 {sc['f1_largest_5']:.3f}  mean {sc['f1_mean']:.3f}",
              flush=True)
    return out


def main(names: list[str]) -> None:
    out = json.loads(OUT.read_text()) if OUT.exists() else {"datasets": {}}
    for name in names:
        out["datasets"][name] = run(name)
        out.update(environment=environment(), k=K, resolution=RESOLUTION, n_iterations=N_ITERATIONS,
                   seeds=len(SEEDS), paper_seeds=len(PAPER_SEEDS))
        OUT.write_text(json.dumps(out, indent=2))
        print(f"-> {OUT} ({name})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or NAMES)
