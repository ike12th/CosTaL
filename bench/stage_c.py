"""Stage C (docs/03 §5, §6): how many Leiden iterations does CosTaL actually need, and does
igraph's native implementation give the same answer faster?

The original CosTaL hard-codes ``n_iterations=10`` (``bc()`` even uses 100). Leiden is now the
dominant cost (docs/03 §8.2), and the Leiden paper's advice is to iterate until convergence,
which usually takes a few passes. This is an experiment, not engineering: every variant runs on
the same exact CosTaL graph with several seeds and is scored on

- the RBConfiguration quality (the objective being optimized; higher is better),
- agreement with the paper setting (leidenalg, n_iterations=10) under the same seed, next to
  the paper setting's own seed-to-seed agreement (the noise floor),
- ARI / NMI against the manual gating labels, and wall time.

Usage:
    python bench/stage_c.py            # deduplicated Levine, all 161,443 cells
    python bench/stage_c.py 50000      # a deduplicated subsample

Output: bench/out/stage_c.json.
"""

from __future__ import annotations

import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stage_a import SEEDS, environment, label_scores, load_levine, ours_graph  # noqa: E402

OUT = HERE / "out" / "stage_c.json"
RESOLUTION = 0.8
PAPER = ("leidenalg", 10)
VARIANTS = [("leidenalg", n) for n in (1, 2, 3, 5, 10, -1)] + [("igraph", n) for n in (2, 10, -1)]


def run(graph, impl: str, n_iterations: int, seed: int) -> tuple[np.ndarray, float]:
    import leidenalg as la

    t0 = time.perf_counter()
    if impl == "leidenalg":
        part = la.find_partition(graph, la.RBConfigurationVertexPartition, weights="weight",
                                 n_iterations=n_iterations, resolution_parameter=RESOLUTION,
                                 seed=seed)
        membership = part.membership
    else:
        random.seed(seed)  # python-igraph draws from Python's random module
        membership = graph.community_leiden(objective_function="modularity", weights="weight",
                                            resolution=RESOLUTION,
                                            n_iterations=n_iterations).membership
    return np.asarray(membership), time.perf_counter() - t0


def quality(graph, membership: np.ndarray) -> float:
    """RBConfiguration quality of any membership, so both implementations are scored alike."""
    import leidenalg as la

    return la.RBConfigurationVertexPartition(
        graph, initial_membership=membership.tolist(), weights="weight",
        resolution_parameter=RESOLUTION,
    ).quality()


def main(n: int | None) -> None:
    import igraph
    import leidenalg
    from sklearn.metrics import adjusted_rand_score

    x, labels = load_levine(n, dedup=True)
    _, _, graph, t_graph = ours_graph(x, 10)
    print(f"N={len(x)}: graph in {t_graph['knn_weights_graph_s']:.2f} s, {graph.ecount()} edges",
          flush=True)

    runs = {}
    for impl, iters in VARIANTS:
        out = [run(graph, impl, iters, s) for s in SEEDS]
        runs[(impl, iters)] = out
        print(f"  {impl:<9} n_iterations={iters:>3}: "
              f"{np.mean([t for _, t in out]):6.1f} s/run", flush=True)

    paper = [m for m, _ in runs[PAPER]]
    rows = []
    for (impl, iters), out in runs.items():
        ms = [m for m, _ in out]
        scores = [label_scores(labels, m) for m in ms]
        rows.append({
            "impl": impl,
            "n_iterations": iters,
            "seconds": float(np.mean([t for _, t in out])),
            "quality": float(np.mean([quality(graph, m) for m in ms])),
            "ari_vs_paper_same_seed": float(np.mean([adjusted_rand_score(a, b)
                                                     for a, b in zip(ms, paper)])),
            "ari_seed_to_seed": float(np.mean([adjusted_rand_score(a, b)
                                               for a, b in itertools.combinations(ms, 2)])),
            **{f"labels_{key}": {"mean": float(np.mean([s[key] for s in scores])),
                                 "sd": float(np.std([s[key] for s in scores]))}
               for key in ("ari", "nmi", "clusters")},
        })
    for r in rows:
        print(f"  {r['impl']:<9} {r['n_iterations']:>3}  {r['seconds']:6.1f} s  "
              f"Q={r['quality']:.2f}  vs paper {r['ari_vs_paper_same_seed']:.3f}  "
              f"seed-to-seed {r['ari_seed_to_seed']:.3f}  "
              f"ARI {r['labels_ari']['mean']:.3f}±{r['labels_ari']['sd']:.3f}  "
              f"clusters {r['labels_clusters']['mean']:.1f}", flush=True)

    OUT.write_text(json.dumps({
        "environment": {**environment(), "igraph": igraph.__version__,
                        "leidenalg": leidenalg.version},
        "n": len(x), "k": 10, "resolution": RESOLUTION, "seeds": list(SEEDS),
        "graph": {"edges": graph.ecount(), **t_graph},
        "variants": rows,
    }, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
