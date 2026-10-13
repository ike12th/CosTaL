"""docs/05 §1.1, step 0, done properly: do rare populations get swallowed?

The first pass (bench/stage_c2.py) measured each population's share in its dominant cluster. That
can't see absorption: a rare population swallowed whole by a big cluster still sits entirely in one
cluster and scores 1.0. Here each gated population is scored by the F1 of its best-matching cluster,
which also penalizes the cluster being mostly someone else. Run on the published CosTaL graph
(cosine selection, Tanimoto weights) across resolutions, because ARI favors low resolutions that
merge small populations (bench/out/stage_c3.json).

Leiden: leidenalg, 2 iterations (reproduces the paper's 10, docs/03 §9.1), 3 seeds.
Usage: python bench/rare_populations.py
Output: bench/out/rare_populations.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stage_a import environment, label_scores, load_levine  # noqa: E402
from stage_c2 import build  # noqa: E402

import costal_plus as cp  # noqa: E402

OUT = HERE / "out" / "rare_populations.json"
RESOLUTIONS = [0.2, 0.4, 0.8, 1.2]
SEEDS = range(3)


def population_f1(labels: np.ndarray, membership: np.ndarray, gated_only: bool) -> dict[int, float]:
    """For each gated population, the best F1 over clusters. Precision counts either all cells of
    a cluster or only its gated cells (ungated cells are not wrong, just unlabeled)."""
    gated = labels >= 0
    sizes = np.bincount(membership[gated] if gated_only else membership,
                        minlength=membership.max() + 1)
    out = {}
    for p in np.unique(labels[gated]):
        in_p = labels == p
        hits = np.bincount(membership[in_p], minlength=len(sizes))
        precision = hits / np.maximum(sizes, 1)
        recall = hits / in_p.sum()
        f1 = np.where(hits > 0, 2 * precision * recall / (precision + recall + 1e-300), 0.0)
        out[int(p)] = float(f1.max())
    return out


def main() -> None:
    from scipy.stats import spearmanr

    x, labels = load_levine(None, dedup=True)
    gated = labels >= 0
    graph = build(cp.knn(x, 10), "tanimoto")
    abundance = {int(p): float(np.mean(labels[gated] == p)) for p in np.unique(labels[gated])}
    pops = sorted(abundance, key=abundance.get)
    results = []
    for res in RESOLUTIONS:
        ms = [leiden_at(graph, s, res) for s in SEEDS]
        scores = [label_scores(labels, m) for m in ms]
        entry = {"resolution": res,
                 "ari": float(np.mean([s["ari"] for s in scores])),
                 "clusters": float(np.mean([s["clusters"] for s in scores]))}
        for mode in ("all_cells", "gated_only"):
            f1s = [population_f1(labels, m, mode == "gated_only") for m in ms]
            per_pop = {str(p): float(np.mean([f[p] for f in f1s])) for p in pops}
            rho, pval = spearmanr([abundance[p] for p in pops], [per_pop[str(p)] for p in pops])
            entry[mode] = {
                "f1_by_population": per_pop,
                "spearman_abundance_vs_f1": {"rho": float(rho), "p": float(pval)},
                "f1_rarest_5_mean": float(np.mean([per_pop[str(p)] for p in pops[:5]])),
                "f1_largest_5_mean": float(np.mean([per_pop[str(p)] for p in pops[-5:]])),
            }
            e = entry[mode]
            print(f"res={res:<4} {mode:<10} ARI {entry['ari']:.3f} clusters {entry['clusters']:.1f}  "
                  f"F1 rarest-5 {e['f1_rarest_5_mean']:.3f}  largest-5 {e['f1_largest_5_mean']:.3f}  "
                  f"Spearman(abundance, F1) {rho:+.2f} (p={pval:.2g})", flush=True)
        results.append(entry)
    for mode in ("all_cells", "gated_only"):
        print(f"per population, {mode} (abundance: F1 at " + ", ".join(map(str, RESOLUTIONS)) + "):")
        for p in pops:
            print(f"  pop {p:>2} {abundance[p]:.4f}: "
                  + "  ".join(f"{r[mode]['f1_by_population'][str(p)]:.3f}" for r in results))
    OUT.write_text(json.dumps({"environment": environment(), "abundance": abundance,
                               "seeds": list(SEEDS), "results": results}, indent=2))
    print(f"-> {OUT}")


def leiden_at(graph, seed: int, resolution: float) -> np.ndarray:
    import leidenalg as la

    return np.asarray(la.find_partition(graph, la.RBConfigurationVertexPartition, weights="weight",
                                        n_iterations=2, resolution_parameter=resolution,
                                        seed=seed).membership)


if __name__ == "__main__":
    main()
