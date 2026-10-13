"""docs/05 §2.2, experiment #1: does local scale adaptation recover rare populations?

The original `find_communities(..., local=True)` builds the graph with UMAP's fuzzy simplicial set
(`smooth_knn_dist` + `compute_membership_strengths`), but hands it Tanimoto *similarities* as if
they were ascending *distances*, so the nearest neighbor is treated as the farthest (docs/05 §2.2).
Three graphs on the same exact kNN (cosine selection, as published):

- `published`: Tanimoto weights, fuzzy union (the default path);
- `local_as_shipped`: the original `local=True`, bug included (similarities as distances);
- `local_fixed`: distance = 1 − T, each row's ρᵢ / σᵢ fitted on the ascending distances, membership
  exp(−(d − ρᵢ)/σᵢ), then the same fuzzy union.

The original also keeps both directions of every edge in the local path (no `triu`); scaling all
weights by 2 leaves RBConfiguration's optimum unchanged, so all three use one edge per pair here.

Scored over a resolution sweep by ARI / NMI and the per-population best-match F1 of
bench/rare_populations.py (labeled cells only), summarized as the mean F1 of the 5 rarest and the
5 largest populations — the trade-off the global resolution cannot escape (docs/05 §1.1).

Data: deduplicated Levine_32 (14 populations), Levine_13 (24), GSE84133 human3 (14 types, 1–1130
cells). Leiden: leidenalg RBConfiguration; 2 iterations and 3 seeds on the cytometry sets
(reproduces 10, docs/03 §9.1), 10 iterations and 5 seeds on GSE84133.

Usage: python bench/local_scaling.py [dataset ...]
Output: bench/out/local_scaling.json (merged per dataset)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from datasets import load, preprocess  # noqa: E402
from rare_populations import population_f1  # noqa: E402
from stage_a import environment, label_scores, load_levine  # noqa: E402
from small_datasets import leiden  # noqa: E402

import costal_plus as cp  # noqa: E402
from costal_plus import _costal_plus as ext  # noqa: E402

OUT = HERE / "out" / "local_scaling.json"
K = 10
RESOLUTIONS = [0.2, 0.4, 0.8, 1.2, 2.0]
GRAPHS = ("published", "local_as_shipped", "local_fixed")


def local_weights(g: cp.KnnGraph, w: np.ndarray, fixed: bool) -> np.ndarray:
    """UMAP membership strengths per (i, idx[i, j]), as an (n, k) array in the kNN's order."""
    from umap.umap_ import compute_membership_strengths, smooth_knn_dist

    if not fixed:  # what the original does: similarities in place of distances, in .clu order
        d = w.astype(np.float32)
        sigmas, rhos = smooth_knn_dist(d, float(K), local_connectivity=1.0)
        _, _, vals, _ = compute_membership_strengths(g.idx.astype(np.int32), d, sigmas, rhos)
        return vals.reshape(g.n, K).astype(np.float64)
    d = 1.0 - w
    sigmas, rhos = smooth_knn_dist(np.sort(d, axis=1).astype(np.float32), float(K), local_connectivity=1.0)
    rhos, sigmas = rhos[:, None].astype(np.float64), sigmas[:, None].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.exp(-(d - rhos) / sigmas)
    return np.where((d - rhos <= 0) | (sigmas == 0), 1.0, vals)


def graph_of(g: cp.KnnGraph, kind: str):
    import igraph as ig

    w = cp.reweight(g, "tanimoto")
    if kind != "published":
        w = local_weights(g, w, fixed=kind == "local_fixed")
    rows, cols, ew = ext._fuzzy_union(g.idx, np.ascontiguousarray(w))
    return ig.Graph(n=g.n, edges=np.column_stack([rows, cols]).tolist(), edge_attrs={"weight": ew.tolist()})


def data(name: str) -> tuple[np.ndarray, np.ndarray, int, range]:
    if name == "Levine_32":
        x, labels = load_levine(None, dedup=True)
        return x, labels, 2, range(3)
    ds = load(name)
    x, kept = preprocess(ds)
    return (x, ds.labels[kept], 2, range(3)) if ds.kind == "cytometry" else (x, ds.labels[kept], 10, range(5))


def run(name: str) -> dict:
    x, labels, n_iterations, seeds = data(name)
    g = cp.knn(x, K)
    lab = labels[labels >= 0]
    abundance = {int(c): float(np.mean(lab == c)) for c in np.unique(lab)}
    order = sorted(abundance, key=abundance.get)
    rare, large = order[:5], order[-5:]
    rows = []
    for kind in GRAPHS:
        graph = graph_of(g, kind)
        for res in RESOLUTIONS:
            ms = [leiden(graph, s, res, n_iterations) for s in seeds]
            scores = [label_scores(labels, m) for m in ms]
            f1s = [population_f1(labels, m, gated_only=True) for m in ms]
            per_pop = {p: float(np.mean([f[p] for f in f1s])) for p in order}
            row = {"graph": kind, "resolution": res,
                   **{key: {"mean": float(np.mean([s[key] for s in scores])),
                            "sd": float(np.std([s[key] for s in scores]))}
                      for key in ("ari", "nmi", "clusters")},
                   "f1_rarest_5": float(np.mean([per_pop[p] for p in rare])),
                   "f1_largest_5": float(np.mean([per_pop[p] for p in large])),
                   "f1_by_population": {str(p): v for p, v in per_pop.items()}}
            rows.append(row)
            print(f"  {name:<10} {kind:<17} res={res:<4} ARI {row['ari']['mean']:.3f}±{row['ari']['sd']:.3f}  "
                  f"NMI {row['nmi']['mean']:.3f}  clusters {row['clusters']['mean']:5.1f}  "
                  f"F1 rarest-5 {row['f1_rarest_5']:.3f}  largest-5 {row['f1_largest_5']:.3f}", flush=True)
    return {"n": int(len(x)), "abundance": {str(p): abundance[p] for p in order},
            "leiden": {"n_iterations": n_iterations, "seeds": list(seeds)}, "runs": rows}


def main(names: list[str]) -> None:
    out = json.loads(OUT.read_text()) if OUT.exists() else {"datasets": {}}
    for name in names:
        out["datasets"][name] = run(name)
        out.update(environment=environment(), k=K, resolutions=RESOLUTIONS)
        OUT.write_text(json.dumps(out, indent=2))
        print(f"-> {OUT} ({name})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["GSE84133", "Levine_32", "Levine_13"])
