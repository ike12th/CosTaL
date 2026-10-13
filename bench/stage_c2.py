"""Stage C′ (docs/04 §5, §5.1; docs/05 §1.1): is the "+" worth it, and does the row norm help?

One script, one set of Leiden runs, three questions, all on deduplicated Levine (161,443 cells):

1. **The 2×2 of docs/04 §5** — select ∈ {cosine, tanimoto} × weight ∈ {cosine, tanimoto}, with
   Tanimoto selection on the raw rows. `cosine/tanimoto` is the published CosTaL. Reported: ARI /
   NMI against the gating labels over several Leiden seeds, plus how much the neighbor sets change
   when the selection metric changes.
2. **The row-norm question of docs/04 §5.1** — on Levine (the positive control: per-channel
   arcsinh, row norm = overall marker abundance), how much of ‖x‖ do the gating labels explain
   (η²), and does Tanimoto weighting beat cosine weighting at the same selection?
3. **docs/05 §1.1, step 0** — is there a resolution-limit problem at all? Per gated population,
   recall = the share of its cells that land in its dominant cluster, against abundance. A
   systematically lower recall for rare populations would justify the methodology track.

Leiden: leidenalg RBConfiguration, resolution 0.8, n_iterations from the command line (default 10,
the paper's setting). Usage:

    python bench/stage_c2.py [n_iterations]

Output: bench/out/stage_c2.json.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stage_a import SEEDS, environment, label_scores, load_levine  # noqa: E402

import costal_plus as cp  # noqa: E402
from costal_plus import _costal_plus as ext  # noqa: E402

OUT = HERE / "out" / "stage_c2.json"
K = 10
RESOLUTION = 0.8


def build(g: cp.KnnGraph, weight: str):
    import igraph as ig

    w = cp.reweight(g, weight)
    rows, cols, ew = ext._fuzzy_union(g.idx, w)
    return ig.Graph(n=g.n, edges=np.column_stack([rows, cols]).tolist(),
                    edge_attrs={"weight": ew.tolist()})


def leiden(graph, seed: int, n_iterations: int) -> np.ndarray:
    import leidenalg as la

    part = la.find_partition(graph, la.RBConfigurationVertexPartition, weights="weight",
                             n_iterations=n_iterations, resolution_parameter=RESOLUTION, seed=seed)
    return np.asarray(part.membership)


def population_recall(labels: np.ndarray, membership: np.ndarray) -> dict[int, float]:
    """Share of each gated population's cells that fall in its dominant cluster."""
    out = {}
    for p in np.unique(labels[labels >= 0]):
        clusters = membership[labels == p]
        out[int(p)] = float(np.bincount(clusters).max() / len(clusters))
    return out


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """Share of the variance of `values` explained by the group means."""
    total = np.var(values) * len(values)
    between = sum(len(v) * (v.mean() - values.mean()) ** 2
                  for v in (values[groups == g] for g in np.unique(groups)))
    return float(between / total)


def main(n_iterations: int) -> None:
    from scipy.stats import spearmanr
    from sklearn.metrics import adjusted_rand_score

    x, labels = load_levine(None, dedup=True)
    gated = labels >= 0
    t0 = time.perf_counter()
    knn = {"cosine": cp.knn(x, K), "tanimoto": cp.knn(x, K, select="tanimoto")}
    t_knn = time.perf_counter() - t0
    same = np.mean([set(a) == set(b) for a, b in zip(knn["cosine"].idx.tolist(),
                                                      knn["tanimoto"].idx.tolist())])
    jac = np.mean([len(set(a) & set(b)) / len(set(a) | set(b))
                   for a, b in zip(knn["cosine"].idx.tolist(), knn["tanimoto"].idx.tolist())])
    print(f"kNN (both metrics) {t_knn:.1f} s; rows with identical neighbor sets {same:.3f}, "
          f"mean Jaccard {jac:.3f}", flush=True)

    cells = {}
    for select, weight in itertools.product(("cosine", "tanimoto"), ("cosine", "tanimoto")):
        graph = build(knn[select], weight)
        ms = [leiden(graph, s, n_iterations) for s in SEEDS]
        cells[(select, weight)] = ms
        scores = [label_scores(labels, m) for m in ms]
        print(f"  select={select:<8} weight={weight:<8} ARI "
              f"{np.mean([s['ari'] for s in scores]):.3f}±{np.std([s['ari'] for s in scores]):.3f}  "
              f"NMI {np.mean([s['nmi'] for s in scores]):.3f}", flush=True)

    paper = cells[("cosine", "tanimoto")]
    grid = []
    for (select, weight), ms in cells.items():
        scores = [label_scores(labels, m) for m in ms]
        grid.append({
            "select": select, "weight": weight,
            "published_costal": (select, weight) == ("cosine", "tanimoto"),
            **{key: {"mean": float(np.mean([s[key] for s in scores])),
                     "sd": float(np.std([s[key] for s in scores])),
                     "values": [float(s[key]) for s in scores]}
               for key in ("ari", "nmi", "clusters")},
            "ari_vs_published_same_seed": float(np.mean([adjusted_rand_score(a, b)
                                                         for a, b in zip(ms, paper)])),
        })

    # docs/05 §1.1: per-population recall against abundance, for the published CosTaL
    abundance = {int(p): float(np.mean(labels[gated] == p)) for p in np.unique(labels[gated])}
    recalls = [population_recall(labels, m) for m in paper]
    per_pop = [{"population": p, "abundance": abundance[p],
                "recall_mean": float(np.mean([r[p] for r in recalls])),
                "recall_min": float(np.min([r[p] for r in recalls]))}
               for p in sorted(abundance, key=abundance.get)]
    rho, pval = spearmanr([r["abundance"] for r in per_pop], [r["recall_mean"] for r in per_pop])
    print("  per-population recall (published CosTaL):", flush=True)
    for r in per_pop:
        print(f"    pop {r['population']:>2}  abundance {r['abundance']:.4f}  "
              f"recall {r['recall_mean']:.3f} (min {r['recall_min']:.3f})", flush=True)
    print(f"  Spearman(abundance, recall) = {rho:.3f} (p = {pval:.3g})", flush=True)

    # docs/04 §5.1: what does the row norm encode on Levine?
    norm = np.linalg.norm(x, axis=1)
    # Σ asinh(max(raw, 0)/5) = Σ max(x, 0), since asinh is odd and monotone
    abundance_cell = np.clip(x, 0, None).sum(axis=1)
    norm_stats = {
        "eta2_of_row_norm_on_gating_labels": eta_squared(norm[gated], labels[gated]),
        "spearman_row_norm_vs_abundance": float(spearmanr(norm, abundance_cell)[0]),
    }
    print(f"  row norm: {norm_stats}", flush=True)

    OUT.write_text(json.dumps({
        "environment": environment(),
        "n": len(x), "k": K, "resolution": RESOLUTION, "n_iterations": n_iterations,
        "seeds": list(SEEDS),
        "selection_change": {"rows_identical": float(same), "mean_jaccard": float(jac),
                             "knn_seconds_both": t_knn},
        "grid": grid,
        "per_population_recall": per_pop,
        "spearman_abundance_vs_recall": {"rho": float(rho), "p": float(pval)},
        "row_norm": norm_stats,
    }, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
