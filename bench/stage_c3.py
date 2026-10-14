"""Stage C′, follow-up: is cosine weighting's lead over Tanimoto weighting real, or a resolution effect?

At the paper's resolution 0.8, cosine edge weights beat the published Tanimoto weights by ~0.1 ARI
on deduplicated Levine (bench/out/stage_c2.json). Tanimoto compresses weak edges non-uniformly, so
it also shifts the effective resolution, and a single-resolution comparison can't separate the two.
This sweeps the resolution for both weightings (cosine selection, as published) and compares them
at their best resolution and across the whole range.

- Data: deduplicated Levine (161,443 cells) and the raw file (265,627 rows, with the duplicated
  gated cells; docs/02 Addendum), since the paper may have used the latter.
- Leiden: igraph's native implementation (2 iterations, 5 seeds) over the full grid, and leidenalg
  (2 iterations, which reproduces the paper's 10 — docs/03 §9.1; 3 seeds) on deduplicated data.
- Per run: ARI / NMI against the gating labels, cluster count, and per-population recall.

Usage: python bench/stage_c3.py
Output: bench/out/stage_c3.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stage_a import environment, label_scores, load_levine  # noqa: E402
from stage_c2 import build, population_recall  # noqa: E402

import costal as cp  # noqa: E402

OUT = HERE / "out" / "stage_c3.json"
RESOLUTIONS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
IGRAPH_SEEDS = range(5)
LEIDENALG_SEEDS = range(3)


def run(graph, impl: str, resolution: float, seed: int) -> np.ndarray:
    if impl == "igraph":
        random.seed(seed)
        return np.asarray(graph.community_leiden(objective_function="modularity", weights="weight",
                                                 resolution=resolution, n_iterations=2).membership)
    import leidenalg as la

    return np.asarray(la.find_partition(graph, la.RBConfigurationVertexPartition, weights="weight",
                                        n_iterations=2, resolution_parameter=resolution,
                                        seed=seed).membership)


def sweep(name: str, dedup: bool, impls: dict[str, range]) -> list[dict]:
    x, labels = load_levine(None, dedup=dedup)
    g = cp.knn(x, 10)
    graphs = {w: build(g, w) for w in ("cosine", "tanimoto")}
    rows = []
    for impl, seeds in impls.items():
        for weight, graph in graphs.items():
            for res in RESOLUTIONS:
                ms = [run(graph, impl, res, s) for s in seeds]
                scores = [label_scores(labels, m) for m in ms]
                recalls = [population_recall(labels, m) for m in ms]
                pops = sorted(recalls[0])
                row = {
                    "data": name, "impl": impl, "weight": weight, "resolution": res,
                    **{key: {"mean": float(np.mean([s[key] for s in scores])),
                             "sd": float(np.std([s[key] for s in scores]))}
                       for key in ("ari", "nmi", "clusters")},
                    "population_recall": {str(p): float(np.mean([r[p] for r in recalls]))
                                          for p in pops},
                }
                rows.append(row)
                print(f"  {name:<5} {impl:<9} {weight:<8} res={res:<4} "
                      f"ARI {row['ari']['mean']:.3f}±{row['ari']['sd']:.3f}  "
                      f"NMI {row['nmi']['mean']:.3f}  clusters {row['clusters']['mean']:.1f}",
                      flush=True)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(r["data"], r["impl"]) for r in rows})
    for data, impl in keys:
        for weight in ("cosine", "tanimoto"):
            sub = [r for r in rows if (r["data"], r["impl"], r["weight"]) == (data, impl, weight)]
            best = max(sub, key=lambda r: r["ari"]["mean"])
            out.append({"data": data, "impl": impl, "weight": weight,
                        "best_resolution": best["resolution"], "best_ari": best["ari"],
                        "best_nmi": best["nmi"], "clusters_at_best": best["clusters"]["mean"],
                        "mean_ari_over_grid": float(np.mean([r["ari"]["mean"] for r in sub]))})
    return out


def main() -> None:
    rows = sweep("dedup", True, {"igraph": IGRAPH_SEEDS, "leidenalg": LEIDENALG_SEEDS})
    rows += sweep("raw", False, {"igraph": IGRAPH_SEEDS})
    summary = summarize(rows)
    for s in summary:
        print(f"  BEST {s['data']:<5} {s['impl']:<9} {s['weight']:<8} res={s['best_resolution']:<4} "
              f"ARI {s['best_ari']['mean']:.3f}±{s['best_ari']['sd']:.3f}  "
              f"mean over grid {s['mean_ari_over_grid']:.3f}", flush=True)
    OUT.write_text(json.dumps({"environment": environment(), "k": 10, "resolutions": RESOLUTIONS,
                               "summary": summary, "runs": rows}, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
