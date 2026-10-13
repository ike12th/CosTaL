"""Confirm docs/04 §5.1's Levine_13 finding with more seeds: does magnitude beat pure cosine?

bench/small_datasets.py found, with 3 seeds, that every select/weight setup using Tanimoto beats
cosine/cosine by ~0.03 ARI at the best resolution on Levine_13. This reruns the 2×2 with 10 seeds
around the best resolutions and compares setups **paired by seed and resolution** (the same Leiden
seed on two graphs is not independent noise, but it is the fairest pairing available).

Leiden: leidenalg RBConfiguration, 2 iterations (reproduces 10, docs/03 §9.1), run in parallel.

Usage: python bench/magnitude_confirm.py [dataset]   (default Levine_13)
Output: bench/out/magnitude_confirm.json (merged per dataset)
"""

from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from local_scaling import data  # noqa: E402
from rare_populations import population_f1  # noqa: E402
from stage_a import environment, label_scores  # noqa: E402
from stage_c2 import build  # noqa: E402

import CosTaL as cp  # noqa: E402

OUT = HERE / "out" / "magnitude_confirm.json"
K = 10
RESOLUTIONS = [0.8, 1.0, 1.2, 1.5, 2.0]
SEEDS = range(10)
SETUPS = list(itertools.product(("cosine", "tanimoto"), ("cosine", "tanimoto")))

_GRAPHS = {}


def _init(n: int, graphs: dict) -> None:
    import igraph as ig

    for key, (rows, cols, w) in graphs.items():
        _GRAPHS[key] = ig.Graph(n=n, edges=np.column_stack([rows, cols]).tolist(), edge_attrs={"weight": w.tolist()})


def _leiden(job: tuple) -> np.ndarray:
    import leidenalg as la

    key, res, seed = job
    return np.asarray(la.find_partition(_GRAPHS[key], la.RBConfigurationVertexPartition, weights="weight",
                                        n_iterations=2, resolution_parameter=res, seed=seed).membership)


def run(name: str) -> dict:
    x, labels, _, _ = data(name)
    knn = {"cosine": cp.knn(x, K), "tanimoto": cp.knn(x, K, select="tanimoto")}
    graphs = {}
    for select, weight in SETUPS:
        g = build(knn[select], weight)
        rows, cols = np.array(g.get_edgelist()).T
        graphs[f"{select}/{weight}"] = (rows, cols, np.asarray(g.es["weight"]))
    jobs = [(key, r, s) for key in graphs for r in RESOLUTIONS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=7, mp_context=mp.get_context("spawn"), initializer=_init,
                             initargs=(len(x), graphs)) as pool:
        ms = list(pool.map(_leiden, jobs))
    lab = labels[labels >= 0]
    order = sorted(np.unique(lab), key=lambda c: np.mean(lab == c))
    rare = [int(c) for c in order[: len(order) // 4]]
    ari, f1r = {}, {}
    for (key, r, s), m in zip(jobs, ms):
        ari[key, r, s] = label_scores(labels, m)["ari"]
        f = population_f1(labels, m, gated_only=True)
        f1r[key, r, s] = float(np.mean([f[c] for c in rare]))
    out = {"n": int(len(x)), "seeds": list(SEEDS), "resolutions": RESOLUTIONS, "setups": {}}
    base = "cosine/cosine"
    for key in graphs:
        per_res = {}
        for r in RESOLUTIONS:
            a = np.array([ari[key, r, s] for s in SEEDS])
            d = a - np.array([ari[base, r, s] for s in SEEDS])
            per_res[str(r)] = {"ari_mean": float(a.mean()), "ari_sd": float(a.std()),
                               "rare_quarter_f1_mean": float(np.mean([f1r[key, r, s] for s in SEEDS])),
                               "paired_diff_vs_cosine_cosine": {"mean": float(d.mean()), "sd": float(d.std()),
                                                                "wins": int(np.sum(d > 0)), "of": len(d)}}
        best = max(per_res, key=lambda r: per_res[r]["ari_mean"])
        # per seed, the best ARI over the resolutions: what a user tuning the resolution would get
        per_seed_best = np.array([max(ari[key, r, s] for r in RESOLUTIONS) for s in SEEDS])
        base_best = np.array([max(ari[base, r, s] for r in RESOLUTIONS) for s in SEEDS])
        out["setups"][key] = {"by_resolution": per_res, "best_resolution": float(best),
                              "best_ari_mean": per_res[best]["ari_mean"], "best_ari_sd": per_res[best]["ari_sd"],
                              "per_seed_best_ari": {"mean": float(per_seed_best.mean()), "sd": float(per_seed_best.std())},
                              "per_seed_best_diff_vs_cosine_cosine": {
                                  "mean": float((per_seed_best - base_best).mean()),
                                  "wins": int(np.sum(per_seed_best > base_best)), "of": len(SEEDS)}}
        s = out["setups"][key]
        print(f"  {name} {key:<17} best {s['best_ari_mean']:.3f}±{s['best_ari_sd']:.3f} @ {best}  "
              f"per-seed best {s['per_seed_best_ari']['mean']:.3f}±{s['per_seed_best_ari']['sd']:.3f}  "
              f"vs cos/cos {s['per_seed_best_diff_vs_cosine_cosine']['mean']:+.3f} "
              f"(wins {s['per_seed_best_diff_vs_cosine_cosine']['wins']}/{len(SEEDS)})  "
              f"rare-quarter F1 @best {per_res[best]['rare_quarter_f1_mean']:.3f}", flush=True)
    return out


def main(names: list[str]) -> None:
    out = json.loads(OUT.read_text()) if OUT.exists() else {"datasets": {}}
    for name in names:
        out["datasets"][name] = run(name)
        out.update(environment=environment(), k=K)
        OUT.write_text(json.dumps(out, indent=2))
        print(f"-> {OUT} ({name})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["Levine_13"])
