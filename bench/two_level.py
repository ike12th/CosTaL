"""The paper's own answer to rare populations, as a baseline for docs/05 §2.3: two-level CosTaL.

The CosTaL supplement ("CosTaL Multilevel Clustering") describes an iterative option: cluster, then
cluster each resulting population again, skipping populations smaller than a threshold (default 500
cells); with two levels it gave the best F1 on Mosmann_rare and Nilsson_rare. The implementation
(`CosTaL.clustering_iterative`) was never released (PyPI stops at 0.0.2.4), so this reconstructs it
from that description:

- level 1: the published CosTaL graph (exact cosine kNN, Tanimoto, fuzzy union), Leiden at 0.8;
- level 2: for every level-1 cluster with ≥ threshold cells, the same pipeline on that subset alone
  (the kNN recomputed exactly within the cluster, docs/05 §2.1), Leiden at 0.8;
- thresholds 500 (the supplement's default) and 50 (the value in the author's later notebook).

Leiden: leidenalg RBConfiguration, 2 iterations (reproduces 10, docs/03 §9.1), 4 seeds, scored like
bench/multiscale.py (ARI, best-match F1 of the 5 rarest / 5 largest populations, mean F1).

Usage: python bench/two_level.py [dataset ...]
Output: bench/out/two_level.json (merged per dataset)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from local_scaling import data  # noqa: E402
from multiscale import scores  # noqa: E402
from small_datasets import leiden  # noqa: E402
from stage_a import environment  # noqa: E402
from stage_c2 import build  # noqa: E402

import costal_plus as cp  # noqa: E402

OUT = HERE / "out" / "two_level.json"
K = 10
RESOLUTION = 0.8
THRESHOLDS = [500, 50]
SEEDS = range(4)


def two_level(x: np.ndarray, graph, seed: int, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    level1 = leiden(graph, seed, RESOLUTION, 2)
    labels = level1.astype(np.int64).copy()
    next_label = labels.max() + 1
    for c in np.unique(level1):
        cells = np.flatnonzero(level1 == c)
        if len(cells) < max(threshold, K + 2):
            continue
        sub = leiden(build(cp.knn(x[cells], K), "tanimoto"), seed, RESOLUTION, 2)
        if sub.max() == 0:
            continue
        labels[cells] = next_label + sub
        next_label += sub.max() + 1
    return level1, np.unique(labels, return_inverse=True)[1]


def run(name: str) -> dict:
    x, labels, _, _ = data(name)
    lab = labels[labels >= 0]
    abundance = {int(c): float(np.mean(lab == c)) for c in np.unique(lab)}
    order = sorted(abundance, key=abundance.get)
    rare, large = order[:5], order[-5:]
    graph = build(cp.knn(x, K), "tanimoto")
    out = {"n": int(len(x)), "levels": {}}
    keys = ("ari", "nmi", "clusters", "f1_rarest_5", "f1_largest_5", "f1_mean")
    for threshold in THRESHOLDS:
        one, two = [], []
        for seed in SEEDS:
            m1, m2 = two_level(x, graph, seed, threshold)
            one.append(scores(labels, m1, rare, large))
            two.append(scores(labels, m2, rare, large))
        for tag, runs in (("one_level", one), (f"two_level_{threshold}", two)):
            if tag in out["levels"]:
                continue
            out["levels"][tag] = {k: float(np.mean([r[k] for r in runs])) for k in keys}
            s = out["levels"][tag]
            print(f"  {name:<9} {tag:<15} ARI {s['ari']:.3f}  clusters {s['clusters']:5.1f}  F1 rarest-5 "
                  f"{s['f1_rarest_5']:.3f}  largest-5 {s['f1_largest_5']:.3f}  mean {s['f1_mean']:.3f}", flush=True)
    return out


def main(names: list[str]) -> None:
    out = json.loads(OUT.read_text()) if OUT.exists() else {"datasets": {}}
    for name in names:
        out["datasets"][name] = run(name)
        out.update(environment=environment(), k=K, resolution=RESOLUTION, seeds=list(SEEDS))
        OUT.write_text(json.dumps(out, indent=2))
        print(f"-> {OUT} ({name})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["Levine_32", "Levine_13", "GSE84133", "GSE74672"])
