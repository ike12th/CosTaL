"""docs/05 §2.3, experiment #2: multi-resolution stability — can stable clusters from several
resolutions beat any single resolution?

The single global resolution trades rare populations against large ones (docs/05 §1.1). On Levine,
taking each population at its own best resolution would give the 5 rarest a mean F1 of 0.50 *and*
the 5 largest 0.99, where any single resolution gets 0.28 / 0.98 or 0.49 / 0.81
(bench/out/rare_populations.json). This tries to get there without labels:

1. **Ensemble.** Leiden (leidenalg RBConfiguration, 2 iterations) at resolutions 0.1–3.2 (ratio √2)
   × 4 seeds, all on the same exact kNN graph. Partitions are cached under ``$COSTAL_DATA/cache``.
2. **Stability of a cluster** C from the partition at resolution index r: the mean, over every other
   partition at resolutions r−1, r, r+1, of the best Jaccard between C and any of its clusters.
3. **Base partition**: the resolution whose seeds agree most (mean pairwise ARI), and at it the
   seed partition closest to the others (the medoid).
4. **Refinement**: going from coarse to fine resolutions, a candidate cluster with stability ≥ τ
   and at least ``MIN_SIZE`` cells is carved out of the current cluster that hosts it, if it is
   nested in that host (≥ 80% of C inside it) and is a proper part of it (C ≤ 90% of the host).

**Second rule (conductance)**, added after the first one failed to transfer: start from the
coarsest resolution (its medoid seed) and, coarse to fine, carve a candidate (≥ ``MIN_SIZE`` cells,
nested and proper as above) out of its current host if less than φ_max of its edge weight connects
to the rest of the host: φ(C | H) = w(C, H∖C) / vol(C). No seed-stability threshold and no base
selection. φ_max = 0.04 was fixed from the gap between sub-populations and fragments on Levine_32
before running the other sets; 0.02–0.05 is reported for sensitivity.

Labels are used only for scoring: ARI / NMI on labeled cells, and the per-population best-match F1
of bench/rare_populations.py, summarized as the 5 rarest and the 5 largest populations. Reported
next to every single resolution of the ensemble and the per-population oracle above, for
τ ∈ {0.6, 0.7, 0.8, 0.85, 0.9, 0.95}.

Usage: python bench/multiscale.py [dataset ...]     (Levine_32, Levine_13, GSE84133, GSE74672)
       MULTISCALE_LEIDEN=igraph python bench/multiscale.py ...   (igraph's native Leiden, 2 iterations)
Output: bench/out/multiscale.json, or multiscale_igraph.json (merged per dataset)
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from datasets import CACHE  # noqa: E402
from local_scaling import data  # noqa: E402
from rare_populations import population_f1  # noqa: E402
from stage_a import environment, label_scores  # noqa: E402
from stage_c2 import build  # noqa: E402

import CosTaL as cp  # noqa: E402

IMPL = os.environ.get("MULTISCALE_LEIDEN", "leidenalg")
assert IMPL in ("leidenalg", "igraph"), IMPL
OUT = HERE / "out" / ("multiscale.json" if IMPL == "leidenalg" else "multiscale_igraph.json")
K = 10
RESOLUTIONS = [round(0.1 * 2 ** (i / 2), 3) for i in range(11)]  # 0.1 … 3.2
SEEDS = range(4)
N_ITERATIONS = 2
TAUS = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
MIN_SIZE = 2 * K
PHI_MAX = 0.04  # fixed on Levine_32 (docs/05 §2.3)
PHIS = [0.02, 0.03, 0.04, 0.05]
NESTED, PROPER = 0.8, 0.9

_GRAPH = None


def _init(n: int, rows: np.ndarray, cols: np.ndarray, weights: np.ndarray) -> None:
    import igraph as ig

    global _GRAPH
    _GRAPH = ig.Graph(n=n, edges=np.column_stack([rows, cols]).tolist(), edge_attrs={"weight": weights.tolist()})


def _leiden(job: tuple[float, int]) -> np.ndarray:
    import leidenalg as la

    res, seed = job
    if IMPL == "igraph":
        random.seed(seed)
        return np.asarray(_GRAPH.community_leiden(objective_function="modularity", weights="weight",
                                                  resolution=res, n_iterations=N_ITERATIONS).membership,
                          dtype=np.int32)
    return np.asarray(la.find_partition(_GRAPH, la.RBConfigurationVertexPartition, weights="weight",
                                        n_iterations=N_ITERATIONS, resolution_parameter=res,
                                        seed=seed).membership, dtype=np.int32)


def ensemble(name: str, x: np.ndarray) -> np.ndarray:
    """(len(RESOLUTIONS), len(SEEDS), n) memberships, cached."""
    path = CACHE / f"multiscale_{name}{'' if IMPL == 'leidenalg' else '_igraph'}.npz"
    if path.exists():
        z = np.load(path)
        if z["resolutions"].tolist() == RESOLUTIONS and z["seeds"].tolist() == list(SEEDS):
            return z["memberships"]
    g = build(cp.knn(x, K), "tanimoto")
    rows, cols = np.array(g.get_edgelist()).T
    jobs = [(r, s) for r in RESOLUTIONS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context("spawn"), initializer=_init,
                             initargs=(g.vcount(), rows, cols, np.asarray(g.es["weight"]))) as pool:
        ms = list(pool.map(_leiden, jobs))
    memberships = np.stack(ms).reshape(len(RESOLUTIONS), len(SEEDS), -1)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, memberships=memberships, resolutions=np.array(RESOLUTIONS), seeds=np.array(list(SEEDS)))
    return memberships


def adjacency(x: np.ndarray):
    """Symmetric weighted adjacency (scipy CSR) of the published graph, and the weighted degrees."""
    import scipy.sparse as sp

    g = build(cp.knn(x, K), "tanimoto")
    e = np.array(g.get_edgelist())
    w = np.asarray(g.es["weight"])
    n = g.vcount()
    a = sp.csr_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))
    return a, np.asarray(a.sum(axis=1)).ravel()


def refine_conductance(ms: np.ndarray, a, deg: np.ndarray, phi_max: float) -> tuple[np.ndarray, int]:
    """The second rule: coarse to fine, carve C out of its host H if w(C, H∖C) / vol(C) ≤ phi_max."""
    R, S, _ = ms.shape
    sizes0 = [len(np.unique(ms[0, s])) for s in range(S)]
    labels = ms[0, int(np.argsort(sizes0)[len(sizes0) // 2])].astype(np.int64).copy()  # median-size seed
    next_label = labels.max() + 1
    carved = 0
    for r in range(1, R):
        cands = []
        for s in range(S):
            m = ms[r, s]
            sizes = np.bincount(m)
            for c in np.flatnonzero(sizes >= MIN_SIZE):
                cands.append((s, c))
        scored = []
        for s, c in cands:
            cells = ms[r, s] == c
            hosts = np.bincount(labels[cells])
            host = int(hosts.argmax())
            in_host = cells & (labels == host)
            host_mask = labels == host
            if hosts[host] < NESTED * cells.sum() or in_host.sum() > PROPER * host_mask.sum():
                continue
            phi = a[in_host][:, host_mask & ~in_host].sum() / deg[in_host].sum()
            if phi <= phi_max:
                scored.append((phi, s, c))
        for phi, s, c in sorted(scored):
            cells = ms[r, s] == c
            hosts = np.bincount(labels[cells])
            host = int(hosts.argmax())
            in_host = cells & (labels == host)
            host_mask = labels == host
            if hosts[host] < NESTED * cells.sum() or in_host.sum() > PROPER * host_mask.sum():
                continue  # the host changed since scoring
            if a[in_host][:, host_mask & ~in_host].sum() / deg[in_host].sum() > phi_max:
                continue
            labels[in_host] = next_label
            next_label += 1
            carved += 1
    return np.unique(labels, return_inverse=True)[1], carved


def best_jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each cluster of partition a, the best Jaccard with any cluster of partition b."""
    na, nb = a.max() + 1, b.max() + 1
    m = np.bincount(a.astype(np.int64) * nb + b, minlength=na * nb).reshape(na, nb).astype(np.float64)
    sa, sb = m.sum(1), m.sum(0)
    jac = m / (sa[:, None] + sb[None, :] - m)
    return jac.max(1)


def stabilities(ms: np.ndarray) -> np.ndarray:
    """(R, S) object array of per-cluster stability vectors."""
    R, S, _ = ms.shape
    out = np.empty((R, S), dtype=object)
    for r in range(R):
        for s in range(S):
            others = [(r2, s2) for r2 in range(max(0, r - 1), min(R, r + 2)) for s2 in range(S)
                      if (r2, s2) != (r, s)]
            out[r, s] = np.mean([best_jaccard(ms[r, s], ms[r2, s2]) for r2, s2 in others], axis=0)
    return out


def base_partition(ms: np.ndarray) -> tuple[int, int, list[float]]:
    from sklearn.metrics import adjusted_rand_score

    R, S, _ = ms.shape
    agreement = []
    for r in range(R):
        aris = [adjusted_rand_score(ms[r, a], ms[r, b]) for a in range(S) for b in range(a + 1, S)]
        agreement.append(float(np.mean(aris)))
    r = int(np.argmax(agreement))
    medoid = max(range(S), key=lambda s: sum(adjusted_rand_score(ms[r, s], ms[r, t]) for t in range(S) if t != s))
    return r, medoid, agreement


def refine(ms: np.ndarray, stab: np.ndarray, r0: int, s0: int, tau: float) -> tuple[np.ndarray, int]:
    labels = ms[r0, s0].astype(np.int64).copy()
    next_label = labels.max() + 1
    carved = 0
    R, S, _ = ms.shape
    for r in range(r0 + 1, R):
        cands = []
        for s in range(S):
            m = ms[r, s]
            sizes = np.bincount(m)
            for c in np.flatnonzero((stab[r, s] >= tau) & (sizes >= MIN_SIZE)):
                cands.append((-stab[r, s][c], s, c))
        for _, s, c in sorted(cands):
            cells = ms[r, s] == c
            hosts = np.bincount(labels[cells])
            host = int(hosts.argmax())
            in_host = cells & (labels == host)
            host_size = int(np.sum(labels == host))
            if hosts[host] < NESTED * cells.sum() or in_host.sum() > PROPER * host_size:
                continue
            labels[in_host] = next_label
            next_label += 1
            carved += 1
    return np.unique(labels, return_inverse=True)[1], carved


def scores(labels: np.ndarray, membership: np.ndarray, rare: list[int], large: list[int]) -> dict:
    f1 = population_f1(labels, membership, gated_only=True)
    s = label_scores(labels, membership)
    return {"ari": float(s["ari"]), "nmi": float(s["nmi"]), "clusters": int(s["clusters"]),
            "f1_rarest_5": float(np.mean([f1[p] for p in rare])),
            "f1_largest_5": float(np.mean([f1[p] for p in large])),
            "f1_mean": float(np.mean(list(f1.values()))),
            "f1_by_population": {str(p): float(v) for p, v in f1.items()}}


def run(name: str) -> dict:
    x, labels, _, _ = data(name)
    ms = ensemble(name, x)
    lab = labels[labels >= 0]
    abundance = {int(c): float(np.mean(lab == c)) for c in np.unique(lab)}
    order = sorted(abundance, key=abundance.get)
    rare, large = order[:5], order[-5:]

    single = []
    for r, res in enumerate(RESOLUTIONS):
        per_seed = [scores(labels, ms[r, s], rare, large) for s in SEEDS]
        single.append({"resolution": res, **{k: float(np.mean([p[k] for p in per_seed]))
                                              for k in ("ari", "nmi", "clusters", "f1_rarest_5",
                                                        "f1_largest_5", "f1_mean")},
                       "f1_by_population": {str(p): float(np.mean([q["f1_by_population"][str(p)] for q in per_seed]))
                                            for p in order}})
        print(f"  {name:<9} single res={res:<5} ARI {single[-1]['ari']:.3f}  clusters {single[-1]['clusters']:5.1f}  "
              f"F1 rarest-5 {single[-1]['f1_rarest_5']:.3f}  largest-5 {single[-1]['f1_largest_5']:.3f}  "
              f"mean {single[-1]['f1_mean']:.3f}", flush=True)
    oracle = {str(p): max(s["f1_by_population"][str(p)] for s in single) for p in order}
    oracle_summary = {"f1_rarest_5": float(np.mean([oracle[str(p)] for p in rare])),
                      "f1_largest_5": float(np.mean([oracle[str(p)] for p in large])),
                      "f1_mean": float(np.mean(list(oracle.values())))}
    print(f"  {name:<9} oracle (each population at its best resolution): {json.dumps(oracle_summary)}", flush=True)

    stab = stabilities(ms)
    r0, s0, agreement = base_partition(ms)
    print(f"  {name:<9} seed agreement by resolution {np.round(agreement, 3).tolist()} -> base res "
          f"{RESOLUTIONS[r0]} (seed {s0})", flush=True)
    refined = []
    for tau in TAUS:
        m, carved = refine(ms, stab, r0, s0, tau)
        sc = scores(labels, m, rare, large)
        refined.append({"tau": tau, "carved": carved, **sc})
        print(f"  {name:<9} multiscale tau={tau}: carved {carved:3d}  ARI {sc['ari']:.3f}  clusters "
              f"{sc['clusters']:3d}  F1 rarest-5 {sc['f1_rarest_5']:.3f}  largest-5 {sc['f1_largest_5']:.3f}  "
              f"mean {sc['f1_mean']:.3f}", flush=True)
    a, deg = adjacency(x)
    conductance = []
    for phi_max in PHIS:
        m, carved = refine_conductance(ms, a, deg, phi_max)
        sc = scores(labels, m, rare, large)
        conductance.append({"phi_max": phi_max, "carved": carved, **sc})
        print(f"  {name:<9} conductance phi_max={phi_max}: carved {carved:3d}  ARI {sc['ari']:.3f}  clusters "
              f"{sc['clusters']:3d}  F1 rarest-5 {sc['f1_rarest_5']:.3f}  largest-5 {sc['f1_largest_5']:.3f}  "
              f"mean {sc['f1_mean']:.3f}", flush=True)
    return {"n": int(len(x)), "abundance": {str(p): abundance[p] for p in order},
            "conductance": conductance,
            "seed_agreement": agreement, "base": {"resolution": RESOLUTIONS[r0], "seed": s0},
            "single_resolution": single, "oracle": {"by_population": oracle, **oracle_summary},
            "multiscale": refined}


def main(names: list[str]) -> None:
    out = json.loads(OUT.read_text()) if OUT.exists() else {"datasets": {}}
    for name in names:
        out["datasets"][name] = run(name)
        out.update(environment=environment(), k=K, resolutions=RESOLUTIONS, seeds=list(SEEDS),
                   leiden=IMPL, n_iterations=N_ITERATIONS, min_size=MIN_SIZE, nested=NESTED, proper=PROPER)
        OUT.write_text(json.dumps(out, indent=2))
        print(f"-> {OUT} ({name})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["Levine_32", "Levine_13", "GSE84133", "GSE74672",
                          "Samusik_01", "Nilsson_rare", "Mosmann_rare"])
