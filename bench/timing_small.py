"""End-to-end timings on the paper's datasets at equal results: the original code path vs CosTaL+.

Per dataset (paper preset, k = 10), on an otherwise idle machine:

- original: the `bc()` path (bench/stage_a.py `original_graph`: phenograph's brute-force cosine kNN,
  then the original `map_tani` and `build_graph`) + leidenalg at the paper's settings (resolution
  0.8, 10 iterations). The Linux `p_knng` path can't run on macOS; docs/02 §1.1 has its numbers.
- CosTaL+ `leiden="paper"`: exact kNN + Tanimoto + fuzzy union in Rust, then the same leidenalg call.
- CosTaL+ with igraph's native Leiden, 2 iterations, on the same graph (docs/03 §9.1): the
  `leiden="fast"` of the first runs, now `leiden="igraph"` (keys `costal_plus_fast*`).
- `leiden="fast"`, CosTaL's own Leiden (docs/07 §5): the public `find_communities` call timed whole
  (kNN, weights, fuzzy union, Leiden; no igraph graph), key `fast_native`.

Equal results are checked in the same run, not assumed: the share of cells whose neighbor sets are
identical, and the ARI between the original's and CosTaL+'s partitions at the same Leiden seed (1.0 =
identical). Accuracy against the labels: ARI on labeled cells, and for the one-population sets
(Nilsson_rare, Mosmann_rare) the best-match F1 of the rare population.

Preprocessing (scanpy, identical for both) is reported separately. Sets under 100k cells take the best
of 3 runs; larger ones run once (the original kNN alone takes minutes). Levine_32 is deduplicated
(docs/02 addendum).

Usage: python bench/timing_small.py [dataset ...]
       python bench/timing_small.py --fast-only [dataset ...]   (only `fast_native`, merged into the
                                                              existing results)
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

from datasets import NAMES, RARE, load, preprocess  # noqa: E402
from rare_populations import population_f1  # noqa: E402
from stage_a import environment, label_scores, load_levine, original_graph, ours_graph  # noqa: E402

OUT = HERE / "out" / "timing_small.json"
K = 10


def leiden_paper(graph) -> tuple[float, np.ndarray]:
    import leidenalg as la

    t0 = time.perf_counter()
    m = la.find_partition(graph, la.RBConfigurationVertexPartition, weights="weight", n_iterations=10,
                          resolution_parameter=0.8, seed=0).membership
    return time.perf_counter() - t0, np.asarray(m)


def leiden_fast(graph) -> tuple[float, np.ndarray]:
    random.seed(0)
    t0 = time.perf_counter()
    m = graph.community_leiden(objective_function="modularity", weights="weight", resolution=0.8,
                               n_iterations=2).membership
    return time.perf_counter() - t0, np.asarray(m)


def fast_native(x: np.ndarray, labels: np.ndarray, rare: int | None) -> dict:
    """`find_communities(leiden="fast")` end to end, best of 3 under 100k cells."""
    import costal as cp

    best, m = float("inf"), None
    for _ in range(1 if x.shape[0] > 100_000 else 3):
        t0 = time.perf_counter()
        m = cp.find_communities(x, K, leiden="fast", seed=0)
        best = min(best, time.perf_counter() - t0)
    return {"end_to_end_s": best, "accuracy": accuracy(labels, m, rare)}


def matrix(name: str) -> tuple[np.ndarray, np.ndarray, float, int | None]:
    """Preprocessed matrix, labels, preprocessing seconds, and the rare population's label (if any)."""
    if name == "Levine_32":
        x, labels = load_levine(None, dedup=True)
        return np.asarray(x, dtype=np.float32), labels, 0.0, None
    ds = load(name)
    t0 = time.perf_counter()
    x, kept = preprocess(ds)
    rare = ds.classes.index(RARE[name]) if name in RARE else None
    return x, ds.labels[kept], time.perf_counter() - t0, rare


def accuracy(labels: np.ndarray, m: np.ndarray, rare: int | None) -> float:
    if rare is not None:
        return float(population_f1(labels, m, gated_only=True)[rare])
    return float(label_scores(labels, m)["ari"])


def run(name: str) -> dict:
    from sklearn.metrics import adjusted_rand_score

    x, labels, t_pp, rare = matrix(name)
    repeat = 1 if x.shape[0] > 100_000 else 3
    best = {}
    for _ in range(repeat):
        idx_o, _, graph_o, t_o = original_graph(x, K)
        t0 = time.perf_counter()
        g, _, graph, _ = ours_graph(x, K)  # timed with the igraph construction, as the original's build_graph
        t_c = time.perf_counter() - t0
        t_lo, m_o = leiden_paper(graph_o)
        t_lp, m_p = leiden_paper(graph)
        t_lf, m_f = leiden_fast(graph)
        sample = {"original_knn_s": t_o["knn_s"], "original_weights_graph_s": t_o["weights_graph_s"],
                  "original_leiden_paper_s": t_lo, "costal_plus_graph_s": t_c,
                  "costal_plus_leiden_paper_s": t_lp, "costal_plus_leiden_fast_s": t_lf}
        best = {k: min(v, best.get(k, v)) for k, v in sample.items()}
    same = float(np.mean([set(a) == set(b) for a, b in zip(idx_o.tolist(), g.idx.tolist())]))
    agreement = float(adjusted_rand_score(m_o, m_p))
    acc = {"original": accuracy(labels, m_o, rare), "costal_plus_paper": accuracy(labels, m_p, rare),
           "costal_plus_fast": accuracy(labels, m_f, rare)}
    total = {
        "original": best["original_knn_s"] + best["original_weights_graph_s"] + best["original_leiden_paper_s"],
        "costal_plus_paper": best["costal_plus_graph_s"] + best["costal_plus_leiden_paper_s"],
        "costal_plus_fast": best["costal_plus_graph_s"] + best["costal_plus_leiden_fast_s"],
    }
    out = {"cells": int(x.shape[0]), "features": int(x.shape[1]), "repeat": repeat,
           "preprocess_s": t_pp, **best, "end_to_end_s": total,
           "identical_neighbor_sets": same, "partition_ari_original_vs_costal_plus_same_seed": agreement,
           "accuracy": acc, "accuracy_metric": "rare-population F1" if rare is not None else "ARI",
           "speedup_graph": (best["original_knn_s"] + best["original_weights_graph_s"]) / best["costal_plus_graph_s"],
           "speedup_end_to_end_paper": total["original"] / total["costal_plus_paper"],
           "speedup_end_to_end_fast": total["original"] / total["costal_plus_fast"]}
    out["fast_native"] = fast_native(x, labels, rare)
    out["fast_native"]["speedup_end_to_end"] = total["original"] / out["fast_native"]["end_to_end_s"]
    print(f"{name:<12} {x.shape[0]:>7} x {x.shape[1]:<4}  original {total['original']:8.2f} s  "
          f"CosTaL+ paper {total['costal_plus_paper']:7.2f} s ({out['speedup_end_to_end_paper']:5.1f}x)  "
          f"fast {total['costal_plus_fast']:6.2f} s ({out['speedup_end_to_end_fast']:6.1f}x)  "
          f"graph {out['speedup_graph']:6.1f}x | same neighbors {same:.4f}, partition ARI {agreement:.3f} | "
          f"{out['accuracy_metric']} orig {acc['original']:.3f} / paper {acc['costal_plus_paper']:.3f} / "
          f"fast {acc['costal_plus_fast']:.3f}", flush=True)
    return out


def main(names: list[str], fast_only: bool) -> None:
    out = json.loads(OUT.read_text()) if OUT.exists() else {"datasets": {}}
    for name in names:
        if fast_only:
            entry = out["datasets"][name]
            x, labels, _, rare = matrix(name)
            entry["fast_native"] = fast_native(x, labels, rare)
            entry["fast_native"]["speedup_end_to_end"] = (entry["end_to_end_s"]["original"]
                                                          / entry["fast_native"]["end_to_end_s"])
            print(f"{name:<12} fast (native) {entry['fast_native']['end_to_end_s']:7.3f} s "
                  f"({entry['fast_native']['speedup_end_to_end']:6.1f}x)  {entry['accuracy_metric']} "
                  f"{entry['fast_native']['accuracy']:.3f}", flush=True)
        else:
            out["datasets"][name] = run(name)
        out.update(environment=environment(), k=K)
        OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--fast-only"]
    main(args or NAMES[1:] + ["Nilsson_rare", "Samusik_01", "Levine_13", "Levine_32", "Mosmann_rare"],
         fast_only="--fast-only" in sys.argv)
