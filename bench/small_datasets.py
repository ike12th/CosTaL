"""The paper's small benchmark datasets: accuracy, the row-norm question, and the scRNA-seq presets.

For each dataset of bench/datasets.py (Levine_13 and four scRNA-seq sets; sizes and labels there):

1. **Parity with the original** (scRNA-seq, paper preset): the original graph-building code (the
   bc() path, bench/stage_a.py `original_graph`) next to CosTaL+'s on the same matrix — neighbor
   sets, edge weights, and Leiden partitions at the paper's settings (resolution 0.8, 10
   iterations) for the same seeds.
2. **The row-norm question of docs/04 §5.1**: Spearman of ‖x‖ with total counts and with genes
   detected (scRNA-seq) or with overall marker abundance (cytometry), and η² of ‖x‖ on the labels.
3. **The 2×2 of docs/04 §5** (select ∈ {cosine, tanimoto} × weight ∈ {cosine, tanimoto}) over a
   resolution sweep, scored by ARI / NMI and per-class best-match F1 (bench/rare_populations.py)
   on the labeled cells.
4. **The presets of docs/04 §6.7** (scRNA-seq): `paper` (log1p of raw counts) and
   `normalize_total` (median library size, then log1p), through the same 2×2 and sweep.

Leiden: leidenalg RBConfiguration. scRNA-seq: 10 iterations (the paper's), 5 seeds. Levine_13
(167,044 cells): 2 iterations (reproduces 10, docs/03 §9.1), 3 seeds.

Usage: python bench/small_datasets.py [dataset ...]   (default: all)
Output: bench/out/small_datasets.json (merged per dataset)
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

from datasets import NAMES, PAPER_TABLE2, describe, load, preprocess  # noqa: E402
from rare_populations import population_f1  # noqa: E402
from stage_a import environment, label_scores, original_graph  # noqa: E402
from stage_c2 import build, eta_squared  # noqa: E402

import costal as cp  # noqa: E402

OUT = HERE / "out" / "small_datasets.json"
K = 10
RESOLUTIONS = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
PAPER_RESOLUTION = 0.8


def leiden(graph, seed: int, resolution: float, n_iterations: int) -> np.ndarray:
    import leidenalg as la

    return np.asarray(la.find_partition(graph, la.RBConfigurationVertexPartition, weights="weight",
                                        n_iterations=n_iterations, resolution_parameter=resolution,
                                        seed=seed).membership)


def settings(kind: str) -> tuple[int, range]:
    return (10, range(5)) if kind == "scrnaseq" else (2, range(3))


def norm_stats(ds, x: np.ndarray, kept: np.ndarray, labels: np.ndarray) -> dict:
    from scipy.stats import spearmanr

    norm = np.linalg.norm(x.astype(np.float64), axis=1)
    lab = labels >= 0
    out = {"row_norm_cv": float(norm.std() / norm.mean()),
           "eta2_of_row_norm_on_labels": eta_squared(norm[lab], labels[lab])}
    if ds.kind == "scrnaseq":
        raw = ds.x[kept]
        out["spearman_row_norm_vs_total_counts"] = float(spearmanr(norm, np.asarray(raw.sum(axis=1)).ravel())[0])
        out["spearman_row_norm_vs_genes_detected"] = float(spearmanr(norm, np.asarray((raw > 0).sum(axis=1)).ravel())[0])
        out["eta2_of_log_total_counts_on_labels"] = eta_squared(
            np.log(np.asarray(raw.sum(axis=1)).ravel()[lab]), labels[lab])
    else:
        out["spearman_row_norm_vs_abundance"] = float(spearmanr(norm, np.clip(x, 0, None).sum(axis=1))[0])
    return out


def parity(x: np.ndarray, labels: np.ndarray, seeds: range) -> dict:
    """The original graph-building code vs CosTaL+ on the same matrix, at the paper's settings."""
    from sklearn.metrics import adjusted_rand_score

    idx_o, w_o, graph_o, t_o = original_graph(x, K)
    g = cp.knn(x, K)
    w = cp.reweight(g, "tanimoto")
    graph = build(g, "tanimoto")
    same = np.mean([set(a) == set(b) for a, b in zip(idx_o.tolist(), g.idx.tolist())])
    # compare weights by neighbor id, since the two may order tied neighbors differently
    w_diff = max(abs(dict(zip(io, wo))[j] - wj) for io, wo, ig, wg in zip(idx_o.tolist(), w_o.tolist(),
                                                                         g.idx.tolist(), w.tolist())
                 if set(io) == set(ig) for j, wj in zip(ig, wg))
    ms_o = [leiden(graph_o, s, PAPER_RESOLUTION, 10) for s in seeds]
    ms = [leiden(graph, s, PAPER_RESOLUTION, 10) for s in seeds]
    return {
        "rows_with_identical_neighbor_sets": float(same),
        "max_abs_weight_diff": float(w_diff),
        "same_seed_partition_ari": [float(adjusted_rand_score(a, b)) for a, b in zip(ms_o, ms)],
        "original_ari": [float(label_scores(labels, m)["ari"]) for m in ms_o],
        "ours_ari": [float(label_scores(labels, m)["ari"]) for m in ms],
    }


def sweep(knn: dict, labels: np.ndarray, n_iterations: int, seeds: range) -> list[dict]:
    rows = []
    classes = np.unique(labels[labels >= 0])
    for select, weight in itertools.product(("cosine", "tanimoto"), ("cosine", "tanimoto")):
        graph = build(knn[select], weight)
        for res in RESOLUTIONS:
            ms = [leiden(graph, s, res, n_iterations) for s in seeds]
            scores = [label_scores(labels, m) for m in ms]
            f1s = [population_f1(labels, m, gated_only=True) for m in ms]
            rows.append({
                "select": select, "weight": weight, "resolution": res,
                **{key: {"mean": float(np.mean([s[key] for s in scores])),
                         "sd": float(np.std([s[key] for s in scores]))}
                   for key in ("ari", "nmi", "clusters")},
                "f1_by_class": {str(c): float(np.mean([f[int(c)] for f in f1s])) for c in classes},
            })
    return rows


def summarize(rows: list[dict], abundance: dict[str, float]) -> list[dict]:
    rare = sorted(abundance, key=abundance.get)[: max(1, len(abundance) // 4)]
    out = []
    for select, weight in itertools.product(("cosine", "tanimoto"), ("cosine", "tanimoto")):
        sub = [r for r in rows if (r["select"], r["weight"]) == (select, weight)]
        best = max(sub, key=lambda r: r["ari"]["mean"])
        at_paper = next(r for r in sub if r["resolution"] == PAPER_RESOLUTION)
        out.append({
            "select": select, "weight": weight,
            "ari_at_0.8": at_paper["ari"], "nmi_at_0.8": at_paper["nmi"],
            "clusters_at_0.8": at_paper["clusters"]["mean"],
            "best_resolution": best["resolution"], "best_ari": best["ari"], "best_nmi": best["nmi"],
            "mean_ari_over_grid": float(np.mean([r["ari"]["mean"] for r in sub])),
            "rarest_quarter_f1_at_0.8": float(np.mean([at_paper["f1_by_class"][c] for c in rare])),
            "rarest_quarter_f1_best_over_grid": float(max(np.mean([r["f1_by_class"][c] for c in rare])
                                                          for r in sub)),
        })
    return out


def run(name: str) -> dict:
    ds = load(name)
    n_iterations, seeds = settings(ds.kind)
    result = {"kind": ds.kind, "classes": ds.classes, "paper_table2": PAPER_TABLE2[name],
              "leiden": {"n_iterations": n_iterations, "seeds": list(seeds)}, "presets": {}}
    for preset in (("paper", "normalize_total") if ds.kind == "scrnaseq" else ("paper",)):
        x, kept = preprocess(ds, preset)
        labels = ds.labels[kept]
        entry = {"data": describe(ds, x, kept), "row_norm": norm_stats(ds, x, kept, labels)}
        lab = labels[labels >= 0]
        abundance = {str(c): float(np.mean(lab == c)) for c in np.unique(lab)}
        entry["abundance"] = abundance
        t0 = time.perf_counter()
        knn = {"cosine": cp.knn(x, K)}
        entry["knn_cosine_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        knn["tanimoto"] = cp.knn(x, K, select="tanimoto")
        entry["knn_tanimoto_s"] = time.perf_counter() - t0
        entry["selection_change_mean_jaccard"] = float(np.mean(
            [len(set(a) & set(b)) / len(set(a) | set(b))
             for a, b in zip(knn["cosine"].idx.tolist(), knn["tanimoto"].idx.tolist())]))
        if preset == "paper" and ds.kind == "scrnaseq":
            entry["parity_with_original"] = parity(x, labels, seeds)
            p = entry["parity_with_original"]
            print(f"{name} parity: identical neighbor sets {p['rows_with_identical_neighbor_sets']:.4f}, "
                  f"max weight diff {p['max_abs_weight_diff']:.2e}, same-seed partition ARI "
                  f"{np.round(p['same_seed_partition_ari'], 3).tolist()}", flush=True)
        print(f"{name} [{preset}] {json.dumps(entry['data'])}\n  row norm {json.dumps(entry['row_norm'])}",
              flush=True)
        rows = sweep(knn, labels, n_iterations, seeds)
        entry["summary"] = summarize(rows, abundance)
        entry["runs"] = rows
        for s in entry["summary"]:
            print(f"  select={s['select']:<8} weight={s['weight']:<8} ARI@0.8 {s['ari_at_0.8']['mean']:.3f}"
                  f"±{s['ari_at_0.8']['sd']:.3f}  best {s['best_ari']['mean']:.3f} @ {s['best_resolution']}"
                  f"  grid mean {s['mean_ari_over_grid']:.3f}  rare-F1@0.8 {s['rarest_quarter_f1_at_0.8']:.3f}",
                  flush=True)
        result["presets"][preset] = entry
    return result


def main(names: list[str]) -> None:
    out = json.loads(OUT.read_text()) if OUT.exists() else {"datasets": {}}
    for name in names:
        out["datasets"][name] = run(name)
        out["environment"] = environment()
        out["k"], out["resolutions"] = K, RESOLUTIONS
        OUT.write_text(json.dumps(out, indent=2))
        print(f"-> {OUT} ({name})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or NAMES[1:] + NAMES[:1])  # the scRNA-seq sets first, the slow Levine_13 last
