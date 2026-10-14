"""Stage A acceptance (docs/03 §6): every item reports both correctness and speed.

Usage (use a native arm64 interpreter on Apple Silicon; numbers under Rosetta don't count):
    python bench/stage_a.py sweep     # tile size x matmul backend (N=50k)
    python bench/stage_a.py speed     # timings at N=10k/50k/100k/265k; bitwise check vs brute force for N<=20k
    python bench/stage_a.py parity    # deduplicated N=50k vs the original bc() path: neighbors, weights, graph, Leiden
    python bench/stage_a.py full      # all 161,443 deduplicated cells end to end + Leiden; ARI/NMI vs gating labels
    python bench/stage_a.py parity-raw / full-raw   # same, on the raw (not deduplicated) file (265,627 rows)
    python bench/stage_a.py dup-effect   # 5 Leiden seeds with and without deduplication: effect of duplicate cells

sweep / speed use the raw file, so the sizes line up with docs/02.

Results are merged per subcommand into bench/out/stage_a.json.
"""

from __future__ import annotations

import itertools
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# On CI, write to a separate file so results don't mix with the local (M1) numbers in the repo
OUT = Path(os.environ.get("STAGE_A_OUT", HERE / "out" / "stage_a.json")).resolve()
REFERENCE = HERE / "reference" / "CosTaL-0.0.2.4"
K = 10
SEED = 0
SEEDS = range(5)

import costal as cp  # noqa: E402
from costal import _costal as ext  # noqa: E402

NON_MARKER = {
    "Time", "Cell_length", "DNA1", "DNA2", "Viability",
    "file_number", "event_number", "label", "individual",
}


def load_levine(n: int | None, seed: int = SEED, dedup: bool = True):
    """Levine_32dim: the 32 markers after arcsinh(x/5) (float64) and the manual gating labels
    (-1 for ungated cells).

    The upstream file concatenates the FCS files of the 14 gated populations with
    NotDebrisSinglets (the parent gate, which holds every cell), so each of the 104,184 gated
    cells appears twice: once with a label and once without, with bit-identical marker values
    (docs/02 Addendum). With ``dedup=True`` each (file_number, event_number, individual) keeps
    one row, taking the label from the labeled copy, which gives 161,443 unique cells."""
    import fcsparser

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, df = fcsparser.parse(str(HERE / "data" / "Levine_32dim_notransform.fcs"), reformat_meta=True)
    markers = [c for c in df.columns if c not in NON_MARKER]
    assert len(markers) == 32, markers
    x = np.arcsinh(df[markers].to_numpy(dtype=np.float64) / 5)
    labels = df["label"].to_numpy()
    if dedup:
        key = df[["file_number", "event_number", "individual"]].to_numpy()
        order = np.argsort(np.isnan(labels), kind="stable")  # labeled rows first, so deduplication keeps them
        _, first = np.unique(key[order], axis=0, return_index=True)
        keep = np.sort(order[first])
        x, labels = x[keep], labels[keep]
    labels = np.where(np.isnan(labels), -1, labels).astype(np.int64)
    if n is not None and n < len(x):
        keep = np.sort(np.random.default_rng(seed).choice(len(x), n, replace=False))
        x, labels = x[keep], labels[keep]
    return np.ascontiguousarray(x), labels


def timed(fn, repeat: int = 1):
    """Return (best time in seconds, result of the last run)."""
    best, out = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def environment() -> dict:
    import sysconfig

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "costal_plus": cp.__version__,
        "threads": os.cpu_count(),
        "default_matmul": cp.default_matmul(),
        "l2_bytes_per_thread": cp.l2_bytes_per_thread(),
        "tile_budget_bytes": cp.tile_budget_bytes(),
        "veclib_maximum_threads": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "build": sysconfig.get_config_var("SOABI"),
    }


def save(section: str, payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[section] = {"environment": environment(), **payload}
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"→ {OUT.relative_to(HERE.parent)} [{section}]")


# ------------------------------------------------------------------ sweep


def cmd_sweep() -> None:
    x, _ = load_levine(50_000, dedup=False)
    x32 = np.ascontiguousarray(x, dtype=np.float32)
    ext._knn_exact(x32[:5000], K)  # warm up the thread pool
    backends = ["gemm"] + (["accelerate"] if cp.default_matmul() == "accelerate" else [])
    rows = []
    for backend, rb, cb in itertools.product(backends, [16, 32, 64, 128, 256], [256, 1024, 4096]):
        t, (_, stats) = timed(
            lambda: ext._knn_exact(x32, K, row_block=rb, col_block=cb, matmul=backend), repeat=2
        )
        tile_kb = rb * cb * 4 / 1024
        rows.append({"matmul": backend, "row_block": rb, "col_block": cb,
                     "tile_kb": tile_kb, "seconds": t})
        print(f"  {backend:<10} rb={rb:<4} cb={cb:<5} tile={tile_kb:7.0f} KB  {t:7.3f} s", flush=True)
    t, (_, stats) = timed(lambda: ext._knn_exact(x32, K), repeat=2)
    print(f"  default ({stats['matmul']}, rb={stats['row_block']}, cb={stats['col_block']}) {t:.3f} s")
    save("sweep", {"n": len(x), "k": K, "grid": rows,
                   "default": {"seconds": t, **dict(stats)}})


# ------------------------------------------------------------------ speed

# The original's numbers from docs/02 (same machine; swap was full during that run, so absolute values are slow)
DOCS02 = {
    10_000: {"blas_blocked": 2.15, "phenograph_bruteforce": 22.62},
    50_000: {"blas_blocked": 40.67, "phenograph_bruteforce": 74.97},
    100_000: {"matmul_only_lower_bound": 26.60, "matmul_plus_topk": 225.32},
    265_627: {"blas_blocked": 1098.69},
}


def cmd_speed() -> None:
    full, _ = load_levine(None, dedup=False)
    rng = np.random.default_rng(SEED)
    results = []
    ext._knn_exact(np.ascontiguousarray(full[:5000], dtype=np.float32), K)  # warm up
    for n in [10_000, 20_000, 50_000, 100_000, len(full)]:
        keep = np.sort(rng.choice(len(full), n, replace=False)) if n < len(full) else slice(None)
        x32 = np.ascontiguousarray(full[keep], dtype=np.float32)
        repeat = 3 if n <= 50_000 else 1
        t_knn, ((idx, cos, norms), stats) = timed(lambda: ext._knn_exact(x32, K), repeat)
        t_w, w = timed(lambda: ext._reweight(idx, cos, norms, "tanimoto"), repeat)
        t_g, edges = timed(lambda: ext._fuzzy_union(idx, w), repeat)
        row = {"n": n, "knn_s": t_knn, "tanimoto_s": t_w, "graph_s": t_g,
               "total_s": t_knn + t_w + t_g, "edges": int(len(edges[0])),
               "fallback_rows": stats["fallback_rows"],
               "row_block": stats["row_block"], "col_block": stats["col_block"],
               "docs02": DOCS02.get(n)}
        if n <= 20_000:
            t_bf, bf = timed(lambda: ext._knn_bruteforce(x32, K))
            row["bruteforce_s"] = t_bf
            row["bitwise_equal_to_bruteforce"] = bool(
                np.array_equal(bf[0], idx) and np.array_equal(bf[1], cos)
            )
        results.append(row)
        print(f"  N={n:>7}  knn {t_knn:8.3f} s  tanimoto {t_w:6.3f} s  graph {t_g:6.3f} s  "
              f"fallback={stats['fallback_rows']}  "
              + (f"bitwise={row['bitwise_equal_to_bruteforce']}" if "bitwise_equal_to_bruteforce" in row else ""),
              flush=True)
    save("speed", {"k": K, "rows": results})


# ------------------------------------------------------------------ parity


def original_clustering():
    """The original CosTaL clustering module (the 0.0.2.4 source in bench/reference), loaded under its
    own name next to CosTaL 2.0 (bench/original.py)."""
    from original import clustering  # noqa: PLC0415

    return clustering()


def original_graph(x: np.ndarray, k: int):
    """The graph-building part of the original bc(), calling the original functions as is (the path
    CosTaL actually takes on non-Linux platforms)."""
    clustering = original_clustering()
    from phenograph import bruteforce_nn  # noqa: PLC0415

    t0 = time.perf_counter()
    d, idx = bruteforce_nn.knnsearch(x, k + 1, "cosine")
    idx = idx[:, 1:]
    t_knn = time.perf_counter() - t0
    rows = list(itertools.chain.from_iterable(itertools.repeat(i, k) for i in range(len(x))))
    columns = list(itertools.chain.from_iterable(idx))
    t0 = time.perf_counter()
    weights = clustering.map_tani(rows, columns, x)
    graph = clustering.build_graph(rows, columns, weights, (len(x), len(x)))
    t_graph = time.perf_counter() - t0
    return idx, np.asarray(weights).reshape(len(x), k), graph, {"knn_s": t_knn, "weights_graph_s": t_graph}


def leiden(graph, seed: int = SEED):
    """The original Leiden settings: RBConfiguration, resolution 0.8, n_iterations 10 (docs/04 §0.0 item 5)."""
    import leidenalg as la

    t0 = time.perf_counter()
    part = la.find_partition(graph, la.RBConfigurationVertexPartition, weights="weight",
                             n_iterations=10, resolution_parameter=0.8, seed=seed)
    return np.asarray(part.membership), time.perf_counter() - t0


def label_scores(labels: np.ndarray, membership: np.ndarray) -> dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    gated = labels >= 0
    return {"ari": adjusted_rand_score(labels[gated], membership[gated]),
            "nmi": normalized_mutual_info_score(labels[gated], membership[gated]),
            "clusters": int(len(np.unique(membership)))}


def ours_graph(x: np.ndarray, k: int):
    import igraph as ig

    t0 = time.perf_counter()
    g = cp.knn(x, k)
    w = cp.reweight(g, "tanimoto")
    rows, cols, ew = ext._fuzzy_union(g.idx, w)
    t_pipe = time.perf_counter() - t0
    graph = ig.Graph(n=len(x), edges=np.column_stack([rows, cols]).tolist(), edge_attrs={"weight": ew.tolist()})
    return g, w, graph, {"knn_weights_graph_s": t_pipe, "stats": g.meta["stats"]}


def cmd_parity(n: int = 50_000, dedup: bool = True) -> None:
    from sklearn.metrics import adjusted_rand_score

    x, labels = load_levine(n, dedup=dedup)
    print(f"  N={n} dedup={dedup}: original bc() graph (phenograph brute-force kNN)...", flush=True)
    o_idx, o_w, o_graph, o_time = original_graph(x, K)
    print(f"    {o_time}", flush=True)
    g, w, graph, t_ours = ours_graph(x, K)
    print(f"  CosTaL+ graph {t_ours['knn_weights_graph_s']:.3f} s", flush=True)

    # Neighbor sets: compare row by row; for rows that differ, check whether it is a near-tie
    same_rows = np.array([set(a) == set(b) for a, b in zip(o_idx, g.idx)])
    diff_rows = np.flatnonzero(~same_rows)
    x64 = x / np.linalg.norm(x, axis=1, keepdims=True)
    gaps = []
    for i in diff_rows:
        only_o = sorted(set(o_idx[i]) - set(g.idx[i]))
        only_c = sorted(set(g.idx[i]) - set(o_idx[i]))
        c_o = x64[only_o] @ x64[i]
        c_c = x64[only_c] @ x64[i]
        gaps.append(float(np.max(np.abs(np.sort(c_o) - np.sort(c_c)))))
    # Edge weights: compare Tanimoto on edges both sides have (the original uses float64 raw data,
    # we use the float32-converted data)
    common = same_rows
    o_sorted = np.take_along_axis(o_w, np.argsort(o_idx, axis=1), axis=1)[common]
    c_sorted = np.take_along_axis(w, np.argsort(g.idx, axis=1), axis=1)[common]
    w_rel = float(np.max(np.abs(o_sorted - c_sorted) / np.abs(o_sorted)))

    # The original bc() drops self with idx[:, 1:]: with exact duplicate cells the duplicate can sort
    # ahead of the cell itself, which then stays in its own neighbor list while a real neighbor is lost
    self_in = np.array([i in o_idx[i] for i in range(n)])
    _, inv, counts = np.unique(x, axis=0, return_inverse=True, return_counts=True)
    in_dup_group = counts[inv.ravel()] > 1

    # Leiden is itself random: run several seeds and use "same graph, different seed" agreement as the noise baseline
    print(f"  Leiden x {len(SEEDS)} seeds x two graphs...", flush=True)
    runs_o = [leiden(o_graph, s) for s in SEEDS]
    runs_c = [leiden(graph, s) for s in SEEDS]
    m_o, m_c = [m for m, _ in runs_o], [m for m, _ in runs_c]

    def mean_ari(pairs):
        return float(np.mean([adjusted_rand_score(a, b) for a, b in pairs]))

    def summary(ms):
        scores = [label_scores(labels, m) for m in ms]
        return {key: {"mean": float(np.mean([s[key] for s in scores])),
                      "sd": float(np.std([s[key] for s in scores]))}
                for key in ("ari", "nmi", "clusters")}

    payload = {
        "n": n, "k": K, "dedup": dedup, "seeds": list(SEEDS),
        "rows_in_exact_duplicate_groups": int(in_dup_group.sum()),
        "original_rows_containing_self": int(self_in.sum()),
        "neighbour_rows_identical": float(same_rows.mean()),
        "rows_differing": int(len(diff_rows)),
        "rows_differing_because_original_kept_self": int((~same_rows & self_in).sum()),
        "max_cos_gap_in_differing_rows": max(gaps) if gaps else 0.0,
        "tanimoto_max_rel_diff": w_rel,
        "edges": {"original": o_graph.ecount(), "ours": graph.ecount()},
        "partition_ari": {
            "ours_vs_original_same_seed": mean_ari(zip(m_c, m_o)),
            "original_seed_to_seed": mean_ari(itertools.combinations(m_o, 2)),
            "ours_seed_to_seed": mean_ari(itertools.combinations(m_c, 2)),
        },
        "original": {**o_time, "leiden_s": float(np.mean([t for _, t in runs_o])),
                     "labels": summary(m_o)},
        "ours": {**t_ours, "leiden_s": float(np.mean([t for _, t in runs_c])),
                 "labels": summary(m_c)},
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=float))
    save("parity" if dedup else "parity_raw", payload)


def cmd_full(dedup: bool = True) -> None:
    x, labels = load_levine(None, dedup=dedup)
    g, w, graph, t_ours = ours_graph(x, K)
    print(f"  graph {t_ours['knn_weights_graph_s']:.2f} s  {t_ours['stats']}", flush=True)
    m, t_l = leiden(graph)
    payload = {"n": len(x), "k": K, "dedup": dedup, **t_ours, "leiden_s": t_l, "edges": graph.ecount(),
               **label_scores(labels, m)}
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=float))
    save("full" if dedup else "full_raw", payload)


def cmd_dup_effect() -> None:
    """Effect of duplicate cells on accuracy: build the graph with and without deduplication, run
    Leiden with several seeds, and score against the gating labels."""
    payload = {"k": K, "seeds": list(SEEDS)}
    for name, dedup in (("dedup", True), ("raw", False)):
        x, labels = load_levine(None, dedup=dedup)
        _, _, graph, t = ours_graph(x, K)
        runs = [leiden(graph, s) for s in SEEDS]
        scores = [label_scores(labels, m) for m, _ in runs]
        payload[name] = {
            "n": len(x),
            "leiden_s": float(np.mean([t for _, t in runs])),
            **{key: {"mean": float(np.mean([s[key] for s in scores])),
                     "sd": float(np.std([s[key] for s in scores])),
                     "values": [float(s[key]) for s in scores]}
               for key in ("ari", "nmi", "clusters")},
        }
        print(f"  {name}: n={len(x)}  ARI {payload[name]['ari']['mean']:.3f} ± {payload[name]['ari']['sd']:.3f}  "
              f"NMI {payload[name]['nmi']['mean']:.3f} ± {payload[name]['nmi']['sd']:.3f}", flush=True)
    save("duplication_effect", payload)


if __name__ == "__main__":
    {
        "sweep": cmd_sweep,
        "speed": cmd_speed,
        "parity": cmd_parity,
        "parity-raw": lambda: cmd_parity(dedup=False),
        "full": cmd_full,
        "full-raw": lambda: cmd_full(dedup=False),
        "dup-effect": cmd_dup_effect,
    }[sys.argv[1]]()
