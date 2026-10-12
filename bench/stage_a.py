"""阶段 A 验收（docs/03 §6）：每一项都同时给正确性和速度。

用法（必须用原生 arm64 解释器，Rosetta 下的数字不作数）：
    python bench/stage_a.py sweep     # 瓦片尺寸 × matmul 后端（N=50k）
    python bench/stage_a.py speed     # N=10k/50k/100k/265k 计时；N≤20k 与暴力解逐位比对
    python bench/stage_a.py parity    # 去重数据 N=50k 与原版 bc() 路径对照：邻居集、边权、图、Leiden 划分
    python bench/stage_a.py full      # 去重后全部 161,443 个细胞端到端 + Leiden，对门控标签报 ARI/NMI
    python bench/stage_a.py parity-raw / full-raw   # 同上，但用未去重的原始文件（265,627 行）
    python bench/stage_a.py dup-effect   # 去重前后各跑 5 个 Leiden seed，看重复细胞对 ARI/NMI 的影响

sweep / speed 用未去重的原始文件，规模与 docs/02 对齐。

结果按子命令合并写进 bench/out/stage_a.json。
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
OUT = HERE / "out" / "stage_a.json"
REFERENCE = HERE / "reference" / "CosTaL-0.0.2.4"
K = 10
SEED = 0
SEEDS = range(5)

import costal_plus as cp  # noqa: E402
from costal_plus import _costal_plus as ext  # noqa: E402

NON_MARKER = {
    "Time", "Cell_length", "DNA1", "DNA2", "Viability",
    "file_number", "event_number", "label", "individual",
}


def load_levine(n: int | None, seed: int = SEED, dedup: bool = True):
    """Levine_32dim：arcsinh(x/5) 后的 32 个 marker（float64）与人工门控标签（未门控为 -1）。

    上游文件把 14 个门控群体的 FCS 和 NotDebrisSinglets（父门，含全部细胞）直接拼接，
    于是 104,184 个已门控细胞各出现两次：一份带标签、一份不带，marker 值逐位相同
    （docs/02 附记）。``dedup=True`` 时每个 (file_number, event_number, individual)
    只留一行、标签取带标签那份，得到 161,443 个唯一细胞。"""
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
        order = np.argsort(np.isnan(labels), kind="stable")  # 带标签的排前面，去重时留它
        _, first = np.unique(key[order], axis=0, return_index=True)
        keep = np.sort(order[first])
        x, labels = x[keep], labels[keep]
    labels = np.where(np.isnan(labels), -1, labels).astype(np.int64)
    if n is not None and n < len(x):
        keep = np.sort(np.random.default_rng(seed).choice(len(x), n, replace=False))
        x, labels = x[keep], labels[keep]
    return np.ascontiguousarray(x), labels


def timed(fn, repeat: int = 1):
    """返回 (最好一次的秒数, 最后一次的结果)。"""
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
    ext._knn_exact(x32[:5000], K)  # 预热线程池
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

# docs/02 的原版数字（同一台机器，测试时 swap 已满，绝对值偏慢）
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
    ext._knn_exact(np.ascontiguousarray(full[:5000], dtype=np.float32), K)  # 预热
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


def original_graph(x: np.ndarray, k: int):
    """原版 bc() 的建图部分，原样调用原版函数（CosTaL 在非 Linux 上实际走的路径）。"""
    sys.path.insert(0, str(REFERENCE))
    import importlib  # noqa: PLC0415

    # CosTaL/__init__.py 把同名函数 clustering 导出到包上，会遮住模块，所以按模块名导入
    clustering = importlib.import_module("CosTaL.clustering")
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
    """原版的 Leiden 设置：RBConfiguration、resolution 0.8、n_iterations 10（docs/04 §0.0 第 5 条）。"""
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
    print(f"  N={n} dedup={dedup}: 原版 bc() 建图（phenograph 暴力 kNN）…", flush=True)
    o_idx, o_w, o_graph, o_time = original_graph(x, K)
    print(f"    {o_time}", flush=True)
    g, w, graph, t_ours = ours_graph(x, K)
    print(f"  CosTaL+ 建图 {t_ours['knn_weights_graph_s']:.3f} s", flush=True)

    # 邻居集：逐行比较；不一致的行看是不是近似并列
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
    # 边权：对两边都有的边比较 Tanimoto（原版用 float64 原始数据，我们用 float32 转换后的数据）
    common = same_rows
    o_sorted = np.take_along_axis(o_w, np.argsort(o_idx, axis=1), axis=1)[common]
    c_sorted = np.take_along_axis(w, np.argsort(g.idx, axis=1), axis=1)[common]
    w_rel = float(np.max(np.abs(o_sorted - c_sorted) / np.abs(o_sorted)))

    # 原版 bc() 用 idx[:, 1:] 去自身：有精确重复细胞时，重复者可能排在自身前面，
    # 于是自身留在邻居表里、一个真邻居被丢掉
    self_in = np.array([i in o_idx[i] for i in range(n)])
    _, inv, counts = np.unique(x, axis=0, return_inverse=True, return_counts=True)
    in_dup_group = counts[inv.ravel()] > 1

    # Leiden 自身就有随机性：多跑几个 seed，拿"同一张图换 seed"的一致度当噪声基线
    print(f"  Leiden × {len(SEEDS)} 个 seed × 两张图…", flush=True)
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
    print(f"  建图 {t_ours['knn_weights_graph_s']:.2f} s  {t_ours['stats']}", flush=True)
    m, t_l = leiden(graph)
    payload = {"n": len(x), "k": K, "dedup": dedup, **t_ours, "leiden_s": t_l, "edges": graph.ecount(),
               **label_scores(labels, m)}
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=float))
    save("full" if dedup else "full_raw", payload)


def cmd_dup_effect() -> None:
    """重复细胞对精度评估的影响：去重前后各建一次图，Leiden 跑多个 seed，比对门控标签。"""
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
