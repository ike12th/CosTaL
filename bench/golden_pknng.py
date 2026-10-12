"""在 Linux x86-64 上跑原版 `p_knng_Linux`，拿 L3 黄金输出并与 CosTaL+ 对照（docs/04 §4）。

要回答 docs/04 §4 待核的两件事：

1. **索引起始。** 原版传了 `-readZidx -writeZidx`（按 0 起始读写），读回后却又做了
   `columns - 1`（注释 "bug in L2knng"）。二进制实际按几起始写？
2. **重复细胞上怎么排除自身。** 原版 `bc()` 的 `idx[:, 1:]` 在精确重复细胞上会把自身
   留下（docs/04 §4 第 4 条）。`p_knng` 有没有同样的问题？

只能在 Linux x86-64 上跑（CI 的 ubuntu runner）。用法：

    python bench/golden_pknng.py [N ...]      # 默认 20000 50000；每个 N 跑去重与未去重两份

输出：`bench/out/golden_pknng.json`，以及每次运行的 `.clu`（`bench/out/golden/`）。
"""

from __future__ import annotations

import json
import os
import platform
import stat
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stage_a import K, REFERENCE, load_levine, original_clustering  # noqa: E402

import costal_plus as cp  # noqa: E402

OUT = HERE / "out" / "golden_pknng.json"
CLU_DIR = HERE / "out" / "golden"
PKNNG = REFERENCE / "CosTaL" / "p_knng_Linux"


def run_pknng(x: np.ndarray, tag: str) -> tuple[np.ndarray, np.ndarray, list[int], float]:
    """照原版 find_communities 的写法：L2 归一化后写 .ijv，用同样的参数调 p_knng，读回 .clu。

    返回 (raw_cols, vals, meta, 秒数)。raw_cols 是 .clu 里的原始整数，不做任何 −1。
    """
    from sklearn.preprocessing import normalize

    clustering = original_clustering()
    CLU_DIR.mkdir(parents=True, exist_ok=True)
    ijv = CLU_DIR / f"{tag}.ijv"
    clu = CLU_DIR / f"{tag}.clu"
    clustering.save_as_ijv(str(ijv), normalize(x, norm="l2", axis=1, copy=True))
    PKNNG.chmod(PKNNG.stat().st_mode | stat.S_IXUSR)
    args = [str(PKNNG), "pl2knn", str(ijv), "-k", str(K),
            "-nthreads", str(max(2, os.cpu_count() or 2)), "-readZidx", "-writeZidx", str(clu)]
    t0 = time.perf_counter()
    proc = subprocess.run(args, capture_output=True, text=True)
    seconds = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"p_knng 失败（{proc.returncode}）：{proc.stderr or proc.stdout}")
    ijv.unlink()  # .ijv 只是输入，不留

    # 与原版 build_tani_graph 相同的解析，只是不做 −1
    lines = [line.split() for line in clu.read_text().splitlines()]
    meta = list(map(int, lines[0]))
    flat = [tok for row in lines[1:] for tok in row]
    n, width = meta[0], meta[2] // meta[0]
    raw_cols = np.array(list(map(int, flat[::2])), dtype=np.int64).reshape(n, width)
    vals = np.array(list(map(float, flat[1::2])), dtype=np.float64).reshape(n, width)
    return raw_cols, vals, meta, seconds


def compare(x: np.ndarray, raw_cols: np.ndarray, vals: np.ndarray) -> dict:
    g = cp.knn(x, K)
    n = len(x)
    ours = [set(r) for r in g.idx.tolist()]
    res = {}
    for base in (0, 1):
        cols = raw_cols - base
        same = np.array([set(c) == o for c, o in zip(cols.tolist(), ours)])
        self_in = np.array([i in set(c) for i, c in enumerate(cols.tolist())])
        res[f"as_{base}_based"] = {
            "rows_identical_to_costal_plus": float(same.mean()),
            "rows_containing_self": int(self_in.sum()),
            "min_index": int(cols.min()),
            "max_index": int(cols.max()),
        }
    # 选更像的那种解释，再看不同的行是不是精确 / 近似并列
    base = max((0, 1), key=lambda b: res[f"as_{b}_based"]["rows_identical_to_costal_plus"])
    cols = raw_cols - base
    x64 = x / np.linalg.norm(x, axis=1, keepdims=True)
    gaps = []
    for i in range(n):
        a, b = set(cols[i].tolist()), ours[i]
        if a != b and 0 <= min(a) and max(a) < n:
            only_p = sorted(a - b)
            only_c = sorted(b - a)
            if len(only_p) == len(only_c):
                gaps.append(float(np.max(np.abs(np.sort(x64[only_p] @ x64[i]) - np.sort(x64[only_c] @ x64[i])))))
    # .clu 里的值是不是 cosine，与我们的精排值差多少（按邻居对齐）
    order_p = np.argsort(cols, axis=1)
    order_c = np.argsort(g.idx, axis=1)
    same_rows = np.array([set(c) == o for c, o in zip(cols.tolist(), ours)])
    val_diff = float(np.max(np.abs(
        np.take_along_axis(vals, order_p, 1)[same_rows] - np.take_along_axis(g.cos, order_c, 1)[same_rows]
    ))) if same_rows.any() else None
    sorted_by_value = bool(np.all(np.diff(vals, axis=1) <= 0))
    sorted_by_index = bool(np.all(np.diff(raw_cols, axis=1) > 0))
    return {
        **res,
        "inferred_base": base,
        "rows_differing": int((~same_rows).sum()),
        "max_cos_gap_in_differing_rows": max(gaps) if gaps else 0.0,
        "value_max_abs_diff_vs_costal_plus_cos": val_diff,
        "rows_sorted_by_value_desc": sorted_by_value,
        "rows_sorted_by_index_asc": sorted_by_index,
    }


def main(sizes: list[int]) -> None:
    results = []
    for n in sizes:
        for dedup in (True, False):
            x, _ = load_levine(n, dedup=dedup)
            tag = f"levine_n{n}_{'dedup' if dedup else 'raw'}"
            raw_cols, vals, meta, seconds = run_pknng(x, tag)
            row = {"n": n, "dedup": dedup, "clu_meta": meta, "pknng_s": seconds, **compare(x, raw_cols, vals)}
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "python": sys.version.split()[0], "costal_plus": cp.__version__,
                        "cpus": os.cpu_count()},
        "k": K,
        "runs": results,
    }, indent=2, ensure_ascii=False))
    print(f"→ {OUT}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [20_000, 50_000])
