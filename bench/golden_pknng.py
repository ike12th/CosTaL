"""Run the original `p_knng_Linux` on Linux x86-64 to get the L3 golden output and compare it
with CosTaL+ (docs/04 §4).

It answers the two open items in docs/04 §4:

1. **Index base.** The original passes `-readZidx -writeZidx` (zero-based read and write), yet
   applies `columns - 1` after reading back (comment "bug in L2knng"). Which base does the
   binary actually write?
2. **Self-exclusion with duplicate cells.** The original `bc()`'s `idx[:, 1:]` keeps the cell
   itself when exact duplicates exist (docs/04 §4 item 4). Does `p_knng` have the same problem?

Runs only on Linux x86-64 (the CI ubuntu runner). Usage:

    python bench/golden_pknng.py [N ...]      # default 20000 50000; each N runs deduplicated and raw
    GOLDEN_CLIP=1 python bench/golden_pknng.py 20000   # same, with negative values clipped to 0

Why the clip variant: the pl2knng source (0.2.0, 2021) treats `> 0` as "present" — candidate
accumulators use the sign as a state flag, and the remaining dot product skips features where the
query value is negative — so it is only exact on non-negative data (docs/04 §4).

Output: `bench/out/golden_pknng.json`, plus each run's `.clu` (`bench/out/golden/`).
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

import CosTaL as cp  # noqa: E402

OUT = HERE / "out" / "golden_pknng.json"
CLU_DIR = HERE / "out" / "golden"
PKNNG = REFERENCE / "CosTaL" / "p_knng_Linux"


def run_pknng(x: np.ndarray, tag: str) -> tuple[np.ndarray, np.ndarray, list[int], float]:
    """Follow the original find_communities: write the L2-normalized data to .ijv, call p_knng with
    the same arguments, and read the .clu back.

    Returns (raw_cols, vals, meta, seconds). raw_cols are the raw integers from the .clu, with no −1.
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
        raise RuntimeError(f"p_knng failed ({proc.returncode}): {proc.stderr or proc.stdout}")
    ijv.unlink()  # the .ijv is only input; don't keep it

    # Same parsing as the original build_tani_graph, except no −1
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
    # Pick the interpretation that matches better, then check whether differing rows are exact / near ties
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
    # Are the .clu values cosine, and how far are they from our rescored values (aligned by neighbor)
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
    import os

    clip = os.environ.get("GOLDEN_CLIP") == "1"
    results = []
    for n in sizes:
        for dedup in (True, False):
            x, _ = load_levine(n, dedup=dedup)
            if clip:
                x = np.clip(x, 0, None)
            tag = f"levine_n{n}_{'dedup' if dedup else 'raw'}{'_clip' if clip else ''}"
            raw_cols, vals, meta, seconds = run_pknng(x, tag)
            row = {"n": n, "dedup": dedup, "clip_negatives": clip, "clu_meta": meta,
                   "pknng_s": seconds, **compare(x, raw_cols, vals)}
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = OUT.with_name("golden_pknng_clip.json") if clip else OUT
    out.write_text(json.dumps({
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "python": sys.version.split()[0], "costal_plus": cp.__version__,
                        "cpus": os.cpu_count()},
        "k": K,
        "runs": results,
    }, indent=2, ensure_ascii=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [20_000, 50_000])
