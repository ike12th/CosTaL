#!/usr/bin/env python3
"""Render bench/out/profile.json as markdown tables for docs/02.
Copying numbers by hand invites mistakes, so the tables are always generated."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

LABEL = {
    "S0a load fcs": "S0a Read FCS file",
    "S0b pp (arcsinh/5)": "S0b `pp` preprocessing (arcsinh/5)",
    "S1 save_as_ijv": "S1 `save_as_ijv` (L2 normalization + text write)",
    "S2b write .clu (harness)": "(harness writes .clu, not counted toward CosTaL)",
    "S3 build_tani_graph": "S3 `build_tani_graph` (parse + Tanimoto + igraph)",
    "S4 leiden (n_iter=10)": "S4 Leiden (n_iterations=10)",
}
KNN = {"S2 knn[blas_blocked]": "S2 kNN: blocked BLAS (exact)",
       "S2 knn[bruteforce]": "S2 kNN: phenograph brute force (exact, CosTaL's macOS fallback)",
       "S2 knn[hnsw]": "S2 kNN: hnswlib (approximate)"}
PIPELINE = ["S0a load fcs", "S0b pp (arcsinh/5)", "S1 save_as_ijv",
            "S3 build_tani_graph", "S4 leiden (n_iter=10)"]


def render(data: dict) -> str:
    out: list[str] = []
    for r in data["results"]:
        st = r["stages"]
        knn_keys = [k for k in KNN if k in st]
        out.append(f"\n#### N = {r['n_cells']:,}, d = {r['n_dim']}, k = {r['k']}"
                   f" (graph: {r['n_edges']:,} edges, Leiden yields {r['n_clusters']} clusters)\n")
        out.append("| Stage | Time (s) | Share of non-kNN total |")
        out.append("|---|---:|---:|")
        non_knn = sum(st[k] for k in PIPELINE if k in st)
        for k in PIPELINE:
            if k in st:
                out.append(f"| {LABEL[k]} | {st[k]:.2f} | {st[k]/non_knn:.0%} |")
        out.append(f"| **Non-kNN total** | **{non_knn:.2f}** | 100% |")
        out.append("")
        out.append("| kNN implementation | Time (s) | Share of full pipeline |")
        out.append("|---|---:|---:|")
        for k in knn_keys:
            tot = non_knn + st[k]
            out.append(f"| {KNN[k]} | {st[k]:.2f} | {st[k]/tot:.0%} |")
        out.append("")
        d = r["decomposed_S3"]
        out.append("<details><summary>S3 breakdown</summary>\n")
        out.append("| Sub-step | Time (s) |")
        out.append("|---|---:|")
        for k, v in d.items():
            out.append(f"| {k.strip().removeprefix('[d] ')} | {v:.3f} |")
        out.append("\n</details>")
    return "\n".join(out)


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "out" / "profile.json")
    data = json.loads(src.read_text())
    e = data["env"]
    print(f"> Environment: {e['platform']} / {e['machine']} / {e['cpu_count']} threads · "
          f"Python {e['python']} · numpy {e['numpy']} (BLAS: {e['numpy_blas']})")
    print(render(data))
