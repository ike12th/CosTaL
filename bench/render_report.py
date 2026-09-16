#!/usr/bin/env python3
"""把 bench/out/profile.json 渲染成 markdown 表格，贴进 docs/02。
手抄数字容易抄错，所以表格一律生成。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

LABEL = {
    "S0a load fcs": "S0a 读 FCS 文件",
    "S0b pp (arcsinh/5)": "S0b `pp` 预处理（arcsinh/5）",
    "S1 save_as_ijv": "S1 `save_as_ijv`（L2 归一化 + 写文本）",
    "S2b write .clu (harness)": "（harness 写 .clu，不计入 CosTaL）",
    "S3 build_tani_graph": "S3 `build_tani_graph`（解析 + Tanimoto + igraph）",
    "S4 leiden (n_iter=10)": "S4 Leiden（n_iterations=10）",
}
KNN = {"S2 knn[blas_blocked]": "S2 kNN｜分块 BLAS（精确）",
       "S2 knn[bruteforce]": "S2 kNN｜phenograph 暴力（精确，CosTaL 的 macOS 回退）",
       "S2 knn[hnsw]": "S2 kNN｜hnswlib（近似）"}
PIPELINE = ["S0a load fcs", "S0b pp (arcsinh/5)", "S1 save_as_ijv",
            "S3 build_tani_graph", "S4 leiden (n_iter=10)"]


def render(data: dict) -> str:
    out: list[str] = []
    for r in data["results"]:
        st = r["stages"]
        knn_keys = [k for k in KNN if k in st]
        out.append(f"\n#### N = {r['n_cells']:,}，d = {r['n_dim']}，k = {r['k']}"
                   f"（图：{r['n_edges']:,} 条边，Leiden 出 {r['n_clusters']} 个簇）\n")
        out.append("| 阶段 | 耗时 (s) | 占非 kNN 总时 |")
        out.append("|---|---:|---:|")
        non_knn = sum(st[k] for k in PIPELINE if k in st)
        for k in PIPELINE:
            if k in st:
                out.append(f"| {LABEL[k]} | {st[k]:.2f} | {st[k]/non_knn:.0%} |")
        out.append(f"| **非 kNN 合计** | **{non_knn:.2f}** | 100% |")
        out.append("")
        out.append("| kNN 实现 | 耗时 (s) | 占全流程 |")
        out.append("|---|---:|---:|")
        for k in knn_keys:
            tot = non_knn + st[k]
            out.append(f"| {KNN[k]} | {st[k]:.2f} | {st[k]/tot:.0%} |")
        out.append("")
        d = r["decomposed_S3"]
        out.append("<details><summary>S3 内部拆解</summary>\n")
        out.append("| 子步骤 | 耗时 (s) |")
        out.append("|---|---:|")
        for k, v in d.items():
            out.append(f"| {k.strip().removeprefix('[d] ')} | {v:.3f} |")
        out.append("\n</details>")
    return "\n".join(out)


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "out" / "profile.json")
    data = json.loads(src.read_text())
    e = data["env"]
    print(f"> 环境：{e['platform']} / {e['machine']} / {e['cpu_count']} 线程 · "
          f"Python {e['python']} · numpy {e['numpy']}（BLAS: {e['numpy_blas']}）")
    print(render(data))
