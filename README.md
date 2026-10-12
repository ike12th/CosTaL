# CosTaL+

CosTaL 的高性能重写：cosine kNN 图 → Tanimoto 边权精修 → Leiden 社区发现，
用 Rust 实现，交付为 Python wheel / CLI / C ABI。

**方法来源**：Li Y, Nguyen J, Anastasiu DC, Arriaga EA.
*CosTaL: an accurate and scalable graph-based clustering algorithm for
high-dimensional single-cell data analysis.* Briefings in Bioinformatics, 2023.
https://doi.org/10.1093/bib/bbad157

## 目标

1. **精度不退**：在原论文的单细胞数据集上复跑，指标持平或更好。
2. **速度提升**：给出可复现的加速比（对比原 Python 实现）。
3. **跨领域可用**：同一套 API 能跑人脸 embedding 聚类（见 `nas-manager` 项目的人脸轨）。
4. **快速部署**：`pip install` 即可用；不写 Python 也能用 CLI。

## 现状

**阶段 A 已完成（v0.1.0）**：精确 kNN + 边权 + 对称化进了 Rust，Leiden 仍用 `leidenalg`
（docs/03 §5）。不传可选参数 = 已发表的 CosTaL。

```python
import costal_plus, igraph, leidenalg as la

rows, cols, w = costal_plus.build_graph(X, k=10)        # 精确 cosine kNN → Tanimoto → 模糊并集
G = igraph.Graph(n=len(X), edges=list(zip(rows.tolist(), cols.tolist())),
                 edge_attrs={"weight": w.tolist()})
part = la.find_partition(G, la.RBConfigurationVertexPartition, weights="weight",
                         n_iterations=10, resolution_parameter=0.8)

# 贵的一步只做一次，边权和 k 随便换（docs/04 §6.1）
g = costal_plus.knn(X, k=30)
w_tani = costal_plus.reweight(g, "tanimoto")
w_jacc = costal_plus.reweight(costal_plus.truncate(g, 10), "jaccard")
```

- **精确**：全部 N² 个相似度都算过；最终名次由 f64 固定顺序重算决定，换平台、线程数、
  matmul 后端，结果逐位相同（docs/06 §1）。
- **速度**：Levine_32dim 全部 265,627 行，精确 kNN 6.5 s（原版路线的精确 kNN 在同一台机器上
  是 1098.7 s，docs/02）。数字与对照见 `bench/stage_a.py` 和 `bench/out/stage_a.json`。

- 技术方案：[docs/01_技术规划.md](docs/01_技术规划.md)
- 原版 profile 的真实数字：[docs/02_原版profile报告.md](docs/02_原版profile报告.md)
- 架构与实施计划：[docs/03_架构与计划.md](docs/03_架构与计划.md)
- 选择度量与剪枝策略：[docs/04_选择度量与剪枝策略.md](docs/04_选择度量与剪枝策略.md)
- 分辨率极限与停止准则（方法学线）：[docs/05_分辨率极限与停止准则.md](docs/05_分辨率极限与停止准则.md)
- 跨平台、确定性与后端选择：[docs/06_跨平台与确定性.md](docs/06_跨平台与确定性.md)

## 开发环境

```bash
# 1. Rust（本仓库用 stable，rust-toolchain.toml 已固定 channel 与 target）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"

# 2. 装进当前 python 环境并自检
pip install maturin
pip install -e . --no-build-isolation
python -c "import costal_plus; print(costal_plus.hello())"
costal-plus info

# 3. 一条龙：fmt + clippy + cargo test + pip install -e . + pytest
scripts/dev.sh

# 4. 出 wheel
scripts/build-wheel.sh                       # 当前平台
scripts/build-wheel.sh x86_64-unknown-linux-gnu --zig   # 交叉
```

> **macOS 的坑**：`conda` 发的 python 常常是 x86_64 版（在 Rosetta 下跑）。
> 装扩展时 maturin 会按解释器的架构去编译，所以需要
> `rustup target add x86_64-apple-darwin`（`rust-toolchain.toml` 里已列上）。
> 但**性能测试必须用原生 arm64 解释器**，Rosetta 下的数字不作数。

## 云端测试（GitHub Actions）

仓库是私有的，Actions 分钟有额度（Linux 1 倍、Windows 2 倍、macOS 10 倍），所以分了档：

| 工作流 | 什么时候跑 | 做什么 |
|---|---|---|
| `CI` | 每次推送（只改文档 / 报告时不跑） | Linux 上 fmt + clippy + `cargo test` + pytest（Python 3.9 / 3.13） |
| `CI`（勾选 all_os） | 手动触发、打 `v*` 标签 | 同上，加 macOS 与 Windows |
| `bench` | 手动触发 | x86-64 Linux 上跑 `bench/stage_a.py`（纯 Rust gemm 后端的 x86 数字） |
| `golden` | 手动触发 | x86-64 Linux 上跑原版 `p_knng_Linux`，拿 L3 黄金输出并与 CosTaL+ 对照 |
| `wheels` | 打 `v*` 标签 | 8 个平台的 wheel |

```bash
gh workflow run golden -f sizes="20000 50000"
```

M1 上的性能数字仍在本机测（`bench/out/stage_a.json`）。

## 复现 profile

```bash
python -m venv .venv-arm64 && . .venv-arm64/bin/activate     # 必须是原生 arm64 解释器
pip install numpy scipy igraph leidenalg hnswlib scikit-learn pandas numba fcsparser phenograph umap-learn
python bench/fetch_reference.py        # 拉原版 CosTaL 源码 + Levine_32dim 数据集
python bench/profile_costal.py         # 分阶段计时，结果写 bench/out/profile.json
python bench/check_tanimoto_semantics.py
```

## 目标机器

| | CPU | 内存 | GPU |
|---|---|---|---|
| Mac（主力开发/算力） | Apple M1 Pro (arm64, 8 核) | 16 GB 统一内存 | 14 核 Metal + AMX |
| NAS（存储，弱算力） | Intel Pentium Gold 8505 (6 线程) | 7 GB | Intel UHD GT1 |

交付 wheel 需覆盖 `macos-arm64` 与 `linux-x86_64`。
