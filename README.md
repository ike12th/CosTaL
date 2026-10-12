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

骨架阶段（P0 已完成）。算法尚未实现，但构建链路已经通：
Rust workspace → PyO3 绑定 → maturin wheel → `pip install` → `import costal_plus`。

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
