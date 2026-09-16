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

规划阶段。技术方案见 [docs/01_技术规划.md](docs/01_技术规划.md)。

## 目标机器

| | CPU | 内存 | GPU |
|---|---|---|---|
| Mac（主力开发/算力） | Apple M1 Pro (arm64, 8 核) | 16 GB 统一内存 | 14 核 Metal + AMX |
| NAS（存储，弱算力） | Intel Pentium Gold 8505 (6 线程) | 7 GB | Intel UHD GT1 |

交付 wheel 需覆盖 `macos-arm64` 与 `linux-x86_64`。
