# 原版 CosTaL profile 报告（2026-09-16）

docs/01 §1.1 留了一条前置动作：**先 profile 现有 Python 版，用数字确认
"瓶颈只有一个地方：kNN 建图"；推翻了就改文档**。这是那份数字。

**结论先行：那条判断只在"走精确 kNN"时成立，走近似 kNN 时被推翻。**
一旦换成 HNSW（也就是 docs/01 §2 给大规模数据规划的那条路），
最大的一块变成 **Leiden**（占 77%），kNN 只剩 10%。
所以火力不能"几乎全部投在 kNN"。docs/01 §1.1 已按此改写。

---

## 1. 口径

- **数据**：Levine_32dim（265,627 个细胞 × 32 个表面 marker），
  原论文用过的 CyTOF 基准，取自
  `lmweber/benchmark-data-Levine-32-dim`；小规模行是无放回随机子采样。
- **被测代码**：原版 `CosTaL==0.0.2.4` **自己的函数**
  （`pp` / `save_as_ijv` / `build_tani_graph` / `la_find_partition`），
  不是等价重写——否则测的是新代码。
- **k = 10**，Leiden `n_iterations=10`、`resolution=0.8`、`RBConfigurationVertexPartition`，
  与原版默认一致。
- **一键复现**：
  ```bash
  python bench/fetch_reference.py
  python bench/profile_costal.py
  python bench/split_matmul_topk.py
  python bench/check_tanimoto_semantics.py
  ```
  原始数据在 `bench/out/profile.json`。

### 1.1 kNN 那一格为什么给三个数

原版的精确 kNN 是随包分发的 `p_knng_Linux`——**ELF x86-64 二进制**，
本机（macOS arm64）根本跑不起来，Linux 那条路径在这台机器上测不了。
所以 kNN 一栏给三个实现，把真实成本夹在中间：

| 实现 | 性质 | 说明 |
|---|---|---|
| `blas_blocked` | 精确 | 分块 BLAS matmul + `argpartition`，代表"实现得当的精确 kNN"，也是 Rust 版要打败的下限 |
| `bruteforce` | 精确 | `phenograph.bruteforce_nn`，**CosTaL 在非 Linux 平台实际走的回退路径** |
| `hnsw` | 近似 | hnswlib，docs/01 §2 给大规模数据规划的那条路 |

`p_knng` 带剪枝上界，应当比 `blas_blocked` 快，但仍是精确解、量级同属
"精确"一档；不会掉到 `hnsw` 那一档。

> **环境有噪声**：这台 M1 Pro（16 GB）测试期间 swap 已用满 14.6 GB。
> 绝对秒数会偏慢，**阶段之间的比例**是本报告的结论依据，不是绝对值。

---

## 2. 分阶段耗时

> 环境：macOS-26.5.2-arm64-arm-64bit / arm64 / 8 线程 · Python 3.9.6 · numpy 1.26.4（BLAS: openblas64）

#### N = 10,000，d = 32，k = 10（图：78,521 条边，Leiden 出 13 个簇）

| 阶段 | 耗时 (s) | 占非 kNN 总时 |
|---|---:|---:|
| S0a 读 FCS 文件 | 0.14 | 5% |
| S0b `pp` 预处理（arcsinh/5） | 0.01 | 0% |
| S1 `save_as_ijv`（L2 归一化 + 写文本） | 0.55 | 19% |
| S3 `build_tani_graph`（解析 + Tanimoto + igraph） | 0.14 | 5% |
| S4 Leiden（n_iterations=10） | 2.10 | 71% |
| **非 kNN 合计** | **2.94** | 100% |

| kNN 实现 | 耗时 (s) | 占全流程 |
|---|---:|---:|
| S2 kNN｜分块 BLAS（精确） | 2.15 | 42% |
| S2 kNN｜phenograph 暴力（精确，CosTaL 的 macOS 回退） | 22.62 | 89% |
| S2 kNN｜hnswlib（近似） | 0.37 | 11% |

<details><summary>S3 内部拆解</summary>

| 子步骤 | 耗时 (s) |
|---|---:|
| read+split .clu | 0.009 |
| parse to py lists | 0.018 |
| tanimoto weights | 0.032 |
| scipy sym/prune | 0.017 |
| igraph build | 0.018 |

</details>

#### N = 50,000，d = 32，k = 10（图：400,311 条边，Leiden 出 17 个簇）

| 阶段 | 耗时 (s) | 占非 kNN 总时 |
|---|---:|---:|
| S0a 读 FCS 文件 | 0.10 | 1% |
| S0b `pp` 预处理（arcsinh/5） | 0.02 | 0% |
| S1 `save_as_ijv`（L2 归一化 + 写文本） | 1.78 | 9% |
| S3 `build_tani_graph`（解析 + Tanimoto + igraph） | 0.99 | 5% |
| S4 Leiden（n_iterations=10） | 16.45 | 85% |
| **非 kNN 合计** | **19.33** | 100% |

| kNN 实现 | 耗时 (s) | 占全流程 |
|---|---:|---:|
| S2 kNN｜分块 BLAS（精确） | 40.67 | 68% |
| S2 kNN｜phenograph 暴力（精确，CosTaL 的 macOS 回退） | 74.97 | 79% |
| S2 kNN｜hnswlib（近似） | 1.98 | 9% |

<details><summary>S3 内部拆解</summary>

| 子步骤 | 耗时 (s) |
|---|---:|
| read+split .clu | 0.248 |
| parse to py lists | 0.123 |
| tanimoto weights | 0.213 |
| scipy sym/prune | 0.104 |
| igraph build | 0.115 |

</details>

#### N = 265,627，d = 32，k = 10（图：2,119,067 条边，Leiden 出 17 个簇）

| 阶段 | 耗时 (s) | 占非 kNN 总时 |
|---|---:|---:|
| S0a 读 FCS 文件 | 0.13 | 0% |
| S0b `pp` 预处理（arcsinh/5） | 0.23 | 0% |
| S1 `save_as_ijv`（L2 归一化 + 写文本） | 11.43 | 10% |
| S3 `build_tani_graph`（解析 + Tanimoto + igraph） | 5.61 | 5% |
| S4 Leiden（n_iterations=10） | 102.68 | 86% |
| **非 kNN 合计** | **120.09** | 100% |

| kNN 实现 | 耗时 (s) | 占全流程 |
|---|---:|---:|
| S2 kNN｜分块 BLAS（精确） | 1098.69 | 90% |
| S2 kNN｜hnswlib（近似） | 13.48 | 10% |

<details><summary>S3 内部拆解</summary>

| 子步骤 | 耗时 (s) |
|---|---:|
| read+split .clu | 0.997 |
| parse to py lists | 0.936 |
| tanimoto weights | 1.164 |
| scipy sym/prune | 0.645 |
| igraph build | 1.784 |

</details>

---

## 3. 四条结论

### 3.1 「瓶颈只在 kNN」——**有条件成立，作为通则被推翻**

全量 265,627 细胞：

| 路线 | kNN | 非 kNN | kNN 占比 | 最大的一块 |
|---|---:|---:|---:|---|
| 精确 kNN（`blas_blocked`） | 1098.7 s | 120.1 s | **90%** | kNN |
| 近似 kNN（`hnsw`） | 13.5 s | 120.1 s | **10%** | **Leiden（77%）** |

- 走**精确** kNN 时，原判断成立且余量很大（90%）——科研复现场景确实该死磕 kNN。
- 走**近似** kNN 时，原判断不成立：kNN 掉到 10%，**Leiden 一个人占 77%**
  （102.7 s / 133.6 s）。这正是 docs/01 §2 给大规模数据规划的那条路。

→ **docs/01 §1.1 已改写**：瓶颈是"精确 kNN **或** Leiden"，取决于走哪条 kNN 路线。

### 3.2 「Leiden 近线性，通常不是瓶颈」——**推翻**

`leidenalg` 的绝对耗时随规模涨得很凶，且**从 N=10,000 起就已经是非 kNN 部分的最大块**：

| N | Leiden | 占非 kNN | 相对 N=10k 的倍数 | 图规模倍数 |
|---:|---:|---:|---:|---:|
| 10,000 | 2.10 s | 71% | 1.0× | 1.0× |
| 50,000 | 16.45 s | 85% | 7.8× | 5.1× |
| 265,627 | 102.68 s | 86% | 49.0× | 27.0× |

边数涨 27×，Leiden 时间涨 49×——**超线性**，不是"近线性"。
在全量数据上它比 HNSW 建图贵 7.6 倍。

→ docs/01 §6 第 3 条"Leiden 是否自己写——等 profile 数字"：**数字到了，值得写**。
但注意这是 `leidenalg`（C++ 核心 + Python 绑定）的数字，不是 Leiden 算法本身的下限；
先查 `n_iterations=10` 是不是必要（原版默认写死 10，`bc()` 里甚至写了 100）。

### 3.3 「top-k 会成为新瓶颈」（§1.2）——**成立，而且比预想更极端**

把精确 kNN 拆成 matmul 和 top-k 两半（都用 numpy，matmul 走多线程 BLAS）：

| N | matmul | top-k | top-k 占比 |
|---:|---:|---:|---:|
| 10,000 | 0.11 s | 1.01 s | **90%** |
| 25,000 | 0.49 s | 6.26 s | **93%** |
| 50,000 | 3.98 s | 46.56 s | **92%** |
| 100,000 | 26.60 s | 198.72 s | **88%** |

matmul 只占 7–12%。**精确 kNN 的钱几乎全花在 top-k 选择上**，
`argpartition`（C 实现的 introselect）都救不了——因为它是单线程、
并且要对整行 N 个候选走一遍。

→ §1.2 "这部分自己写才有价值"是对的，而且优先级应该排在 matmul 前面。
→ 并行 + 分块内联 top-k（边算边选，不落 `b × N` 的中间矩阵）是第一刀。
  顺带解决内存：本次 profile 就因为 `argpartition` 的 `b × N` int64 中间数组
  把机器压进了 swap，block 从 1024 降到 256 才测得下去。

### 3.4 一块规划里没写的开销：**文本文件当 IPC**

原版把矩阵写成 `.ijv` 文本交给 `p_knng` 子进程，再把 `.clu` 文本读回来：

| N | S1 `save_as_ijv` | S3 里的 .clu 解析 | 合计 | 占非 kNN |
|---:|---:|---:|---:|---:|
| 10,000 | 0.55 s | 0.03 s | 0.58 s | 20% |
| 50,000 | 1.78 s | 0.37 s | 2.15 s | 11% |
| 265,627 | 11.43 s | 1.93 s | 13.36 s | 11% |

全量数据上这 13.4 s 是 Tanimoto 精修本身（1.16 s）的 **11 倍**。
纯粹是"两个进程之间用文本文件说话"的税，Rust 版做成单进程后**直接归零**，
不需要任何算法工作。

至于 Tanimoto 精修本身：全量 2,119,067 条边只花 1.16 s，
**§1.1 说它是 `O(边数)`、不是瓶颈，这条完全成立**（占非 kNN 的 1%）。

---

## 4. 顺带核实：原版 Tanimoto 到底在哪个空间上算

docs/01 §2.2 把"让幅度携带真实信息"写成**待验假设**。读代码 + 跑数字，
它其实**已经是原版的既有行为**：

- `find_communities` 把 **L2 归一化后**的 X 写进 `.ijv` 交给 `p_knng` —— 邻居**选择**在归一化空间；
- `build_tani_graph(clu, X)` 收到的是**未归一化**的 X，
  `cell_norm = norm(X, axis=1)` 取的是**原始幅度** —— 边**权重**在原空间。

在 Levine_32dim 上实测（20,000 细胞 × k=10 = 200,000 条边）：

| 量 | 值 |
|---|---|
| 行范数 ‖x‖ 范围 | 2.949 – 17.233（max/min = 5.8×） |
| 原版 T（带幅度） | 0.354 – 1.000 |
| 退化形式 `c/(2−c)` | 0.676 – 1.000 |
| 两者最大绝对差 | **0.516** |
| 两者 Pearson r | 0.780 |
| 排序是否一致 | **否** |

→ 幅度**确实在干活**，不是待验假设。
→ 这与 docs/01 §7（来自 nas-manager 人脸线）互为印证：人脸 embedding 被 L2
  归一化，幅度被抹平，Tanimoto 退化成 cosine 的单调变换，所以人脸线只能当
  **阴性对照**。docs/01 §2.2 的措辞已按此更新。

---

## 5. 对 Rust 版的直接影响（改动优先级）

1. **P1 · 并行分块 top-k**——精确 kNN 里 88–93% 的时间在这儿。
   边算边选，不落 `b × N` 中间矩阵（同时解决内存）。
2. **P1 · 单进程，去掉文本 IPC**——全量数据上白送 13.4 s，零算法风险。
3. **P2 · Leiden**——近似路线上的最大块（77%），且实测超线性。
   先试调参（`n_iterations`），再决定是否自己写。
4. **P3 · matmul**——只占精确 kNN 的 7–12%。Accelerate/AMX 接上就行，
   不值得为它先上 Metal。
5. **Tanimoto 精修保持现状即可**——占比 1%，`O(边数)` 的判断成立。
