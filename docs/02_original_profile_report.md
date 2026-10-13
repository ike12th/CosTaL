# Original CosTaL profile report (2026-09-16)

docs/01 §1.1 left one prerequisite action: **first profile the existing Python version, and use numbers to confirm
"there is only one bottleneck: kNN graph construction"; if that is refuted, change the docs**. These are those numbers.

**Bottom line first: that judgment holds only "on the exact kNN route"; on the approximate kNN route it is refuted.**
Once we switch to HNSW (i.e. the route docs/01 §2 plans for large-scale data),
the largest piece becomes **Leiden** (77%), and kNN is left with only 10%.
So the effort cannot go "almost entirely into kNN". docs/01 §1.1 has been rewritten accordingly.

---

## 1. Methodology

- **Data**: Levine_32dim (265,627 cells × 32 surface markers),
  the CyTOF benchmark used in the original paper, taken from
  `lmweber/benchmark-data-Levine-32-dim`; the smaller-scale rows are random subsamples without replacement.
- **Code under test**: the original `CosTaL==0.0.2.4`'s **own functions**
  (`pp` / `save_as_ijv` / `build_tani_graph` / `la_find_partition`),
  not an equivalent rewrite — otherwise we would be measuring the new code.
- **k = 10**, Leiden `n_iterations=10`, `resolution=0.8`, `RBConfigurationVertexPartition`,
  matching the original's defaults.
- **One-command reproduction**:
  ```bash
  python bench/fetch_reference.py
  python bench/profile_costal.py
  python bench/split_matmul_topk.py
  python bench/check_tanimoto_semantics.py
  ```
  The raw data is in `bench/out/profile.json`.

### 1.1 Why the kNN cell gets three numbers

The original's exact kNN is the bundled `p_knng_Linux` — an **ELF x86-64 binary**,
which simply cannot run on this machine (macOS arm64), so the Linux path cannot be measured here.
So the kNN column gives three implementations, bracketing the true cost:

| Implementation | Nature | Notes |
|---|---|---|
| `blas_blocked` | exact | Blocked BLAS matmul + `argpartition`; represents "a well-implemented exact kNN", and is also the floor the Rust version has to beat |
| `bruteforce` | exact | `phenograph.bruteforce_nn`, **the fallback path CosTaL actually takes on non-Linux platforms** |
| `hnsw` | approximate | hnswlib, the route docs/01 §2 plans for large-scale data |

`p_knng` has a pruning upper bound and should be faster than `blas_blocked`, but it is still an exact solution, in the same
"exact" tier by order of magnitude; it will not drop into the `hnsw` tier.

> **Addendum (2026-10-12): `p_knng` source and behavior**
>
> - **The source is readable.** The L2Knng author Anastasiu's [snnlib](https://github.com/davidanastasiu/snnlib)
>   (MIT) contains a C++ implementation: `snnlib/CosKnnIndex.h`, about 1100 lines, with exact/approximate graph construction and pruning.
>   But it is a 2023 rewrite: the whole repository has only 2 commits (the last one, from 2024-04, is "partial"),
>   the README calls itself a work in progress, and the tests cover only the approximate path, with no recall check. `p_knng_Linux`
>   is an earlier C version (symbol prefix `da_csr_*`, compiled with GCC 4.8.5 / RHEL 7, file date 2021-01).
>   **The two are not guaranteed to be bit-identical.** That earlier C source cannot be found on GitHub.
>   **Update (2026-10-13): it is in the author's own archive** — `pl2knng-0.2.0.tgz` (2021,
>   `src/pl2knn.c` and the `da_csr_*` support library), matching the binary's symbols. Reading it
>   explained the one open question about the binary: `pl2knn` uses "value > 0" as a "present"
>   flag, both for candidate accumulators and in the forward dot product, so it assumes
>   non-negative input. With negatives clipped beforehand, the binary matches CosTaL+'s exact kNN
>   on 19,999 of 20,000 Levine rows (the last one is a float32 tie); docs/04 §4 has the details.
> - **The binary is not stripped.** The function names are all there. `strings` alone reads out the modes `pl2knn` / `pl2knn-a` / `pkij`
>   and the parameters `-readZidx -writeZidx -nthreads -permcols`. The original uses `pl2knn` (exact).
> - **Don't decompile it; run it.** What we are missing is "what it actually outputs" (the L3 golden output in docs/04 §4,
>   and the index-origin question recorded there), which decompiling cannot give us. It is an x86-64 Linux program: run it on the NAS
>   (Pentium Gold 8505, x86-64, needs `libgomp`) or on a CI ubuntu runner.
> - **L2Knng is designed for sparse data.** Its pruning is built on an inverted index; Levine_32dim is measured to be 100% dense,
>   so it moves to the sparse backend (docs/06 §5).

> **Addendum (2026-10-13): what the author's archive adds** (NAS, `Edgar_Lab/CosTaL_2026`, read only)
>
> - **The supplement states the assumption.** The paper's LaTeX source (`CosTaL_source/Supplementary.tex`,
>   section "The L2knng and p-L2knng Algorithms") says that "L2knng assumes vectors are non-negative".
>   The golden runs found exactly this (docs/04 §4); Levine's arcsinh data violate it.
> - **1M_neurons timings** (`CosTaL_2023021525_scores.txt`, dataset 25, on the HPC): 3,210–3,612 s
>   per end-to-end run (k = 20 / 25), of which writing the `.ijv` text alone took 436 s
>   (`25_CosTaL_20230121_read_and_write_times.txt`) — the IPC overhead of §3.4 at full scale.
> - **GSE81861 labels** (`48_all_PCA_50_2000.py`): the colour suffix of the column names, which maps
>   one to one onto the 7 cell lines that `bench/datasets.py` uses; the counts were loaded dense,
>   float64. So the 0.6% non-zero difference against Table 2 (docs/03 §10.1) is not the loading.
> - **A multilevel option exists** (supplement, "CosTaL Multilevel Clustering"): re-cluster each
>   population of ≥ 500 cells; two levels gave the best F1 on Mosmann_rare and Nilsson_rare. Its code
>   (`clustering_iterative`, with `threshold`, `damp`, `depth`) was never released; docs/05 §2.3 uses
>   a reconstruction as the baseline for the multi-resolution work.

> **Addendum (2026-10-12): Levine_32dim contains 104,184 pairs of exactly duplicated cells**
>
> The upstream `prepare_data_Levine_32dim.R` concatenates the FCS files of the 14 gated populations with `NotDebrisSinglets`,
> and treats the latter as "unassigned". But `NotDebrisSinglets` is the **parent gate** (all non-debris singlets),
> which itself already contains every gated cell. Both files (`Levine_32dim.fcs`, `_notransform.fcs`) were measured and agree:
>
> - Of the 265,627 rows, there are only **161,443** unique (file_number, event_number, individual);
> - 104,184 keys each appear twice: one copy with a label (rows 0–104,183) and one without (falling within the last 161,443 rows),
>   with all 32 markers bit-identical;
> - So the upstream README's "61% (161,443) unassigned" does not hold: only 57,259 cells truly have no gating label.
>
> Impact:
>
> 1. The timings in this report at N=265,627 remain valid (they measure the 265,627-row scale), but **this dataset actually contains only
>    161,443 cells**.
> 2. Each gated cell's nearest neighbor is its own duplicate (cos = 1), so the density of the gated populations in the kNN graph is doubled.
> 3. The original `bc()` uses `idx[:, 1:]` to drop self. When a duplicate also has distance 0, it may be ranked ahead of self,
>    so self stays in the neighbor list and a true neighbor is dropped (docs/04 §4).
> 4. From now on, every accuracy evaluation deduplicates first (`load_levine(dedup=True)` in `bench/stage_a.py`: keep one row per
>    key, taking the label from the labeled copy). Whether the original paper's results on Levine_32dim were based on this file with duplicates
>    needs to be checked with the authors. **Evidence that they were (2026-10-13)**: the paper reports a Levine_32dim sparsity of
>    26.999%; the share of negative entries is 26.9993% in the raw 265,627-row file and 26.53% after deduplication (docs/04 §4).

> **Noisy environment**: during testing, this M1 Pro (16 GB) had its swap fully used at 14.6 GB.
> Absolute seconds will be on the slow side; this report's conclusions rest on the **ratios between stages**, not on the absolute values.

---

## 2. Per-stage timings

> Environment: macOS-26.5.2-arm64-arm-64bit / arm64 / 8 threads · Python 3.9.6 · numpy 1.26.4 (BLAS: openblas64)

#### N = 10,000, d = 32, k = 10 (graph: 78,521 edges, Leiden yields 13 clusters)

| Stage | Time (s) | Share of non-kNN total |
|---|---:|---:|
| S0a Read FCS file | 0.14 | 5% |
| S0b `pp` preprocessing (arcsinh/5) | 0.01 | 0% |
| S1 `save_as_ijv` (L2 normalization + text write) | 0.55 | 19% |
| S3 `build_tani_graph` (parse + Tanimoto + igraph) | 0.14 | 5% |
| S4 Leiden (n_iterations=10) | 2.10 | 71% |
| **Non-kNN total** | **2.94** | 100% |

| kNN implementation | Time (s) | Share of full pipeline |
|---|---:|---:|
| S2 kNN: blocked BLAS (exact) | 2.15 | 42% |
| S2 kNN: phenograph brute force (exact, CosTaL's macOS fallback) | 22.62 | 89% |
| S2 kNN: hnswlib (approximate) | 0.37 | 11% |

<details><summary>S3 breakdown</summary>

| Sub-step | Time (s) |
|---|---:|
| read+split .clu | 0.009 |
| parse to py lists | 0.018 |
| tanimoto weights | 0.032 |
| scipy sym/prune | 0.017 |
| igraph build | 0.018 |

</details>

#### N = 50,000, d = 32, k = 10 (graph: 400,311 edges, Leiden yields 17 clusters)

| Stage | Time (s) | Share of non-kNN total |
|---|---:|---:|
| S0a Read FCS file | 0.10 | 1% |
| S0b `pp` preprocessing (arcsinh/5) | 0.02 | 0% |
| S1 `save_as_ijv` (L2 normalization + text write) | 1.78 | 9% |
| S3 `build_tani_graph` (parse + Tanimoto + igraph) | 0.99 | 5% |
| S4 Leiden (n_iterations=10) | 16.45 | 85% |
| **Non-kNN total** | **19.33** | 100% |

| kNN implementation | Time (s) | Share of full pipeline |
|---|---:|---:|
| S2 kNN: blocked BLAS (exact) | 40.67 | 68% |
| S2 kNN: phenograph brute force (exact, CosTaL's macOS fallback) | 74.97 | 79% |
| S2 kNN: hnswlib (approximate) | 1.98 | 9% |

<details><summary>S3 breakdown</summary>

| Sub-step | Time (s) |
|---|---:|
| read+split .clu | 0.248 |
| parse to py lists | 0.123 |
| tanimoto weights | 0.213 |
| scipy sym/prune | 0.104 |
| igraph build | 0.115 |

</details>

#### N = 265,627, d = 32, k = 10 (graph: 2,119,067 edges, Leiden yields 17 clusters)

| Stage | Time (s) | Share of non-kNN total |
|---|---:|---:|
| S0a Read FCS file | 0.13 | 0% |
| S0b `pp` preprocessing (arcsinh/5) | 0.23 | 0% |
| S1 `save_as_ijv` (L2 normalization + text write) | 11.43 | 10% |
| S3 `build_tani_graph` (parse + Tanimoto + igraph) | 5.61 | 5% |
| S4 Leiden (n_iterations=10) | 102.68 | 86% |
| **Non-kNN total** | **120.09** | 100% |

| kNN implementation | Time (s) | Share of full pipeline |
|---|---:|---:|
| S2 kNN: blocked BLAS (exact) | 1098.69 | 90% |
| S2 kNN: hnswlib (approximate) | 13.48 | 10% |

<details><summary>S3 breakdown</summary>

| Sub-step | Time (s) |
|---|---:|
| read+split .clu | 0.997 |
| parse to py lists | 0.936 |
| tanimoto weights | 1.164 |
| scipy sym/prune | 0.645 |
| igraph build | 1.784 |

</details>

---

## 3. Four conclusions

### 3.1 "The bottleneck is only in kNN" — **holds conditionally; refuted as a general rule**

Full 265,627 cells:

| Route | kNN | Non-kNN | kNN share | Largest piece |
|---|---:|---:|---:|---|
| Exact kNN (`blas_blocked`) | 1098.7 s | 120.1 s | **90%** | kNN |
| Approximate kNN (`hnsw`) | 13.5 s | 120.1 s | **10%** | **Leiden (77%)** |

- On the **exact** kNN route, the original judgment holds, with a wide margin (90%) — for research reproduction, grinding away at kNN really is the right call.
- On the **approximate** kNN route, the original judgment does not hold: kNN drops to 10%, and **Leiden alone takes 77%**
  (102.7 s / 133.6 s). This is exactly the route docs/01 §2 plans for large-scale data.

→ **docs/01 §1.1 has been rewritten**: the bottleneck is "exact kNN **or** Leiden", depending on which kNN route is taken.

### 3.2 "Leiden is near-linear and usually not the bottleneck" — **refuted**

`leidenalg`'s absolute time grows steeply with scale, and **from N=10,000 on it is already the largest piece of the non-kNN part**:

| N | Leiden | Share of non-kNN | Multiple vs. N=10k | Graph-size multiple |
|---:|---:|---:|---:|---:|
| 10,000 | 2.10 s | 71% | 1.0× | 1.0× |
| 50,000 | 16.45 s | 85% | 7.8× | 5.1× |
| 265,627 | 102.68 s | 86% | 49.0× | 27.0× |

The edge count grows 27×, Leiden time grows 49× — **superlinear**, not "near-linear".
On the full data it costs 7.6 times as much as building the HNSW graph.

→ docs/01 §6 item 3, "Whether to write Leiden ourselves — wait for the profile numbers": **the numbers are in; it is worth writing**.
But note these are numbers for `leidenalg` (C++ core + Python bindings), not a lower bound for the Leiden algorithm itself;
first check whether `n_iterations=10` is necessary (the original hard-codes the default at 10, and `bc()` even sets it to 100).

### 3.3 "top-k will become the new bottleneck" (§1.2) — **holds, and is more extreme than expected**

Splitting exact kNN into its matmul and top-k halves (both in numpy; matmul uses multithreaded BLAS):

| N | matmul | top-k | top-k share |
|---:|---:|---:|---:|
| 10,000 | 0.11 s | 1.01 s | **90%** |
| 25,000 | 0.49 s | 6.26 s | **93%** |
| 50,000 | 3.98 s | 46.56 s | **92%** |
| 100,000 | 26.60 s | 198.72 s | **88%** |

matmul accounts for only 7–12%. **Almost all of exact kNN's cost goes to top-k selection**,
and even `argpartition` (introselect implemented in C) cannot rescue it — because it is single-threaded
and has to make a pass over all N candidates in each row.

→ §1.2's "this part is only worth it if we write it ourselves" is right, and its priority should rank ahead of matmul.
→ Parallel + blocked inline top-k (select while computing, without materializing the `b × N` intermediate matrix) is the first cut.
  It also solves memory along the way: this very profile run pushed the machine into swap because of `argpartition`'s `b × N` int64
  intermediate array, and could only be completed after reducing the block from 1024 to 256.

### 3.4 An overhead the plan did not mention: **text files as IPC**

The original writes the matrix as `.ijv` text, hands it to a `p_knng` subprocess, then reads `.clu` text back:

| N | S1 `save_as_ijv` | .clu parsing in S3 | Total | Share of non-kNN |
|---:|---:|---:|---:|---:|
| 10,000 | 0.55 s | 0.03 s | 0.58 s | 20% |
| 50,000 | 1.78 s | 0.37 s | 2.15 s | 11% |
| 265,627 | 11.43 s | 1.93 s | 13.36 s | 11% |

On the full data this 13.4 s is **11 times** the Tanimoto refinement itself (1.16 s).
It is purely the tax of "two processes talking through text files"; once the Rust version is a single process it **drops straight to zero**,
with no algorithmic work required.

As for the Tanimoto refinement itself: all 2,119,067 edges of the full data take only 1.16 s;
**§1.1's claim that it is `O(edges)` and not a bottleneck holds completely** (1% of non-kNN).

---

## 4. Side check: in which space does the original Tanimoto actually compute?

docs/01 §2.2 wrote "let magnitude carry real information" as a **hypothesis to be tested**. Reading the code and running the numbers shows
it is in fact **already the original's existing behavior**:

- `find_communities` writes the **L2-normalized** X into `.ijv` and hands it to `p_knng` — neighbor **selection** happens in the normalized space;
- `build_tani_graph(clu, X)` receives the **unnormalized** X,
  and `cell_norm = norm(X, axis=1)` takes the **raw magnitude** — edge **weights** are computed in the original space.

Measured on Levine_32dim (20,000 cells × k=10 = 200,000 edges):

| Quantity | Value |
|---|---|
| Row norm ‖x‖ range | 2.949 – 17.233 (max/min = 5.8×) |
| Original T (with magnitude) | 0.354 – 1.000 |
| Degenerate form `c/(2−c)` | 0.676 – 1.000 |
| Max absolute difference between the two | **0.516** |
| Pearson r between the two | 0.780 |
| Same ordering? | **No** |

→ Magnitude **really is doing work**; it is not a hypothesis to be tested.
→ This corroborates docs/01 §7 (from the nas-manager face pipeline): face embeddings are L2-normalized,
  which flattens magnitude, so Tanimoto degenerates into a monotonic transform of cosine and the face pipeline can only serve as a
  **negative control**. The wording of docs/01 §2.2 has been updated accordingly.

---

## 5. Direct implications for the Rust version (change priorities)

1. **P1 · Parallel blocked top-k** — 88–93% of exact kNN's time is spent here.
   Select while computing, without materializing the `b × N` intermediate matrix (which also solves memory).
2. **P1 · Single process, no text IPC** — 13.4 s for free on the full data, zero algorithmic risk.
3. **P2 · Leiden** — the largest piece on the approximate route (77%), and measured to be superlinear.
   First try tuning (`n_iterations`), then decide whether to write our own.
4. **P3 · matmul** — only 7–12% of exact kNN. Hooking up Accelerate/AMX is enough;
   not worth going to Metal for it first.
5. **Keep the Tanimoto refinement as it is** — a 1% share; the `O(edges)` judgment holds.
