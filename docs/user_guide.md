# CosTaL+ user guide

CosTaL+ clusters single-cell data — mass, flow and spectral cytometry, scRNA-seq — with the CosTaL
method (Li et al., *Briefings in Bioinformatics* 2023): an **exact** cosine kNN graph, Tanimoto edge
weights, and Leiden community detection. It is a rewrite of the original `CosTaL` package with the
kNN graph built in Rust: with no options it gives the same clusters as the published method, tens to
hundreds of times faster, on every platform, with no external binary and no temporary files.

Contents

1. [Installation](#1-installation)
2. [Quick start](#2-quick-start)
3. [Preparing data](#3-preparing-data)
4. [The pipeline step by step](#4-the-pipeline-step-by-step)
5. [Rare populations: `find_communities_multiscale`](#5-rare-populations-find_communities_multiscale)
6. [API reference](#6-api-reference)
7. [Command line](#7-command-line)
8. [Choosing parameters](#8-choosing-parameters)
9. [Reproducibility](#9-reproducibility)
10. [Performance and memory](#10-performance-and-memory)
11. [Migrating from CosTaL 0.0.2](#11-migrating-from-costal-002)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Installation

```bash
pip install "costal-plus[all]"
```

`costal-plus` itself needs only numpy. The optional parts are grouped as extras:

| Extra | Installs | Needed for |
|---|---|---|
| `leiden` | igraph, leidenalg | `find_communities`, `find_communities_multiscale`, `costal-plus cluster` |
| `scrnaseq` | scanpy, scikit-misc | `preprocess(..., "scrnaseq")` |
| `io` | pandas, fcsparser, anndata | reading CSV / FCS / h5ad in `costal-plus cluster` |
| `all` | all of the above | |

Check the installation:

```bash
costal-plus info
```

prints the version, the platform and the matrix-multiplication backend (`accelerate` on macOS,
`gemm` elsewhere).

**Platforms.** Wheels are built for macOS (Apple silicon and Intel), Linux (x86-64 and aarch64, glibc
and musl) and Windows (x64 and arm64). One wheel per platform covers CPython 3.9 and later (abi3).

**From source** (any platform with a Rust toolchain):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
git clone https://github.com/ike12th/costal-plus && cd costal-plus
pip install maturin && pip install ".[all]"
```

> **macOS and conda**: some conda Pythons are x86-64 builds running under Rosetta. They work, but run
> about half as fast as a native arm64 Python; `python -c "import platform; print(platform.machine())"`
> should print `arm64` on Apple silicon.

---

## 2. Quick start

### Cytometry

```python
import fcsparser
import costal_plus

_, df = fcsparser.parse("cells.fcs", reformat_meta=True)
markers = [c for c in df.columns if c not in ("Time", "Event_length", "label")]

X = costal_plus.preprocess(df[markers], "mass")      # arcsinh(x/5); "flow": x/150; "spectral flow": x/6000
labels = costal_plus.find_communities(X)             # the published CosTaL
```

### scRNA-seq

```python
import anndata
import costal_plus

ad = anndata.read_h5ad("counts.h5ad")                # raw UMI counts, cells × genes
X, cells = costal_plus.preprocess(ad.X, "scrnaseq", return_cells=True)
labels = costal_plus.find_communities(X)
ad = ad[cells]                                       # cells with < 200 detected genes were dropped
ad.obs["costal"] = labels.astype(str)
```

### From the command line

```bash
costal-plus cluster cells.fcs --platform mass --exclude Time,Event_length -o labels.csv
costal-plus cluster counts.h5ad --platform scrnaseq --leiden fast
costal-plus cluster cells.fcs --platform flow --multiscale        # keeps rare populations
```

The output has one row per input cell: `cell,cluster`.

### Faster, or finer

```python
labels = costal_plus.find_communities(X, leiden="fast")         # igraph's Leiden: 10-60x faster end to end
labels = costal_plus.find_communities(X, resolution=0.4)        # fewer, larger clusters
labels = costal_plus.find_communities_multiscale(X)             # rare populations kept (section 5)
```

---

## 3. Preparing data

CosTaL clusters a cells × features matrix of non-negative-ish, per-feature-transformed values. Give it
what the original `pp()` produced:

| Data | Call | What it does |
|---|---|---|
| Mass cytometry (CyTOF) | `preprocess(X, "mass")` | `arcsinh(x / 5)` |
| Flow cytometry | `preprocess(X, "flow")` | `arcsinh(x / 150)` |
| Spectral flow | `preprocess(X, "spectral flow")` | `arcsinh(x / 6000)` |
| scRNA-seq (raw counts) | `preprocess(X, "scrnaseq")` | `filter_cells(min_genes=200)` → `filter_genes(min_cells=3)` → 2000 HVGs (`seurat_v3`, on counts) → `log1p` |

- `X` may be a numpy array, a pandas DataFrame (its numeric columns are used) or, for scRNA-seq, a
  scipy sparse matrix. Select the marker columns yourself: leave out time, event length, DNA,
  viability, bead and barcode channels unless you mean to cluster on them.
- **scRNA-seq drops cells.** Pass `return_cells=True` to get the indices of the cells kept and subset
  your labels or AnnData with them; CosTaL+ warns (`CostalWarning`) whenever a cell was dropped.
- `preset="normalize_total"` (scRNA-seq) scales each cell to the median library size before `log1p`.
  The paper's preprocessing has no library-size normalization, so its row norms mostly measure
  sequencing depth; on the paper's four scRNA-seq sets this preset was never worse and less
  sensitive to the resolution (docs/04 §6.7). It is opt-in; the default stays the paper's.
- **Don't** feed PCA scores, z-scored or Pearson-residual data, or unit-normalized rows: Tanimoto
  weights use the row norm as "overall abundance", which those transforms remove or change. CosTaL+
  detects these cases and warns (`CostalWarning`), but clusters anyway.

---

## 4. The pipeline step by step

`find_communities` is four steps. Calling them yourself lets you do the expensive one once:

```python
g = costal_plus.knn(X, k=30)                     # exact kNN: the only expensive step
w = costal_plus.reweight(g, "tanimoto")          # edge weights, ~milliseconds
g10 = costal_plus.truncate(g, 10)                # the exact 10-NN, for free

rows, cols, weights = costal_plus.build_graph(X, k=10)   # kNN → Tanimoto → fuzzy union, in one call

import igraph, leidenalg as la
G = igraph.Graph(n=len(X), edges=np.column_stack([rows, cols]), edge_attrs={"weight": weights})
for resolution in (0.2, 0.4, 0.8, 1.6):
    part = la.find_partition(G, la.RBConfigurationVertexPartition, weights="weight",
                             n_iterations=10, resolution_parameter=resolution, seed=0)
```

- **kNN**: each cell's k most similar other cells, by cosine (`select="cosine"`, the paper) or by
  Tanimoto on the raw rows (`select="tanimoto"`). Exact: all N² similarities are computed.
- **Edge weights**: Tanimoto `T = c‖a‖‖b‖ / (‖a‖² + ‖b‖² − c‖a‖‖b‖)` from the cosine `c` and the raw
  row norms (the paper), or the cosine itself, or Jaccard on shared neighbors.
- **Symmetrization**: the fuzzy union `S + Sᵀ − S∘Sᵀ`, exactly as the original.
- **Leiden**: `leidenalg`'s RBConfiguration partition, resolution 0.8, 10 iterations (the paper), or
  igraph's native implementation (`leiden="fast"`).

---

## 5. Rare populations: `find_communities_multiscale`

No single resolution recovers both rare and large populations: a low resolution merges a rare
population into its neighbor, a high one splits large populations into pieces. Across the paper's
datasets the rare populations come out far better at higher resolutions than the paper's 0.8, and the
large ones far better at lower ones (docs/05 §1.1).

`find_communities_multiscale` builds the same exact graph once, runs Leiden at 11 resolutions
(0.1 → 3.2) × 4 seeds, starts from the coarsest partition and, coarse to fine, splits off every finer
cluster that is cohesive on its own: one whose edges leak at most `phi_max` (4%) of their weight into
the rest of the cluster it came from. A real sub-population inside a large cluster has few edges to
the rest; a fragment cut out of a homogeneous population has many.

```python
labels, info = costal_plus.find_communities_multiscale(X, return_info=True)
print(info["clusters"], len(info["carved"]))
```

On the paper's rare-population benchmarks (docs/05 §2.3; `phi_max` fixed in advance on another
dataset):

| Dataset | Published setting (0.8) | Best single resolution | Supplement's two-level option | `find_communities_multiscale` |
|---|---|---|---|---|
| Nilsson_rare, HSC F1 (358 of 44,140 cells) | 0.17 | 0.48 | 0.50 (122 clusters) | **0.53** (53 clusters) |
| Mosmann_rare, activated-cell F1 (109 of 396,460) | 0.18 | 0.56 | 0.64 (131 clusters) | **0.65** (34 clusters) |

It is **not** the published method (`compat="paper"` refuses it) and it costs one kNN plus 44 Leiden
runs (about two minutes for 160,000 cells). It cannot find populations smaller than `2k` cells (20
by default) or populations that are not cohesive in the kNN graph.

---

## 6. API reference

All functions take a cells × features matrix `X` (numpy array or anything `np.asarray` accepts;
converted to float32 internally) and return numpy arrays. Errors are `ValueError` / `TypeError`;
advice is a `costal_plus.CostalWarning`.

### `preprocess(X, platform, *, preset="paper", n_top_genes=2000, return_cells=False, compat=None)`

The original `pp()` (section 3). Returns a dense float32 matrix, plus the kept cell indices with
`return_cells=True`.

| Parameter | Meaning |
|---|---|
| `platform` | `"mass"`, `"flow"`, `"spectral flow"` or `"scrnaseq"` |
| `preset` | scRNA-seq: `"paper"` (default) or `"normalize_total"` |
| `n_top_genes` | scRNA-seq: number of highly variable genes (2000 in the paper) |
| `return_cells` | also return the indices of the cells kept |
| `compat` | `"paper"` refuses anything but the original preprocessing |

### `find_communities(X, k=10, *, select="cosine", weight="tanimoto", normalize=None, resolution=0.8, leiden="paper", n_iterations=None, seed=None, compat=None, n_threads=None)`

The whole method; returns `(n,)` int64 cluster labels. With no optional arguments: the published
CosTaL.

| Parameter | Meaning |
|---|---|
| `k` | neighbors per cell (10 in the paper) |
| `select` | neighbor selection: `"cosine"` (paper) or `"tanimoto"` |
| `weight` | edge weights: `"tanimoto"` (paper), `"cosine"` or `"jaccard"` |
| `normalize` | `"l2"` or `"none"`; the default follows `select` and needs no change |
| `resolution` | Leiden resolution (0.8 in the paper); higher gives more clusters |
| `leiden` | `"paper"`: leidenalg, 10 iterations; `"fast"`: igraph's native Leiden, 2 iterations |
| `n_iterations` | Leiden iterations; negative runs until stable |
| `seed` | Leiden seed, for repeatable clusters |
| `compat` | `"paper"` raises on any setting that departs from the published method |
| `n_threads` | threads for the kNN; default all cores |

### `find_communities_multiscale(X, k=10, *, phi_max=0.04, select="cosine", weight="tanimoto", normalize=None, resolutions=None, seeds=4, leiden="fast", n_iterations=2, min_size=None, compat=None, n_threads=None, return_info=False)`

Section 5. Returns `(n,)` int64 labels, and with `return_info=True` also a dict: `clusters`,
`carved` (one entry per split-off cluster: resolution, seed index, cells, leak φ), and the settings.

| Parameter | Meaning |
|---|---|
| `phi_max` | the leak threshold (0.04); lower splits off fewer clusters |
| `resolutions` | ascending Leiden resolutions; default 0.1 … 3.2, ratio √2 |
| `seeds` | Leiden seeds per resolution, or how many |
| `leiden`, `n_iterations` | as in `find_communities`; defaults `"fast"`, 2 |
| `min_size` | smallest cluster that can be split off; default `2 * k` |

### `knn(X, k=10, *, select="cosine", normalize=None, backend="exact", compat=None, n_threads=None)`

The exact kNN, as a `KnnGraph`:

| Attribute | Content |
|---|---|
| `idx` | `(n, k)` int32 neighbor indices, each row sorted by similarity (ties by index) |
| `cos` | `(n, k)` float64 cosine similarities |
| `norms` | `(n,)` float64 row norms of the raw input |
| `meta` | parameters, kernel statistics and input diagnostics |
| `n`, `k` | sizes |

### `reweight(knn, weight="tanimoto")`

`(n, k)` float64 edge weights on a `KnnGraph`: `"tanimoto"`, `"cosine"` or `"jaccard"`.

### `truncate(knn, k)`

The first `k` neighbors of each row: the exact kNN for the smaller `k`.

### `build_graph(X, k=10, *, select="cosine", weight="tanimoto", normalize=None, backend="exact", symmetrize="fuzzy_union", compat=None, n_threads=None)`

kNN → weights → fuzzy union. Returns the upper-triangular edge list `(rows, cols, weights)`: int32,
int32, float64, sorted by (row, col), without self-loops or zero-weight edges — the same edges and
weights as the original `build_graph`.

### Diagnostics

`costal-plus info` / `costal_plus.default_matmul()`, `l2_bytes_per_thread()`, `tile_budget_bytes()`
report the build and the cache sizes the kernel tiles for. `costal_plus.__version__` is the version.

---

## 7. Command line

```
costal-plus cluster INPUT [-o OUTPUT] [--platform {none,mass,flow,spectral flow,scrnaseq}]
                    [--preset {paper,normalize_total}] [--columns A,B,...] [--exclude A,B,...]
                    [-k K] [--select {cosine,tanimoto}] [--weight {tanimoto,cosine,jaccard}]
                    [--resolution R] [--leiden {paper,fast}] [--seed N]
                    [--multiscale] [--phi-max F] [--seeds N]
                    [--compat paper] [--threads N] [--json]
costal-plus info
```

- **Input**: `.csv`, `.tsv`, `.txt` (optionally `.gz`; header row, non-numeric columns dropped),
  `.fcs`, `.npy`, `.h5ad` (its `.X`). Cells are rows.
- **Output**: `-o labels.csv` (default `<input>_clusters.csv`) with columns `cell,cluster`, or
  `-o labels.npy`. Cells removed by scRNA-seq preprocessing get cluster `-1`.
- `--platform none` (default) clusters the values as they are; give the platform to apply the
  original preprocessing.
- `--json` also prints a one-line JSON summary (cells, clusters, seconds, output) to stdout; progress
  goes to stderr.
- Exit status 0 on success, 2 for invalid settings.

---

## 8. Choosing parameters

- **Resolution matters most.** On the paper's datasets the best resolution ranged from 0.05 to 1.0,
  and ARI varied from 0.45 to 0.97 with it on Levine_32 (docs/03 §10.3). The paper's 0.8 is a
  reasonable start, not a rule: sweep a few values on one `build_graph` (section 4) — it costs only
  Leiden runs.
- **k**: 10 (the paper). Larger k smooths the graph and merges small populations.
- **`leiden="fast"`** gives the same accuracy against the labels on every dataset we measured, at
  10-60x the end-to-end speed, but not the same individual partition as the paper's setting.
  Use it for exploration and sweeps; use the default to reproduce published results.
- **`select="tanimoto"`** chooses neighbors by Tanimoto on the raw rows instead of cosine. On
  low-dimensional panels where populations differ in overall marker abundance (Levine_13: 13
  markers) it recovered rare populations better at equal ARI; elsewhere it tied (docs/04 §5.1).
- **`weight`**: Tanimoto (the paper) was never worse than cosine weights and better on Levine_13.

---

## 9. Reproducibility

- **The kNN is deterministic and exact.** All N² similarities are computed, the ranking is decided
  in fixed-order f64 arithmetic and ties go to the lower index, so `knn` returns bit-identical
  results across machines, operating systems, thread counts and matrix-multiplication backends.
- **Leiden is randomized.** Pass `seed=` for repeatable clusters. The same seed reproduces the same
  partition on the same graph; different seeds typically agree at ARI 0.85-0.95.
- **Parity with the original.** On the paper's datasets, CosTaL+ and the original code give identical
  neighbor sets and, at the same Leiden seed, identical partitions on most datasets (docs/03 §10).
  Where partitions differ, the graphs differ only in the last bits of the edge weights (float32 vs
  float64 arithmetic), which Leiden can amplify; accuracy is the same.
- **`compat="paper"`** turns any departure from the published method into an error, so a script that
  passes it is guaranteed to run the method as published.

---

## 10. Performance and memory

### Speed at identical results

End to end (kNN + weights + graph + Leiden at the paper's settings) on one Apple M1 Pro, against the
original code's own kNN path (docs/03 §10.4):

| Dataset | Cells × features | Original | CosTaL+ | CosTaL+ `leiden="fast"` | Graph only | Same neighbors | Accuracy (original / CosTaL+ / fast) |
|---|---|---:|---:|---:|---:|---:|---|
| E-MTAB-3321 | 124 × 2,000 | 0.06 s | 0.02 s (3.1×) | 0.004 s (17×) | 13× | 100% | ARI 0.687 / 0.687 / 0.687 |
| GSE81861 | 561 × 2,000 | 1.0 s | 0.11 s (9.1×) | 0.02 s (41×) | 40× | 100% | ARI 1.000 / 1.000 / 1.000 |
| GSE74672 | 2,881 × 2,000 | 26 s | 0.76 s (35×) | 0.13 s (197×) | 205× | 100% | ARI 0.495 / 0.495 / 0.492 |
| GSE84133 | 3,605 × 2,000 | 41 s | 1.0 s (40×) | 0.21 s (195×) | 208× | 100% | ARI 0.845 / 0.845 / 0.851 |
| Nilsson_rare | 44,140 × 14 | 32 s | 12 s (2.7×) | 0.90 s (36×) | 31× | 100% | HSC F1 0.152 / 0.153 / 0.189 |
| Samusik_01 | 86,864 × 39 | 91 s | 32 s (2.9×) | 2.3 s (39×) | 39× | 100% | ARI 0.843 / 0.843 / 0.808 |
| Levine_13 | 167,044 × 13 | 165 s | 69 s (2.4×) | 5.8 s (28×) | 26× | 100% | ARI 0.849 / 0.873 / 0.846 |
| Levine_32 | 161,443 × 32 | 300 s | 69 s (4.3×) | 5.0 s (60×) | 45× | 100% | ARI 0.756 / 0.756 / 0.758 |
| Mosmann_rare | 396,460 × 15 | 1,826 s | 587 s (3.1×) | 61 s (30×) | 73× | 100% | activated F1 0.020 / 0.019 / 0.038 |

- "Graph only" is the kNN + weights + graph, where CosTaL+'s work is; with the paper's Leiden
  settings, leidenalg itself is 80–97% of CosTaL+'s time, which is why the end-to-end gain on the
  large cytometry sets is 2.4–4.3×. `leiden="fast"` removes that bottleneck.
- At the same Leiden seed the partitions are identical on six datasets; on Nilsson_rare, Levine_13
  and Mosmann_rare the graphs have the same edges but the original's edge weights carry float32
  rounding (≤ 6×10⁻⁷; CosTaL+ computes them in float64), and Leiden follows a different path from
  there. Accuracy is the same within Leiden's run-to-run noise.
- The original's own path here is its brute-force kNN (what it runs on macOS); the paper's Linux runs
  used the `p_knng` binary, whose timings are in docs/02.

### Memory

CosTaL+ needs about **twice the input matrix** for the kNN — the input as float32 plus one working
copy — plus 12 bytes per neighbor for the result; graph construction and Leiden add a few hundred
bytes per edge (about k edges per cell), most of it inside igraph. Measured peak memory above the
loaded input (Apple M1 Pro):

| Step | Levine_32 (161,443 × 32, 20 MB) | Mosmann_rare (396,460 × 15, 23 MB) | Dense 100,000 × 2,000 (763 MB) |
|---|---:|---:|---:|
| `preprocess(…, "mass")` | 36 MB | 39 MB | 779 MB (the output itself) |
| `knn` | 81 MB | 124 MB | 840 MB |
| `build_graph` | 118 MB | 252 MB | 863 MB |
| `find_communities(leiden="fast")` | 311 MB | 681 MB | 1,036 MB |
| `find_communities_multiscale` | 369 MB | | |

To keep memory low:

- **Pass float32.** Any other dtype is converted to a float32 copy first.
- **scRNA-seq**: give `preprocess` sparse counts (an h5ad's `.X`); scanpy works on a copy, so dense
  counts cost twice. The 2000-gene result is dense: 8 kB per cell.
- Very large dense matrices (millions of cells × thousands of features) need a sparse backend that
  is not part of 1.0; reduce to highly variable genes first.

---

## 11. Migrating from CosTaL 0.0.2

| CosTaL 0.0.2 | CosTaL+ |
|---|---|
| `from CosTaL import clustering` | `import costal_plus` |
| `CosTaL.pp(X, platform="mass")` | `costal_plus.preprocess(X, "mass")` (`"flow"`, `"spectral flow"`, `"scrnaseq"` likewise) |
| `CosTaL.find_communities(X, pp_method="mass", nbr_num=10, resolution=0.8, seed=s)` | `costal_plus.find_communities(costal_plus.preprocess(X, "mass"), k=10, resolution=0.8, seed=s)` |
| `method="tani"` / `"cosine"` / `"jaccard"` | `weight="tanimoto"` / `"cosine"` / `"jaccard"` |
| `n_threads=` | `n_threads=` |
| `partition_type=la.RBConfigurationVertexPartition` | the default; for other partition types call `build_graph` and leidenalg yourself (section 4) |
| `local=True` | not provided (see below); for rare populations use `find_communities_multiscale` |
| `keep_temp_files`, `temp_name`, `verbose` | not needed: no temporary files, no external binary |
| `CosTaL.clustering(X, ...)` | `costal_plus.find_communities(...)` |
| `CosTaL.return_scores(ytrue, ypred)` and other `descriptions` helpers | not included; use `sklearn.metrics` (ARI, AMI, FMI, V-measure) |
| returns a Python list | returns a numpy int64 array |

What changes, and why:

- **No binary, every platform.** The original ran the `p_knng` executable on Linux and fell back to a
  slower brute-force path elsewhere; CosTaL+ is one compiled library on all platforms.
- **Exact kNN on all data.** The `p_knng` binary assumes non-negative input (its own documentation
  says so); arcsinh-transformed cytometry has negative values (27% of Levine_32's entries, 42% of
  Samusik_01's), where it returned slightly wrong neighbors for about 17% of cells. CosTaL+ computes
  the exact cosine kNN the paper describes. On non-negative data (all scRNA-seq) the two agree
  exactly.
- **macOS / Windows results now match Linux.** The original's fallback path kept a cell in its own
  neighbor list when it had an exact duplicate and ran Leiden for 100 iterations instead of 10;
  CosTaL+ follows the published (Linux) path everywhere.
- **Dropped cells are reported.** The original scRNA-seq `pp()` returned only the matrix, so labels
  silently shifted if `filter_cells` removed a cell; `preprocess` warns and returns the kept indices.
- **`local=True`** passed similarities where distances were expected, inverting UMAP's local scaling.
  Fixed and evaluated, it gave no consistent improvement (docs/05 §2.2), so it was not carried over.

---

## 12. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ImportError: find_communities needs igraph` | `pip install "costal-plus[leiden]"` |
| `ImportError: platform='scrnaseq' needs scanpy` | `pip install "costal-plus[scrnaseq]"` |
| `ValueError: ... seurat_v3 ... skmisc` or a numpy dtype-size error from `skmisc` | scikit-misc missing or built for another numpy; on Python 3.9 use `pip install "numpy<2" scikit-misc` |
| `TypeError: sparse input needs the sparse backend` | `knn` takes dense matrices; for scRNA-seq call `preprocess` first (it returns a dense matrix) |
| `ValueError: row ... contains NaN or inf` | clean or impute the input; CosTaL+ never guesses |
| `CostalWarning: row norms are nearly constant` | the rows are unit-normalized, so Tanimoto weights add nothing; cluster the untransformed data |
| `CostalWarning: columns are centered` | PCA / z-scored input; Tanimoto weights change meaning; cluster the preprocessed, uncentered data |
| `CostalWarning: filter_cells ... dropped N cells` | expected for low-quality cells; use `return_cells=True` to align labels |
| Clusters differ between runs | pass `seed=` |
| Too many / too few clusters | change `resolution` (section 8) |
