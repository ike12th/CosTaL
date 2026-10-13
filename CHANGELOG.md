# Changelog

All notable changes to CosTaL. Versions follow [semantic versioning](https://semver.org/) with one
project rule (docs/04 §6.5): **changing what any default does is a major version**, and
`compat="paper"` — the method exactly as published — is never removed.

## 2.0.0 — 2026-10

A rewrite of CosTaL 1.0 (published on PyPI as 0.0.2.4) with the kNN graph in Rust. 1.0 code keeps working: the same names,
arguments and import paths (`find_communities(X, pp_method=…, nbr_num=…)`, `clustering`, `pp`,
`return_scores`, `from CosTaL.clustering import …`).

### The published method, reimplemented

- Exact cosine kNN in Rust: every N² similarity, ranked in fixed-order f64 arithmetic with ties to the
  lower index, so results are bit-identical across platforms, thread counts and matrix-multiplication
  backends (Accelerate on macOS, a pure-Rust gemm elsewhere). No external binary, no temporary files.
- Tanimoto edge weights from the raw row norms and the original's fuzzy-union symmetrization, edge for
  edge identical to the original's scipy pipeline.
- `find_communities()`: the whole method; with no options, the published CosTaL (leidenalg
  RBConfiguration, resolution 0.8, 10 iterations). `leiden="fast"` uses igraph's native Leiden.
- `preprocess()`: the original `pp()` for mass, flow and spectral cytometry and scRNA-seq; reports the
  cells scRNA-seq filtering drops (the original silently misaligned labels when it dropped one).
- Reproduces the paper: on its scRNA-seq datasets the preprocessing matches the paper's Table 2, and
  neighbors and Leiden partitions are identical to the original code's (docs/03 §10).

### Beyond the paper (all opt-in)

- `select="tanimoto"`: choose neighbors by Tanimoto on the raw rows; better on rare populations on
  low-dimensional panels whose populations differ in abundance (docs/04 §5.1).
- `preset="normalize_total"` for scRNA-seq: library-size normalization before `log1p` (docs/04 §6.7).
- `find_communities_multiscale()`: Leiden over resolutions 0.1–3.2, keeping every finer cluster that
  leaks at most 4% of its edge weight into its parent. Recovers rare populations better than any
  single resolution on six of seven benchmark datasets, including the paper's Nilsson_rare and
  Mosmann_rare (docs/05 §2.3).
- `knn()`, `reweight()`, `truncate()`, `build_graph()`: the pipeline in steps, so the kNN is computed
  once for many weights, k and resolutions.

### Installation and input

- `pip install CosTaL` installs everything the package uses (igraph, leidenalg, scanpy, scikit-misc,
  anndata, pandas, scikit-learn, scipy); no extras to choose. On Windows on Arm, the packages without
  wheels there are skipped.
- `read_fcs()`: FCS 2.0 / 3.0 / 3.1 files, read value for value as fcsparser reads them (checked on the
  Levine files and on synthetic float, double and masked-integer files in both byte orders). It
  replaces the fcsparser dependency, whose current release pins numpy below 2.

### Command line

- `costal cluster`: CSV/TSV, FCS, `.npy` or `.h5ad` in, one label per cell out, with the original
  preprocessing, `--multiscale` and `--compat paper`. `costal info` reports the build.

### Speed and memory

- 13–208x faster graph construction than the original's own kNN path on the paper's datasets, at
  identical neighbors; 17–197x end to end with `leiden="fast"` at the same accuracy (docs/03 §10.4).
- Peak memory about twice the input matrix for the kNN; input diagnostics stream in 8 MB chunks.

### Differences from 1.0

- The original's Linux `p_knng` binary assumes non-negative input and returns slightly wrong
  neighbors on arcsinh-transformed cytometry with negative values; 2.0 computes the exact kNN
  the paper describes everywhere.
- The original's macOS/Windows fallback kept duplicate cells in their own neighbor lists and ran
  Leiden for 100 iterations; 2.0 follows the published Linux path on every platform.
- `local=True` (UMAP-style local scaling) is not carried over: the original passed similarities where
  distances were expected, and the corrected version gave no consistent gain (docs/05 §2.2).
- `build_graph()` is the 2.0 function (kNN → weights → fuzzy union from `X`, int32 row and column
  indices); 1.0's `build_graph(rows, columns, weights, shape)` and its other internals (`save_as_ijv`,
  `build_tani_graph`, `bc`, `map_tani`) are gone.
- `find_communities` called the 2.0 way returns a numpy array; called with 1.0's arguments it
  returns a list, as before.
- The evaluation helpers of `CosTaL.descriptions` are kept and match 1.0.
