# CosTaL+

A high-performance rewrite of CosTaL: cosine kNN graph → Tanimoto edge reweighting → Leiden
community detection, implemented in Rust and delivered as a Python wheel / CLI / C ABI.

**Method**: Li Y, Nguyen J, Anastasiu DC, Arriaga EA.
*CosTaL: an accurate and scalable graph-based clustering algorithm for
high-dimensional single-cell data analysis.* Briefings in Bioinformatics, 2023.
https://doi.org/10.1093/bib/bbad157

## Goals

1. **No loss of accuracy**: rerun on the original paper's single-cell data sets with equal or better scores.
2. **Speed**: reproducible speedups against the original Python implementation.
3. **Cross-domain**: the same API clusters face embeddings (the face pipeline of the `nas-manager` project).
4. **Easy deployment**: works after `pip install`; the CLI works without writing Python.

## Status

**Stages A, A+ and C are done (v0.1.0)**: exact kNN (cosine, or Tanimoto with
`select="tanimoto"`) + edge weights + symmetrization are in Rust; Leiden still uses `leidenalg`
(docs/03 §5). No optional arguments = the published CosTaL. Stage C measured that 2 Leiden
iterations reproduce the paper's 10 at a third of the cost, and igraph's native Leiden is 38×
faster (docs/03 §9).

```python
import costal_plus

labels = costal_plus.find_communities(X, k=10)          # the published CosTaL, end to end
labels = costal_plus.find_communities(X, leiden="fast")  # igraph's native Leiden, ~38x faster clustering

# Or step by step, handing the graph to Leiden yourself:
import igraph, leidenalg as la
rows, cols, w = costal_plus.build_graph(X, k=10)        # exact cosine kNN → Tanimoto → fuzzy union
G = igraph.Graph(n=len(X), edges=list(zip(rows.tolist(), cols.tolist())),
                 edge_attrs={"weight": w.tolist()})
part = la.find_partition(G, la.RBConfigurationVertexPartition, weights="weight",
                         n_iterations=10, resolution_parameter=0.8)

# Do the expensive step once, then swap edge weights and k freely (docs/04 §6.1)
g = costal_plus.knn(X, k=30)
w_tani = costal_plus.reweight(g, "tanimoto")
w_jacc = costal_plus.reweight(costal_plus.truncate(g, 10), "jaccard")
```

- **Exact**: all N² similarities are computed; the final ranking comes from fixed-order f64
  recomputation, so results are bit-identical across platforms, thread counts and matmul
  backends (docs/06 §1).
- **Speed**: on all 265,627 rows of Levine_32dim, exact kNN takes 6.5 s (the original route's
  exact kNN took 1098.7 s on the same machine, docs/02). Numbers and comparisons are in
  `bench/stage_a.py` and `bench/out/stage_a.json`.

Design documents:

- Technical plan: [docs/01_technical_plan.md](docs/01_technical_plan.md)
- Measured profile of the original: [docs/02_original_profile_report.md](docs/02_original_profile_report.md)
- Architecture and implementation plan: [docs/03_architecture_and_plan.md](docs/03_architecture_and_plan.md)
- Selection metric and pruning: [docs/04_selection_metric_and_pruning.md](docs/04_selection_metric_and_pruning.md)
- Resolution limit and stopping criteria (methodology track): [docs/05_resolution_limit_and_stopping.md](docs/05_resolution_limit_and_stopping.md)
- Cross-platform, determinism and backend choice: [docs/06_cross_platform_and_determinism.md](docs/06_cross_platform_and_determinism.md)
- Research report on data transformations in single-cell and spatial omics: [reports/single_cell_and_spatial_transforms.md](reports/single_cell_and_spatial_transforms.md)

## Development

```bash
# 1. Rust (this repo uses stable; rust-toolchain.toml pins the channel and targets)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"

# 2. Install into the current python environment and self-check
pip install maturin
pip install -e . --no-build-isolation
python -c "import costal_plus; print(costal_plus.hello())"
costal-plus info

# 3. All in one: fmt + clippy + cargo test + pip install -e . + pytest
scripts/dev.sh

# 4. Build wheels
scripts/build-wheel.sh                       # current platform
scripts/build-wheel.sh x86_64-unknown-linux-gnu --zig   # cross-compile
```

> **macOS pitfall**: the python shipped by `conda` is often an x86_64 build (running under
> Rosetta). maturin compiles for the interpreter's architecture, so you need
> `rustup target add x86_64-apple-darwin` (already listed in `rust-toolchain.toml`).
> But **performance measurements must use a native arm64 interpreter**; numbers under Rosetta
> don't count.

## Cloud testing (GitHub Actions)

The repository is private and Actions minutes are budgeted (Linux 1x, Windows 2x, macOS 10x),
so the workflows are tiered:

| Workflow | When it runs | What it does |
|---|---|---|
| `CI` | every push (not when only docs / reports change) | fmt + clippy + `cargo test` + pytest on Linux (Python 3.9 / 3.13) |
| `CI` (with all_os) | manual dispatch, `v*` tags | the same, plus macOS and Windows |
| `bench` | manual dispatch | `bench/stage_a.py` on x86-64 Linux (x86 numbers for the pure-Rust gemm backend) |
| `golden` | manual dispatch | runs the original `p_knng_Linux` on x86-64 Linux for the L3 golden output and compares it with CosTaL+ |
| `wheels` | `v*` tags | wheels for 8 platforms |

```bash
gh workflow run golden -f sizes="20000 50000"
```

M1 performance numbers are still measured locally (`bench/out/stage_a.json`).

## Reproducing the profile

```bash
python -m venv .venv-arm64 && . .venv-arm64/bin/activate     # must be a native arm64 interpreter
pip install numpy scipy igraph leidenalg hnswlib scikit-learn pandas numba fcsparser phenograph umap-learn
python bench/fetch_reference.py        # fetch the original CosTaL source + the Levine_32dim data set
python bench/profile_costal.py         # stage-by-stage timings, written to bench/out/profile.json
python bench/check_tanimoto_semantics.py
```

## Target machines

| | CPU | Memory | GPU |
|---|---|---|---|
| Mac (main development / compute) | Apple M1 Pro (arm64, 8 cores) | 16 GB unified memory | 14-core Metal + AMX |
| NAS (storage, little compute) | Intel Pentium Gold 8505 (6 threads) | 7 GB | Intel UHD GT1 |

Delivered wheels must cover `macos-arm64` and `linux-x86_64`.
