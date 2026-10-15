# CosTaL

**Fast, exact graph-based clustering for single-cell data** — mass, flow and spectral cytometry and
scRNA-seq: an exact cosine kNN graph, Tanimoto edge weights, Leiden community detection.

> Li Y, Nguyen J, Anastasiu DC, Arriaga EA. *CosTaL: an accurate and scalable graph-based clustering
> algorithm for high-dimensional single-cell data analysis.* Briefings in Bioinformatics 24(3),
> bbad157 (2023). https://doi.org/10.1093/bib/bbad157

**CosTaL 2.0** rewrites CosTaL 1.0 (published on PyPI as 0.0.2.4) with the kNN graph computed in Rust:

- **The published method's results.** With no options, CosTaL 2.0 is the published CosTaL: on nine
  of the paper's datasets it finds the same neighbors as 1.0 for 100% of cells, and at the same
  Leiden seed the same clusters on six; on the other three, 1.0's float32 rounding of the edge
  weights (≤ 6×10⁻⁷) sends Leiden elsewhere, at the same accuracy.
- **Speed.** Graph construction is 13–208× faster than 1.0's own path. With CosTaL's own Leiden,
  written in Rust (`leiden="fast"`), the whole run is 23–420× faster, at the same accuracy.
- **Every platform.** macOS, Linux and Windows, with no external binary and no temporary files.
- **1.0 code keeps working.** The same names, arguments and import paths.
- **A command line** for clustering a file without writing code.
- **Opt-in extras beyond the paper.** `find_communities_multiscale` recovers rare populations that
  any single resolution loses.

## Install

```bash
pip install --upgrade CosTaL
costal info          # confirms the build
```

> **CosTaL 2 is not on PyPI yet** (the upload is pending). Until it is, `pip install CosTaL`
> installs 1.0; install the [latest release](https://github.com/ike12th/CosTaL/releases/latest)
> from GitHub:
>
> ```bash
> pip install --upgrade CosTaL --find-links https://github.com/ike12th/CosTaL/releases/expanded_assets/v2.1.0
> ```

One command installs everything it uses: igraph and leidenalg (Leiden), scanpy and scikit-misc
(scRNA-seq preprocessing), pandas and anndata (file readers), scikit-learn and scipy; FCS files are
read by CosTaL itself (`costal.read_fcs`).
Python ≥ 3.9 on macOS (Apple silicon, Intel), Linux (x86-64, aarch64) and Windows (x64, arm64);
on Windows on Arm, leidenalg, scanpy, scikit-misc and anndata have no wheels yet, so `leiden="fast"`
and the cytometry workflow work there. Building from source: [docs/development.md](docs/development.md).

## Use

```python
import costal

X = costal.preprocess(raw, "mass")          # "mass", "flow", "spectral flow" or "scrnaseq"
labels = costal.find_communities(X)         # the published CosTaL

labels = costal.find_communities(X, leiden="fast", resolution=0.4)   # faster; fewer, larger clusters
labels = costal.find_communities_multiscale(X)                       # keeps rare populations
labels = costal.leiden(n, rows, cols, weights)                        # Leiden on any edge list
```

```bash
costal cluster cells.fcs --platform mass --exclude Time,Event_length -o labels.csv
costal cluster counts.h5ad --platform scrnaseq --leiden fast
```

For scRNA-seq, `preprocess(counts, "scrnaseq", return_cells=True)` also returns which cells survived
quality filtering, so labels stay aligned. More in the **[user guide](docs/user_guide.md)**:
data preparation, the pipeline step by step, the full API and command line, choosing parameters,
reproducibility, and memory.

## Upgrading from 1.0

1.0 code runs unchanged. 2.0's import name is lowercase, `import costal`, as Python packages
usually are; 1.0's `import CosTaL` still works and gives the very same package:

```python
import CosTaL
labels = CosTaL.find_communities(X, pp_method="mass", nbr_num=10, resolution=0.8)   # a list, as before
scores = CosTaL.return_scores(truth, labels)
```

The [upgrade guide](docs/user_guide.md#11-upgrading-from-costal-10) lists every 1.0 function and
what changed. Two differences change results in rare cases, and both fix bugs in 1.0:

- 1.0's `p_knng` binary assumes non-negative input, so on arcsinh-transformed cytometry it returned
  slightly wrong neighbors. 2.0 computes the exact kNN the paper describes.
- 1.0's macOS/Windows path kept a duplicated cell in its own neighbor list. 2.0 excludes it, on
  every platform.

## Speed at identical results

Measured end to end on one Apple M1 Pro: kNN, weights, graph, and Leiden at the paper's settings.
"1.0" is its own kNN path. Details in [docs/03 §10.4](docs/03_architecture_and_plan.md).

| Dataset | Cells × features | 1.0 | 2.0 | 2.0, `leiden="fast"` | Graph only | Same neighbors |
|---|---|---:|---:|---:|---:|---:|
| GSE84133 (pancreas) | 3,605 × 2,000 | 41 s | 1.0 s (40×) | 0.098 s (420×) | 208× | 100% |
| Samusik_01 | 86,864 × 39 | 91 s | 32 s (2.9×) | 0.89 s (102×) | 39× | 100% |
| Levine_32 | 161,443 × 32 | 300 s | 69 s (4.3×) | 2.3 s (128×) | 45× | 100% |
| Mosmann_rare | 396,460 × 15 | 1,826 s | 587 s (3.1×) | 11.5 s (159×) | 73× | 100% |

With the paper's Leiden settings, leidenalg itself takes 80–97% of 2.0's time; `leiden="fast"`
removes that bottleneck. It is CosTaL's own implementation of the same Leiden algorithm: 2.6–3.3×
faster than igraph's, at the same modularity and accuracy over 20 seeds, and it gives the same
clusters for a seed on every platform and at any thread count
([docs/07](docs/07_leiden_implementations.md)). Accuracy against the labels is the same within
Leiden's run-to-run noise in every case.

## Documentation

- [User guide](docs/user_guide.md): installation, quick start, API, command line, parameters, upgrading.
- [Changelog](CHANGELOG.md).
- [Development](docs/development.md): building, testing, benchmarks, releasing.
- Design notes and every measurement behind this README (written under the working name "CosTaL+" for 2.0;
  "the original" there is 1.0):
  - [01 Technical plan](docs/01_technical_plan.md)
  - [02 Profile of 1.0](docs/02_original_profile_report.md)
  - [03 Architecture and results](docs/03_architecture_and_plan.md)
  - [04 Selection metric and pruning](docs/04_selection_metric_and_pruning.md)
  - [05 Resolution limit](docs/05_resolution_limit_and_stopping.md)
  - [06 Cross-platform determinism](docs/06_cross_platform_and_determinism.md)
  - [07 Leiden implementations](docs/07_leiden_implementations.md): why leidenalg is slow, how igraph's Leiden works, a native Leiden

## Citing

Please cite the CosTaL paper above. [CITATION.cff](CITATION.cff) has the details.

## License

MIT, see [LICENSE](LICENSE).
