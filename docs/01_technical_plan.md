# CosTaL+ Technical Plan (v0, 2026-09-16)

> This document is a **plan**, not an implementation record. Every judgment states its basis; anything refuted by
> measurement must be corrected as it is, and claims already disproven must not be kept.

## 1. Three judgments that decide success or failure

### 1.1 There are two bottlenecks, depending on which kNN route is taken

> **Rewritten per measurements (2026-09-16).** The original text said "there is only one bottleneck: kNN graph
> construction, so nearly all of the rewrite's firepower goes into kNN". The profile numbers (Levine_32dim, all 265,627 cells,
> see [docs/02](02_original_profile_report.md)) refuted half of that:

| Route | kNN | Non-kNN | kNN share | Largest piece |
|---|---:|---:|---:|---|
| Exact kNN | 1098.7 s | 120.1 s | 90% | kNN |
| Approximate kNN (HNSW) | 13.5 s | 120.1 s | 10% | **Leiden (77%)** |

Measured conclusions for the three steps of CosTaL:

- **Tanimoto refinement** is indeed `O(edges)`: the 2.11 million edges of the full dataset take only 1.16 s, 1% of the non-kNN time.
  This **holds**, and the idea of reusing `a·b` and `‖·‖²` stays.
- **Leiden is near-linear and usually not the bottleneck**: **refuted**. `leidenalg` is measured to be superlinear
  (edges up 27×, time up 49×); from N=10k onward it is the largest piece of the non-kNN part,
  and on the full dataset it costs 7.6 times as much as building the HNSW graph.
- **kNN graph construction** eats the bulk of the time only on the **exact** route.

**So the firepower is split in two**: on the exact route, attack kNN (88–93% of which is top-k, see §1.2);
on the approximate route, attack Leiden. There is also a free win the plan did not mention: the original passes matrices
between Python and the `p_knng` subprocess through **text files**; on the full dataset this IPC tax is 13.4 s,
11 times the Tanimoto refinement itself, and it drops to zero once everything runs in a single process.

### 1.2 kNN is essentially blocked matrix multiplication + top-k selection

matmul has very mature acceleration paths (Apple: Accelerate uses AMX automatically; x86: MKL/OpenBLAS + AVX2/512).
**What is easy to overlook is top-k**: once matmul is optimized, naive sorting becomes the new bottleneck.
→ Use a heap / introselect + SIMD for top-k; this is the part worth writing ourselves.

**Confirmed by measurement (2026-09-16), and more extreme than expected**: splitting exact kNN into its two halves,
top-k takes 88–93% and matmul only 7–12% (N=10k/25k/50k/100k, numpy + multithreaded BLAS).
Even numpy's `argpartition` (introselect in C) cannot save it: it is single-threaded and has to pass over all
N candidates of each row.
→ **top-k takes priority over matmul**; the approach is parallelism + blocked inline selection (select while computing,
without materializing the `b × N` intermediate matrix), which also solves the memory problem: during profiling it was exactly this
intermediate matrix that pushed the 16 GB machine into swap.

### 1.3 Be honest about "AI hardware acceleration": the ANE is not usable

The Apple Neural Engine **has no public general-purpose GEMM interface**; it can only run model graphs through CoreML
and cannot be used for arbitrary kNN computation.

Viable acceleration options:

| Platform | Viable | Not viable |
|---|---|---|
| Apple Silicon | Accelerate/AMX (already in use via BLAS), Metal compute shaders, NEON | ANE |
| x86 + NVIDIA GPU | CUDA, AVX2/AVX-512 | — |

→ The roadmap lists only GPU/SIMD/BLAS; **do not list the ANE**.

→ Likewise, **no TypeScript / WebGPU** (2026-10-12): on a Mac, WebGPU ultimately translates to
Metal too, which Rust can use directly; bringing in a JS runtime would also break offline use after `pip install`.
See [docs/06 §6](06_cross_platform_and_determinism.md).

## 2. Layered design

```
CosTaL+ (Rust + PyO3/maturin)
├── Preprocessing layer  normalization strategy (L2 / none / magnitude carries quality), optional dimensionality reduction (PCA / random projection)
├── kNN layer            ① exact: p-L2knng rewrite (pruning upper bound; needed for research reproducibility)
│                        ② approximate: HNSW / IVF (the route for large scale)
│                        both implementations sit behind the same trait; callers switch with one line
├── Refinement layer     Tanimoto edge weights, fused with cosine into **one single-pass** kernel
│                        (reuses a·b and ‖·‖², avoiding a second traversal of the edge list)
├── Community layer      Leiden (bind leidenalg first; write our own only after profiling proves it is the bottleneck)
└── Backends             CPU: BLAS + SIMD(NEON/AVX) · GPU: Metal(Apple) / CUDA(NVIDIA)
```

### 2.1 Why Rust rather than C++

- maturin builds the wheels and cargo manages dependencies; cross-compiling to `macos-arm64` and `linux-x86_64`
  covers exactly the target machines;
- in parallel kNN, memory safety saves a great deal of debugging time.

Cost: the BLAS / Metal bindings have to be wired up by hand; the work is bounded and acceptable.

### 2.2 Magnitude semantics (an experiment shared with the face pipeline)

Tanimoto `T = a·b / (‖a‖² + ‖b‖² − a·b)` **degenerates on L2-normalized vectors** to
`T = c / (2 − c)`, a strictly monotone transform of cosine: neighbor ranking is unchanged, and **only the shape of the edge weights changes**
(weak edges are penalized harder: c=0.9→0.818, c=0.5→0.333, c=0.3→0.176).

→ So the value of Tanimoto lies in the **edge weights**, not in neighbor selection.
→ Only if magnitude is allowed to carry real information does the `‖a‖²+‖b‖²` term actually do any work.

**Verified (2026-09-16): this is not a hypothesis awaiting testing but the existing behavior of the original**: reading `clustering.py`
shows that `find_communities` writes the **L2-normalized** X into `.ijv` for `p_knng` to select neighbors,
but `build_tani_graph(clu, X)` receives the **unnormalized** X, and `cell_norm` takes the
**raw magnitude**; neighbor selection happens in normalized space, edge weights in the original space. Measured on Levine_32dim
(200k edges): row norms span 5.8×; between the original's T and the degenerate form `c/(2−c)` the maximum difference is 0.516,
Pearson r=0.780, and the rankings disagree. Magnitude really is doing work. Details in [docs/02 §4](02_original_profile_report.md).
→ Both modes must still be runnable and comparable (the face pipeline is the side where magnitude is flattened, see §7).

## 3. Deliverables

1. **Python wheel** (maturin): usable right after `pip install`, for callers such as `nas-manager`;
2. **Standalone CLI**: runs without writing any Python; **done (2026-10-13)**: `costal-plus cluster` reads
   CSV/TSV, FCS, `.npy` and `.h5ad`, applies the original `pp()` and writes one label per cell
   (167,044-cell FCS end to end in 10 s with `--leiden fast`);
3. **C ABI**: can be embedded in other hosts in the future.

The wheels must cover `macos-arm64` and `linux-x86_64`; in practice CI builds a wider set,
so callers don't have to pick an environment (abi3-py39: one wheel per platform covers CPython 3.9+):

| OS | Architecture |
|---|---|
| macOS | arm64, x86_64 |
| Linux (glibc / musl) | x86_64, aarch64 |
| Windows | x64, arm64 |

> **macOS pitfall**: the Python shipped by conda is often the x86_64 build (running under Rosetta),
> so installing the extension requires `rustup target add x86_64-apple-darwin`. But **performance numbers must come from
> a native arm64 interpreter**; numbers taken under Rosetta do not count.

## 4. Validation (what makes it "CosTaL+" rather than "yet another clustering library")

Two benchmark sets, both with **one-command reproduction**:

1. **Single-cell**: rerun on the datasets used in the original paper and report accuracy (compared with the original) + speedup;
2. **Faces**: the face embeddings from `nas-manager`, to show it works across domains.

Without these two sets of numbers, it is just a library.

## 5. Relationship with nas-manager

- CosTaL+ is an **independent repository with an independent lifecycle**;
- nas-manager installs the wheel via pip and **depends only on the public API**;
- nas-manager first gets the face track working with `hnswlib` + `leidenalg` and **pins down the interface**;
  once CosTaL+ matures, only the implementation needs to be swapped.

→ The two tracks do not block each other: the face pipeline does not wait for Rust, and when the Rust is written, a real caller and real data are already on hand as the benchmark.

## 6. Undecided matters (nothing left implied)

1. Whether exact kNN is really needed: depends on how much weight the research-reproducibility use case carries; to be decided;
2. How far the Metal backend goes (matmul only, or offload top-k as well);
3. ~~Whether to write Leiden ourselves: wait for the profile numbers;~~
   **The numbers are in (docs/02 §3.2): worth writing.** On the approximate route it takes 77%, and it is measured to be superlinear.
   But first check whether `n_iterations=10` is necessary (the original hard-codes a default of 10, and in `bc()` it is even 100);
   tuning alone may eat most of the cost;
4. Whether dimensionality reduction goes into the default pipeline (PCA changes the similarity semantics; needs experimental support);
5. ~~Version/API stability policy (when to lock 1.0).~~
   **Decided (2026-09-20): route (b).** Defaults may change at 1.0 (if the §5 experiments
   show that another combination is better), but **the major version number must bump with them**, and the `compat="paper"`
   escape hatch exists from day one and is never removed. See [docs/04 §6.5](04_selection_metric_and_pruning.md).

---

## 7. Empirical findings (2026-09-16, from nas-manager T85)

The full CosTaL pipeline was run on face embeddings (151 faces / 8 identities / hnswlib + igraph + leidenalg),
yielding a **generalizable boundary condition**:

> **The benefit of Tanimoto refinement depends on "whether magnitude carries information".**

- **When magnitude carries information** (single-cell counts / flow cytometry intensities): the `‖a‖²+‖b‖²` term genuinely takes part in discrimination;
  this is where the original paper's "distinguishing vectors with the same direction but different magnitudes" comes from, and it is where CosTaL's value lies.
- **When magnitude is flattened** (L2-normalized vectors, e.g. ArcFace face embeddings):
  `T = c/(2−c)` degenerates into a **strictly monotone transform** of cosine (derivative `2/(2−c)² > 0`);
  neighbor ranking is unchanged, only the threshold scale changes. Step ② becomes a no-op.

Measured evidence: `normalized × cosine, t=0.25` and `normalized × tanimoto, t=0.15`
give the same operating point (`T=0.15 ↔ c=0.2609`), with nearly identical metrics
(0.933/0.701/0.960 vs 0.935/0.675/0.960).

### 7.1 Implications for benchmark design (revises §4)

Face data **should not** serve as a showcase for "better across domains too", but as a **negative control**:
it shows that when magnitude carries no information, CosTaL **degenerates to the cosine baseline without getting worse**.
The real performance claims must still come from data in which magnitude carries information (single-cell).

### 7.2 Two more measured observations

1. **resolution is the most valuable knob** (on face data): Leiden resolution
   1.0 → accuracy 0.664, 2.0 → 0.846. The default resolution merges different identities into the same community.
2. **The "small k preserves quality" selling point cannot be verified on face data**: the 8 identities are too far apart from each other;
   with threshold ≥0.25, k=5..30 give the same solution row for row. To verify this selling point, the benchmark must use datasets with
   **blurry inter-class boundaries**; this is a hard requirement for choosing the §4 benchmarks.

### 7.3 Incidental engineering facts

- On an M1 Pro, the CoreML EP of `onnxruntime` produces output **identical image by image** to the CPU EP
  (cosine 1.00000), but on 12 samples it gives **no speedup** (2.73s vs 2.75s).
  Do not take inference acceleration for granted; measure it.
