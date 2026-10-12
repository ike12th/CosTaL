# Cross-platform, determinism, and backend selection (v0, 2026-10-12)

> Background: the fused blocking in docs/03 was designed with one M1 Pro in mind, but the delivery surface is
> macOS / Linux / Windows, desktop and server (docs/01 §3). These classes of machines differ in cache structure,
> matrix libraries, and core types. This document answers: **how can the same code be fast on every platform,
> and give the same answer on every platform?**
>
> The order is deliberate: **first guarantee a consistent answer (§1), then talk about speed (§2–§4)**. Once §1 is right,
> all later tuning affects only speed, not output.

> **Exactness statement**: nothing in this document changes "exact kNN". Either every pair is computed in full (§1–§4),
> or only candidates that **provably** cannot enter the top k are skipped (the conservative threshold and pruning of
> docs/04 §2–§3, the L2Knng exact mode of §5). Approximate methods (HNSW, docs/03 Stage E) are opt-in only and
> **never enter the default path** — if the default path is not exact, it is not CosTaL.

---

## 1. Consistent results: full f32 coarse selection + f64 rescoring

### 1.1 The problem

- matmul uses f32 (docs/04 §4.1). Different matrix libraries (Accelerate / OpenBLAS / MKL /
  pure-Rust kernels) differ in accumulation order and FMA usage, so their rounding differs; even the same library
  can differ across tile shapes and instruction sets.
- "Near-ties" around rank k can therefore flip ⇒ **the same data builds two slightly different graphs
  on Mac and on Linux**. The tie-break in docs/04 §4.1 handles only **exact equality**;
  it cannot handle "order flips caused by rounding".
- Worse: change the tile size (§2), the thread count, or the backend (§5), and the result may change too —
  **tuning and correctness become entangled**, and every round of tuning means re-verifying correctness.

### 1.2 Approach

```
full f32 coarse selection: the blocked matmul still computes all N² similarities pair by pair,
                           but each row keeps the top-(k+m); m is a margin (initially m = k)
     ↓ k+m candidates per row
f64 rescoring: for these k+m candidates, recompute a·b in f64, in a fixed order by dimension index
     ↓
take the top k by f64 value; on ties, take the smaller original index (docs/04 §4.1)
```

"Coarse selection" means coarse in **precision** (f32), not coarse in **scope**: every candidate is computed; there is no approximation.

- **Why this is consistent across platforms**: f64 rescoring is an ordinary Rust loop. By default Rust
  **does not contract floating-point operations** (it will not fuse a multiply and an add into an FMA on its own), and without
  fast-math it does not reorder floating-point accumulation either, so the same code is bit-identical on x86 and ARM.
- **Cost**: `N·(k+m)·d` f64 multiply-adds. Full Levine: 265,627 × 20 × 32 ≈ 1.7×10⁸,
  on the order of milliseconds.
- **Remaining risk**: the true rank-k neighbor falls outside the top k+m under f32. That requires the f32 errors
  of m candidates to all skew in the same direction. **Quantify it with tests, don't rely on belief**: in the L2 brute-force
  comparison, count the "coarse-selection misses"; the expected count is always 0.
- **Now replaced by a provable bound (2026-10-12, during the Stage A implementation)**: the margin m alone is not enough — with many exact ties
  (integer data, duplicate rows), the tie group at rank k can be larger than k+m. Each row is now checked: every candidate that did not enter the heap
  has an f32 value no greater than the heap root, so its true cosine is no greater than heap root + ε (ε = (d+4)·2⁻²³, derived from the
  dot-product error of f32 normalization under any accumulation order, with a factor-of-2 margin). If the rank-k value after rescoring is strictly above this bound ⇒ the result is exact;
  otherwise, for that row, collect all candidates with f32 value ≥ rank-k value − ε and rescore them. **The remaining risk is therefore zero**; the only cost is
  that these rows are slower. The number of triggers is recorded in the statistics (`fallback_rows`); on Levine it is 0 at every scale.
- Edge weights are also computed from the rescored f64 cos (converted to f32 only at output), so Tanimoto edge weights are likewise consistent across platforms.

### 1.3 What it unlocks

- The tile size of §2, the matrix library of §3, the thread count, and the dense/sparse backend of §5 **no longer affect the output**,
  and can be tuned per platform with confidence.
- The L1/L2 correctness nets of docs/04 §4 can require **entry-by-entry identity** on all three CI systems,
  plus one new item: **the outputs of the three platforms are compared entry by entry against each other**.
- `truncate()` in docs/04 §6.1 holds: after rescoring, each row is totally ordered by (f64 similarity descending, original index ascending),
  so any prefix is the exact answer for a smaller k.
- L3 (comparison against the original `p_knng`) can still only achieve "identical except for near-ties": `p_knng`
  has its own precision and accumulation order. How large this gap is must be measured with golden output (docs/04 §4).

---

## 2. Cache and tile size

### 2.1 Why cache matters more than compute at d=32

Full N=265,627: N² ≈ 7.06×10¹⁰ candidate pairs. Each similarity is written by matmul (4 bytes)
and then read by top-k (4 bytes).

- If tiles **spill to memory**: ≈ 565 GB must be moved; at the M1 Pro's ~200 GB/s that is ≈ 3 s,
  the same order of magnitude as the computation itself;
- If tiles **stay in cache**: this traffic essentially disappears.

The smaller d is, the less computation per similarity (only 32 multiply-adds at d=32), and the larger the share of data movement.

> The above is an **estimate, not measured**. The Stage A bench has to provide real numbers.

### 2.2 Cache structure by platform

| Platform | L1D per core | L2 | L3 |
|---|---|---|---|
| M1 Pro (this machine, measured with `sysctl`) | 128 KB (performance cores) | 12 MB, shared by 3 performance cores (two clusters) | `sysctl` does not report an L3; there is also a system-level cache, SLC (about 24 MB according to reviews, shared by CPU/GPU) |
| Desktop x86 (typical for Windows) | 32–48 KB | Private per core: Intel 2015–2020 256 KB, newer 1–2 MB; AMD Zen 2/3 512 KB, Zen 4 1 MB | A dozen to several tens of MB, shared by all cores |
| Server x86 (typical for Linux) | 32–48 KB | Private per core, 512 KB–2 MB | Very large, but split among dozens of cores; on AMD, partitioned by CCD (8 cores) |

> Rows two and three are typical values; specific models will vary.

### 2.3 Principle: target the "per-core private share"; L3 is only a fallback

- **Don't target L3**: L3 is shared by all cores, and on servers there are other people's processes too, so how much you get is unpredictable.
- **Target the per-core private L2**: size each thread's working set (tile + column block + heap) to it.
- **The original 256×4096 (4 MB per thread) of docs/03 §2.2 is withdrawn**: on the M1, 3 performance cores × 4 MB
  exactly fill the 12 MB shared L2, leaving no room for column blocks and heaps; on old x86 with a 256 KB L2 it spills outright.

Three tiers:

1. **Floor value**: size the tile at half of the smallest common private L2 (256 KB), ≈128 KB
   (e.g. 32×1024 f32s). Not optimal, but it never spills to memory on any platform.
2. **Runtime detection**: on Linux read `/sys/devices/system/cpu/cpu*/cache/`, on macOS read
   `sysctl hw.perflevel*`, on Windows call `GetLogicalProcessorInformationEx`;
   scale the tile up according to the "L2 per thread" (M1: 12 MB / 3 cores = 4 MB; the tile takes half).
   If detection fails, use the floor value.
3. **(Optional) first-run self-test**: try 3 sizes, about 30 ms each, store the result locally,
   and use it directly from then on (the idea behind FFTW's wisdom).

Because of §1, these three tiers affect only speed, not output.

> **Measured (2026-10-12, M1 Pro, N=50k, docs/03 §8.3)**: tiles between 128 KB and 2 MB form a plateau
> (differences < 15%); ≤ 64 KB is clearly slower (each matmul call has a fixed overhead). The default derived from tier 2
> (per-thread L2 4 MB → tile 2 MB) is the fastest setting in the sweep. The tier-1 floor of 128 KB falls within the plateau.
> Implemented: detection on macOS (`sysctl`) and Linux (`/sys`); **Windows currently just uses the floor value**;
> `GetLogicalProcessorInformationEx` is still to be added. Tier 3 (first-run self-test) is not done — the plateau is wide enough that it is not needed for now.

### 2.4 Alternative: register-level fusion (not assumed for Stage A)

The other route is never to materialize the tile at all: finish the dot product in SIMD registers and compare it directly with the threshold.
The working set shrinks to a few KB, completely independent of cache size. The cost is writing a separate microkernel for each of
NEON / AVX2 / AVX-512, and on Mac the AMX cannot be used (AMX is reachable only through Accelerate).
**Stage A does BLAS + tiles first; whether to take this route is decided after the bench.**

---

## 3. Matrix libraries

| Platform | What to use | Rationale |
|---|---|---|
| macOS | Accelerate (ships with the system, uses AMX) | Zero dependencies, and fastest |
| Linux / Windows | Pure-Rust matrix kernel; candidate: the `gemm` crate | Picks AVX2 / AVX-512 / NEON automatically at runtime; depends on no external library |

- **Don't bundle OpenBLAS / MKL**: stuffing them into manylinux / musllinux / Windows wheels,
  and also dealing with thread oversubscription (competing for cores with the OpenBLAS that numpy brings into the user's process),
  costs far more engineering than it gains.
- At d=32, matmul accounts for only 7–12% of exact kNN (docs/02 §3.3); a pure-Rust kernel is sufficient.
  **Bench first, then decide.**
- `gemm` crate: crates.io 0.19.0 (2025-11); both candle and faer depend on it (checked 2026-10-12).
- **Measured (2026-10-12, M1 Pro)**: Accelerate is about 2.2× faster than `gemm` (0.43 s vs
  0.93 s, each at its best tile size, N=50k), so the macOS default of Accelerate holds. `gemm`'s performance on x86 is not yet measured — that waits for
  the Linux / Windows CI machines or the NAS. The two produce bit-identical output (`tests/test_knn.py`).

---

## 4. Concurrency details

1. **Big/little cores → work stealing.** Apple (P cores + E cores) and Intel 12th gen onward (including the NAS's
   Pentium Gold 8505) all have big/little cores. Splitting work evenly leaves the fast cores idle, waiting for the slow ones ⇒ cut row blocks small
   and use rayon's work stealing.
2. **Cache lines → align uniformly to 128 bytes.** Apple uses 128 bytes (on this machine
   `sysctl hw.cachelinesize` = 128), x86 uses 64 bytes. Per-thread **writable** state such as heaps and thresholds
   is aligned uniformly to 128 bytes (e.g. `crossbeam_utils::CachePadded`),
   so that two threads never modify the same cache line (false sharing). On x86 this costs only a little extra space.
3. **Symmetry → halve the matmul (experimental).** cos(i,j) = cos(j,i): compute only the upper-triangular tiles,
   and feed each value to both row i's heap and row j's heap. Row j's heap may belong to another thread ⇒
   each thread first writes to a local buffer, and the buffers are merged at the end. The merged result is determined by §1's rescoring and original-index tie-break,
   independent of the thread count. Both the gain and the merge overhead need to be measured.
4. **NUMA (dual-socket servers): small impact, not handled specially.** The input X is only 34 MB and read-only;
   each thread allocates its own heap (which lands in local memory on first write).

---

## 5. Two backends, dense / sparse, and `backend="auto"`

### 5.1 Two kinds of data

Levine_32dim, measured (2026-10-12, all 265,627 cells, after `arcsinh(x/5)`):
**nonzero fraction 1.000** (32/32 nonzero in every row); **27.0% of elements are negative**, 99.8% of rows contain negative values,
minimum about −0.058.

| Data | Examples | Backend |
|---|---|---|
| Low-dimensional dense | CyTOF / flow cytometry (Levine: 32 dims, no zeros); prior-knowledge projections (pathway / TF activity, gene programs, §5.3) | Blocked matmul + top-k (Stage A) |
| High-dimensional sparse | Gene-level scRNA-seq (the original's `pp(..., 'scrnaseq')`: 2000 highly variable genes, log1p, sparse CSR) | L2Knng-style inverted index + pruning (Stage S, **exact mode only**) |

L2Knng's pruning is built on an inverted index and is designed for sparse high-dimensional data (docs/02 §1.1 Addendum).
On 32-dimensional fully dense data, every inverted list has N elements, and pruning saves only part of the
32 multiply-adds, which cannot beat BLAS tiles — **this is a judgment, not a measurement**.

### 5.2 Constraints on auto dispatch

1. **The threshold has to be measured.** Whether sparse pays off depends on the fraction of zeros **and** the dimensionality. The crossover point can only
   be measured on the same set of data once both backends are written ⇒ `auto` comes after Stage S.
2. **Dispatch must be deterministic, visible, and overridable.** The rules look only at the input itself (whether it is in a sparse format, nonzero fraction, dimensionality),
   so the same input always takes the same backend; which backend was actually taken is written into the output metadata; `backend="dense"` /
   `"sparse"` can force a choice. Both backends **share the same f64 rescoring of §1**, so as long as both candidate pools
   contain the true top k, the outputs are identical entry by entry, and dispatch only affects speed.
3. **`compat="paper"` is treated separately.** The original makes no distinction by data and always uses `p_knng`. Our output can differ from
   `p_knng` only on near-ties (§1.3); by how much is to be quantified with golden output.
4. **(Added 2026-10-12) Two more signals: whether there are negative values, and whether the columns are centered.** Any column-centered input
   (PCA, Pearson residuals, z-score) is effectively dense, so it **goes straight to the dense backend, no threshold measurement needed**.
   Pearson residuals in raw gene space are a memory trap: 1 million cells × 2,000 genes × f32
   is about 8 GB (our own estimate; Lause 2021 reported 4.5 GB on PBMC); users should be advised to do PCA first
   or switch to shifted log. Imaging panels of 300–500 genes fall near the sparse/dense boundary, which makes them exactly the right data
   for measuring the `auto` threshold.

### 5.3 Prior-knowledge projection: projecting genes onto "biologically meaningful things"

> Rewritten 2026-10-12 based on the survey. The first draft said "taking functional units as the unit"; the more accurate description is:
> **use biological entities defined by prior knowledge (signaling pathways, transcription factor regulons, gene programs) as coordinate axes,
> and project each cell's gene expression onto them**, describing cell state in that space.

Three kinds of approaches:

| Category | Representatives | How the projection works | Output dimensionality |
|---|---|---|---|
| Activity inference (linear projection) | PROGENy (signaling pathways), CollecTRI (TF regulons), computed via decoupler; AUCell / ssGSEA (gene set scoring) | Prior weight matrix × expression (PROGENy is a single matrix multiply) | PROGENy 14 pathways; CollecTRI covers 1,186 TFs (43,175 signed regulatory interactions); gene sets can number in the thousands |
| Prior-constrained latent space | VEGA, expiMap (VAEs whose decoders are wired by gene program); Spectra (matrix factorization, known gene sets + newly discovered programs) | Learned, but each dimension corresponds to a known program | Number of programs, typically a few hundred |
| One embedding per program | Tripso (self-supervised transformer, bioRxiv 2026-03) | One vector per cell per program | A cell is a **set** of vectors, not one |

**Is it still sparse? No.** The input gene matrix is still sparse, but the projected "cell × entity" matrix
is **low- to medium-dimensional and dense**: each score aggregates tens to hundreds of genes, so the zeros left where individual genes go undetected
are averaged out. AUCell on small gene sets, and scores from non-negative factorizations such as Spectra, may contain quite a few zeros, but the dimensionality itself
is low, so they still go to the dense backend.

Implications for CosTaL (**judgment, to be measured**):

1. **Magnitude carries information.** The size of an activity score is the strength of the activity and has not been flattened by normalization — exactly the side on which the Tanimoto
   refinement acts (docs/01 §7), so such representations may be a very natural input for CosTaL. The exception is
   program usages normalized per cell to sum to 1 (e.g. cNMF's normalized usages), where magnitude information is partly erased.
2. **Both positive and negative.** PROGENy and CollecTRI activity scores are signed; the revised guardrail in docs/04 §2.4
   does not depend on the input's sign, so it works.
3. **Wide range of dimensionality.** At 14 dims it is low-dimensional dense, like CyTOF; at thousands of dims, matmul's share of exact kNN
   rises noticeably (docs/05 §4.1 item 5), and only then does a GPU discussion make sense.
4. **"Multiple vectors per cell" does not fit the current kNN directly.** Representations like Tripso can either build one
   kNN graph per program and then run multilayer (multiplex) Leiden (`leidenalg` supports multilayer graph optimization), or be concatenated into
   a single vector. This is a new design question, **undecided, not assumed in advance**.
5. Consistent with docs/05 §4.2: these are all **optional input representations** and do not enter the default path. The default path is still
   "exact kNN on the matrix the user provides".

### 5.4 Dispatch by "input class", not by platform (added 2026-10-12)

[The survey report](../reports/single_cell_and_spatial_transforms.md) groups the mainstream transforms across platforms in 2023–2026
into five classes. Both the backend choice and whether Tanimoto is meaningful are determined by the class, independent of platform:

| Class | Representatives | Sparse | Can be negative | Meaning of the row norm | Backend | Tanimoto |
|---|---|---|---|---|---|---|
| **A** zero-preserving, non-negative, high-dimensional | scRNA shifted log, `log1p(raw)`; sequencing-based spatial bins / cells; 5K/6K/WTx panels | Yes | No | Concentration + residual depth; RNA content / density when raw or area-normalized | Sparse | Informative but confounded by depth / segmentation; needs measurement |
| **B** zero-preserving, non-negative, low-dimensional | Imaging panels of 300–500 genes | Moderate (about 60–85% zeros, estimated) | No | Same as A | Threshold to be measured | Same as A |
| **C** low-dimensional, uncentered abundance | CyTOF / flow / spectral arcsinh, IMC `asinh(x/1)`, MIBI | No | CyTOF: yes, but tiny | **Overall marker abundance** | Dense | **Matches CosTaL's original intent; strongest evidence** |
| **D** centered, signed | PCA, Pearson residuals, z-score, non-unit-normalized foundation-model embeddings | No | Yes | Distance from the data center; for foundation models the meaning of the norm is unknown | Dense | Semantics change; T can be negative |
| **E** already row-normalized | scGPT / UCE / SCimilarity, fastMNN, Pixie | Depends on input | Depends on input | Constant | Dense | ≡ `c/(2−c)`, no new information |

The four quantities needed for the classification (row-norm CV, column mean / standard deviation, fraction of negative values, nonzero fraction) already have to be written into the
metadata (docs/04 §6.6), so dispatch and diagnostics share a single pass.

---

## 6. Roads not taken

- **TypeScript / WebGPU**: TS can reach the GPU only through WebGPU, which on Mac is ultimately translated to Metal anyway
  (Deno's WebGPU is built on Rust's `wgpu`). Rust can use `wgpu` or Metal directly,
  one layer fewer; and bringing in a JS runtime conflicts with "works offline after `pip install`". If we later do browser-side
  visualization (e.g. the nas-manager UI), that is a frontend product, not an acceleration route.
- **Decompiling `p_knng` (REA / Ghidra and the like)**: the source is readable and the binary is not stripped; what we lack is
  golden output, which we get by running it, not by decompiling. See docs/02 §1.1 Addendum.
- **Terminology**: **Metal** is Apple's GPU programming interface (the counterpart of NVIDIA's CUDA), available only on Apple
  devices; **Accelerate / AMX** is a CPU-side matrix coprocessor, not a GPU.
  The GPU backend is still docs/03 Stage F.

---

## 7. What this adds to the docs/03 stage table

| Stage | Increment |
|---|---|
| **A** | Ship §1's f64 rescoring from day one; tiles per §2.3 (floor value + detection); bench sweeps tile sizes; non-Apple platforms use §3's pure-Rust kernel; §4 items 1 and 2 are written by the book from the start |
| **A experiments** | §4 item 3 (symmetry halving), §2.4 (register-level fusion): keep or drop after the bench |
| **CI** | Outputs of the three systems compared against each other entry by entry (§1.3) |
| **S (new)** | Sparse backend: L2Knng-style inverted index + pruning, **exact mode only**; snnlib as the reference implementation (docs/02 §1.1 Addendum); acceptance-tested on gene-level scRNA-seq |
| **After S** | `backend="auto"`, threshold measured (§5.2) |
