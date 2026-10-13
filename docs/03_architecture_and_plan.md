# CosTaL+ Architecture and Implementation Plan (v0, 2026-09-20)

> **Name.** These design notes were written while the rewrite was called *CosTaL+*; it is released as
> **CosTaL 1.0**, the next version of CosTaL (0.0.2 → 1.0). "CosTaL+" below means CosTaL 1.0.

> Background: [docs/01](01_technical_plan.md) sets the direction; [docs/02](02_original_profile_report.md) has the measured numbers.
> This document answers "**exactly how to build it, and in what order**".
>
> **Decided**: exact kNN stays (research reproduction needs it). So the effort goes into the exact route:
> the first cut targets top-k (88–93% of exact kNN).

---

## 1. The architecture in one sentence

**Collapse the original's "five rounds of shuttling data in and out of processes" into "one in-memory pipeline".**

The original (every step fully materializes the data once):

```
X ──pp──> X' ──L2 normalize──> write .ijv text file (11.4s)
                                  │
                          p_knng subprocess (exact kNN)
                                  │
                           write .clu text file
                                  │
          read back + Python list parsing + per-edge Tanimoto (5.6s)
                                  │
              scipy sparse-matrix symmetrization + igraph graph build
                                  │
                              leidenalg (102.7s)
```

CosTaL+ (one process; the similarity matrix **never exists in full**):

```
X ──> normalize + cache the raw norms ‖·‖
        │
        └─> blocked pipeline: for each small tile (row block b × column block c)
              ① BLAS computes the b×c similarities (only a few MB, stays in cache)
              ② immediately merged into each row's top-k heap
              ③ tile discarded ← the key point: the large b×N matrix never exists
        │
        └─> Tanimoto on the k survivors (reusing the a·b and ‖·‖² already at hand)
        │
        └─> emit the symmetrized CSR edge list directly
                                  │
                              Leiden
```

What gets eliminated, per the numbers in docs/02: text IPC 13.4s, Python edge-list assembly 5.6s,
and the b×N memory round trip (the real source of top-k's 88–93%).

---

## 2. Why top-k is slow now, and how to make it fast

This is the most valuable single spot in the whole project, so it gets its own section.

### 2.1 Status quo: what numpy does

```python
sim = Xn[block] @ Xn.T          # materializes a b×N float32 matrix
part = np.argpartition(sim, k)  # one introselect pass over all N values of each row
```

At N=265,627 one row is 265,627 float32 values (1 MB); at b=256 that is **272 MB**.
This memory is **written once, then read back again**, and the M1 Pro's memory bandwidth is limited.
`argpartition` is implemented in C, but it:

1. **is single-threaded** — it uses 1 of the 8 cores;
2. **has to see the whole row** — introselect is O(N) but with a sizeable constant, and its memory-access pattern is cache-unfriendly;
3. **allocates extra memory** — it returns a b×N **int64** index array (265,627×256×8 = 544 MB).
   This array is what pushed the 16 GB machine into swap during this profile.

So of that 88–93%, **very little is actual computation; most of it is memory bandwidth and single-threading**.

### 2.2 The CosTaL+ approach: fused blocking

Instead of materializing b×N, loop over **small (b × c) tiles**:

```
for row block B in 0..N step b:          // parallelizable, one row block per thread
    heaps[b] ← empty k-element min-heap
    for column block C in 0..N step c:   // b×c sized to each core's own share of L2, see docs/06 §2.3
        tile = Xn[B] · Xn[C]ᵀ             // BLAS sgemm, the tile stays in L2
        for each row i, each candidate j in tile:
            if tile[i][j] > heaps[i].min:          // the vast majority are rejected right here
                heaps[i].replace_min(j, tile[i][j])
    // the tile is discarded when it goes out of scope
```

> **Revised (2026-10-12).** The original text said b≈256, c≈4096 (4 MB per thread).
> That only barely holds on the M1: 3 performance cores × 4 MB exactly fills the 12 MB shared L2,
> leaving no room for column blocks and heaps; on older x86 with a 256 KB L2 it simply overflows.
> Tile size is now "a safe default + runtime probing", see docs/06 §2.
> Also, each row actually keeps top-(k+m) and hands it to f64 rescoring, which guarantees consistent results across platforms, see docs/06 §1.

Why this is fast:

- **b×N no longer exists**: peak memory drops from 272 MB to one tile (hundreds of KB to a few MB, depending on the platform),
  which stays in cache the whole time and does not consume memory bandwidth. As a side effect this fixes the OOM — **memory no longer grows with N**.
- **The rejection path is extremely short**: at k=10, N=260k, the probability that a candidate makes the top-10 is about
  10/265627 ≈ 0.004%. In other words, **99.996% of candidates are discarded after a single floating-point comparison**,
  and the branch predictor learns this very accurately. That is an order of magnitude cheaper than "scan the whole row, then select".
- **Naturally parallel**: row blocks are fully independent, so 8 cores simply run 8 ways (rayon).
  On the numpy path matmul is multithreaded but top-k is single-threaded; that asymmetry is gone.
- **matmul is still handed to an existing matrix library** (macOS: Accelerate/AMX; other platforms:
  a pure-Rust kernel, see docs/06 §3); we only write
  steps ② and ③ — exactly what §1.2 means by "this is the part worth writing ourselves".

Expected gain: turn exact kNN from "matmul 7–12% + top-k 88–93%" into
"matmul is the bulk + top-k is a rounding error". The matmul part is already backed by BLAS,
so **total time should approach the matmul column of the profile**
(N=100k: 26.6s rather than 225s). This is the number the first stage has to deliver.

### 2.3 A semantic detail that must not go wrong

**Neighbors are selected by cosine, edge weights are computed with Tanimoto, and the order must not be reversed.**

This is exactly what the original does: `p_knng` selects neighbors by cosine on **L2-normalized** data,
and `build_tani_graph` then computes the Tanimoto edge weights from the **unnormalized** raw norms
(verified in docs/02 §4). Because the raw norms span a 5.8× range,
**putting Tanimoto into the top-k comparator would change the selected neighbor set**,
and the result would no longer be CosTaL. Hence:

```
top-k comparator ── looks only at cosine
     ↓ (k survivors)
Tanimoto ── T = d / (‖a‖² + ‖b‖² − d), where d = ‖a‖·‖b‖·cos
```

`‖·‖` is cached along the way during normalization, O(N), once. This is the "reuse a·b and ‖·‖²" of §1.1;
measured at 1% of the time, it really is a rounding error.

---

## 3. Where pruning (p-L2knng) fits

The fused blocking of §2.2 is an **unpruned exact solution**: it dutifully computes all N² inner products.
The value of p-L2knng is **skipping whole batches of candidates that cannot make the top-k** — using a norm upper bound to prove
"even the largest possible inner product in this block cannot reach the current k-th place", so the whole block is skipped.

The key architectural judgment:

> **Pruning and blocked BLAS are two candidate-generation strategies for the same exact answer.**
> They produce exactly the same result; only the number of skipped candidates differs.

So they sit **behind the same trait**, and this naturally gives us a handy way to verify:

```
BlockedExact (unpruned, slow but obviously correct)
      ↑ compared edge by edge, must match exactly
PrunedExact (pruned, fast)
```

**Build the unpruned version first.** Reasons:

1. It already captures the main gain of §2.2 (top-k is no longer the bottleneck), so the numbers get banked first;
2. It is the pruned version's **golden reference** — pruning bugs are all "missed a neighbor that should have made the top-k",
   and without a reference such bugs are simply invisible; the metrics just quietly get a tiny bit worse;
3. The gain from pruning depends heavily on the data distribution (dimension, sparsity, norm distribution);
   how much it saves on 32-dimensional dense data like Levine_32dim is **something we only know after measuring** —
   we cannot write it all first and only then discover it barely saves anything.

> **Addendum (2026-10-12): L2Knng itself is not a fit for Levine.** L2Knng's pruning is built
> on an inverted index and was designed for sparse high-dimensional data; Levine_32dim, measured on the full set, is **100% dense**
> (32/32 nonzeros in every row, docs/06 §5.1), so the inverted index has nothing to grab onto here.
> On Levine, then, all Stage D can hope for is docs/04 §3's norm window (tanimoto) and
> cone bound (cosine) — both are provable bounds and remain exact. L2Knng moves to the sparse backend
> (Stage S, docs/06 §5), using only its exact mode.

---

## 4. Layering and crate split

```
costal-core/              pure Rust, knows nothing about Python, can be cargo test'ed / bench'ed on its own
│
├── prep      normalization + norm cache. Output: (Xn: row-major f32, norms: Vec<f32>)
│
├── knn       trait KnnIndex { fn search(&self, k) -> Neighbors }
│             ├── exact::Blocked   fused blocking of §2.2                 ← first cut
│             ├── exact::Pruned    norm-window / cone-bound pruning       ← second cut (docs/04 §3)
│             ├── exact::Sparse    L2Knng-style inverted index, sparse input ← Stage S (docs/06 §5)
│             └── approx::Hnsw     large scale / face pipeline, opt-in    ← later, never the default
│             Neighbors = { idx: Vec<u32>, cos: Vec<f32> } (row-major N×k)
│
├── weight    cos + norms → Tanimoto / Jaccard / raw cosine
│             (fused at the knn exit, no second pass over the edge list)
│
├── graph     symmetrization + self-loop removal + CSR. Replaces the original's scipy + igraph glue
│
└── cluster   Leiden. **Not in the first stage**, see §5
```

```
costal-py/                only type conversion and error mapping, no logic
│  What is exposed to Python follows a "minimal usable surface":
│    build_graph(X, k, method, backend) -> (rows, cols, weights)
│    find_communities(X, ...)           -> labels      ← once the cluster layer is in place
```

### 4.1 Why core and py are separate

- **Benchmarks run on their own**: `cargo bench` does not need to start a Python interpreter,
  so the feedback loop while optimizing kernels is much shorter;
- **A C ABI comes almost for free** (docs/01 §3 item 3): core is already pure Rust,
  so adding a `costal-c` crate that exports `extern "C"` is enough, with no refactoring;
- **Bugs in the binding layer and bugs in the algorithm stay separate.**

### 4.2 Which layer the backends (BLAS / SIMD / GPU) live in

Not a separate layer, but **an implementation detail inside the knn layer**:

- `exact::Blocked` calls the matrix library's `sgemm` — Apple goes through Accelerate (which uses AMX automatically),
  other platforms use a pure-Rust kernel, and OpenBLAS/MKL are not bundled (changed 2026-10-12, reasons in docs/06 §3).
  Selected with `cfg`, transparent to the layers above;
- The SIMD (NEON/AVX) for the top-k merge step also lives here;
- Metal/CUDA will later be **a fourth `KnnIndex` implementation**, not a cross-cutting layer.
  As for the "how far to take Metal" question in docs/01 §6 item 2:
  thanks to the trait boundary, we can do only matmul first and later offload top-k along with it,
  without affecting the other layers.

---

## 5. What about Leiden (first stage: hands off)

The numbers in docs/02 §3.2 say Leiden is worth writing, but **not now**; the reason is ordering:

- On the exact route Leiden is only 10% (102.7s out of 1098.7s);
  only after kNN has been knocked down from 1098.7s will Leiden's share surface;
- `leidenalg` is a mature C++ implementation; our own first version would most likely be slower and fail to match its results,
  while "no loss of accuracy" is this project's first goal;
- **There is a possibly free optimization nobody has tried**: the original hard-codes `n_iterations` to 10
  (in `bc()` it is even 100). The Leiden paper recommends iterating until convergence,
  which usually stabilizes within 2–3 rounds. If the 2-round and 10-round partitions agree, that saves 80% right here.
  **This is an experiment, not coding work, and it should come before writing a Rust Leiden.**

So the first stage's deliverable looks like this:

```python
rows, cols, weights = costal_plus.build_graph(X, k=10)   # Rust
G = ig.Graph(...); la.find_partition(G, ...)             # still Python
```

This way **we get the full end-to-end speedup without writing a single line of Leiden**,
and because we share the same Leiden implementation with the original, the accuracy comparison is clean
(partition differences can only come from the graph, not from the clustering algorithm).

---

## 6. Implementation order and the acceptance criteria for each step

Every stage must report two numbers, **correctness** and **speed**; if either is missing, the stage is not done.
The data is always Levine_32dim (265,627 cells, with 14 manual gating labels).

| Stage | What | Correctness criterion | Speed criterion |
|---|---|---|---|
| **A** | `exact::Blocked` + fused Tanimoto + `build_graph` emitting the edge list | Neighbor sets **identical entry by entry** to brute force (full comparison for N≤20k); end-to-end ARI/NMI on par with the original CosTaL | Compared against docs/02's 1098.7s (N=260k). Goal: approach the matmul column |
| **B** | `graph` layer moves into Rust (symmetrization + CSR) | Edge list matches Stage A + scipy | Eliminates docs/02's 5.6s |
| **C** | Leiden `n_iterations` experiment | Pairwise ARI between the partitions at 2/3/5/10 rounds | If 2 rounds suffice, saves ~80% of the 102.7s |
| **D** | `exact::Pruned` (norm window / cone bound, docs/04 §3) | **Identical entry by entry to Stage A** (this is a hard gate) | Speedup relative to A; if <1.5×, not worth keeping |
| **E** | `approx::Hnsw` behind the trait (opt-in, never the default) | Recall vs A; how much ARI drops | For large-scale use / the face pipeline (nas-manager) |
| **F** | Metal / CUDA backend | Same as A | Whether to do it depends on matmul's share after A |
| **S** | `exact::Sparse` (L2Knng-style, exact mode), followed by `backend="auto"` | Identical entry by entry to brute force | Speedup relative to A on gene-level scRNA-seq |

> **C comes before D** on purpose: C is an experiment (a few hours), D is real engineering (a few days).
> Take the cheap gain first, then see whether the expensive one is still worth doing.
>
> **2026-10-12**: Stage A carries f64 rescoring and the cross-platform tile strategy from day one;
> Stage S (sparse backend) is new. Details in docs/06 §7.

### 6.1 What we will not do (to avoid scope creep)

- **Do not touch `pp` preprocessing**: measured at 0.23s, a 0% share. Keeping it on the Python side is fine.
- **Do not write our own BLAS**: matmul is only 7–12% of exact kNN. macOS uses Accelerate,
  other platforms use an existing pure-Rust kernel (docs/06 §3); OpenBLAS/MKL are not bundled either.
- **No dimensionality reduction** (docs/01 §6 item 4): PCA changes the similarity semantics,
  so it stays out of the default pipeline until the "no loss of accuracy" goal is settled.

---

## 7. Which items of docs/01 §6 this architecture answers

| §6 open item | Current status |
|---|---|
| 1. Is exact kNN really needed | **Decided: yes**. Effort goes into the exact route; the first cut is top-k |
| 2. How far to take Metal | Deferred. The trait boundary lets it come later; decide after Stage A based on matmul's share |
| 3. Whether to write Leiden ourselves | Run the Stage C tuning experiment first; no rush to write it |
| 4. Whether dimensionality reduction enters the default pipeline | No, see §6.1 |
| 5. Version/API stability | Stage A exposes a single function, `build_graph`, a small surface that is easy to change; `find_communities` is decided once the cluster layer is in place |

---

## 8. Stage A results (2026-10-12)

> Environment: M1 Pro (6 performance cores + 2 efficiency cores, 16 GB), macOS 26.5.2, native arm64 Python 3.9.6,
> matmul via Accelerate. 8.4 GB of swap was in use during the tests, so the absolute seconds are on the conservative side.
> Commands and raw numbers: `bench/stage_a.py`, `bench/out/stage_a.json`. The results were produced by the code at commit 3d3e8d2
> (the extension was built before the version number was changed to 0.1.0, so the version field in the json reads 0.0.1; the code is the same).

### 8.1 Correctness

| Check | Result |
|---|---|
| Against a Rust f64 brute-force solution (Levine subsets, N = 10k, 20k) | **Bit-identical** (neighbors and cosine) |
| Synthetic data: tile size × matmul backend × layout × thread count | All bit-identical (`cargo test`, `pytest`) |
| Many exact ties (integer data, duplicate rows, zero rows) | Bit-identical, thanks to the fallback below |
| Tanimoto / Jaccard / fuzzy union vs the original code | Bit-identical (`tests/test_knn.py`) |
| Against the original `bc()` path (deduplicated Levine, N = 50k) | Neighbor sets **100% identical**; same edge count (401,424); max relative Tanimoto difference 6e-8 (our input is f32) |
| Same, Leiden partitions (5 seeds) | With the same seed, the two sides' partitions have **ARI = 1.0 (all 5 seeds)**; against the gating labels both sides score ARI 0.804 ± 0.073, NMI 0.875 ± 0.026 |

Leiden's own agreement across seeds is only ARI 0.833 — **Leiden's randomness on the same graph is much larger than the
difference between the two implementations** (which is zero). Accuracy numbers from a single seed cannot be used to compare methods.

**A design correction relative to docs/06 §1.2**: the margin m alone is not enough. With many exact ties, the tie group at k-th place can be larger than
k+m, and the coarse selection's tie-breaking can then drop the candidate that should win by original index (this actually happened on the integer test data). It is now
a provable bound: a candidate that did not enter the heap has a true cosine no greater than heap root + ε (ε = (d+4)·2⁻²³); k-th place strictly above this bound
⇒ exact, otherwise the whole row is collected and rescored. **On Levine the fallback count is 0 at every size.**

### 8.2 Speed (raw files, not deduplicated; sizes aligned with docs/02)

| N | Exact kNN | Edge weights + symmetrization | docs/02 reference |
|---:|---:|---:|---|
| 10,000 | 0.022 s | 0.001 s | blocked BLAS 2.15 s; phenograph 22.6 s |
| 50,000 | 0.31 s | 0.006 s | blocked BLAS 40.7 s; phenograph 75.0 s |
| 100,000 | 1.02 s | 0.017 s | numpy matmul alone 26.6 s (the "approach it" target originally set for this stage) |
| 265,627 | **6.47 s** | 0.043 s | blocked BLAS **1098.7 s** (≈170×) |

- The original's writing `.ijv` + parsing `.clu` + building the graph (S1 + S3, 17.0 s on the full set) now takes 43 ms.
- On deduplicated Levine 50k: the original `bc()` graph build takes 29.0 s (phenograph 28.0 s of that), CosTaL+ 0.35 s.
- **The bottleneck has moved to Leiden**: on all 265,627 rows, graph build 7.6 s and Leiden 112 s; on the 161,443 cells after deduplication,
  graph build 3.0 s and Leiden 69 s. **Stage C (the `n_iterations` experiment) is now the most valuable next step.**

### 8.3 Tiles and matmul backends (N = 50k sweep, verifying docs/06 §2–§3)

- Accelerate is about 2.2× faster than pure-Rust `gemm` (0.43 s vs 0.93 s, each at its best tile).
- Tiles between 128 KB–2 MB form a plateau (< 15% apart); ≤ 64 KB is clearly slower (fixed overhead per matmul call).
- The cache-derived default (4 MB of L2 per thread → 2 MB tile, 512 rows × 1024 columns) was the fastest setting in the sweep: 0.42 s.
- The combined cost per candidate is about 3 cycles (including matmul; a rough estimate assuming 8 cores at 3.2 GHz). How that splits between matmul and the scan
  **was not measured separately**; docs/04 §1's "about 1 cycle/candidate" is still an estimate.

### 8.4 Duplicate cells in Levine_32dim (docs/02 addendum)

In the upstream file every gated cell appears twice (104,184 pairs); the true cell count is 161,443.

- The original `bc()`'s `idx[:, 1:]` breaks on duplicate cells (docs/04 §4 item 4); without deduplication, the differences from the original
  all come from this and from exact ties between the copies — **not a single row is a genuine neighbor difference**.
- **Duplicates inflate the accuracy evaluation** (full set, CosTaL default parameters, 5 Leiden seeds):

| Data | Cells | ARI (vs gating labels) | NMI |
|---|---:|---:|---:|
| Deduplicated | 161,443 | 0.766 ± 0.008 | 0.860 ± 0.004 |
| Raw file (with duplicates) | 265,627 | 0.861 ± 0.029 | 0.880 ± 0.015 |

ARI is inflated by about 0.095, far beyond the seed-to-seed variation. A plausible explanation: the cells being scored are exactly the ones that were duplicated,
so the density of their populations was doubled. **Only CosTaL was measured here**; whether other methods are inflated the same way on this file, and whether the ranking
would change, was not measured. From now on, CosTaL+ accuracy evaluations always use the deduplicated data (docs/02 addendum).

**Qualified (2026-10-12, `bench/out/stage_c3.json`):** the inflation depends on the setting. It is leidenalg at
the paper's resolution 0.8. With igraph's Leiden at 0.8 the raw file scores 0.80 against 0.79 deduplicated
(cosine or Tanimoto weights), and at the best resolution (0.2) both score 0.97.

### 8.5 x86-64 Linux and the original `p_knng` (GitHub Actions, 2026-10-12)

Run on GitHub's x86-64 runners (2 vCPUs, pure-Rust `gemm` backend) by the `bench` and `golden`
workflows; raw numbers in `bench/out/stage_a.linux-x86_64.json`, `bench/out/golden_pknng.json`
and `bench/out/pknng_analysis.json`.

**Correctness carries over to x86.** Bit-identical to the f64 brute force at N = 10k and 20k. On
deduplicated Levine 50k, neighbor sets are 100% identical to the original `bc()`, same-seed Leiden
partitions are identical (ARI = 1.0 for all 5 seeds), and the scores against the gating labels are
exactly the M1's (ARI 0.804 ± 0.073) — the graph is identical end to end across the two platforms
and matmul backends, as docs/06 §1 requires. CI also passes on macOS, Linux and Windows.

| N | Exact kNN (x86, 2 vCPUs, gemm) | M1 Pro (8 cores, Accelerate) |
|---:|---:|---:|
| 10,000 | 0.15 s | 0.022 s |
| 50,000 | 2.91 s | 0.31 s |
| 100,000 | 11.0 s | 1.02 s |
| 265,627 | 73.7 s | 6.47 s |

On the same x86 runner, the original `bc()` graph build at deduplicated 50k takes 37.7 s
(phenograph 36.5 s) versus 3.2 s for CosTaL+.

**Cache detection fails on these VMs.** `l2_bytes_per_thread()` returned `None`, so tiles fell back
to the 128 KB floor (32 rows); the sweep's best tile (256 × 256, 256 KB) is ~20% faster
(2.41 s vs 2.90 s at 50k). Detection via CPUID would cover both these VMs and Windows (docs/06 §2.3).
**Fixed the same day** with CPUID detection and square-ish tiles: a rerun detected 256 KB per thread and
its default tile (128 × 256) came within 2% of the sweep's best (2.98 s vs 2.93 s at 50k); that runner's
CPU was slower overall, so absolute times are not comparable across runs.

**The original `p_knng` is not exact on Levine** (mean recall@10 98.2%, 17% of rows differ), but the
effect on clustering is within Leiden's seed-to-seed noise. Details and the pending decision on
what `compat="paper"` pins: docs/04 §4 "Results".

---

## 9. Stage C and stage A+ results (2026-10-12)

> Same M1 Pro as §8 (swap in use). Deduplicated Levine (161,443 cells), the exact CosTaL graph
> (1,309,514 edges), leidenalg 0.12.0 / python-igraph 1.0.0, resolution 0.8, 5 seeds per variant.
> Script and raw numbers: `bench/stage_c.py`, `bench/out/stage_c.json`.

### 9.1 Stage C: how many Leiden iterations, and which implementation

| Implementation | n_iterations | s / run | Quality (RBConfiguration) | ARI vs paper, same seed | Seed-to-seed ARI | Label ARI |
|---|---:|---:|---:|---:|---:|---:|
| leidenalg | 1 | 16.1 | 2,165,219 | 0.943 | 0.840 | 0.766 ± 0.010 |
| leidenalg | 2 | 21.3 | 2,176,846 | **0.988** | 0.884 | 0.766 ± 0.009 |
| leidenalg | 3 | 32.4 | 2,177,111 | 0.990 | 0.887 | 0.766 ± 0.009 |
| leidenalg | 5 | 40.5 | 2,177,798 | 0.995 | 0.893 | 0.766 ± 0.009 |
| **leidenalg (paper)** | **10** | **65.5** | 2,178,405 | 1.000 | 0.898 | 0.766 ± 0.008 |
| leidenalg | until stable | 269.4 | 2,179,489 | 0.990 | 0.908 | 0.766 ± 0.008 |
| igraph | 2 | **1.7** | 2,173,157 | 0.871 | 0.847 | 0.793 ± 0.049 |
| igraph | 10 | 4.8 | 2,175,627 | 0.880 | 0.858 | 0.792 ± 0.049 |
| igraph | until stable | 15.8 | 2,176,500 | 0.883 | 0.864 | 0.792 ± 0.049 |

- **Two iterations reproduce the paper setting.** With the same seed, leidenalg at 2 iterations agrees
  with 10 iterations at ARI 0.988 — closer than the paper setting agrees with *itself* across seeds
  (0.898) — with identical accuracy and quality within 0.07%, in a third of the time. §5 guessed ~80%
  savings; the measured saving is 67%. Iterating until stable buys 0.05% more quality at 4× the cost.
- **igraph's native Leiden is 38× faster** than the paper setting (1.7 s vs 65.5 s) with no loss of
  accuracy against the labels (0.79 vs 0.77, within the seed spread), but its partitions differ more
  from leidenalg's (0.87–0.88) than leidenalg's do across seeds, and it reaches slightly lower quality.
  It is a different implementation, not a faster run of the same one.
- **Decision (under delegated authority).** `compat="paper"` keeps leidenalg with 10 iterations
  (docs/04 §0.0 item 5). When the cluster layer (`find_communities`) arrives, it gets a fast setting
  (igraph native, 2 iterations) as an explicit option; making it the default would be a default change
  and needs a major version bump (docs/04 §6.5). For resolution sweeps (docs/05 §2.3), igraph's speed
  is what makes them cheap.
- Leiden is now dominated by the choice of implementation, not by kNN: with igraph, the full pipeline on
  161,443 cells is about 3 s of graph building plus 1.7 s of clustering.

### 9.2 Stage A+: Tanimoto selection

`select="tanimoto"` (raw magnitudes; docs/04 §2) is implemented behind the same generic kernel:
a conservative per-block threshold chosen by the sign of cᵢ (§2.4, block extremes precomputed), T on the
rare path, f64 rescoring with the original's `tani_from_cos`, and an ε for the fallback derived for
Tanimoto scores. It is bit-identical to the Tanimoto brute force on synthetic data (all block sizes,
backends, layouts, thread counts; signed and tie-heavy data included) and on Levine 20k, and runs
3.4 s on 161,443 cells against 2.45 s for cosine.

### 9.3 Stage C′: the 2×2, the row norm, and rare populations

Details in docs/04 §5.1 and docs/05 §1.1; in short, on deduplicated Levine:

- Tanimoto selection replaces half of the neighbors but not the accuracy; Tanimoto and cosine weights
  are equal at their best resolution. The apparent 0.1-ARI lead of cosine weights at the paper's 0.8
  is a resolution effect specific to leidenalg. The row norm is mostly within-population variation
  (η² = 0.097). Defaults unchanged.
- The resolution dominates (ARI 0.45–0.97), and rare populations are systematically lost at any single
  resolution (Spearman(abundance, F1) up to +0.97) — the methodology track of docs/05 is justified.

### 9.4 Stage D: norm-window pruning (Tanimoto selection)

On the norm-sorted layout, each row block scans column blocks outward from its own norms and stops each
side once the norms leave every row's window `ρ + 1/ρ ≤ (1+τ)/τ` (docs/04 §3). Exact (bit-identical
with pruning on and off, and to the brute force).

| Data (161,443 cells) | Pruned | Unpruned | Speedup | Tiles skipped |
|---|---:|---:|---:|---:|
| Levine, deduplicated | 3.24 s | 4.32 s | 1.33× | 2.6% |
| Levine, rows rescaled by U(1, 20) | 1.58 s | 7.69 s | 4.9× | 55.5% |

On Levine the gain is mostly the outward scan order (heaps fill with high-T candidates early), not the
skipping, and 1.33× is below the §6 gate of 1.5×. **Kept anyway (delegated decision):** it is exact and
cheap, and it pays most on data whose magnitudes really vary, which is the data Tanimoto selection is
meant for. Separately, the norm-sorted layout itself is what makes Tanimoto selection fast at all:
without it, 161,443 cells take 34.9 s instead of 3.8 s (looser per-block thresholds). Cosine selection
is unaffected (no window; the cone bounds of docs/04 §3 remain to do).

## 10. The paper's other datasets (2026-10-13)

> Loaders: `bench/datasets.py` (raw files on the external drive, `$COSTAL_DATA`; sources listed in the
> module). Experiments: `bench/small_datasets.py`, numbers in `bench/out/small_datasets.json`.
> Chosen by size, following the author's request to stay small: the four smallest scRNA-seq sets of
> the paper's Table 2 and Levine_13; GSE110823 and 1M_neurons wait for the sparse backend (stage S),
> Samusik / Nilsson / Mosmann / Giordani are not available offline.

### 10.1 The original preprocessing is reproduced exactly

`preprocess()` is the original `pp()` (`filter_cells(min_genes=200)` → `filter_genes(min_cells=3)` →
2000 HVGs by `seurat_v3` → `log1p`), with scanpy 1.10.3 / scikit-misc 0.3.1. Against the paper's Table 2:

| Dataset | Cells | Original features | Non-zeros (ours / paper) | Labels |
|---|---:|---:|---|---|
| E-MTAB-3321 (Goolam) | 124 | 41,480 | 144,499 / 144,499 ✓ | 5 stages (SDRF) |
| GSE81861 (Li, cell lines) | 561 | 57,241 | 213,416 / 214,688 (−0.6%) | 7 cell lines |
| GSE74672 (Romanov) | 2,881 | 24,341 | 821,520 / 821,520 ✓ | 7 `level1 class` |
| GSE84133 (Baron, human3) | 3,605 | 20,125 | 792,390 / 792,390 ✓ | 14 `assigned_cluster` |
| Levine_13 | 167,044 | 13 | 2,171,567 / 1,689,738 | 24 gated (81,747 cells) |

- Three of four scRNA-seq sets match to the entry, so the paper's preprocessing, gene filter and HVG
  selection are reproduced, and so are its cell counts (no cell fails `filter_cells`; the original
  `pp()` returns only the matrix, which would misalign labels if one did — the loader tracks it).
- GSE81861 differs at the HVG boundary (some of the 2000 genes differ; not a dtype effect — float32 / float64,
  dense / sparse give the same set; RSEM's non-integer counts may meet `seurat_v3` differently across
  scanpy versions).
- **Levine_13 is open.** The notransform file has 29.26% negative entries and no zeros after
  arcsinh, where the paper reports a sparsity of 22.188%; no threshold or rounding of this file gives
  that number (for Levine_32 the paper's 26.999% is exactly the share of negatives, docs/04 §4). The
  paper probably used another version of the file. **Question for the author.**
  *Update (2026-10-13):* Table 2's "non-zeros" count positive entries on all four other cytometry sets
  (Levine_32, Samusik_01, Nilsson_rare, Mosmann_rare: exact matches), so for Levine_13 the paper's
  1,689,738 positives against our 1,536,151 mean different raw values, not a different transform (any
  arcsinh keeps the sign). HDCytoData's `Levine_13dim_SE` (ExperimentHub EH2242) is identical to the
  GitHub `Levine_13dim_notransform.fcs` value for value, so it is not that either; nor is the
  pre-transformed `Levine_13dim.fcs` of the same GitHub repository (checked 2026-10-13: it is
  `arcsinh(raw/5)` to 5×10⁻⁷, same labels, the same 1,536,151 positives). None of the three public
  versions gives the paper's 1,689,738, so the paper's Levine_13 matrix came from somewhere else
  (another export, or a different intermediate). **Closed (2026-10-13):** the author does not recall
  the source, and no public version reproduces it. It does not affect any comparison here, all of
  which run both methods on the same matrix.

### 10.2 Parity with the original on scRNA-seq

On the paper preset, the original graph-building code (the `bc()` path) and CosTaL+ give **100%
identical neighbor sets** on all four scRNA-seq sets, edge weights equal within 5×10⁻⁷ (float32
input), and **identical Leiden partitions for every seed** at the paper's settings (resolution 0.8,
10 iterations, 5 seeds; same-seed ARI 1.000 everywhere). Together with the golden run (docs/04 §4:
`p_knng` is exact on non-negative input, and all of these are non-negative), CosTaL+ reproduces the
published scRNA-seq pipeline.

### 10.3 Accuracy of the published method

ARI against the labels (labeled cells only) for the published combination (cosine selection,
Tanimoto weights), at the paper's resolution 0.8 and at the best resolution of the grid 0.05–2.0
(`bench/out/small_datasets.json`; Levine_32 from §9 / `stage_c3.json`, deduplicated):

| Dataset | ARI at 0.8 | Best ARI (resolution) |
|---|---|---|
| Levine_32 | 0.766 | 0.968 (0.2) |
| Levine_13 | 0.799 ± 0.053 | 0.846 (1.0) |
| E-MTAB-3321 | 0.687 | 0.883 (0.05) |
| GSE81861 | 1.000 | 1.000 |
| GSE74672 | 0.473 ± 0.027 | 0.662 (0.1) |
| GSE84133 | 0.857 ± 0.007 | 0.938 (0.2) |

The paper's fixed 0.8 is far from the best resolution on four of six sets; the best resolution ranges
over 0.05–1.0 (docs/05 is about why no single value can be right). What the 2×2, the row norm and the
presets do to these numbers is in docs/04 §5.1 and §6.7.

### 10.4 End-to-end time at identical results

`bench/timing_small.py` (rewritten 2026-10-13 to check equality in the same run), idle M1 Pro,
k = 10, best of 3 under 100k cells, one run above. "Original" is the `bc()` path that CosTaL takes
on macOS (phenograph's brute-force cosine kNN + `map_tani` + `build_graph`), **not** the Linux
`p_knng` path the paper timed (docs/02 §1.1); Leiden at the paper's settings for both. Same neighbors:
share of cells whose neighbor sets are identical; accuracy: ARI on labeled cells, or the rare
population's best-match F1 for the one-population sets.

| Dataset | Cells × features | Original | CosTaL+ | CosTaL+ `leiden="fast"` | Graph only | Same neighbors | Same-seed partition ARI | Accuracy (original / CosTaL+ / fast) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| E-MTAB-3321 | 124 × 2,000 | 0.06 s | 0.02 s (3.1×) | 0.004 s (17×) | 13× | 100% | 1.000 | ARI 0.687 / 0.687 / 0.687 |
| GSE81861 | 561 × 2,000 | 1.0 s | 0.11 s (9.1×) | 0.02 s (41×) | 40× | 100% | 1.000 | ARI 1.000 / 1.000 / 1.000 |
| GSE74672 | 2,881 × 2,000 | 26 s | 0.76 s (35×) | 0.13 s (197×) | 205× | 100% | 1.000 | ARI 0.495 / 0.495 / 0.492 |
| GSE84133 | 3,605 × 2,000 | 41 s | 1.0 s (40×) | 0.21 s (195×) | 208× | 100% | 1.000 | ARI 0.845 / 0.845 / 0.851 |
| Nilsson_rare | 44,140 × 14 | 32 s | 12 s (2.7×) | 0.90 s (36×) | 31× | 100% | 0.610 | HSC F1 0.152 / 0.153 / 0.189 |
| Samusik_01 | 86,864 × 39 | 91 s | 32 s (2.9×) | 2.3 s (39×) | 39× | 100% | 1.000 | ARI 0.843 / 0.843 / 0.808 |
| Levine_13 | 167,044 × 13 | 165 s | 69 s (2.4×) | 5.8 s (28×) | 26× | 100% | 0.961 | ARI 0.849 / 0.873 / 0.846 |
| Levine_32 (dedup.) | 161,443 × 32 | 300 s | 69 s (4.3×) | 5.0 s (60×) | 45× | 100% | 1.000 | ARI 0.756 / 0.756 / 0.758 |
| Mosmann_rare | 396,460 × 15 | 1,826 s | 587 s (3.1×) | 61 s (30×) | 73× | 100% | 0.916 | activated F1 0.020 / 0.019 / 0.038 |

- **Same neighbors everywhere, same partitions on six of nine.** On Nilsson_rare the two graphs have
  identical edge lists (324,397 edges) and every weight differs by at most 5.7×10⁻⁷ (relative
  5.8×10⁻⁷): the original computes Tanimoto in float32 from the float32 rows, CosTaL+ in float64.
  Leiden amplifies that rounding into a different partition at the same seed (ARI 0.61), at the same
  accuracy (HSC F1 0.152 vs 0.153). Levine_13 and Mosmann_rare are the same case (0.96, 0.92).
- **Where the time goes.** The graph is 13–208× faster. With the paper's Leiden settings, leidenalg
  is 79–97% of CosTaL+'s time (95–97% on the large cytometry sets), so end to end it is 2.4–4.3× on
  those and 3–40× on scRNA-seq; with `leiden="fast"` it is 17–197× at comparable accuracy (one seed
  each; Samusik_01 0.808 vs 0.843 is within the ±0.05 seed-to-seed spread igraph showed on Levine,
  §9.1).
- Preprocessing (scanpy, the same for both) takes 0.3–5 s on the scRNA-seq sets.

The earlier version of this table (same day, five datasets, no equality check) is superseded.
