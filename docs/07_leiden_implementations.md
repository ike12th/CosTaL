# Leiden implementations: leidenalg, igraph, and a native Leiden for CosTaL (v0, 2026-10-14)

> Background: with the kNN graph in Rust, Leiden is now most of CosTaL 2.0's run time. At the paper's
> settings leidenalg takes 80–97% of it (README, "Speed at identical results"). This note answers three
> questions: **why is leidenalg so much slower than igraph, how does igraph's Leiden spend its time, and
> what would a Leiden written for CosTaL have to do to beat it?**
>
> Sources read at the tags that CosTaL's environment installs: leidenalg `0.12.0` and its C++ core
> libleidenalg `0.13.0`; igraph C core `1.0.0` and python-igraph `1.0.0` (which vendors exactly that
> core, commit `b9b5739`). For comparison: the Leiden authors' Java reference, CWTSLeiden/networkanalysis
> (`master`). References read `repo@tag:path:line`; LA = vtraag/leidenalg@0.12.0, LL = vtraag/libleidenalg@0.13.0,
> IG = igraph/igraph@1.0.0, PIG = igraph/python-igraph@1.0.0.
>
> **Licenses.** leidenalg is GPL-3 and igraph GPL-2.0-or-later; CosTaL is MIT. CosTaL calls them as
> separately installed packages and ships none of their code. A Leiden written for CosTaL must therefore
> be a clean-room implementation: from the paper (Traag, Waltman & van Eck 2019, Sci Rep 9:5233,
> Algorithm A.2) and the Java reference, which is MIT-licensed. It must not be a translation of
> leidenalg or igraph. These notes describe their behavior so that we know what to match and what to
> avoid, not code to port.

---

## 1. Measurements (Levine_32, Apple M1 Pro)

161,443 deduplicated cells × 32 markers, k = 10, Tanimoto weights, fuzzy union: 1,309,514 undirected
edges. Resolution 0.8, seeds 0 and 1.

| Step | Time |
|---|---:|
| exact cosine kNN (Rust) | 2.30 s |
| Tanimoto weights + fuzzy union (Rust) | 0.02 s |
| `import igraph` on first use in a process | ≈ 0.5 s (1.2–1.4 s from a cold disk cache) |
| `_igraph()`: python-igraph `Graph` from the edge arrays | 0.40 s |
| igraph `community_leiden`, 2 iterations | 1.5 s |
| igraph `community_leiden`, 10 iterations | 5.2–6.8 s |
| leidenalg `find_partition`, 2 iterations | 17.8–20.2 s |
| leidenalg `find_partition`, 10 iterations | 57.2–59.9 s |

Accuracy against the gated labels (ARI, labeled cells): leidenalg 0.755–0.776 at both 2 and 10
iterations; igraph 0.758–0.771 at both. Two consequences:

- **At equal iterations igraph is about 12× faster than leidenalg.** Most of `leiden="fast"`'s speed
  comes from the implementation, not from running 2 iterations instead of 10 (that accounts for
  about 3×).
- **leidenalg with `n_iterations=2` is 3× faster than the paper's 10 at the same accuracy.** It is
  available today as `find_communities(X, n_iterations=2)`, but it departs from the published setting.

An earlier quick measurement put "graph build" at 0.83 s. That figure included the first
`import igraph` (`_igraph` imports it lazily); the build itself is 0.40 s.

---

## 2. Why leidenalg is slow

### 2.1 The Linux and macOS wheels are compiled without optimization

- **Evidence (macOS arm64 wheel, verified).** In a `sample` profile of `find_partition`, 65% of busy
  samples are in out-of-line libc++ helpers (`vector::operator[]`, `size()`, `push_back`,
  `vector<bool>` references, `__fill_n`). An optimizing compiler inlines these into single
  instructions. The disassembly of the one-line `MutableVertexPartition::csize` spills its arguments
  to the stack and calls `vector::size()` and `operator[]`: this is -O0 code generation.
- **Cause (verified).** `LA:scripts/build_libleidenalg.sh` configures libleidenalg with no
  `CMAKE_BUILD_TYPE`, and `LL:CMakeLists.txt` sets no default. Linux and macOS use single-config
  generators (Makefiles), where the script's `cmake --build . --config Release` has no effect. The
  sibling `build_igraph.sh` passes `-DCMAKE_BUILD_TYPE=Release`, so the igraph core inside the same
  wheel is optimized. Windows uses the multi-config Visual Studio generator, where `--config Release`
  applies, so the Windows wheels are optimized.
- **Upstream (verified).** Reported on 2026-10-14 by another developer in vtraag/leidenalg#242, with
  four open PRs: leidenalg#241 (Release in the build script), libleidenalg#26 (Release by default),
  #27 (remove a per-call allocation in the refinement), #28 (CSR adjacency). They report identical
  memberships and about 4× from the build fix alone, 5–6× with all four, on a 20k-node graph.
- **Expected effect on CosTaL:** about 3.5–5× faster `leiden="paper"` with identical partitions once a
  fixed leidenalg is released, with no change in CosTaL. Levine_32 estimate: 4–6 s at 2 iterations,
  12–17 s at 10 (still 2–3× igraph).
- The library file is named `liblibleidenalg.0.11.1.dylib` but contains libleidenalg 0.13.0 code:
  `LL:etc/cmake/version.cmake` runs `git describe` without `--tags`, and the 0.12.0/0.13.0 tags are
  lightweight.

### 2.2 Design costs that an optimized build keeps

- **Two neighbor scans per node visit.** For an undirected graph, separate caches for "all" and "out"
  neighbor communities (`LL:src/MutableVertexPartition.cpp:799-899`) each rescan the adjacency.
  `move_node` then loops over the neighbors twice more, to maintain identical "to" and "from" totals.
  That is about 5–6 edge touches per neighbor per moved visit, against about 2 in igraph.
- **One candidate entry per edge, not per community.** `cache_neigh_communities` checks for a nonzero
  accumulated weight *after* adding, so every edge is pushed, duplicates included (`:861-862`).
- **An unused O(n_communities) allocation per refined node.** `get_neigh_comms(v, mode, constrained)`
  zero-fills a `vector<bool>(n_communities)` on every call and never uses it (`comm_added[comm];` is
  a no-op, `:904`). At level 0 every node is its own community, so this costs O(n²) per iteration:
  about 407 M word stores on Levine_32. PR #27 removes it.
- **Adjacency copied from igraph on every cache miss.** Neighbor queries go through igraph's lazy
  lists and are copied into a one-node `std::vector` cache (`LL:src/GraphHelper.cpp:493-568`).
- **Per level:** `init_admin` over the full original graph, a new igraph graph through `igraph_create`
  (aggregates can hold parallel edges), size-sorted renumbering with a heap allocation per community.

### 2.3 leidenalg 0.12.0 changed its refinement: a reproducibility issue

- libleidenalg 0.13.0 changed the default `refine_consider_comms` from `ALL_NEIGH_COMMS` (merge each
  singleton into the best neighboring sub-community with gain ≥ 0) to `RAND_NEIGH_COMM` (draw one
  neighbor edge at random and accept its sub-community if the gain is ≥ 0) (`LL:src/Optimiser.cpp:20`;
  `ALL_NEIGH_COMMS` at libleidenalg 0.11.1 and 0.12.0, verified).
- leidenalg 0.11.0 bundles libleidenalg 0.12.0 and leidenalg 0.12.0 bundles 0.13.0 (verified in
  `scripts/build_libleidenalg.sh` at both tags). **The same seed gives different partitions under
  leidenalg ≤ 0.11 and 0.12.** CosTaL 1.0's published results predate 0.12.
- CosTaL's own 1.0-vs-2.0 parity results (docs/03 §10) ran both versions against the same installed
  leidenalg, so they are unaffected.
- Done: `costal info` prints the leidenalg and igraph versions, and the user guide's
  reproducibility section asks users to record them.
- Neither leidenalg version has the paper's well-connectedness tests or its randomized selection
  ∝ exp(ΔH/θ) in the refinement. libleidenalg#24 (open) raises this.

---

## 3. How igraph's Leiden runs, and where its time goes

### 3.1 Call path

- `PIG:src/igraph/community.py:455-537`: `objective_function="modularity"` sets
  `normalize_resolution=True`. The comparison is case-sensitive (`:523`): `"Modularity"` would silently
  skip the normalization. CosTaL passes lower case.
- `PIG:src/_igraph/graphobject.c:13829-13852`: node weights = strength (self-loops excluded), and
  γ' = γ / Σ strength = γ / 2m. **igraph's modularity `resolution` is the same γ as leidenalg's
  `resolution_parameter`**: the objectives are equal up to the factor 1/2m (`LL:src/RBConfigurationVertexPartition.cpp:47,100-160`).
- β = 0.01 by default; the start is singletons. A negative `n_iterations` means "until an iteration
  moves nothing"; `n_iterations=0` returns singletons (`IG:src/community/leiden.c:1253-1261`).
- `PIG` never releases the GIL, and **its RNG is Python's `random` module**: every draw is a Python
  call (`PIG:src/_igraph/random.c:167-237`). Measured cost about 6%. Two Leiden runs in Python threads
  therefore serialize; only processes run in parallel, and each needs its own copy of the graph.
- The C core is built Release (`-O3`) with LTO (`PIG:setup.py:287`; `IG:etc/cmake/BuildType.cmake`).
  `leiden.c` has no OpenMP.

### 3.2 One iteration (`IG:src/community/leiden.c:803-1022`)

1. **Fast local moving** (`:50-240`): vertices in a shuffled FIFO queue. Candidates: one empty cluster
   (always offered), then each neighbor cluster, found with a dense accumulator, a touched list and
   a bitset (no hashing, no allocation per neighbor). A vertex moves only on a strictly positive gain.
   When a vertex moves, its stable neighbors outside the new cluster are re-queued.
2. **Refinement** (`:318-544`), once per cluster, sequentially. Every vertex starts as a singleton
   and is visited once in random order. It may move only if it is still a singleton and *well
   connected* to its cluster: E(v, S−v) ≥ γ' k_v (K_S − k_v). Target sub-clusters must also be well
   connected. Among candidates with ΔH ≥ 0, one is drawn with probability ∝ exp(ΔH/β). If the sum of
   exponentials overflows (ΔH ≳ 7 in raw weight units, which happens at aggregated levels), the best
   candidate is taken greedily.
3. **Aggregation** (`:584-700`): one aggregate node per refined cluster. Edges are summed with the
   same accumulator, internal weight is dropped, and the aggregate starts from the *unrefined*
   partition (the Leiden rule). The edge list then becomes a new `igraph_t` through `igraph_create`,
   which sorts it twice (`igraph_vector_int_pair_order`). If the refinement merged nothing, the level
   aggregates by the unrefined partition instead. Levels continue while the number of clusters is
   smaller than the number of nodes.
4. **Quality** is recomputed at the end of every iteration (`:1013`), although only the last value is
   returned.

### 3.3 Where the time goes

- **Memory access per neighbor visit.** The incidence list gives an int64 edge id. The other endpoint
  comes from `to[e]`/`from[e]` (int64, random), the weight from `weights[e]` (double, random, by edge
  id), then `membership[u]` (int64, random) and the accumulator. That is 3–4 cache lines per entry,
  against 8 bytes streamed plus 4 bytes random for an interleaved (u32 neighbor, f32 weight) CSR.
- **Rebuilt every level of every iteration:** the incidence list (n allocations, `:885`), the
  aggregate `igraph_t` with its two sorts, and the per-iteration quality pass.
- **Refinement makes three passes** over each vertex's full adjacency and filters out the neighbors
  outside the cluster.
- **Iterations after the first cost about the same as the first** (0.65–0.7 s each on a synthetic
  graph of this size), because each one redoes all of level 0 even when the partition no longer changes.
- **Profile shares** (sampled, 2 runs × 10 iterations): 85% inlined into `igraph_community_leiden`;
  4.8% `pair_order` (aggregation); 2% `exp` (refinement); 1.7% `igraph_incident` (incidence-list rebuilds);
  1.2% binary search (refinement sampling); 2.3% malloc/free.

### 3.4 A deviation from the reference refinement (verified)

When a vertex is taken out of its singleton for evaluation, igraph zeroes the singleton's node weight
and count but not its external edge weight (`IG:src/community/leiden.c:414-420`). The Java reference
zeroes both (`LocalMergingAlgorithm.java`, "Remove the currently selected node from its current
cluster"). If the vertex then stays, its external weight is added again and ends up doubled. That
makes later well-connectedness tests on that sub-cluster easier to pass than the paper intends. A
Leiden for CosTaL follows the reference here. This could be reported to igraph; nothing has been posted.

---

## 4. Semantic differences at a glance

| | Paper (Alg. A.2) / Java reference | igraph 1.0 | leidenalg 0.12.0 |
|---|---|---|---|
| Objective (CosTaL's call) | modularity with γ | same; γ/2m internally | same, unnormalized |
| Empty cluster in local moving | always a candidate | always, first | only if the node is not alone; last |
| Move threshold | gain > 0 | gain > 0 | gain > 10·DBL_EPSILON |
| Refinement eligibility | singleton and well connected | same | singleton only |
| Refinement target | well connected, ∝ exp(ΔH/θ) over ΔH ≥ 0 | same, β = 0.01; greedy on overflow | one random neighbor edge; skip with prob. 1/(k+1); accept if ΔH ≥ 0 |
| External weight of the emptied singleton | reset | not reset (§3.4) | n/a |
| Refinement merged nothing | aggregate the unrefined partition | same | stop |
| RNG | own generator | Python `random` via python-igraph | its own MT19937 (`seed`; 0 behaves as 4357) |
| Final labels | — | first appearance | sorted by community size |

The two implementations optimize the same function but take different paths, so their partitions
differ. On Levine_32 their accuracy is the same (§1).

---

## 5. A native Leiden for CosTaL: proposed design (not started)

Goal: `leiden="fast"` and multiscale faster than igraph, at igraph's accuracy, with the same seed
giving the same partition on every platform and at every thread count. `leiden="paper"` keeps calling
leidenalg: bit-identical leidenalg partitions would require reproducing its RNG, candidate order,
renumbering and multigraph aggregation, which means following its GPL code line by line.

**Data.**
- Read the fuzzy-union output directly as a CSR built once per run: u32 neighbor, f32 weight
  (Tanimoto ∈ (0, 1]), f64 accumulators. No igraph object, no Python lists.
- Drop zero-weight edges. A strength-0 vertex can otherwise be merged at random during refinement.
- Reject a total weight of 0, non-finite weights and self-loops with a clear error.

**Local moving.**
- Sequential FIFO queue over a shuffled order. Candidate clusters found with a dense accumulator, a
  touched list and an epoch-stamped array (no clearing pass). One scan of the adjacency per visit.
- An empty cluster is always a candidate, and a vertex moves only on a strictly positive gain.
- RNG: a counter-based or small-state generator in Rust (e.g. PCG or Xoshiro), seeded from
  (seed, iteration, level). No Python calls.

**Refinement (paper and Java reference).**
- Well-connectedness tests for both the vertex and its target, and sampling ∝ exp((ΔH − ΔH_max)/β).
  Shifting by the maximum avoids overflow and keeps the sampling defined at every level, where igraph
  falls back to greedy.
- The emptied singleton's external weight is reset, as in the reference.
- Scan only in-cluster neighbors: one pass gathers them per cluster.
- Update the external weight from the accumulator, not with a third pass.
- **Parallel and still deterministic.** Clusters are independent given the unrefined partition. Each
  is refined with its own RNG stream derived from (seed, iteration, level, cluster id), and refined
  ids are numbered by a prefix sum in cluster order. The result does not depend on the thread count.

**Aggregation.**
- Counting sort of the vertices by refined cluster, then each aggregate row filled with a dense
  accumulator straight into the next level's CSR.
- No edge-list sort, no re-indexing, parallel by rows, deterministic.
- Node weight = Σ strength. The internal weight is dropped (modularity does not need it).

**Iterations.**
- `n_iterations=k` runs exactly k iterations; a negative value runs until no vertex moves.
- No quality computation unless asked for.
- Canonical labels: first appearance in the original vertex order.

**Many runs.**
- `find_communities_multiscale`'s 11 resolutions × 4 seeds run as parallel jobs on one shared,
  read-only CSR, with the GIL released.
- This is not possible with python-igraph (GIL held, Python RNG) without copying the graph into
  separate processes.

**Not planned:** parallel local moving (GVE-Leiden style). It changes the trajectory with the thread
schedule, and CosTaL's graphs are small enough that the sequential version is not the bottleneck.

### 5.1 Expected gains (estimates, to be measured)

| | igraph today (Levine_32) | Native, estimated |
|---|---:|---:|
| Graph hand-off | 0.40 s (+ ≈ 0.5 s first import) | ≈ 0 |
| Leiden, 2 iterations | 1.5 s | 0.3–0.8 s (layout, no rebuilds; refinement in parallel) |
| `leiden="fast"` end to end | ≈ 4.2–4.7 s | ≈ 2.7–3.2 s (the kNN's 2.3 s stays) |
| Multiscale (44 runs + carving) | 110 s | ≈ 15–25 s (runs in parallel on 8 cores) |

The single-run gain is modest because the kNN is now about half of the run. The multiscale gain is
the main reason to do this.

### 5.2 Validation before it can become the default

1. **Quality.** On the 9 datasets of docs/03 §10, with 20 seeds each: modularity distribution at
   least igraph's, and ARI and per-population F1 against the labels within igraph's seed-to-seed
   spread.
2. **Guarantees.** Every cluster is connected (the Leiden guarantee). The null check of docs/05 §3.3
   still gives one cluster.
3. **Determinism.** Identical labels for 1, 2, 8 and 16 threads, and on macOS, Linux and Windows
   (as for the kNN, docs/06).
4. **Edge cases.** Isolated vertices, disconnected components, uniform weights, n = 1–3, a graph
   with a single edge.
5. **Speed and memory.** Faster than igraph per iteration on every dataset; peak memory no more than
   the CSR plus O(n).

Rollout:
- Ship as a new `leiden=` option first.
- Make it the default for `"fast"` and multiscale only after (1)–(5) pass.
- `"paper"` stays leidenalg.
