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

- **At equal iterations igraph is about 12× faster than leidenalg.** Most of igraph's speed
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
Leiden for CosTaL follows the reference here. In practice it changes no partition on Nilsson_rare or
Levine_32 (§5.2): a node allowed to move has already passed the same test that a target must pass.
It could still be reported to igraph; nothing has been posted.

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

## 5. CosTaL's own Leiden: `leiden="fast"` (implemented 2026-10-14)

`crates/costal-core/src/leiden.rs`; `leiden="fast"` in `find_communities` and (the default)
`find_communities_multiscale`. igraph's implementation stays available as `leiden="igraph"`, and
`leiden="paper"` keeps calling leidenalg: bit-identical leidenalg partitions would require
reproducing its RNG, candidate order, renumbering and multigraph aggregation, i.e. following its GPL
code line by line.

### 5.1 Design

**Algorithm.** The paper's Algorithm A.2, with the details the paper leaves open taken from the
authors' MIT-licensed Java reference (`FastLocalMovingAlgorithm`, `LocalMergingAlgorithm`,
`LeidenAlgorithm`):
- modularity as CPM on node weights k_i (strength) with γ/2m, so γ means what it means in leidenalg
  and igraph;
- fast local moving from a queue in random order, with one empty cluster always a candidate and
  moves only on a strictly positive gain;
- refinement inside each cluster from singletons, with the node and target well-connectedness tests
  and sampling ∝ exp(gain/β), β = 0.01 in edge-weight units as in igraph;
- aggregation on the refined partition, the next level starting from the unrefined one, and the
  unrefined partition aggregated when the refinement merges nothing;
- `n_iterations=k` runs k iterations; a negative value runs until an iteration moves nothing, capped
  at 1,000.

**Data.** The fuzzy-union edge list becomes one CSR (u32 neighbors, f64 weights) with no igraph
object and no Python lists. Zero-weight edges are dropped; self-loops, out-of-range nodes and
negative or non-finite weights are errors. A graph without edges returns singletons.

**Determinism.** The same seed gives the same partition at any thread count and on every platform:
- local moving is sequential, with xoshiro256** seeded from (seed, iteration, level);
- the clusters are refined in parallel, each with its own stream (seed, iteration, level, cluster),
  and refined ids are numbered by (cluster, sub-cluster);
- aggregate rows are built independently and sum duplicate edges in member order;
- `exp` is our own polynomial, because `libm` results can differ in the last bit.

**Many runs.** `leiden_many_into` runs the multiscale ensemble (11 resolutions × 4 seeds) on one
shared graph:
- one task per thread takes the runs in turn, each run sequential inside; with more threads than
  runs (a large server), the spare threads share each run's refinement and aggregation;
- each run writes its membership straight into its row of the output;
- levels under 5,000 nodes are always sequential (rayon's dispatch costs about 1 ms per run on a
  500-node graph).

**Memory.** Every buffer lives in a per-thread `Workspace` that only grows, and is reused across
levels, iterations and runs:
- **Why reuse.** Allocating per level fragmented the heap when runs at different resolutions
  followed each other. On Levine_32, 44 runs on one thread peaked at 618 MB above the edge list,
  growing about 11 MB per run. 44 identical runs did not grow, so this was fragmentation, not a
  leak.
- **Exact sizes.** Aggregate graphs are counted before they are filled, so they are allocated at
  their exact size; grown by doubling, the reused graphs kept up to twice what they needed.
- **No subgraph copy.** The refinement reads each cluster's edges in place rather than copying
  the cluster's subgraph, which at low resolutions is nearly the whole graph, once per thread.
- **Result.** Peak above the edge list for the 44 runs on Levine_32 is 109 MB on 1 thread, 267 MB
  on 4 and 584 MB on 10, about 48 MB per concurrent run. `n_threads` trades time for memory.
- **No cap by default.** One run per thread, whatever the graph's size: CosTaL targets servers
  where memory and cores are plentiful (decided 2026-10-14). On a smaller machine `n_threads`
  lowers both. The exact-size counting pass costs nothing measurable (44 runs on Levine_32: best
  6.66 s without it, 6.67 s with it), so it stays.
- **Results unchanged.** None of this changes a result: the digests of `tests/test_leiden_native.py`
  are the same before and after.

**Not done:** parallel local moving. leiden-rs's colouring scheme changes the algorithm (§6.2); a
speculative window (evaluate about 1,000 queued nodes in parallel, commit them in queue order after
re-validating) would reproduce the sequential result exactly, if profiling ever shows local moving
dominating.

Changes made after reading the other Rust implementations (§6): an emptied cluster's weight is reset
to exactly 0 (rounding residue let nodes hop into "empty" clusters, which slowed `n_iterations<0`:
43–47 iterations to converge on Levine_32 before, 29–33 after); the iteration cap went from 10,000 to
1,000 and a `converged` flag was added; the refinement makes one pass over each cluster's own edges;
`Csr` is only built through its validating constructor; the sampling never falls on an ineligible
candidate.

### 5.2 Validation

**Rust unit tests** (`leiden.rs`, 17): planted partitions, a ring of cliques, two triangles; every
cluster connected and no two adjacent clusters gaining from a merge after every run (Leiden's
γ-connectivity and γ-separation); quality non-decreasing with iterations; after `n_iterations<0`, no
single node gaining from a move; the best of 10 seeds reaching the exact optimum (all 4,140
partitions of 8 nodes) on at least 90% of 40 random graphs; scaling weights and β by 4 leaving the
partition bit-identical; identical partitions at 1, 2, 3 and 8 threads on a 10,000-node weighted
graph; `leiden_many` equal to single runs; edge cases.

**Python tests** (`tests/test_leiden_native.py`): modularity equal to igraph's to 1e-12; optimization
at least as good as igraph's over 10 seeds; the public API; and two fixed partitions checked by
digest, so that the wheel tests on macOS, Linux and Windows catch any platform difference.

**Against igraph on the paper's nine datasets** (`bench/leiden_native.py`,
`bench/out/leiden_native.json`; resolution 0.8, 2 iterations, 20 seeds each; times exclude
igraph's graph conversion, which costs another 0.4 s on Levine_32):

| Dataset | Cells | igraph | native | Modularity igraph / native | ARI igraph / native |
|---|---:|---:|---:|---|---|
| GSE84133 | 3,605 | 0.009 s | 0.006 s | 0.8128 / 0.8128 | 0.854 / 0.857 |
| GSE74672 | 2,881 | 0.007 s | 0.005 s | 0.8183 / 0.8180 | 0.475 / 0.463 |
| GSE81861, E-MTAB-3321 | 561, 124 | ≤ 1 ms | ≤ 6 ms | identical | identical |
| Nilsson_rare | 44,140 | 0.200 s | 0.077 s | 0.8627 / 0.8610 | 0.001 / 0.001 |
| Samusik_01 | 86,864 | 0.731 s | 0.222 s | 0.8685 / 0.8683 | 0.837 / 0.837 |
| Levine_32 | 161,443 | 1.463 s | 0.521 s | 0.8664 / 0.8661 | 0.775 / 0.781 |
| Levine_13 | 167,044 | 1.402 s | 0.528 s | 0.8624 / 0.8619 | 0.818 / 0.804 |
| Mosmann_rare | 396,460 | 5.760 s | 2.174 s | 0.8369 / 0.8365 | 0.000 / 0.000 |

- **Speed:** 2.6–3.3× igraph per run on the cytometry sets.
- **Quality:** the native modularity came out slightly lower on 7 of 7 sets with a difference
  (0.0001–0.0017, each within one seed-to-seed SD). On fresh seeds the gap is noise: 100 seeds on
  Nilsson_rare give −0.00008 ± 0.00034 (Mann–Whitney p = 0.60), and 40 seeds on Levine_32 give
  −0.00016 ± 0.00045 (p = 0.72).
- Two candidate causes were ruled out by switching them on in a copy of the code. igraph's
  unreset external weight (§3.4) changed no partition at all on Nilsson_rare or Levine_32. igraph's
  greedy choice when exp overflows moved the mean modularity by at most 0.0005.
- ARI and per-population F1 match within the seed spread on every set.
- The two smallest sets take a few milliseconds either way. With levels under 5,000 nodes
  sequential, a 500-node graph takes 0.6 ms.

**Multiscale** (one ensemble each; `multiscale_igraph` / `multiscale_native` in the same JSON): 6.6–9×
faster (Levine_32 68 s → 8.8 s, Levine_13 78 s → 11 s, Mosmann_rare 268 s → 41 s), with the same
accuracy. The one large difference in a single run (Nilsson_rare, HSC F1 0.325 native vs 0.506
igraph) is seed noise. Over five disjoint 4-seed sets both swing widely: native 0.33–0.48 (mean
0.414), igraph 0.35–0.51 (mean 0.432). On Levine_32 the two agree (0.616 vs 0.618). The null check of
docs/05 §3.3 still gives one cluster.

**End to end** (`find_communities(leiden="fast")`, `bench/timing_small.py --fast-only`,
`fast_native` in `bench/out/timing_small.json`): Levine_32 2.3 s (5.0 s with igraph), Levine_13
2.5 s (5.8 s), Samusik_01 0.89 s (2.3 s), Nilsson_rare 0.29 s (0.90 s), Mosmann_rare 11.5 s (61 s).
Against CosTaL 1.0 (docs/03 §10.4) that is 23–420× instead of 17–197×. On the two smallest sets the
total is a few to tens of milliseconds and the kNN's millisecond jitter decides the ratio.

---

## 6. Other Rust implementations (read 2026-10-14)

Read from their published sources (all permissively licensed), to learn from their mistakes. None
is closer to the paper than §5. What they contributed is mostly a list of failure modes that §5.2's
tests now guard against.

### 6.1 single-clustering (SingleRust, BSD-3-Clause, 0.7.0)

- Single-cell oriented: CSR with u32 ids and f32 weights.
- Its 0.7.0 rewrite fixed six bugs: self-loop strength counted once on aggregates, γ doubled (/4m
  instead of /2m), candidates taken from a `HashSet` (non-deterministic), the singleton-only rule on
  the wrong statement (disconnected clusters), O(n) work per move, and ignored `n_iterations`.
- Its refinement is not the paper's: no well-connectedness tests, the node's own singleton never
  sampled, θ scaled by the mean weight.
- Measured by its own open PR: 1.33× igraph single-threaded, 1.55–1.70× with parallel refinement.
- Its strength is verification: leidenalg fixtures committed as JSON, brute force on 6–9-node
  graphs, random property tests, and scale and relabelling invariance. §5.2 adopted the brute-force,
  scaling and property tests.

### 6.2 leiden-rs (MIT or Apache-2.0, 0.8.1)

- Parallel local moving by graph colouring. Each colour class is evaluated against frozen totals and
  committed without re-checking. It oscillated and is now capped at 100 rounds; it has no queue and
  no empty-cluster move.
- Its parallel aggregation merges per-thread hash maps, so the floating-point sums depend on
  scheduling. Its determinism tests use unit weights, which hides this.
- Its refinement is greedy and lets any node move, so the connectivity guarantee its README claims
  does not hold.
- No speed numbers.

### 6.3 Others

- **fa-leiden-cd:** Louvain plus a split of disconnected communities, with full-modularity
  recomputation per move.
- **graphrs:**
  - the RNG cannot be seeded;
  - a separate γ = 0.05 is hard-coded for the well-connectedness test;
  - sampling is nearly uniform;
  - an unshifted `exp` can overflow and panic.
- **hit-leiden:** incremental Leiden for dynamic graphs (Lin et al., arXiv:2601.08554). It does not
  carry over to a resolution sweep, where every node's gain changes. The one idea that does is a
  warm start, which `leiden::leiden_from` provides.
