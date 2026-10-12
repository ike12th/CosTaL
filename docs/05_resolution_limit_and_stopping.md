# Resolution limit and stopping criterion (v0, 2026-09-20)

> This document is the **methodology track**, not the acceleration track. The goal is "a better CosTaL", not "a faster CosTaL".
> The two tracks' deliverables must be kept separate: the default path continues to reproduce the published CosTaL (versioning policy (b),
> docs/04 §6.5); everything in this document is an opt-in mode with its own evaluation.
>
> **Good news**: everything in this document touches only the **graph layer and the clustering layer**, not the kNN kernel.
> It does not conflict at all with Stage A/D in docs/03 and can proceed in parallel.

---

## 1. Root cause: the resolution limit

A single global resolution cannot fit data whose local density is heterogeneous. In the modularity framework this is the
**resolution limit** (Fortunato & Barthélemy, PNAS 2007) — RBConfiguration's
resolution parameter can only **shift** this limit, not remove it.

Single-cell data is hit hardest: CD4 T cells at 30% abundance and HSCs at 0.1% abundance have to be separated
with the same resolution, which is fundamentally impossible. A high resolution shatters the large populations; a low resolution swallows the rare ones.

## 1.1 Step 0: first prove the disease exists (before any methods work)

**Levine_32dim has 14 manually gated populations with highly imbalanced abundances — a ready-made, ideal test.**

The diagnostic:

```
For each gated population, compute per-population recall; plot grouped by abundance
```

- **Recall is systematically lower for rare populations** ⇒ the resolution limit really is hurting CosTaL, and this document is worth the investment;
- **No systematic trend** ⇒ this whole front can be dropped; the errors come from elsewhere (look elsewhere).

This step is ~20 lines, uses the already-downloaded data, and runs in a few minutes. **Do it first; don't skip it.**

---

## 2. Three candidate paths

All three attack the same disease, but they intervene at different points.

### 2.1 Recursive splitting (iteratively split large clusters into smaller ones)

**Especially cheap in our architecture.** After splitting into m subclusters, the total cost of the subproblems is `Σnᵢ² ≈ N²/m` —
the whole recursion tree adds up to a mere fraction of one top-level pass. The subclusters' kNN can also directly use the global kNN's
**induced subgraph** (for points inside a large cluster, the k nearest neighbors are mostly inside the cluster anyway; only boundary points lose edges), at nearly zero cost.

> **Exactness caveat (2026-10-12)**: the induced subgraph is **not** the exact kNN within the subcluster — boundary points lose edges,
> but the next-ranked in-cluster neighbor is not filled in. To uphold "exact kNN", a core claim of CosTaL, the subcluster should
> recompute exact kNN, at the cost above, `Σnᵢ² ≈ N²/m`, which is still cheap. The induced subgraph can only be
> an explicitly labeled approximate option.

But it has two hard flaws:

**① Stopping criterion.** The decisive one. Leiden will **always** produce a partition for any graph — you can always keep splitting.
Without a stopping criterion, "recursion depth" is just a new knob that is harder to interpret than resolution — **epistemically a step backward**.

> Prior work along this path puts its contribution **entirely** in the stopping criterion, not in the recursion:
> scSHC (Grabski et al., Nat Methods 2023, hypothesis testing),
> CHOIR (permutation/classifier tests), SCCAF (self-projection classifier accuracy).
> **So if we take this path, the contribution must be in "should this cut be made"; the recursion itself is nothing new.**

**② Greedy and irreversible.** Cells misassigned at the top level never get a chance to be reconsidered.
**This is especially dangerous for rare populations** — they are exactly the ones most likely to be absorbed into a large cluster at the top level.

### 2.2 Local scale adaptation (the reliable version of "local attention")

**First, separate the concepts: without training there is no attention.** The defining element of attention is the learned
query/key projections; remove the training and what remains is **local adaptive normalization** — something far more mature
and far more reliable:

- Zelnik-Manor & Perona, self-tuning spectral clustering (2004):
  `σᵢ` = distance to the k-th nearest neighbor, affinity `exp(−d²ᵢⱼ/(σᵢσⱼ))`;
- **UMAP's fuzzy simplicial set**: `smooth_knn_dist` solves for each point's `ρᵢ` (nearest-neighbor distance) and
  `σᵢ` (such that the row's membership sum = `log₂k`), membership `exp(−(d−ρᵢ)/σᵢ)`.

> **CosTaL already implements the latter**: `find_communities(..., local=True)` →
> `build_graph_umap` → `u.smooth_knn_dist` + `u.compute_membership_strengths`.
> It's just that the default is `local=False`, and the profile in docs/02 did not test this path either.
>
> **So the question is not "whether to build it" but "it's already there, just never evaluated".** This should be experiment #1.

**Before evaluating, check one thing first** (alpha-stage code, off by default, possibly intentional):

```python
distances = np.array(weights).reshape(...)     # weights are Tanimoto coefficients (similarities)
sigmas, rhos = u.smooth_knn_dist(distances, ...)   # expects distances, and assumes each row is ascending
```

What is passed in is **similarity** (larger = closer, descending), whereas `smooth_knn_dist` expects **distance** (smaller = closer, ascending).
Unless it is converted somewhere else, the membership semantics under `local=True` are inverted. **Take a look before evaluating.**

> **Confirmed (2026-10-12): it is inverted.** The `weights` computed by `build_tani_graph` are Tanimoto
> similarities, passed as-is to `build_graph_umap`; the latter directly `reshape`s them into `distances` and hands them to
> `smooth_knn_dist`, with no conversion in between. And `smooth_knn_dist` (umap source; the docstring
> states the input is a "sorted list of distances") takes the first nonzero value of each row as `ρᵢ`, i.e. it assumes each row is
> distances in ascending order. So `local=True` currently treats the most similar neighbor as the farthest.
> **Evaluating it before it is fixed is meaningless.** Fix: first convert to distances (e.g. `1 − T`), sort each row ascending,
> then pass to `smooth_knn_dist`.

**We recommend not doing true attention (softmax row normalization)**, for two reasons:

1. Row normalization destroys the premise "edge weight = similarity, globally comparable". Modularity's null model assumes
   weights are globally comparable; once each point's out-edge sum is normalized, what is being optimized is no longer modularity
   but something closer to the **Markov Stability / random walk** framework. That isn't wrong, but it is **changing the objective function**,
   not adding a module — if we do it, we must acknowledge that head-on and redo the entire evaluation.
2. "Learning" requires labels or a self-supervised objective ⇒ introduces training ⇒ directly conflicts with "no loss of accuracy + one-command reproduction".

### 2.3 Multi-resolution stability / consensus

Sweep Leiden over a whole range of resolutions and keep the clusters that are stable across scales
(Markov Stability, clustree, consensus clustering).

At the same time it sidesteps both hard flaws of recursion: **non-greedy** (no irreversible splits) and **no stopping-criterion problem**
(stability replaces significance testing). The downside is that Leiden has to run many times.

**And this is exactly the sweet spot of what we are building**: `knn()` / `reweight()` / `truncate()`
are already separated (docs/04 §6.1), **kNN is computed only once**, and sweeping resolution only reruns Leiden.
Add the two few-hour tasks from docs/03 (switching to igraph's native `community_leiden`,
tuning `n_iterations`), and sweeping 20 resolutions is entirely affordable.

Moreover, these Leiden runs are independent of each other and share the same read-only graph, so **they can run concurrently**, with speedup close to the core count
(for caveats about performance/efficiency cores see docs/06 §4). kNN is still computed only once, and is still exact.

### 2.4 Ranking

| | What | Cost | Why it ranks here |
|---|---|---|---|
| **0** | The diagnostic in §1.1 | ~20 lines | First prove the disease exists |
| **1** | Evaluate `local=True` (first fix the issue in §2.2: semantics confirmed inverted) | Almost zero | Already implemented, non-greedy, `O(N·k)`, no added knob |
| **2** | Multi-resolution stability | Medium | Non-greedy, no stopping-criterion problem, favored by the architecture |
| **3** | Recursive splitting **+ significance-based stopping criterion** | High | Only worth it if 1 and 2 both fall short, and the contribution must be in the criterion |

---

## 3. Stopping criterion: the hard part is statistics, not representation

### 3.1 The real difficulty is post-selection inference

The stopping criterion is hard for one very specific reason: **the same data is used both to find the cut and to test it**.
This is the classic post-selection inference problem. A permutation null model (what scSHC does) is a remedy, not the most rigorous option.

Statistics has a direct answer: **selective inference for clustering**. The Gao, Bien & Witten line of work
("Selective inference for hierarchical clustering", JASA, recent) derives **exact p-values** conditional on
"the partition being produced by the clustering itself".
(The exact form and assumptions need to be checked against the original paper.)

> **What makes a stopping criterion definitive is not a better embedding, but handling post-selection inference correctly.**
> Changing the representation only changes what is being tested; changing the statistical framework is what changes the "definitive" part itself.

### 3.2 Two structural advantages CosTaL has here

This is the part most worth turning into "CosTaL+":

1. **Tanimoto edge weights are bounded in `[0,1]`.** Building a null model and setting thresholds on a bounded similarity
   is much easier than on an unbounded distance — the scale problem simply does not exist.
2. **A fixed-k kNN graph has constant degree.** The degree-preserving null model (the one modularity uses)
   collapses to a minimal form here, and permutation tests are easy to do as well.

**The graph structure we already have is itself better suited than "distance matrix + threshold" for attaching a rigorous test.**
This is more solid than riding on the momentum of an external model.

---

## 4. Where foundation model embeddings fit

### 4.1 They will not make the stopping criterion more definitive

| # | Problem | Explanation |
|---|---|---|
| 1 | **The space is not calibrated** | Transformer representation spaces are anisotropic; regions are arbitrarily stretched/compressed. "Distance > threshold" merely swaps in a **ruler with no markings**, no more justified than PCA space |
| 2 | **Zero-shot performance is not necessarily better** | Multiple independent evaluations found that scGPT / Geneformer zero-shot embeddings underperform scVI on clustering/cell-type tasks, and sometimes underperform HVG+PCA (Kedzierska et al. "limits of zero-shot foundation models in single-cell biology" is representative; verify the conclusion yourself) |
| 3 | **Circularity ← the most dangerous** | If the calibration comes from the FM's pretraining distribution, it **inherits the model's priors**: confident about types seen in the atlas, blind to **unseen** ones. Yet discovering rare/novel populations is the very reason methods like this exist. **Using a model that has only seen known types to calibrate "is this a new population" is the worst kind of circularity** |
| 4 | **Conflicts with three selling points** | Works right after `pip install`, one-command reproduction, deterministic results — plugging in a checkpoint with hundreds of millions of parameters breaks all three |
| 5 | **Architectural consequence** | FM embeddings are typically 512-dimensional. By the cost balance in docs/03, at d=512 matmul : top-k ≈ 16:1, **so that route has to go to GPU**. Not a blocker, but worth knowing |

### 4.2 But it is a legitimate input representation — so decouple

```
Stopping criterion ── must be a statement about the "graph", not about the "embedding space"
Representation     ── HVG+PCA / scVI / FM embedding are each just one kind of input that produces this graph
```

This way, FM embeddings are **one of the optional inputs**: they can be supported, benchmarked, and compared,
but **the criterion does not inherit their priors**. We get the benefits (if any) without being held hostage by their blind spots.

Consistent with the versioning policy already decided: the default path must run to completion **offline** after `pip install`.

### 4.3 Incidentally: CosTaL's main battleground has no use for FMs

CosTaL's main benchmark is **CyTOF / flow cytometry (Levine_32dim, 32 markers)**.
Single-cell foundation models are essentially all for scRNA-seq; a 32-dimensional protein panel
is a small and interpretable input space to begin with — **there is no FM for it, and none is needed**.
This issue only applies on the scRNA-seq path.

---

## 5. Delivery boundary

| | Scope |
|---|---|
| Default path | Reproduces the published CosTaL, **unaffected by this document** (versioning policy (b)) |
| Every item in this document | Opt-in mode, with its own evaluation |
| Layers touched | **Only the graph layer and the clustering layer**; the kNN kernel is untouched |
| Relationship to the docs/03 stage table | Fully parallel; neither blocks the other |
