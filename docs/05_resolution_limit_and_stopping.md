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

> **Done (2026-10-12) — the disease exists.** Deduplicated Levine, the published CosTaL graph, leidenalg
> (2 iterations, which reproduces the paper's 10 — docs/03 §9.1), 3 seeds; `bench/rare_populations.py`,
> numbers in `bench/out/rare_populations.json`.
>
> A first pass scored each population by its share in its dominant cluster (`bench/stage_c2.py`) and found
> no trend (Spearman −0.14). **That metric can't see absorption**: a rare population swallowed whole by a
> big cluster still sits in one cluster and scores 1.0. Scored instead by the F1 of its best-matching
> cluster (gated cells only, as for ARI):
>
> | Resolution | ARI | F1, 5 rarest | F1, 5 largest | Spearman(abundance, F1) |
> |---:|---:|---:|---:|---|
> | 0.2 | 0.968 | 0.28 | 0.98 | +0.97 (p = 5×10⁻⁹) |
> | 0.4 | 0.945 | 0.32 | 0.97 | +0.96 (p = 3×10⁻⁸) |
> | 0.8 (paper) | 0.768 | 0.44 | 0.89 | +0.64 (p = 0.014) |
> | 1.2 | 0.642 | 0.49 | 0.81 | +0.31 (p = 0.29) |
>
> Rare populations are systematically worse recovered, and the global resolution trades one end against
> the other: raising it recovers rare populations (population 12: F1 0.10 → 0.78 from 0.2 to 1.2) while
> splitting the largest (populations 7 and 10: 0.99 → 0.64–0.73). ARI, dominated by the large
> populations, hides this and prefers 0.2. **By this section's own rule, the methodology track is worth
> the investment.** Caveat: four rare populations (5, 14, 6, 3) stay below F1 0.55 at every resolution,
> so for them the resolution limit is not the whole story — they may not be separable in this graph.

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

> **The paper already has this path (found 2026-10-13, docs/02 addendum):** the supplement's "CosTaL
> Multilevel Clustering" re-clusters every population of ≥ 500 cells, a size floor as the stopping
> rule; two levels gave the best F1 on Mosmann_rare and Nilsson_rare. Reconstructed as a baseline in
> `bench/two_level.py` (§2.3).

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

> **Evaluated (2026-10-13): no consistent gain, not shipped.** `bench/local_scaling.py`, numbers in
> `bench/out/local_scaling.json`. Three graphs on the same exact kNN: `published`, `local_as_shipped`
> (the original `local=True`, bug included) and `local_fixed` (distance 1 − T, ρᵢ/σᵢ fitted on
> ascending rows, then the same fuzzy union); resolutions 0.2–2.0; F1 is the per-population best match
> of §1.1.
>
> | Data | Graph | Best ARI | Best (F1 rarest-5 + F1 largest-5)/2 |
> |---|---|---:|---|
> | Levine_32 | published | 0.968 | 0.666 (0.441 / 0.890 at 0.8) |
> | | local_as_shipped | 0.963 | 0.679 (0.478 / 0.880 at 0.8) |
> | | local_fixed | 0.952 | 0.671 (0.480 / 0.862 at 0.8) |
> | Levine_13 | published | 0.799 | 0.543 (0.161 / 0.925 at 1.2) |
> | | local_as_shipped | 0.789 | 0.522 (0.138 / 0.906 at 2.0) |
> | | local_fixed | 0.799 | 0.530 (0.157 / 0.903 at 2.0) |
> | GSE84133 | published | 0.938 | 0.573 (0.280 / 0.867 at 1.2) |
> | | local_as_shipped | 0.944 | 0.556 (0.139 / 0.973 at 0.8) |
> | | local_fixed | 0.927 | 0.612 (0.251 / 0.973 at 0.4) |
>
> - Local scaling moves along the rare-vs-large trade-off rather than past it: +0.04 on GSE84133 (whose
>   "rare" types have 1–14 cells, so that F1 is noisy), +0.005 on Levine_32, −0.013 on Levine_13, and
>   best ARI equal or lower everywhere. It also shifts the effective resolution (fewer clusters at
>   the same γ) and doubles the seed spread on Levine_32.
> - The shipped bug is mild in practice: the inverted memberships still rank a row's neighbors
>   consistently, so `local_as_shipped` lands close to the published graph. Fixing it doesn't help
>   either. Item 1 of §2.4 is closed; the resolution limit is not a per-point scale problem here.

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
**Measured (docs/03 §9):** on all 161,443 deduplicated Levine cells, igraph's native Leiden takes 1.7 s per
run (2 iterations) and leidenalg with 2 iterations 21 s, so a 20-resolution sweep costs well under a
minute with igraph.

Moreover, these Leiden runs are independent of each other and share the same read-only graph, so **they can run concurrently**, with speedup close to the core count
(for caveats about performance/efficiency cores see docs/06 §4). kNN is still computed only once, and is still exact.

> **First attempt (2026-10-13): works on Levine_32, does not generalize.** `bench/multiscale.py`,
> numbers in `bench/out/multiscale.json`. Leiden (leidenalg, 2 iterations) at 11 resolutions 0.1–3.2
> (ratio √2) × 4 seeds on the published graph. A cluster's **stability** is the mean best-match
> Jaccard against every other partition at the same and adjacent resolutions. The **base** is the
> resolution whose seeds agree most (its medoid seed); going from coarse to fine, a cluster with
> stability ≥ τ and ≥ 2k cells is **carved out** of the current cluster that hosts it, if nested in it
> (≥ 80%) and a proper part of it (≤ 90%). No labels are used until scoring. The **oracle** row takes
> each population's F1 at its own best single resolution — what a perfect multi-scale selection could
> reach.
>
> | Data | Partition | ARI | Clusters | F1, 5 rarest | F1, 5 largest | F1, mean |
> |---|---|---:|---:|---:|---:|---:|
> | Levine_32 | single, best ARI (0.283) | 0.966 | 10.5 | 0.276 | 0.983 | 0.644 |
> | | single, best mean F1 (0.8) | 0.765 | 18.8 | 0.450 | 0.887 | 0.688 |
> | | **multiscale τ = 0.85** | **0.952** | 19 | **0.614** | **0.978** | **0.773** |
> | | multiscale τ = 0.9 | 0.967 | 14 | 0.489 | 0.984 | 0.737 |
> | | oracle | | | 0.607 | 0.985 | 0.812 |
> | Levine_13 | single, best ARI (1.131) | 0.827 | 19.0 | 0.111 | 0.925 | 0.487 |
> | | multiscale τ = 0.85 / 0.9 | 0.710 / 0.717 | 19 / 15 | 0.181 / 0.054 | 0.814 / 0.817 | 0.439 / 0.408 |
> | | oracle | | | 0.181 | 0.941 | 0.536 |
> | GSE84133 | single, best mean F1 (1.131) | 0.815 | 11.2 | 0.280 | 0.900 | 0.651 |
> | | multiscale τ = 0.9 | 0.849 | 12 | 0.280 | 0.904 | 0.652 |
> | GSE74672 | single, best mean F1 (0.8) | 0.468 | 14.5 | 0.767 | 0.755 | 0.748 |
> | | multiscale τ = 0.9 | 0.483 | 22 | 0.694 | 0.730 | 0.710 |
>
> - **On Levine_32 it does what the oracle promised**: at τ = 0.85 the 5 rarest populations reach the
>   oracle's F1 (0.61 vs 0.28 at the best-ARI resolution) while the 5 largest and ARI barely move —
>   no single resolution comes close on the rare populations without losing ≥ 0.1 F1 on the large ones.
> - **It does not transfer.** On GSE84133 it lands on the single-resolution trade-off curve (its rare
>   types have 1–14 cells, below the 2k carving floor, and the oracle shows no single resolution finds
>   them either); on GSE74672 it is below the curve — also against the finer `level2` neuron labels
>   (69 classes: multiscale ARI 0.39 at τ = 0.9 vs 0.48 for the best single resolution), so the extra
>   clusters are not hidden subtypes. On Levine_13 the base selection fails: seeds agree only 0.62–0.75
>   at *every* resolution (0.94–0.98 on Levine_32), so "the most reproducible resolution" (0.2, ARI
>   0.59) is no signal, and everything built on it inherits the bad start.
> - **τ is not transferable**: on Levine_32 the result jumps from ARI 0.60 to 0.95 between τ = 0.8 and
>   0.85 (fragments of large populations with stability 0.8–0.85 stop being carved); on the other sets
>   no τ beats the best single resolution on more than one summary. τ = 0.85 / 0.9 was picked on
>   Levine_32 and then applied unchanged; the other three sets are honest held-out checks.
> - **Not shipped.** Two diagnosed failure points for a next attempt: the base selection (use the
>   stability curve's shape, not its maximum, or skip the base and select clusters globally), and τ
>   (calibrate it against the dataset's own stability distribution, e.g. against the stability of
>   splits of clusters that are themselves stable). Any next version must be fixed on one dataset and
>   checked on the others the same way.

> **Second attempt (2026-10-13): a conductance rule — better, transfers to one of three held-out sets
> fully and to one partly.** Same ensemble and script (`refine_conductance`). Two changes, both aimed at
> the diagnosed failures: no base selection (start from the coarsest resolution) and no seed-stability
> threshold. Instead a candidate C is carved out of its current host H if less than φ_max of its edge
> weight goes to the rest of the host, φ(C | H) = w(C, H∖C) / vol(C): a real sub-population is
> cohesive, while a fragment of a large population shares a long boundary with its sibling. Over all
> candidates, φ separates good sub-clusters from fragments about as well as stability on Levine_32
> (Spearman with the best-match F1 −0.74 vs +0.71), but on a scale that holds across datasets (good
> sub-clusters ≤ 0.02–0.04, fragments 0.05–0.13). **φ_max = 0.04 was fixed on Levine_32** (the gap
> there) before running the other three.
>
> | Data | Partition | ARI | Clusters | F1, 5 rarest | F1, 5 largest | F1, mean |
> |---|---|---:|---:|---:|---:|---:|
> | Levine_32 (design) | single, best ARI (0.283) | 0.966 | 10.5 | 0.276 | 0.983 | 0.644 |
> | | single, best mean F1 (0.8) | 0.765 | 18.8 | 0.450 | 0.887 | 0.688 |
> | | **conductance, φ_max = 0.04** | 0.876 | 38 | **0.622** | 0.912 | **0.785** |
> | | conductance, φ_max = 0.02 | 0.943 | 23 | 0.609 | 0.954 | 0.770 |
> | GSE84133 (held out) | single, best ARI (0.2) | 0.936 | 7.0 | 0.139 | 0.973 | 0.552 |
> | | single, best mean F1 (1.131) | 0.815 | 11.2 | 0.280 | 0.900 | 0.651 |
> | | **conductance, φ_max = 0.04** | **0.950** | 9 | **0.280** | **0.974** | **0.677** |
> | Levine_13 (held out) | single, best ARI (1.131) | 0.827 | 19.0 | 0.111 | 0.925 | 0.487 |
> | | single, best mean F1 (2.263) | 0.768 | 29.0 | 0.162 | 0.878 | 0.498 |
> | | **conductance, φ_max = 0.04** | 0.769 | 46 | **0.315** | 0.908 | **0.550** |
> | GSE74672 (held out) | single, best mean F1 (0.4) | 0.566 | 9.5 | 0.728 | 0.809 | 0.745 |
> | | conductance, φ_max = 0.04 | 0.480 | 26 | 0.689 | 0.723 | 0.705 |
>
> - **GSE84133: better than every single resolution on every summary at once** — ARI above the best
>   single resolution, the rare and large populations at the per-population oracle (0.281 / 0.976).
> - **Levine_13: the rare populations jump** (0.315, above the single-resolution oracle of 0.181,
>   because carving combines clusters no single partition has), mean F1 +0.05 over any single
>   resolution, at an ARI cost (0.77 vs 0.83 at the best-ARI resolution).
> - **GSE74672: no gain** — below the best single resolution on the `level1` labels and level with it
>   on the 69 finer labels (ARI 0.34 vs 0.35 at a similar cluster count).
> - Sensitivity: φ_max 0.02–0.05 changes little on GSE84133 and Levine_32's F1; Levine_13 needs ≥ 0.03
>   for the rare gain, GSE74672 is best at 0.02. So φ_max is a real knob, but a readable one ("at most
>   4% of a sub-cluster's edge weight leaks into the rest of its parent").
> - **Same with igraph's Leiden** (`MULTISCALE_LEIDEN=igraph`, `bench/out/multiscale_igraph.json`),
>   φ_max = 0.04 against igraph's own single resolutions: Levine_32 rarest-5 F1 0.62 vs 0.28 at the
>   best-ARI resolution (ARI 0.88 vs 0.97); GSE84133 ARI 0.950 vs 0.945 with rare and large F1 at the
>   oracle; Levine_13 rarest-5 0.31 vs 0.12–0.15 and mean F1 0.55 vs 0.51, ARI 0.79 vs 0.86; GSE74672
>   below the best single resolution. The 44-run ensemble costs 80 s serial on 161k cells (leidenalg:
>   ~2.5 min on 6 cores).
> - **Against the paper's own multilevel option** (`bench/two_level.py`, numbers in
>   `bench/out/two_level.json`; reconstructed from the supplement: CosTaL at 0.8, then CosTaL again,
>   with the kNN recomputed, inside every cluster of ≥ 500 cells; leidenalg, 4 seeds):
>
>   | Data | One level (0.8) | Two levels, ≥ 500 | Conductance, φ_max = 0.04 |
>   |---|---|---|---|
>   | Levine_32 | ARI 0.765 · rare 0.450 · large 0.887 · mean 0.688 | 0.189 · 0.647 · 0.337 · 0.500 (135 clusters) | 0.876 · 0.622 · 0.912 · 0.785 |
>   | Levine_13 | 0.797 · 0.111 · 0.863 · 0.470 | 0.224 · 0.338 · 0.364 · 0.378 (149) | 0.769 · 0.315 · 0.908 · 0.550 |
>   | GSE84133 | 0.853 · 0.139 · 0.917 · 0.590 | 0.396 · 0.140 · 0.692 · 0.510 (21) | 0.950 · 0.280 · 0.974 · 0.677 |
>   | GSE74672 | 0.468 · 0.767 · 0.755 · 0.748 | 0.243 · 0.767 · 0.621 · 0.653 (24) | 0.480 · 0.689 · 0.723 · 0.705 |
>
>   Two levels reach about the same rare-population F1 as the conductance rule on the cytometry sets,
>   but by shredding everything (135–149 clusters; the largest populations fall to F1 0.34–0.36). That
>   fits what the paper used it for — the F1 of one rare population at a time on Mosmann_rare and
>   Nilsson_rare — and shows why a stopping rule matters (§2.1 ①): the size floor stops nothing on
>   cytometry-sized data. The conductance rule keeps the large populations intact while getting the
>   rare ones.
> - **The paper's rare-population benchmarks (2026-10-13; all held out, φ_max still 0.04).**
>   Samusik_01, Nilsson_rare and Mosmann_rare from HDCytoData (docs/03 §10.1; Table 2 reproduced
>   exactly), leidenalg ensembles, against the reconstructed two-level option of the supplement — the
>   method the paper itself used on Nilsson_rare and Mosmann_rare:
>
>   | Data | Paper setting (0.8) | Best single resolution | Two levels, ≥ 500 | Conductance, φ_max = 0.04 |
>   |---|---|---|---|---|
>   | Nilsson_rare, HSC F1 (358 of 44,140) | 0.169 (18 clusters) | 0.479 (3.2; 50) | 0.500 (122) | **0.531 (53)**; 0.598 at φ_max 0.02 (30) |
>   | Mosmann_rare, activated F1 (109 of 396,460) | 0.175 (16) | 0.560 (1.6; 26) | 0.635 (131) | **0.653 (34)**, the same for φ_max 0.02–0.05 |
>   | Samusik_01, ARI · rare-5 · large-5 · mean F1 | 0.833 · 0.241 · 0.898 · 0.608 | best ARI (0.4): 0.902 · 0.170 · 0.918 · 0.567 | 0.243 · 0.523 · 0.395 · 0.514 (104) | **0.860 · 0.429 · 0.908 · 0.712** (33); at 0.03: 0.905 · 0.429 · 0.919 · 0.714 |
>
>   On both one-population benchmarks the conductance rule finds the rare population better than any
>   single resolution and better than the paper's two-level option, with a quarter of its clusters.
>   On Samusik_01 (24 populations) it lifts the 5 rarest above the single-resolution oracle (0.43 vs
>   0.37) and matches the oracle's mean F1, at −0.04 ARI (φ_max 0.03 gives the same F1 with ARI above
>   every single resolution).
> - **Tally over seven datasets, φ_max = 0.04 fixed in advance:** better than the best single
>   resolution on the rare populations on six (Levine_32, Levine_13, GSE84133, Samusik_01,
>   Nilsson_rare, Mosmann_rare), no gain on GSE74672; ARI within 0.05 of the best single resolution
>   except Levine_13 (−0.06).
> - **Decision (2026-10-13, under delegated authority): ship it as an opt-in mode**, the condition set
>   above (benchmarks with known rare populations) now being met. It does not touch the default path
>   (versioning policy (b)): `compat="paper"` refuses it.
> - **Shipped as `costal_plus.find_communities_multiscale(X, phi_max=0.04, ...)`** (igraph's Leiden by
>   default, `leiden="paper"` for leidenalg). Its partitions are identical to `bench/multiscale.py`'s
>   (checked on GSE84133 and Levine_32). With igraph the held-out results hold: Nilsson_rare HSC F1
>   0.579 vs 0.513 for igraph's best single resolution, Mosmann_rare 0.632 vs 0.492, Samusik_01
>   rare-5 0.379 vs 0.225 at the best-ARI resolution (`bench/out/multiscale_igraph.json`). 110 s for
>   161k cells on the M1 Pro.

### 2.4 Ranking

| | What | Cost | Why it ranks here |
|---|---|---|---|
| **0** | The diagnostic in §1.1 | ~20 lines | First prove the disease exists |
| **1** | ~~Evaluate `local=True` (first fix the issue in §2.2: semantics confirmed inverted)~~ **Done: no consistent gain (§2.2)** | Almost zero | Already implemented, non-greedy, `O(N·k)`, no added knob |
| **2** | Multi-resolution stability — **seed-stability rule: Levine_32 only; conductance rule: better on rare populations in 6 of 7 datasets, shipped opt-in (§2.3)** | Medium | Non-greedy, no stopping-criterion problem, favored by the architecture |
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
