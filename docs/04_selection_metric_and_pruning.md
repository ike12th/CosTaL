# Selection metric and pruning strategy (v0, 2026-09-20)

> **Name.** These design notes were written while the rewrite was called *CosTaL+*; it is released as
> **CosTaL 2.0**, the next major version after CosTaL 1.0 (published on PyPI as 0.0.2.4). "CosTaL+" below
> means CosTaL 2.0, and "the original" means CosTaL 1.0.

> Answers two questions:
> 1. Should cosine selection (+ Tanimoto edge weights) and Tanimoto selection be built as two versions?
> 2. How do we write it without sacrificing speed or verifiability?
>
> **Answer: not two versions, but one kernel + two comparators + one data-layout decision.**

---

## 0. First, clarify: three orthogonal knobs

The original CosTaL already exposes one of them as a parameter (`method='tani'|'jaccard'|'cosine'`),
but it only controls **edge weights**. The full design space is three-dimensional:

| Knob | Values | Affects | Original |
|---|---|---|---|
| `normalize` | L2 / raw magnitude | similarity semantics | L2 when selecting neighbors, raw when computing edge weights |
| `select` | cosine / tanimoto | **neighbor set** | always cosine |
| `weight` | tanimoto / cosine / jaccard | **edge weights** | configurable, default tanimoto |

**The published CosTaL = the (normalize=L2, select=cosine, weight=tanimoto) cell.**
This cell must be the default, and it must be reproduced entry for entry — this is the "no loss of accuracy" contract.
All other cells are opt-in.

### 0.0 The default is a contract, not just two knobs

For "no parameters passed = the published CosTaL" to hold, **more than select and weight** has to be pinned.
Every item below changes the output; every one must go into the defaults and must have a regression test:

| # | What is pinned | Original's value | What happens if it isn't pinned |
|---|---|---|---|
| 1 | `select` / `weight` | cosine / tanimoto | neighbor set or edge weights change |
| 2 | **The two-sidedness of normalization** | L2 normalization for neighbor selection, **raw norms** for edge weights | edge weights degenerate to `c/(2−c)` (docs/02 §4) |
| 3 | **Graph symmetrization formula** | `S + Sᵀ − S∘Sᵀ` (fuzzy union), **not** `(S+Sᵀ)/2` | edge weights systematically too small, community structure changes |
| 4 | Self-loops and the meaning of k | self-loops removed; k excludes self | one edge too many/too few per row |
| 5 | Leiden parameters | `RBConfigurationVertexPartition`,<br>`resolution=0.8`, `n_iterations=10` | number of clusters changes directly |
| 6 | tie-break | see §4.1 | two runs on the same input give different results |

Item 3 is especially easy to miss: in the original `build_graph`, the line
`csr = csr + csr.T - csr.multiply(csr.T)` still has a commented-out
`(csr + csr.T) * 0.5` above it — **the two are not equivalent**; copy the wrong one and the metrics quietly drop a little.

### 0.1 A degenerate cell that must be handled explicitly

When `normalize=L2`, `A=B=1`, and `T = c/(2−c)` is a strictly monotonic transform of cosine,
so **`select=tanimoto` is exactly equivalent to `select=cosine`** (docs/01 §2.2 / §7).

This is not a bug; it is a mathematical fact. But if it isn't handled, users will report "tanimoto selection has no effect".
So:

```
if normalize == L2 and select == tanimoto:
    → short-circuit to cosine, and warn once explaining why
```

**A test oracle comes for free along the way**: this identity can be written as a property test —
the output of `(L2, tanimoto, *)` must be **identical, entry for entry,** to that of `(L2, cosine, *)`.
The moment the Tanimoto comparator's algebra is wrong, this assertion blows up. See §4.

**Generalization (2026-10-12)**: the same holds when the input is **itself** already unit-normalized per row — embeddings from scGPT / UCE / SCimilarity,
the output of fastMNN (`cos.norm=TRUE`), and Pixie's per-pixel normalization all fall into this category.
Then `r ≡ 1`, Tanimoto is identically `c/(2−c)`, and neither selection nor edge weights carry any information beyond cosine.
The criterion is a row-norm coefficient of variation ≈ 0; record it in the metadata and warn once (§6.6; rationale in the
[survey report](../reports/single_cell_and_spatial_transforms.md)).

---

## 1. Decision one: one generic kernel, monomorphized at compile time

The comparator **must not** use runtime enum / dyn dispatch. The budget for the whole inner loop is estimated at
**about 1 cycle/candidate**; inserting a single unpredictable branch could double it.

> **Correction (2026-10-12)**: the original text said "1.2 cycles/candidate (measured in docs/03 §2)",
> but docs/03 contains no such measurement, and no source can be found in the git history either. **This is an estimate, not a measurement**;
> the real number comes from the Stage A bench. The conclusion (the comparator must be statically dispatched) is unaffected.

```rust
trait Selector {
    /// Per-row precomputed constants. cosine: only τᵢ; tanimoto: cᵢ and cᵢ·Aᵢ
    type RowState: Copy;
    fn row_state(norm_sq: f32, tau: f32) -> Self::RowState;
    /// Conservative scalar threshold per column block, see §2
    fn tile_threshold(st: Self::RowState, col_norm_sq_min: f32) -> f32;
    /// Exact re-check after a (rare) pass
    #[inline(always)]
    fn exact_pass(st: Self::RowState, cand_norm_sq: f32, sim: f32) -> bool;
}
```

`match` once at the call entry point; from there on, everything is statically dispatched:

```rust
match select {
    Select::Cosine   => run_blocked::<CosineSel>(...),
    Select::Tanimoto => run_blocked::<TanimotoSel>(...),
}
```

`#[inline(always)]` + monomorphization ⇒ the compiler generates two specialized loops, with **zero branches** in the inner loop.
The cost is one extra copy of the code in the binary (a few KB), in exchange for zero runtime overhead.

---

## 2. Decision two: conservative scalar threshold + exact re-check on the rare path

### 2.1 The problem

The Tanimoto criterion `d > cᵢ·(Aᵢ + Bⱼ)` contains a `Bⱼ` — **the threshold varies per candidate**.
Written directly, that is one extra FMA in the inner loop (`d − cᵢ·Bⱼ > cᵢ·Aᵢ`), making the top-k half 25–60% more expensive.
This can be avoided.

### 2.2 Approach

For a column block, take the minimum squared norm within the block, `B_min`, and define the **conservative scalar threshold**:

```
θᵢ = cᵢ · (Aᵢ + B_min)      ← computed once per row per block, not in the inner loop
```

Because `Bⱼ ≥ B_min` (when `cᵢ ≥ 0`; when `cᵢ < 0`, use `B_max` instead, see §2.4), we have `θᵢ ≤ cᵢ(Aᵢ+Bⱼ)`,
so **every candidate that should pass is guaranteed to pass θᵢ** — only false positives, no false negatives.

The inner loop is then identical to cosine's, **1 comparison**:

```
if tile[i][j] > θᵢ {          // 99.99%+ rejected here (~1 cycle, to be measured)
    if Sel::exact_pass(st, B[j], tile[i][j]) {   // rare path, cost doesn't matter
        heap_insert(...);
        θᵢ = Sel::tile_threshold(st, B_min);     // τᵢ changed, recompute
    }
}
```

**Conclusion: the inner-loop cost of Tanimoto selection ≈ that of cosine selection.** The price is just one scalar,
`B_min`, per column block, plus one extra multiply-add on the rare path.

> **Measured (2026-10-12, stage A+):** on deduplicated Levine (161,443 cells) Tanimoto selection takes
> 3.4 s against 2.45 s for cosine (+38%, not "≈"), bit-identical to the Tanimoto brute force and with
> zero fallback rows. The block extremes are precomputed per column block, so the bound does not depend
> on the norm-sorted layout. §3's norm-window pruning (stage D) is what targets this path.

### 2.3 The key to tightening the conservative threshold: sort by norm

How tight `θᵢ` is depends on the spread of norms within a column block. Hence:

> **Data-layout decision: during preprocessing, reorder the rows by ascending ‖x‖, and map back to the original order at the end via a permutation table.**

Norms within a column block are nearly identical ⇒ `B_min ≈ Bⱼ` ⇒ the conservative threshold is nearly the exact threshold, with very few false positives.

This sort **pays off twice**: it also makes the norm-window pruning of §3 possible.
One `O(N log N)` sort (about 20 ms for 260k rows) buys both.

### 2.4 A correctness guardrail: choose B_min / B_max by the sign of τᵢ

> **Rewritten based on measurements (2026-10-12).** The original text said "flow/mass cytometry data after the arcsinh transform is all non-negative,
> so the `select=tanimoto` entry point validates that the input is non-negative". The premise is wrong: on the full Levine_32dim, after
> `arcsinh(x/5)`, **27.0% of elements are negative and 99.8% of rows contain negative values** (minimum about −0.058,
> docs/06 §5.1). Implemented as originally written, the main benchmark would be rejected outright and the §5 experiments could not run;
> moreover, the original CosTaL computes Tanimoto on exactly this kind of data with negative values, so rejecting negatives would actually break reproduction.

The range of Tanimoto is `[−1/3, 1]`, so `1+τᵢ > 0`, and `cᵢ = τᵢ/(1+τᵢ)` has the same sign as `τᵢ`.
The conservative threshold only needs to pick its bound by the sign of `cᵢ`:

```
cᵢ ≥ 0: θᵢ = cᵢ·(Aᵢ + B_min)     ← Bⱼ ≥ B_min ⇒ θᵢ ≤ cᵢ(Aᵢ+Bⱼ)
cᵢ < 0: θᵢ = cᵢ·(Aᵢ + B_max)     ← Bⱼ ≤ B_max ⇒ θᵢ ≤ cᵢ(Aᵢ+Bⱼ)
```

In both cases there are only false positives and no false negatives, so the selection result is still exact. This branch is evaluated **once per row per block**
(τᵢ only changes on the rare path) and never enters the inner loop, so the cost conclusion of §2.2 is unchanged.

→ **The sign of the input is not validated.** In practice τᵢ can only be negative while the heap is not yet full: until it fills, the Tanimoto
lower bound −1/3 is used, giving `cᵢ = −1/2`, and `θᵢ = −(Aᵢ+B_max)/2 ≤ −‖a‖‖b‖ ≤ dᵢⱼ` always holds,
which is equivalent to "let everything through" — exactly the right behavior.

---

## 3. Decision three: norm-window pruning — a free bonus exclusive to Tanimoto

From `d = ‖a‖‖b‖·cos` and `ρ = ‖b‖/‖a‖`:

```
T ≥ τ   ⟺   cos ≥ c·(ρ + 1/ρ) ,   c = τ/(1+τ)
```

Since `cos ≤ 1`, this gives a pruning rule that is completely independent of direction:

```
ρ + 1/ρ > (1+τ)/τ   ⟹   the candidate cannot reach τ whatever its direction; skip the whole block
```

At τ=0.5 the threshold is `ρ+1/ρ > 3`, i.e. **ρ > 2.618 or ρ < 0.382**.

This only holds when **τᵢ > 0** (the derivation uses `c > 0`). While the heap is just starting to fill and τᵢ is still ≤ 0,
nothing is pruned (see §2.4) — on data with negative values (such as Levine) this phase lasts longer, so the pruning rate has to be measured.

Rows are already sorted by norm (§2.3) ⇒ each query only needs to scan one **contiguous norm window**;
column blocks outside the window **don't even need the matmul**.

**On Levine_32dim this rule will really bite**: the measured row-norm range is 2.949–17.233 (5.8×,
docs/02 §4). The actual pruning rate depends on the value τᵢ converges to, and has to be measured (§5).

> Note that this rule **only holds for `select=tanimoto`**. With `select=cosine` the norm is not part of the criterion,
> so window pruning does not apply — the cosine route has to rely on cone/cluster upper bounds (docs/03 §3, Stage D).
> Both pruning mechanisms sit behind the same `KnnIndex` trait without interfering with each other.

---

## 4. Decision four: a three-layer correctness net

Speed can be tuned over time; **if the neighbor set is wrong, the whole project is pointless**. Three layers, from cheap to expensive:

| Layer | Content | Scale | Frequency |
|---|---|---|---|
| **L1 identity** | `(L2, tanimoto) ≡ (L2, cosine)` identical entry for entry | N≤2000 | every CI run |
| **L2 brute-force reference** | a naive Rust implementation with no blocking and no pruning; every (normalize × select × weight) combination compared entry for entry | N≤20000 | every CI run |
| **L3 reference against the original** | default cell `(L2, cosine, tanimoto)` compared entry for entry against the original CosTaL's neighbor sets | Levine subset | every release |

The practical constraint on L3: the original's `p_knng` is a Linux x86-64 binary and cannot run on this machine
(docs/02 §1.1). But CosTaL's fallback path on non-Linux, `bc()`, goes through
`phenograph.bruteforce_nn`, **which does run on this machine**, so it can serve as a stand-in for L3;
the real `p_knng` comparison goes into Linux CI.

> **Open items (to verify, 2026-10-12):**
>
> 1. **Index base.** The original passes `-writeZidx` (write 0-based) when calling `p_knng`,
>    yet after reading the result back it also does `columns - 1` (comment "bug in L2knng"). If the binary really wrote 0-based,
>    every neighbor in the original would be off by one; but then −1 would appear, scipy would error out building the sparse matrix,
>    and the paper's runs could not have gone through — so most likely the binary actually writes 1-based and that −1 is correct.
>    **This bears on the default-value contract and must be pinned down with golden output**: run `p_knng_Linux` on the NAS (x86-64) or on a CI
>    ubuntu runner (docs/02 §1.1 addendum).
> 2. **`bc()` can only stand in for the default cell.** In `bc(method='cosine')`, `d` does not drop the self column
>    (`idx` does), so weights and edge counts don't line up; moreover `d` is a **distance** from `cdist` (ascending),
>    and using it directly as an edge weight inverts near and far. The default `method='tani'` is unaffected.
> 3. **Differences from `p_knng` can only come from near-ties** (docs/06 §1.3); measure their size with golden output.
>    **Refuted — see "Results" below.**
> 4. **(measured) `bc()`'s `idx[:, 1:]` breaks on exact duplicate cells.** It assumes column 0 is the cell itself,
>    but a duplicate is also at distance 0 from it and may rank first. Every gated cell in Levine_32dim has a duplicate
>    (docs/02 addendum): in the raw (not deduplicated) 50k subsample, 7,236 rows are in duplicate groups, and **exactly half (3,618 rows) keep themselves
>    in the neighbor list**, dropping one true neighbor (afterwards `setdiag(0)` removes the self-loop, so these rows actually end up with only k−1 neighbors).
>    CosTaL+ excludes self by index and is unaffected. **After deduplication, the neighbor sets are 100% identical to `bc()`'s** (docs/03 §8.1).
>    Whether `p_knng` has the same problem awaits the golden output (item 1).

> **Results (2026-10-12, `golden` workflow on a GitHub x86-64 runner + `bench/pknng_analysis.py`;
> raw numbers in `bench/out/golden_pknng.json` and `bench/out/pknng_analysis.json`).**
> `p_knng_Linux` was run with the original's exact arguments on deduplicated and raw Levine
> subsamples of 20k and 50k cells.
>
> - **Item 1 — resolved: `.clu` is 1-based** despite `-writeZidx`. Read as 1-based, indices span
>   exactly 0…n−1 and no row contains itself; read as 0-based, indices reach n (out of range).
>   The original's `columns - 1` is correct, so **the published pipeline has no off-by-one.**
> - **Item 4 — resolved: `p_knng` excludes self correctly**, also on raw data with duplicate cells
>   (0 rows contain themselves). The self-exclusion bug exists only in the non-Linux `bc()` fallback.
> - **Item 3 — refuted: `p_knng` is not an exact cosine kNN on Levine.** Against CosTaL+'s exact
>   kNN (deduplicated, 50k): mean recall@10 is **98.2%**, **17.1% of rows** have at least one
>   different neighbor, and the worst row gets 7 of 10. The differences are not ties: in differing
>   rows the cosine gap reaches 2–3×10⁻³, and even on rows with identical neighbor sets the reported
>   similarity differs from the true cosine (median 2×10⁻⁵, max 3.2×10⁻³, in both directions).
>   ~~Cause unknown.~~ **Cause found (2026-10-13): `pl2knn` assumes non-negative input** — see
>   the next item. (The first attempt compared p_knng's output on *signed* input with the cosine
>   of clipped rows, which fits worse — max 6–8×10⁻³ — because p_knng does not clip either; it
>   drops *some* negative contributions, depending on the order it accumulates in.)
> - **Item 3, cause — confirmed: `pl2knn` treats `> 0` as "present".** The 0.2.0 source
>   (`src/pl2knn.c`, found in the author's archive; docs/02 addendum) uses the sign of a value as
>   a flag in two places. A candidate's partial dot product is kept only while it is positive
>   (`if (accum[lcid] > 0.0)` at line 732; at line 877, `accum ≤ 0` resets the candidate to
>   "not seen"), and the forward pass adds a product only when the query's value is positive
>   (`if (qrow[rind[j]] > 0)` at line 816). Both are correct for non-negative data and silently
>   wrong for signed data, where a partial sum can dip below zero and recover. **Test**: the
>   `golden` workflow with `clip=true` (`GOLDEN_CLIP=1`, Levine 20k; raw numbers in
>   `bench/out/golden_pknng_clip.json`) clips negatives to 0 *before* both p_knng and CosTaL+ see
>   the data. Deduplicated: **19,999 of 20,000 rows identical**; the one differing row has a
>   cosine gap of 9×10⁻⁸ (a float32 tie), and the reported values differ from the exact cosine by
>   at most 7×10⁻⁷ (float32 output). Raw (with duplicate cells): 98.5% identical; all 298
>   differing rows have gaps ≤ 1×10⁻⁷, i.e. exact duplicates whose tie order differs. So on
>   non-negative input `p_knng` *is* an exact cosine kNN, and the whole Levine discrepancy comes
>   from the negative entries that arcsinh leaves in Levine_32dim (26.53% of entries deduplicated,
>   26.9993% on the raw 265,627 rows). Side finding: that raw-file share is exactly the paper's
>   reported Levine_32dim "sparsity" of 26.999%. The original `calculate_sparsity` counts exact
>   zeros, and the file has none, so the paper's number was computed on data with negatives
>   zeroed (or counted as ≤ 0) — and on the raw file with its duplicate cells, since the
>   deduplicated share is 26.53%.
> - **Clustering impact: within Leiden's own noise.** Building the graph exactly as the published
>   Linux path does (p_knng's neighbors and values → original `tani_from_cos` → original
>   `build_graph`) and running Leiden with 5 seeds next to the exact graph: same-seed partition
>   agreement is ARI 0.886, about the same as Leiden agreeing with itself across seeds
>   (0.83–0.87). Against the gating labels: p_knng graph ARI 0.770 ± 0.019 / NMI 0.862 ± 0.009,
>   exact graph ARI 0.804 ± 0.073 / NMI 0.875 ± 0.026 — not distinguishable with 5 seeds.
>
> **Consequence for the default-value contract (§0.0, §6.5) — decided 2026-10-12** (under authority
> delegated by the author). `compat="paper"` cannot mean "bit-identical to the shipped binary": the
> binary's deviation from its own stated method has no identified cause, so it can't be reproduced
> deliberately. **`compat="paper"` pins the method as published** — exact cosine kNN, with every
> other item of §0.0 — and this section stays as the record of how the shipped `p_knng` deviates
> from it. Evidence that this costs nothing in accuracy: the clustering comparison above.
>
> **Reaffirmed 2026-10-13, with the cause known.** The deviation could now be reproduced (port
> the two sign checks), but it should not be: it is a non-negative-data algorithm applied to
> signed data, not part of the method the paper describes. Two consequences: (1) on data without
> negatives — every scRNA-seq dataset of the paper (log1p of counts) — the shipped binary was an
> exact cosine kNN, so `compat="paper"` reproduces those results up to float32 ties; (2) on
> cytometry data with negatives (Levine_32dim, and any other arcsinh-transformed set with
> negative entries), the paper's numbers came from the slightly perturbed graph, whose effect on clustering is within Leiden's noise (above).

### 4.1 Deterministic tie-break (must be settled first, otherwise all three layers above fail)

The matmul must use f32 (docs/03 §3③), with relative error ~1e-7, so ties will appear around rank k.
The rule is fixed:

```
When similarities are equal, take the smaller column index.
```

Note that the sort (§2.3) scrambles the original indices — **the tie-break must be decided by original index,
not by post-sort position**; otherwise the results change whenever k or the thread count changes.

> **Addendum (2026-10-12): the tie-break acts on the values after f64 rescoring** (docs/06 §1).
> f32 is only responsible for the full coarse selection of k+m candidates; the final ranking is determined by similarities recomputed in f64 in a fixed order.
> Otherwise, different platforms' matrix libraries round differently, which can flip the order around rank k — that is not exact equality,
> and the tie-break cannot handle it.

---

## 5. Experiment design: is this "+" actually worth it

`select=tanimoto` is something CosTaL never did. The paper's rationale for Tanimoto is
"distinguishing the same direction with different magnitudes" — **this rationale applies to neighbor selection no less than to edge weights**,
yet today it is only used for edge weights. This can be tested directly.

Levine_32dim, 14 manually gated populations, report ARI / NMI:

| | `weight=cosine` | `weight=tanimoto` |
|---|---|---|
| `select=cosine` | pure cosine baseline | **the published CosTaL** |
| `select=tanimoto` | new | new |

All run under `normalize=raw magnitude` (under L2 the second row degenerates, see §0.1).
By-products: the actual pruning rate of the §3 norm window, and how much the neighbor set changes when `select` is swapped.

**Three possible outcomes, each to be reported as it is**:
1. `select=tanimoto` is better → this is CosTaL+'s methodological contribution;
2. on par → the magnitude information is already fully used by the edge weights, **which is itself a clean conclusion**,
   and it saves a knob;
3. worse → cosine's neighbor set + Tanimoto's edge-weight shape is an interesting combination,
   and the original's choice was right. **Report it as it is; don't hide it.**

### 5.1 The more fundamental question: are row norms useful at all (added 2026-10-12)

Tanimoto can be rewritten as `T = c/(r + 1/r − c)`, `r = ‖a‖/‖b‖` — **the only information it has beyond cosine
comes from the norm ratio r**. And in the 2023–2026 survey, **no published study has tested "whether row norms can
improve clustering"** ([survey report](../reports/single_cell_and_spatial_transforms.md)). CosTaL+'s benchmark
can be the first to answer it. For each preprocessing:

1. Measure `Spearman(‖x‖, log total UMI)`, `Spearman(‖x‖, number of detected genes)`, and the η² of `‖x‖` on cell-type
   labels — to see whether the norm records technical quantities or biology;
2. At the same operating point, compare the ARI / NMI of cosine edge weights against Tanimoto edge weights;
3. **Positive control**: Levine (per-channel arcsinh, row norm = overall marker abundance);
   **negative control**: already unit-normalized embeddings (§0.1; same reasoning as the face experiment in docs/01 §7);
4. **Spatial data is evaluated with two kinds of labels**: spatial-domain labels and cell-type labels. Library size varies with tissue region
   (Bhuva 2024), so magnitude may help the former and hurt the latter.

Another directly relevant finding: the original scRNA-seq preprocessing has **no** `normalize_total` before `log1p`,
so the row norms there mainly record sequencing depth. How much of the Tanimoto gain the original paper observed on scRNA-seq
comes from depth and how much from biology currently cannot be told apart — this experiment can answer exactly that. See §6.7.

> **Measured on Levine (2026-10-12; `bench/stage_c2.py`, `bench/stage_c3.py`; deduplicated, k = 10;
> numbers in `bench/out/stage_c2.json`, `stage_c3.json`).**
>
> - **The row norm is mostly within-population variation:** the gating labels explain η² = 0.097 of its
>   variance; it tracks overall marker abundance (Spearman 0.81).
> - **The §5 2×2 at the paper's resolution 0.8** (leidenalg, 5 seeds), select/weight:
>   cosine/cosine 0.862 ± 0.052, cosine/tanimoto (published) 0.766 ± 0.009, tanimoto/cosine
>   0.769 ± 0.010, tanimoto/tanimoto 0.764 ± 0.011. Changing the selection metric replaces half of the
>   neighbors (mean Jaccard 0.495; 1.2% of rows identical) but leaves accuracy unchanged.
> - **The 0.1 gap between the weightings is a resolution effect, not structure.** Sweeping the resolution
>   over 0.2–2.0, at each setup's best resolution (0.2) the weightings are equal: leidenalg 0.962 vs 0.968,
>   igraph 0.969 vs 0.970, raw file with igraph 0.969 vs 0.968. Cosine weights lead only with leidenalg in
>   the middle range (0.6–1.0), where Tanimoto weights push it to more clusters at the same resolution;
>   with igraph there is no gap at any resolution.
> - **Outcome 2 of §5 ("tie") on this dataset:** on Levine, magnitude adds no signal for the gated
>   populations, in selection or in weighting. Defaults are unchanged (the published method stays the
>   default under §6.5, and nothing measured here beats it at its best resolution). Caveat: this is one
>   dataset, and η² already said magnitude was unlikely to help there; data where magnitude differs
>   between populations (§6.7 presets, spatial data) remains open.
> - **The resolution matters far more:** ARI spans 0.45–0.97 over 0.2–2.0, and docs/05 §1.1 shows why no
>   single value is right (rare populations need a higher one than large populations).

> **Measured on the paper's scRNA-seq sets (2026-10-13; `bench/small_datasets.py`, numbers in
> `bench/out/small_datasets.json`; k = 10, leidenalg with the paper's 10 iterations, 5 seeds,
> resolutions 0.05–2.0; data and labels in docs/03 §10).**
>
> Row norm under the paper preset (log1p of raw counts) and under `normalize_total`:
>
> | Dataset | ρ(‖x‖, total counts) paper → nt | η²(‖x‖ \| labels) paper → nt | η²(log total counts \| labels) |
> |---|---|---|---:|
> | E-MTAB-3321 | +0.88 → +0.38 | 0.55 → 0.77 | 0.13 |
> | GSE81861 | +0.84 → +0.54 | 0.59 → 0.60 | 0.48 |
> | GSE74672 | +0.93 → −0.26 | 0.07 → 0.59 | 0.21 |
> | GSE84133 | +0.79 → +0.22 | 0.68 → 0.82 | 0.41 |
>
> The 2×2 (select/weight), ARI at each setup's best resolution, paper preset (`normalize_total` in
> parentheses):
>
> | Dataset | cos/cos | **cos/tani (published)** | tani/cos | tani/tani |
> |---|---:|---:|---:|---:|
> | E-MTAB-3321 | 0.883 (0.883) | 0.883 (0.883) | 0.791 (0.883) | 0.883 (0.883) |
> | GSE81861 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
> | GSE74672 | 0.683 (0.676) | 0.662 (0.675) | 0.701 (0.702) | 0.704 (0.664) |
> | GSE84133 | 0.939 (0.939) | 0.938 (0.945) | 0.933 (0.941) | 0.932 (0.943) |
>
> - **Under the paper preset the row norm is mostly sequencing depth** (ρ with total counts
>   0.79–0.93), as §5.1 expected. `normalize_total` removes most of that, and what is left tracks the
>   cell type more, not less (η² up on three of four sets; on GSE74672 from 0.07 to 0.59).
> - **Still a tie.** At each setup's best resolution the four cells lie within 0.04 ARI of each other,
>   with no consistent winner (GSE74672: Tanimoto selection +0.02–0.04 under the paper preset, −0.01 to
>   +0.03 under `normalize_total`, against a seed SD of 0.01–0.05), except one small-sample case:
>   E-MTAB-3321 (124 cells), tani/cos 0.79 vs 0.88, only under the paper preset. Magnitude carries
>   label information here, but the direction already carries it too: Tanimoto adds no separable
>   signal. Together with Levine_32, this is outcome 2 of §5 on five datasets (but see Levine_13 below).
>   Defaults unchanged.
> - **Selection changes the neighbors more under the paper preset** (mean Jaccard between cosine and
>   Tanimoto neighbor sets 0.46–0.74) than under `normalize_total` (0.84–0.94): the depth term is what
>   Tanimoto selection mostly reacts to.

> **Levine_13 — the first dataset where magnitude helps (2026-10-13; same script, 3 seeds, 2 Leiden
> iterations; 24 gated populations, 13 markers).** Here the row norm explains **η² = 0.89** of the
> label variance (Levine_32: 0.10) and tracks overall marker abundance (Spearman 0.95).
>
> | select/weight | Best ARI (resolution) | Mean ARI over the grid | Best F1, rarest quarter |
> |---|---|---:|---:|
> | cosine/cosine | 0.817 ± 0.021 (1.5) | 0.675 | 0.184 |
> | **cosine/tanimoto (published)** | 0.846 ± 0.009 (1.0) | 0.704 | 0.194 |
> | tanimoto/cosine | 0.850 ± 0.019 (1.2) | 0.712 | 0.229 |
> | tanimoto/tanimoto | 0.852 ± 0.022 (1.5) | 0.713 | 0.239 |
>
> With 3 seeds every setup using magnitude looked ~0.03 ARI ahead of pure cosine. **Rechecked with 10
> seeds** (`bench/magnitude_confirm.py`, resolutions 0.8–2.0, comparisons paired by seed; numbers in
> `bench/out/magnitude_confirm.json`), the effect is real but smaller:
>
> | select/weight | Per-seed best ARI | vs cosine/cosine, paired | F1 rarest quarter at 1.5 / 2.0 |
> |---|---:|---|---|
> | cosine/cosine | 0.829 ± 0.024 | — | 0.14 / 0.15 |
> | **cosine/tanimoto (published)** | 0.851 ± 0.015 | +0.022, better on 8 of 10 seeds | 0.14 / 0.17 |
> | tanimoto/cosine | 0.839 ± 0.034 | +0.010, 6 of 10 | 0.21 / 0.23 |
> | tanimoto/tanimoto | 0.844 ± 0.022 | +0.015, 8 of 10 | 0.21 / 0.22 |
>
> - **Tanimoto weights help a little on ARI** (+0.02 for the published method; largest at the
>   paper's 0.8–1.0, +0.07 paired) — the published method's rationale, now
>   measured, on the one dataset where the row norm carries the labels (η² 0.89) and 13 markers leave
>   the direction short of information.
> - **Tanimoto selection doesn't add ARI but recovers rare populations better**: the rarest quarter's
>   F1 is higher at every resolution from 0.8 to 2.0 (0.15–0.23 vs 0.07–0.17), at equal ARI.
> - So outcome 2 of §5 is not universal: on low-dimensional panels where populations differ in overall
>   abundance, magnitude adds signal. Defaults stay as published (Tanimoto weights; that is also what
>   the data favor); `select="tanimoto"` is worth trying on such panels when rare populations matter.
>   Caveat: one dataset.
> - **Control: the same check on Levine_32** (η² 0.10). Tanimoto selection's rarest-quarter F1 is higher
>   only at 0.8 (0.38–0.40 vs 0.31–0.33) and level from 1.0 to 2.0; ARI on this grid just follows the
>   resolution effect above (every setup's best lies at 0.2, below this grid). The rare-population
>   gain tracks how much of the labels the row norm carries.

---

## 6. API shape

### 6.1 Split into "the expensive step" and "the free step"

kNN is the only expensive part; edge weights are a pure transform on `(cos, ‖a‖, ‖b‖)`, `O(N·k)`,
about 10 ms in Rust. So we split them apart, letting callers swap edge weights **on the same kNN**:

```python
knn = costal_plus.knn(X, k=10, select="cosine", normalize="l2")
#   → knn.idx (N,k) int32 · knn.cos (N,k) f64 · knn.norms (N,) f64
#     (f64: the rescored values are kept as is, so reweight is bit-identical to the one-step result, docs/06 §1)

e1 = costal_plus.reweight(knn, "tanimoto")   # ~10 ms
e2 = costal_plus.reweight(knn, "cosine")     # ~10 ms
e3 = costal_plus.reweight(knn, "jaccard")    # exception, see 6.3

knn5 = costal_plus.truncate(knn, k=5)        # take each row's prefix, ~0 cost

costal_plus.build_graph(X, k=10)             # one-step convenience entry point
costal_plus.find_communities(X, k=10)        # end-to-end (pending the cluster layer)
```

Benefits: the 2×2 experiment of §5 drops from "running kNN four times" to **running kNN twice and producing six sets of edge weights**;
nas-manager's face pipeline can likewise sweep weight and resolution repeatedly on a single kNN graph
(docs/01 §7.2: resolution is the most valuable knob, exactly the thing that needs repeated sweeps).

`truncate()` (added 2026-10-12; docs/05 §2.3 already references it): the same goes for sweeping k —
compute once at k_max, and for smaller k just take the prefix. This holds on the premise that each row is totally ordered by
(f64 similarity descending, original index ascending) (docs/06 §1);
then the prefix **is** the exact answer for the smaller k, not an approximation.

### 6.2 Defaults

```python
knn(X, k=10, select="cosine", normalize=None, backend="exact")
build_graph(X, k=10, select="cosine", weight="tanimoto",
            normalize=None, backend="exact", symmetrize="fuzzy_union")
```

`normalize=None` resolves to `"l2"` for cosine and `"none"` for Tanimoto (stage A+, 2026-10-12):
cosine is scale-invariant, and Tanimoto selection only means something on raw magnitudes. Asking
explicitly for `select="tanimoto", normalize="l2"` runs cosine and warns (§0.1).

**Passing no optional parameters ⇒ the output must be identical, entry for entry, to the published CosTaL** (all six items of §0.0 pinned).

### 6.3 Two constraints that must go into the documentation

1. **reweight ≠ changing the selection.** Reweighting only changes the numbers on the edges; it cannot change which edges are in the graph.
   `select=cosine` + `weight=tanimoto` (the published CosTaL) and
   `select=tanimoto` + `weight=tanimoto` are **two different graphs** —
   the `Bⱼ` in the Tanimoto criterion pushes candidates with large norms out of the top k
   (docs/02 §4 measured that the two rankings disagree, Pearson r=0.780).
2. **`weight=jaccard` is not in the free tier.** Converting tanimoto↔cosine only needs
   `(cos, ‖a‖, ‖b‖)`, all of which are at hand; Jaccard has to compute intersections of neighbors' neighbors,
   which takes another pass over the kNN result, on the order of `O(N·k²)`
   (the original's `map_jaccard` opens a multiprocessing Pool for this).

### 6.4 Why not two functions / two binaries

1. The expensive parts (matmul, blocking, parallelism, graph assembly, IO) are **100% shared**;
   splitting them means maintaining the same code twice;
2. The three-layer verification of §4 **has to run over all combinations anyway**; splitting into two entry points just adds a layer of dispatch;
3. Most importantly: **future pruning backends have to serve both selection metrics at once**
   (the §3 norm window for tanimoto, the cone bound for cosine).
   If we fork at the selection-metric level, the pruning layer is forced to be written twice.
   The fork point belongs in the **comparator** (a dozen or so lines), not in **versions**.

### 6.5 Compatibility escape hatch (corresponds to docs/01 §6 item 5)

**Decided (2026-09-20): route (b).** The defaults **may** change — if the §5 experiments show
that another combination is better, people who `pip install` should get the better thing directly, instead of having it
locked behind a flag forever. But with two hard constraints:

1. **Changing a default requires a major-version bump** (0.x → 1.0 → 2.0). When someone writes
   "we used CosTaL v2.1", the major version alone is enough to determine the default semantics.
2. **`compat="paper"` exists from day one and is never removed.** The reproduction path is explicit and never breaks.

So from day one we provide:

```python
costal_plus.knn(X, k=10, compat="paper")   # explicitly locks in all of bbad157's semantics
```

That way, even if the defaults later change because of the §5 experiment results, **the reproduction path always exists and is explicit**.

**What "paper" means (decided 2026-10-12, §4 "Results")**: the method as published — exact cosine
kNN plus the six items of §0.0 — not the output of the `p_knng` binary that shipped with CosTaL
0.0.2.4, which is measurably inexact on Levine (recall@10 98.2%) for reasons not yet identified.

### 6.6 Input diagnostics recorded in the metadata (added 2026-10-12)

The return value of `knn()` carries input diagnostics, **recorded only; the data is not modified**:

| Field | Purpose |
|---|---|
| Coefficient of variation of row norms | ≈ 0 ⇒ input already unit-normalized, Tanimoto ≡ `c/(2−c)`, warn once (§0.1 generalization) |
| Column means, column standard deviations | columns already centered (PCA, z-score, Pearson residuals) ⇒ warn: the meaning of the row norm has shifted from "abundance" to "distance from the data center" |
| Fraction of negative values | on signed input the lower bound of T is −1/3; the §3 norm window is only usable when τ > 0 |
| Fraction of nonzeros | basis for dispatching to the dense / sparse backend (docs/06 §5.2) |

**How to handle edges with T < 0: undecided.** `compat="paper"` keeps them as is (that is what the original does, no clipping);
the other modes record the number of negative edges in the metadata for now, and wait for the §5.1 experimental data to decide whether to clip them to 0 or drop them.

logicle input (FlowJo's display default, FlowSOM's `transformFunction` default): by definition, data value 0
does not map to 0, and all cells share a positive offset, so cosine's origin loses its meaning of "no signal".
**Measure it once with flowCore first**, then decide whether to reject it or re-center it.

### 6.7 Two scRNA-seq presets (added 2026-10-12)

| Preset | Steps | Status |
|---|---|---|
| `compat="paper"` | `filter_cells` → `filter_genes` → `seurat_v3` selects 2000 HVGs (raw counts) → `log1p` | bit-for-bit reproduction of the original, never removed |
| `normalize_total` preset | same as above, but with `normalize_total(target_sum=None)` inserted before `log1p` | aligned with the defaults of the three major ecosystems as of 2023–2026 |

- Both **preserve zeros and are non-negative**, and both use the sparse backend, so switching doesn't affect the backend design.
- **No CPM preset**: Ahlmann-Eltze & Huber 2023 and the scverse best practices both explicitly advise against feeding CPM into a shifted log.
- Compare the two with the §5.1 experiment. If the new preset wins, change the default via route (b) of §6.5 and bump the major version.
- Preprocessing stays on the Python side (docs/03 §6.1); this is just one more preset.

> **Implemented and measured (2026-10-13).** `costal_plus.preprocess(X, platform, preset=...)`
> (`compat="paper"` refuses `normalize_total`). Published combination (cosine selection, Tanimoto
> weights), mean ARI over the resolution grid 0.05–2.0 and ARI at the paper's 0.8, paper → `normalize_total`:
>
> | Dataset | Mean ARI over the grid | ARI at 0.8 | Best ARI |
> |---|---|---|---|
> | E-MTAB-3321 | 0.627 → 0.678 | 0.687 → 0.687 | 0.883 → 0.883 |
> | GSE81861 | 0.958 → 0.988 | 1.000 → 1.000 | 1.000 → 1.000 |
> | GSE74672 | 0.478 → 0.487 | 0.473 → 0.476 | 0.662 → 0.675 |
> | GSE84133 | 0.829 → 0.860 | 0.857 → 0.939 | 0.938 → 0.945 |
>
> `normalize_total` is never worse and is less sensitive to the resolution (higher on the grid mean on
> all four; +0.08 at the paper's 0.8 on GSE84133). The gain is modest and the best ARI is unchanged, so
> under versioning policy (b) it stays an opt-in preset; it is the first candidate if a v2 ever
> changes defaults.

### 6.8 An unverified design direction: an external magnitude vector

T depends only on c and r, so **r does not have to come from the matrix used to select neighbors**. `reweight()` could accept
an externally supplied magnitude vector m — raw total counts, cell area, arcsinh abundance — and compute edge weights with
`T = c/(m_a/m_b + m_b/m_a − c)`. That way neighbors can be selected on shifted log or PCA,
while "what magnitude stands for" is injected explicitly as a visible parameter that can be written into a paper's methods section,
rather than letting it change silently with the preprocessing. **Decide whether to build this once the §5.1 data is in.**

## 7. Mapping onto the docs/03 stage table

| Stage | Increment |
|---|---|
| **A** | only `select=cosine`, but build the trait and generic-kernel skeleton in one go; also do the §2.3 sort-by-norm in one go (the cosine route doesn't need it but it does no harm, and the §4.1 tie-break depends on the original-index mapping, so better to get it right from the start) |
| **A+** | add the `select=tanimoto` comparator + the §2.2 conservative threshold + the §2.4 guardrail. **About a dozen or so lines + a set of tests** — **done 2026-10-12** (docs/03 §9) |
| **C′** | the §5 2×2 experiment (run together with the Leiden tuning experiment, sharing one metrics script); the same scripts run the §5.1 row-norm experiment and the §6.7 comparison of the two scRNA presets |
| **D** | the §3 norm window (tanimoto) and cone bound (cosine), behind the same trait |
