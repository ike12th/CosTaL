//! Exact kNN: fused blocked f32 coarse selection over all pairs + f64 rescoring
//! (docs/03 §2.2, docs/06 §1).
//!
//! 1. Row norms (f64), a norm-sorted working layout (docs/04 §2.3), and an L2-normalized f32 copy;
//! 2. **Coarse selection**: for every (row block × column block) tile, run the matmul and merge
//!    the tile into each row's top-(k+m) heap right away; the tile buffer is then reused.
//!    All N² similarities are computed — only the precision is coarse, not the coverage;
//! 3. **Rescoring**: each row's k+m candidates are recomputed in f64 with a fixed order over the
//!    dimensions and ranked by (score desc, original index asc). The final ranking depends only
//!    on this step, so tile sizes, thread count and matmul backend never change the output.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::cache;
use crate::matmul::{self, Matmul};

/// Default column-block width. The row-block height comes from the cache budget (docs/06 §2.3).
pub const DEFAULT_COL_BLOCK: usize = 1024;

/// kNN parameters. `KnnParams::new(k)` gives defaults matching the published CosTaL.
#[derive(Clone, Debug)]
pub struct KnnParams {
    /// Neighbors per row, excluding the row itself (docs/04 §0.0 item 4).
    pub k: usize,
    /// Coarse-selection margin m: each row keeps top-(k+m) for f64 rescoring (docs/06 §1.2).
    pub margin: usize,
    /// Row-block height; `None` derives it from the cache budget.
    pub row_block: Option<usize>,
    /// Column-block width; `None` uses [`DEFAULT_COL_BLOCK`].
    pub col_block: Option<usize>,
    pub matmul: Matmul,
    /// Norm-sorted working layout (docs/04 §2.3). Cosine selection doesn't need it but it is
    /// harmless; the switch exists only to test that the layout doesn't change the output.
    pub sort_by_norm: bool,
}

impl KnnParams {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            margin: k,
            row_block: None,
            col_block: None,
            matmul: Matmul::platform_default(),
            sort_by_norm: true,
        }
    }
}

/// kNN result, all in original row order.
#[derive(Clone, Debug, PartialEq)]
pub struct Knn {
    pub n: usize,
    pub k: usize,
    /// Row-major n×k original indices; each row sorted by (cos desc, index asc).
    pub idx: Vec<u32>,
    /// Row-major n×k cosine after f64 rescoring.
    pub cos: Vec<f64>,
    /// Raw (un-normalized) row norms, f64.
    pub norms: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KnnError {
    Shape { len: usize, n: usize, d: usize },
    BadK { k: usize, n: usize },
    TooManyRows { n: usize },
    NonFinite { row: usize },
    MatmulUnavailable(&'static str),
}

impl fmt::Display for KnnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KnnError::Shape { len, n, d } => {
                write!(f, "data length {len} does not match shape {n}x{d}")
            }
            KnnError::BadK { k, n } => write!(f, "invalid k={k}: need 1 <= k <= n-1 (n={n})"),
            KnnError::TooManyRows { n } => write!(f, "n={n} exceeds the u32 index range"),
            KnnError::NonFinite { row } => write!(f, "row {row} contains NaN or inf"),
            KnnError::MatmulUnavailable(name) => {
                write!(f, "matmul backend {name} is not available on this platform")
            }
        }
    }
}

impl std::error::Error for KnnError {}

/// Comparator for the selection metric (docs/04 §1). The inner loop is monomorphized over it,
/// with a single `match` at the entry point.
///
/// Stage A has only [`Cosine`]. Tanimoto arrives in stage A+: `tile_threshold` gives a
/// conservative threshold from the column block's smallest norm (columns are sorted by norm
/// ascending, so the minimum is at `col_start`), and `score` computes the exact score on the
/// rare path (docs/04 §2).
pub trait Selector: Sync {
    /// Per-row precomputed constants.
    type RowState: Copy;
    fn row_state(&self, row: usize) -> Self::RowState;
    /// Conservative scalar threshold for a column block: a candidate whose tile value does not
    /// exceed it can never enter the heap. `tau` is the row's heap threshold (-inf while the
    /// heap is not full); `col_start` is the block's first column (working order).
    fn tile_threshold(&self, st: Self::RowState, tau: f32, col_start: usize) -> f32;
    /// Rare path: the selection score after passing the threshold (f32, coarse selection only).
    fn score(&self, st: Self::RowState, col: usize, sim: f32) -> f32;
    /// Rescoring score (f64), from the rescored cosine and both raw norms.
    fn rescore(&self, cos: f64, norm_i: f64, norm_j: f64) -> f64;
}

/// Select neighbors by cosine — the selection metric of the published CosTaL.
pub struct Cosine;

impl Selector for Cosine {
    type RowState = ();
    #[inline(always)]
    fn row_state(&self, _row: usize) {}
    #[inline(always)]
    fn tile_threshold(&self, _st: (), tau: f32, _col_start: usize) -> f32 {
        tau
    }
    #[inline(always)]
    fn score(&self, _st: (), _col: usize, sim: f32) -> f32 {
        sim
    }
    #[inline(always)]
    fn rescore(&self, cos: f64, _norm_i: f64, _norm_j: f64) -> f64 {
        cos
    }
}

/// f64 dot product in a fixed order over the dimension index. f32×f32 is exact in f64; Rust
/// does not contract floating-point ops or reorder float sums by default, so this is the single
/// source of cross-platform bit-for-bit agreement (docs/06 §1.2).
#[inline]
pub fn dot64(a: &[f32], b: &[f32]) -> f64 {
    let mut s = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        s += f64::from(*x) * f64::from(*y);
    }
    s
}

/// Cosine for rescoring; a zero vector has cosine 0 with everything (matching zero rows after
/// the original's L2 normalization).
#[inline]
fn cos64(dot: f64, norm_i: f64, norm_j: f64) -> f64 {
    if norm_i == 0.0 || norm_j == 0.0 {
        return 0.0;
    }
    let c = dot / (norm_i * norm_j);
    // Normalize -0.0 to 0.0 so the total order used for sorting agrees with numeric equality
    if c == 0.0 {
        0.0
    } else {
        c
    }
}

/// Raw row norms, f64.
pub fn row_norms64(x: &[f32], n: usize, d: usize) -> Vec<f64> {
    (0..n)
        .into_par_iter()
        .map(|i| {
            let r = &x[i * d..(i + 1) * d];
            dot64(r, r).sqrt()
        })
        .collect()
}

fn validate(x: &[f32], n: usize, d: usize, k: usize) -> Result<(), KnnError> {
    if x.len() != n * d {
        return Err(KnnError::Shape { len: x.len(), n, d });
    }
    if n > u32::MAX as usize {
        return Err(KnnError::TooManyRows { n });
    }
    if k == 0 || k >= n {
        return Err(KnnError::BadK { k, n });
    }
    let bad = (0..n)
        .into_par_iter()
        .find_first(|&i| x[i * d..(i + 1) * d].iter().any(|v| !v.is_finite()));
    match bad {
        Some(row) => Err(KnnError::NonFinite { row }),
        None => Ok(()),
    }
}

/// Side statistics of an exact kNN run. Kept out of [`Knn`] so they don't affect result comparisons.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KnnStats {
    /// Rows whose coarse threshold was too close to the k-th score and that took the
    /// "regather the whole row + rescore" path. Results are still exact, just slower for these
    /// rows; this counts how often the "missed in coarse selection" risk of docs/06 §1.2 was caught.
    pub fallback_rows: usize,
    pub row_block: usize,
    pub col_block: usize,
}

/// Exact kNN, selecting neighbors by cosine (stage A).
pub fn knn_exact(x: &[f32], n: usize, d: usize, params: &KnnParams) -> Result<Knn, KnnError> {
    knn_exact_with_stats(x, n, d, params).map(|(knn, _)| knn)
}

/// Like [`knn_exact`], also returning statistics.
pub fn knn_exact_with_stats(
    x: &[f32],
    n: usize,
    d: usize,
    params: &KnnParams,
) -> Result<(Knn, KnnStats), KnnError> {
    validate(x, n, d, params.k)?;
    if !params.matmul.is_available() {
        return Err(KnnError::MatmulUnavailable(params.matmul.name()));
    }
    Ok(run_blocked(&Cosine, x, n, d, params))
}

/// Read-only state shared by all threads within one call.
struct Ctx<'a, S> {
    sel: &'a S,
    x: &'a [f32],
    /// L2-normalized f32 copy in working order, used only for coarse selection.
    xn: &'a [f32],
    norms: &'a [f64],
    /// Working order → original order.
    perm: &'a [u32],
    n: usize,
    d: usize,
    k: usize,
    cap: usize,
    cb: usize,
    backend: Matmul,
    /// Upper bound on |f32 coarse value − true cosine|. Normalizing to f32 (relative error ≤ u)
    /// plus an f32 dot product in any summation order (≤ γ_d) is on the order of (d+2)·u; we use
    /// (d+4)·2⁻²³, a 2× margin.
    eps: f64,
}

fn run_blocked<S: Selector>(
    sel: &S,
    x: &[f32],
    n: usize,
    d: usize,
    p: &KnnParams,
) -> (Knn, KnnStats) {
    let k = p.k;
    let norms = row_norms64(x, n, d);

    // Working order → original order, ascending by (norm, original index): the stable sort
    // keeps original-index order among equal norms
    let mut perm: Vec<u32> = (0..n as u32).collect();
    if p.sort_by_norm {
        perm.sort_by(|&a, &b| norms[a as usize].total_cmp(&norms[b as usize]));
    }

    let mut xn = vec![0.0f32; n * d];
    xn.par_chunks_mut(d.max(1))
        .zip(perm.par_iter())
        .for_each(|(dst, &orig)| {
            let o = orig as usize;
            let nrm = norms[o];
            if nrm > 0.0 {
                for (v, &s) in dst.iter_mut().zip(&x[o * d..(o + 1) * d]) {
                    *v = (f64::from(s) / nrm) as f32;
                }
            }
        });

    let (rb, cb) = block_sizes(n, p);
    let ctx = Ctx {
        sel,
        x,
        xn: &xn,
        norms: &norms,
        perm: &perm,
        n,
        d,
        k,
        cap: (k + p.margin).min(n - 1),
        cb,
        backend: p.matmul,
        eps: (d as f64 + 4.0) * f64::from(f32::EPSILON),
    };
    let fallbacks = AtomicUsize::new(0);
    let mut w_idx = vec![0u32; n * k];
    let mut w_cos = vec![0.0f64; n * k];
    w_idx
        .par_chunks_mut(rb * k)
        .zip(w_cos.par_chunks_mut(rb * k))
        .enumerate()
        .for_each_init(
            || Scratch::new(rb, cb, ctx.cap),
            |scr, (bi, (out_idx, out_cos))| {
                let r0 = bi * rb;
                let rows = out_idx.len() / k;
                ctx.coarse_block(r0, rows, scr);
                for r in 0..rows {
                    let rows_out = r * k..(r + 1) * k;
                    let fell_back = ctx.finalize_row(
                        r0 + r,
                        r,
                        scr,
                        &mut out_idx[rows_out.clone()],
                        &mut out_cos[rows_out],
                    );
                    if fell_back {
                        fallbacks.fetch_add(1, Ordering::Relaxed);
                    }
                }
            },
        );

    // Working order → original order
    let mut idx = vec![0u32; n * k];
    let mut cos = vec![0.0f64; n * k];
    for (w, &orig) in perm.iter().enumerate() {
        let o = orig as usize;
        idx[o * k..(o + 1) * k].copy_from_slice(&w_idx[w * k..(w + 1) * k]);
        cos[o * k..(o + 1) * k].copy_from_slice(&w_cos[w * k..(w + 1) * k]);
    }
    let stats = KnnStats {
        fallback_rows: fallbacks.into_inner(),
        row_block: rb,
        col_block: cb,
    };
    (
        Knn {
            n,
            k,
            idx,
            cos,
            norms,
        },
        stats,
    )
}

/// Row-block height and column-block width.
fn block_sizes(n: usize, p: &KnnParams) -> (usize, usize) {
    let cb = p.col_block.unwrap_or(DEFAULT_COL_BLOCK).clamp(1, n);
    let rb = p
        .row_block
        .unwrap_or_else(|| {
            let by_cache = cache::tile_budget_bytes() / (cb * std::mem::size_of::<f32>());
            // Give each thread ~8 row blocks so work stealing can balance big and little cores
            // (docs/06 §4)
            let by_balance = n.div_ceil(rayon::current_num_threads() * 8);
            by_cache.min(by_balance).clamp(8, 512)
        })
        .clamp(1, n);
    (rb, cb)
}

/// Per-thread reusable buffers.
struct Scratch {
    tile: Vec<f32>,
    heaps: BlockHeaps,
    cands: Vec<(f64, f64, u32)>,
}

impl Scratch {
    fn new(rb: usize, cb: usize, cap: usize) -> Self {
        Self {
            tile: vec![0.0; rb * cb],
            heaps: BlockHeaps::new(rb, cap),
            cands: Vec::with_capacity(cap),
        }
    }
}

/// Sort candidates by (score desc, original index asc).
fn sort_cands(c: &mut [(f64, f64, u32)]) {
    c.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.2.cmp(&b.2)));
}

impl<S: Selector> Ctx<'_, S> {
    /// Coarse selection for one row block: matmul per column block, merged into each row's heap.
    fn coarse_block(&self, r0: usize, rows: usize, scr: &mut Scratch) {
        let d = self.d;
        scr.heaps.reset();
        let a = &self.xn[r0 * d..(r0 + rows) * d];
        let mut c0 = 0;
        while c0 < self.n {
            let cols = self.cb.min(self.n - c0);
            let tile = &mut scr.tile[..rows * cols];
            let b = &self.xn[c0 * d..(c0 + cols) * d];
            matmul::tile_nt(self.backend, a, rows, b, cols, d, tile);
            for r in 0..rows {
                let st = self.sel.row_state(r0 + r);
                let row = &tile[r * cols..(r + 1) * cols];
                scan_row(self.sel, st, row, c0, r0 + r, &mut scr.heaps, r);
            }
            c0 += cols;
        }
    }

    /// Rescore one row; returns whether the fallback path was taken.
    ///
    /// Every candidate that never entered the heap has an f32 value no greater than the heap
    /// root (the coarse threshold), so its true cosine is at most threshold + ε. As long as the
    /// rescored k-th score is strictly above that bound, no such candidate can enter the top k
    /// or tie, and the result is exact. Otherwise the whole row is regathered — every candidate
    /// with f32 value ≥ k-th score − ε (all candidates with true cosine ≥ the k-th score are
    /// among them) — and rescored together.
    ///
    /// The bound is derived for cosine; stage A+ must re-derive it for Tanimoto scores.
    fn finalize_row(
        &self,
        w_row: usize,
        r: usize,
        scr: &mut Scratch,
        out_idx: &mut [u32],
        out_cos: &mut [f64],
    ) -> bool {
        let k = self.k;
        let i = self.perm[w_row] as usize;
        scr.cands.clear();
        for &wj in scr.heaps.ids_of(r) {
            self.push_rescored(i, wj as usize, &mut scr.cands);
        }
        sort_cands(&mut scr.cands);

        // With cap == n-1 the heap holds every candidate, so nothing was left out
        let mut fell_back = false;
        if self.cap < self.n - 1 {
            let floor = f64::from(scr.heaps.tau(r));
            let s_k = scr.cands[k - 1].0;
            if s_k <= floor + self.eps {
                fell_back = true;
                self.gather_row(w_row, s_k - self.eps, scr);
                sort_cands(&mut scr.cands);
            }
        }
        for (t, &(_, c, j)) in scr.cands.iter().take(k).enumerate() {
            out_idx[t] = j;
            out_cos[t] = c;
        }
        fell_back
    }

    /// Fallback: rescan all columns for one row, gather candidates with f32 value ≥ `min_sim`, and rescore them.
    fn gather_row(&self, w_row: usize, min_sim: f64, scr: &mut Scratch) {
        let d = self.d;
        let i = self.perm[w_row] as usize;
        let Scratch { tile, cands, .. } = scr;
        cands.clear();
        let a = &self.xn[w_row * d..(w_row + 1) * d];
        let mut c0 = 0;
        while c0 < self.n {
            let cols = self.cb.min(self.n - c0);
            let b = &self.xn[c0 * d..(c0 + cols) * d];
            matmul::tile_nt(self.backend, a, 1, b, cols, d, &mut tile[..cols]);
            for (t, &v) in tile[..cols].iter().enumerate() {
                let wj = c0 + t;
                if wj != w_row && f64::from(v) >= min_sim {
                    self.push_rescored(i, wj, cands);
                }
            }
            c0 += cols;
        }
    }

    /// Recompute working-order candidate `wj` in f64 and push it into `cands` with its original index.
    fn push_rescored(&self, i: usize, wj: usize, cands: &mut Vec<(f64, f64, u32)>) {
        let d = self.d;
        let j = self.perm[wj];
        let ju = j as usize;
        let dot = dot64(&self.x[i * d..(i + 1) * d], &self.x[ju * d..(ju + 1) * d]);
        let c = cos64(dot, self.norms[i], self.norms[ju]);
        cands.push((self.sel.rescore(c, self.norms[i], self.norms[ju]), c, j));
    }
}

/// Scan one tile row. Almost every candidate is rejected at the "whole chunk is below the
/// threshold" step; self-exclusion and heap insertion happen only on the rare path.
#[inline(always)]
fn scan_row<S: Selector>(
    sel: &S,
    st: S::RowState,
    row: &[f32],
    c0: usize,
    self_col: usize,
    heaps: &mut BlockHeaps,
    r: usize,
) {
    const LANES: usize = 16;
    let mut th = sel.tile_threshold(st, heaps.tau(r), c0);
    let mut base = c0;
    for chunk in row.chunks(LANES) {
        let mut hit = false;
        for &v in chunk {
            hit |= v > th;
        }
        if hit {
            for (t, &v) in chunk.iter().enumerate() {
                let col = base + t;
                if v > th && col != self_col {
                    heaps.push(r, sel.score(st, col, v), col as u32);
                    th = sel.tile_threshold(st, heaps.tau(r), c0);
                }
            }
        }
        base += chunk.len();
    }
}

/// One fixed-capacity min-heap per row of a row block. The root is the "worst" candidate: a lower
/// score is worse, and on equal scores a larger index is worse.
struct BlockHeaps {
    cap: usize,
    scores: Vec<f32>,
    ids: Vec<u32>,
    lens: Vec<usize>,
}

#[inline(always)]
fn worse(sa: f32, ia: u32, sb: f32, ib: u32) -> bool {
    sa < sb || (sa == sb && ia > ib)
}

impl BlockHeaps {
    fn new(rows: usize, cap: usize) -> Self {
        Self {
            cap,
            scores: vec![0.0; rows * cap],
            ids: vec![0; rows * cap],
            lens: vec![0; rows],
        }
    }

    fn reset(&mut self) {
        self.lens.fill(0);
    }

    /// Heap threshold: -inf while not full, otherwise the root's score.
    #[inline(always)]
    fn tau(&self, r: usize) -> f32 {
        if self.lens[r] < self.cap {
            f32::NEG_INFINITY
        } else {
            self.scores[r * self.cap]
        }
    }

    fn ids_of(&self, r: usize) -> &[u32] {
        &self.ids[r * self.cap..r * self.cap + self.lens[r]]
    }

    fn push(&mut self, r: usize, score: f32, id: u32) {
        let cap = self.cap;
        let s = &mut self.scores[r * cap..(r + 1) * cap];
        let ids = &mut self.ids[r * cap..(r + 1) * cap];
        let len = self.lens[r];
        if len < cap {
            let mut pos = len;
            s[pos] = score;
            ids[pos] = id;
            while pos > 0 {
                let parent = (pos - 1) / 2;
                if !worse(s[pos], ids[pos], s[parent], ids[parent]) {
                    break;
                }
                s.swap(pos, parent);
                ids.swap(pos, parent);
                pos = parent;
            }
            self.lens[r] = len + 1;
            return;
        }
        if !worse(s[0], ids[0], score, id) {
            return;
        }
        s[0] = score;
        ids[0] = id;
        let mut pos = 0;
        loop {
            let left = 2 * pos + 1;
            if left >= cap {
                break;
            }
            let mut w = left;
            if left + 1 < cap && worse(s[left + 1], ids[left + 1], s[left], ids[left]) {
                w = left + 1;
            }
            if !worse(s[w], ids[w], s[pos], ids[pos]) {
                break;
            }
            s.swap(pos, w);
            ids.swap(pos, w);
            pos = w;
        }
    }
}

/// Golden reference: no blocking, no pruning, f64 throughout (layer L2 of docs/04 §4).
/// O(N²·d); for tests and benchmarks only.
pub fn knn_bruteforce(x: &[f32], n: usize, d: usize, k: usize) -> Result<Knn, KnnError> {
    validate(x, n, d, k)?;
    let norms = row_norms64(x, n, d);
    let mut idx = vec![0u32; n * k];
    let mut cos = vec![0.0f64; n * k];
    idx.par_chunks_mut(k)
        .zip(cos.par_chunks_mut(k))
        .enumerate()
        .for_each(|(i, (oi, oc))| {
            let xi = &x[i * d..(i + 1) * d];
            let mut all: Vec<(f64, u32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    (
                        cos64(dot64(xi, &x[j * d..(j + 1) * d]), norms[i], norms[j]),
                        j as u32,
                    )
                })
                .collect();
            all.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            for (t, &(c, j)) in all.iter().take(k).enumerate() {
                oi[t] = j;
                oc[t] = c;
            }
        });
    Ok(Knn {
        n,
        k,
        idx,
        cos,
        norms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reproducible pseudo-random numbers (no extra dependency).
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
        }
    }

    /// Clustered data with negative values, shaped like CyTOF.
    fn clustered(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut g = Lcg(seed);
        let centers: Vec<f32> = (0..8 * d).map(|_| g.next_f32() * 4.0).collect();
        (0..n)
            .flat_map(|i| {
                let c = i % 8;
                (0..d)
                    .map(|t| centers[c * d + t] + g.next_f32() - 0.55)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn params(k: usize) -> KnnParams {
        KnnParams::new(k)
    }

    #[test]
    fn matches_bruteforce_exactly() {
        let (n, d, k) = (1500, 32, 10);
        let x = clustered(n, d, 1);
        let want = knn_bruteforce(&x, n, d, k).unwrap();
        let got = knn_exact(&x, n, d, &params(k)).unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn output_is_independent_of_blocks_backend_layout_and_threads() {
        let (n, d, k) = (900, 17, 7);
        let x = clustered(n, d, 2);
        let want = knn_bruteforce(&x, n, d, k).unwrap();
        let backends = [Matmul::Gemm, Matmul::platform_default()];
        for backend in backends {
            for (rb, cb) in [(1, 1), (8, 64), (33, 129), (512, 4096)] {
                for sort in [false, true] {
                    let p = KnnParams {
                        row_block: Some(rb),
                        col_block: Some(cb),
                        matmul: backend,
                        sort_by_norm: sort,
                        ..params(k)
                    };
                    for threads in [1, 4] {
                        let pool = rayon::ThreadPoolBuilder::new()
                            .num_threads(threads)
                            .build()
                            .unwrap();
                        let got = pool.install(|| knn_exact(&x, n, d, &p).unwrap());
                        assert_eq!(
                            got, want,
                            "{backend:?} rb={rb} cb={cb} sort={sort} t={threads}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn exact_duplicates_break_ties_by_original_index() {
        // 10 distinct vectors, 30 copies each: every row has 29 tied neighbors with cos=1,
        // more than k+m=20
        let (d, k) = (5, 10);
        let mut g = Lcg(3);
        let protos: Vec<f32> = (0..10 * d).map(|_| g.next_f32() - 0.3).collect();
        let n = 300;
        let x: Vec<f32> = (0..n)
            .flat_map(|i| protos[(i % 10) * d..(i % 10 + 1) * d].to_vec())
            .collect();
        let want = knn_bruteforce(&x, n, d, k).unwrap();
        let got = knn_exact(&x, n, d, &params(k)).unwrap();
        assert_eq!(got, want);
        // Row 0's neighbors must be the 10 smallest indices of the same prototype (10, 20, …, 100)
        assert_eq!(&got.idx[..k], &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    }

    #[test]
    fn zero_rows_and_integer_ties() {
        // Small-integer data has many exact ties; mix in a few all-zero rows
        let (n, d, k) = (400, 6, 12);
        let mut g = Lcg(4);
        let mut x: Vec<f32> = (0..n * d).map(|_| (g.next_f32() * 3.0).floor()).collect();
        for row in [0usize, 57, 399] {
            x[row * d..(row + 1) * d].fill(0.0);
        }
        let want = knn_bruteforce(&x, n, d, k).unwrap();
        assert_eq!(knn_exact(&x, n, d, &params(k)).unwrap(), want);
        // A zero row has cosine 0 with everything
        assert!(want.cos[..k].iter().all(|&c| c == 0.0));
    }

    #[test]
    fn k_close_to_n() {
        let (n, d) = (25, 4);
        let x = clustered(n, d, 5);
        for k in [1, 23, 24] {
            assert_eq!(
                knn_exact(&x, n, d, &params(k)).unwrap(),
                knn_bruteforce(&x, n, d, k).unwrap()
            );
        }
    }

    #[test]
    fn rejects_bad_input() {
        let x = vec![1.0f32; 12];
        assert_eq!(
            knn_exact(&x, 4, 3, &params(4)),
            Err(KnnError::BadK { k: 4, n: 4 })
        );
        assert_eq!(
            knn_exact(&x, 4, 3, &params(0)),
            Err(KnnError::BadK { k: 0, n: 4 })
        );
        assert!(matches!(
            knn_exact(&x, 5, 3, &params(1)),
            Err(KnnError::Shape { .. })
        ));
        let mut y = x.clone();
        y[7] = f32::NAN;
        assert_eq!(
            knn_exact(&y, 4, 3, &params(1)),
            Err(KnnError::NonFinite { row: 2 })
        );
    }

    #[test]
    fn heap_keeps_best_with_index_tiebreak() {
        let mut h = BlockHeaps::new(1, 3);
        // Push without checking the threshold: push itself must reject candidates that aren't better
        for (s, id) in [(0.5, 7), (0.9, 3), (0.5, 2), (0.7, 9), (0.5, 1), (0.95, 0)] {
            h.push(0, s, id);
        }
        let mut kept: Vec<u32> = h.ids_of(0).to_vec();
        kept.sort();
        assert_eq!(kept, vec![0, 3, 9]);
    }
}
