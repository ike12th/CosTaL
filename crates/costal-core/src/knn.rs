//! 精确 kNN：融合分块的 f32 全量粗选 + f64 精排（docs/03 §2.2、docs/06 §1）。
//!
//! 1. 行范数（f64）、按范数排序的工作布局（docs/04 §2.3）、L2 归一化的 f32 副本；
//! 2. **粗选**：对每个（行块 × 列块）瓦片做 matmul，立刻并入每行的 top-(k+m) 堆，
//!    瓦片随即复用。全部 N² 个相似度都算过——粗的是精度，不是范围；
//! 3. **精排**：每行 k+m 个候选用 f64、按维度下标的固定顺序重算 cosine，
//!    按（分数降序, 原始索引升序）取前 k。最终名次只由这一步决定，
//!    所以瓦片尺寸、线程数、matmul 后端都不改变输出。

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::cache;
use crate::matmul::{self, Matmul};

/// 默认列块宽度。行块高度由缓存预算推出（docs/06 §2.3）。
pub const DEFAULT_COL_BLOCK: usize = 1024;

/// kNN 参数。`KnnParams::new(k)` 给出与已发表 CosTaL 一致的默认值。
#[derive(Clone, Debug)]
pub struct KnnParams {
    /// 每行的邻居数，不含自身（docs/04 §0.0 第 4 条）。
    pub k: usize,
    /// 粗选余量 m：每行保留 top-(k+m) 交给 f64 精排（docs/06 §1.2）。
    pub margin: usize,
    /// 行块高度；`None` 按缓存预算推出。
    pub row_block: Option<usize>,
    /// 列块宽度；`None` 用 [`DEFAULT_COL_BLOCK`]。
    pub col_block: Option<usize>,
    pub matmul: Matmul,
    /// 按行范数排序的工作布局（docs/04 §2.3）。cosine 选择用不上但无害；
    /// 留着开关只为测试"布局不改变输出"。
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

/// kNN 结果，全部按原始行序。
#[derive(Clone, Debug, PartialEq)]
pub struct Knn {
    pub n: usize,
    pub k: usize,
    /// 行主序 n×k 的原始索引；每行按（cos 降序, 索引升序）排好。
    pub idx: Vec<u32>,
    /// 行主序 n×k，f64 精排后的 cosine。
    pub cos: Vec<f64>,
    /// 原始（未归一化）行范数，f64。
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
            KnnError::Shape { len, n, d } => write!(f, "数据长度 {len} 与形状 {n}x{d} 不符"),
            KnnError::BadK { k, n } => write!(f, "k={k} 不合法：需要 1 ≤ k ≤ n-1（n={n}）"),
            KnnError::TooManyRows { n } => write!(f, "n={n} 超出 u32 索引范围"),
            KnnError::NonFinite { row } => write!(f, "第 {row} 行含 NaN 或 inf"),
            KnnError::MatmulUnavailable(name) => write!(f, "matmul 后端 {name} 在当前平台不可用"),
        }
    }
}

impl std::error::Error for KnnError {}

/// 选择度量的比较器（docs/04 §1）。内层循环按它单态化，入口处 match 一次。
///
/// 阶段 A 只有 [`Cosine`]。Tanimoto 在阶段 A+ 加入：`tile_threshold` 用列块的
/// 最小范数给保守阈值（列已按范数升序排好，`col_start` 处就是最小值），
/// `score` 在罕见路径上算精确分数（docs/04 §2）。
pub trait Selector: Sync {
    /// 每行的预计算常量。
    type RowState: Copy;
    fn row_state(&self, row: usize) -> Self::RowState;
    /// 列块上的保守标量阈值：瓦片值不超过它的候选一定进不了堆。
    /// `tau` 是该行堆的门槛（未满时为 -inf），`col_start` 是列块起点（工作序）。
    fn tile_threshold(&self, st: Self::RowState, tau: f32, col_start: usize) -> f32;
    /// 罕见路径：通过阈值后的选择分数（f32，只用于粗选）。
    fn score(&self, st: Self::RowState, col: usize, sim: f32) -> f32;
    /// 精排分数（f64），由精排后的 cosine 与两端原始范数算出。
    fn rescore(&self, cos: f64, norm_i: f64, norm_j: f64) -> f64;
}

/// 按 cosine 选邻居——已发表 CosTaL 的选择度量。
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

/// f64、按维度下标固定顺序的点积。f32×f32 在 f64 里是精确的；Rust 默认不做浮点收缩、
/// 也不重排浮点累加，所以这是跨平台逐位一致的唯一来源（docs/06 §1.2）。
#[inline]
pub fn dot64(a: &[f32], b: &[f32]) -> f64 {
    let mut s = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        s += f64::from(*x) * f64::from(*y);
    }
    s
}

/// 精排用的 cosine；零向量与任何向量的 cosine 记为 0（与原版 L2 归一化后零行的行为一致）。
#[inline]
fn cos64(dot: f64, norm_i: f64, norm_j: f64) -> f64 {
    if norm_i == 0.0 || norm_j == 0.0 {
        return 0.0;
    }
    let c = dot / (norm_i * norm_j);
    // 把 -0.0 规整成 0.0，排序用的全序才和数值相等一致
    if c == 0.0 {
        0.0
    } else {
        c
    }
}

/// 原始行范数，f64。
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

/// 精确 kNN 的附带统计。不放进 [`Knn`]，免得影响结果比对。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KnnStats {
    /// 粗选门槛离第 k 名太近、改走"整行收集 + 精排"的行数。结果仍然精确，
    /// 只是这些行更慢。它就是 docs/06 §1.2 说的"粗选漏选"风险被兜住的次数。
    pub fallback_rows: usize,
    pub row_block: usize,
    pub col_block: usize,
}

/// 精确 kNN，按 cosine 选邻居（阶段 A）。
pub fn knn_exact(x: &[f32], n: usize, d: usize, params: &KnnParams) -> Result<Knn, KnnError> {
    knn_exact_with_stats(x, n, d, params).map(|(knn, _)| knn)
}

/// 同 [`knn_exact`]，另外返回统计。
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

/// 一次调用里所有线程共享的只读状态。
struct Ctx<'a, S> {
    sel: &'a S,
    x: &'a [f32],
    /// 工作序下 L2 归一化的 f32 副本，只供粗选。
    xn: &'a [f32],
    norms: &'a [f64],
    /// 工作序 → 原始序。
    perm: &'a [u32],
    n: usize,
    d: usize,
    k: usize,
    cap: usize,
    cb: usize,
    backend: Matmul,
    /// f32 粗选值与真实 cosine 之差的上界。归一化到 f32（相对误差 ≤ u）加上任意累加
    /// 顺序的 f32 点积（≤ γ_d），合起来是 (d+2)·u 量级；这里取 (d+4)·2⁻²³，留一倍余量。
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

    // 工作序 → 原始序。按 (范数, 原始索引) 升序：稳定排序保证等范数时按原始索引
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

    // 工作序 → 原始序
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

/// 行块高度与列块宽度。
fn block_sizes(n: usize, p: &KnnParams) -> (usize, usize) {
    let cb = p.col_block.unwrap_or(DEFAULT_COL_BLOCK).clamp(1, n);
    let rb = p
        .row_block
        .unwrap_or_else(|| {
            let by_cache = cache::tile_budget_bytes() / (cb * std::mem::size_of::<f32>());
            // 每个线程至少分到 ~8 个行块，大小核之间靠工作窃取均衡（docs/06 §4）
            let by_balance = n.div_ceil(rayon::current_num_threads() * 8);
            by_cache.min(by_balance).clamp(8, 512)
        })
        .clamp(1, n);
    (rb, cb)
}

/// 每个线程复用的缓冲区。
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

/// 候选按（分数降序, 原始索引升序）排好。
fn sort_cands(c: &mut [(f64, f64, u32)]) {
    c.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.2.cmp(&b.2)));
}

impl<S: Selector> Ctx<'_, S> {
    /// 一个行块的粗选：逐列块做 matmul，并入每行的堆。
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

    /// 精排一行，返回是否走了兜底路径。
    ///
    /// 没进堆的候选，f32 值都不超过堆根（粗选门槛），真实 cosine 因此不超过门槛 + ε。
    /// 只要精排后的第 k 名严格高于这个界，没进堆的候选既进不了前 k 也不可能并列，
    /// 结果就是精确的。否则整行重新收集 f32 值 ≥ 第 k 名 − ε 的全部候选
    /// （真实 cosine ≥ 第 k 名的必在其中），再统一精排。
    ///
    /// 这个界是按 cosine 推的；阶段 A+ 加 Tanimoto 选择时要按它的分数重推。
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

        // cap == n-1 时堆里就是全部候选，不存在没进堆的
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

    /// 兜底：对一行重新扫全部列，收集 f32 值 ≥ `min_sim` 的候选并精排。
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

    /// 对工作序候选 `wj` 用 f64 重算，连同原始索引一起放进 `cands`。
    fn push_rescored(&self, i: usize, wj: usize, cands: &mut Vec<(f64, f64, u32)>) {
        let d = self.d;
        let j = self.perm[wj];
        let ju = j as usize;
        let dot = dot64(&self.x[i * d..(i + 1) * d], &self.x[ju * d..(ju + 1) * d]);
        let c = cos64(dot, self.norms[i], self.norms[ju]);
        cands.push((self.sel.rescore(c, self.norms[i], self.norms[ju]), c, j));
    }
}

/// 扫瓦片的一行。绝大多数候选在"整段都不超过阈值"这一步就被拒掉；
/// 自身排除和堆插入都只在罕见路径上。
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

/// 一个行块里每行一个定容小顶堆。根是"最差"的候选：分数低者更差，同分时索引大者更差。
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

    /// 进堆门槛：未满时为 -inf，满了是根的分数。
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

/// 黄金对照：不分块、不剪枝、全程 f64（docs/04 §4 的 L2 层）。O(N²·d)，只用于测试与基准。
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

    /// 可复现的伪随机数（不引依赖）。
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

    /// 带簇结构、带负值的数据，形状接近 CyTOF。
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
        // 10 个不同向量各复制 30 份：每行有 29 个 cos=1 的并列邻居，多于 k+m=20
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
        // 第 0 行的邻居应是同一原型里索引最小的 10 个（10, 20, …, 100）
        assert_eq!(&got.idx[..k], &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    }

    #[test]
    fn zero_rows_and_integer_ties() {
        // 小整数数据有大量精确并列；再混进几行全零
        let (n, d, k) = (400, 6, 12);
        let mut g = Lcg(4);
        let mut x: Vec<f32> = (0..n * d).map(|_| (g.next_f32() * 3.0).floor()).collect();
        for row in [0usize, 57, 399] {
            x[row * d..(row + 1) * d].fill(0.0);
        }
        let want = knn_bruteforce(&x, n, d, k).unwrap();
        assert_eq!(knn_exact(&x, n, d, &params(k)).unwrap(), want);
        // 零行与谁的 cosine 都是 0
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
        // 不经门槛直接 push：push 自己也必须拒绝不更好的候选
        for (s, id) in [(0.5, 7), (0.9, 3), (0.5, 2), (0.7, 9), (0.5, 1), (0.95, 0)] {
            h.push(0, s, id);
        }
        let mut kept: Vec<u32> = h.ids_of(0).to_vec();
        kept.sort();
        assert_eq!(kept, vec![0, 3, 9]);
    }
}
