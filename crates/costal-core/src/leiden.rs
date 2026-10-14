//! Leiden community detection (Traag, Waltman & van Eck 2019, "From Louvain to Leiden:
//! guaranteeing well-connected communities", Sci Rep 9:5233), working directly on CosTaL's graph.
//!
//! A clean-room implementation of the paper's Algorithm A.2. Where the paper leaves a detail open,
//! it follows the authors' MIT-licensed Java reference (CWTSLeiden/networkanalysis); docs/07 §5.
//! The objective is modularity with resolution γ: node weights are the strengths k_i, and moves
//! are scored as in the constant Potts model with γ / 2m, so γ means what it means in leidenalg's
//! `RBConfigurationVertexPartition` and igraph's `community_leiden(objective_function="modularity")`.
//!
//! **Determinism.** The same seed gives the same partition on every platform and at every thread
//! count:
//! - local moving is sequential;
//! - the refinement runs the clusters in parallel, each with its own random stream derived from
//!   (seed, iteration, level, cluster);
//! - aggregation builds every row on its own and sums duplicate edges in a fixed order;
//! - `exp` is computed here, because platform `libm` results can differ in the last bit.
//!
//! [`leiden_many`] runs independent runs in parallel, each sequential inside, so no more runs than
//! threads are in memory at once.

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Undirected weighted graph in compressed rows: both directions of every edge, no self-loops,
/// positive finite weights. Built only by [`Csr::from_edges`], which checks all of this.
#[derive(Clone, Debug, PartialEq)]
pub struct Csr {
    indptr: Vec<usize>,
    nbr: Vec<u32>,
    w: Vec<f64>,
}

impl Default for Csr {
    /// A graph without nodes.
    fn default() -> Self {
        Csr {
            indptr: vec![0],
            nbr: Vec::new(),
            w: Vec::new(),
        }
    }
}

impl Csr {
    /// From an edge list holding each undirected edge once, such as `graph::fuzzy_union`'s output.
    ///
    /// Rows list neighbors in edge-list order, which is ascending for an upper-triangular list
    /// sorted by (row, col). Edges of weight exactly 0 are dropped: they carry nothing for
    /// modularity, but a node whose only edges weigh 0 could otherwise be merged at random by the
    /// refinement. A pair listed twice counts with the sum of its weights.
    pub fn from_edges(n: usize, rows: &[u32], cols: &[u32], w: &[f64]) -> Result<Csr, String> {
        if rows.len() != cols.len() || rows.len() != w.len() {
            return Err(format!(
                "edge arrays differ in length: {} rows, {} cols, {} weights",
                rows.len(),
                cols.len(),
                w.len()
            ));
        }
        if n > u32::MAX as usize {
            return Err(format!("{n} nodes exceed the u32 node index"));
        }
        let mut indptr = vec![0usize; n + 1];
        for e in 0..rows.len() {
            let (i, j, x) = (rows[e] as usize, cols[e] as usize, w[e]);
            if i >= n || j >= n {
                return Err(format!("edge {e} ({i}, {j}) is out of range (n={n})"));
            }
            if i == j {
                return Err(format!("edge {e} is a self-loop on node {i}"));
            }
            if !(x.is_finite() && x >= 0.0) {
                return Err(format!(
                    "edge {e} has weight {x}; weights must be finite and non-negative"
                ));
            }
            if x > 0.0 {
                indptr[i + 1] += 1;
                indptr[j + 1] += 1;
            }
        }
        for v in 0..n {
            indptr[v + 1] += indptr[v];
        }
        let mut fill = indptr[..n].to_vec();
        let mut nbr = vec![0u32; indptr[n]];
        let mut ww = vec![0f64; indptr[n]];
        for e in 0..rows.len() {
            let (i, j, x) = (rows[e] as usize, cols[e] as usize, w[e]);
            if x > 0.0 {
                nbr[fill[i]] = j as u32;
                ww[fill[i]] = x;
                fill[i] += 1;
                nbr[fill[j]] = i as u32;
                ww[fill[j]] = x;
                fill[j] += 1;
            }
        }
        Ok(Csr { indptr, nbr, w: ww })
    }

    /// Number of nodes.
    pub fn n(&self) -> usize {
        self.indptr.len() - 1
    }

    /// Number of undirected edges.
    pub fn n_edges(&self) -> usize {
        self.nbr.len() / 2
    }

    #[inline]
    fn row(&self, v: usize) -> (&[u32], &[f64]) {
        let (a, b) = (self.indptr[v], self.indptr[v + 1]);
        (&self.nbr[a..b], &self.w[a..b])
    }

    /// Weighted degree of every node.
    pub fn strengths(&self) -> Vec<f64> {
        (0..self.n()).map(|v| self.row(v).1.iter().sum()).collect()
    }
}

/// Leiden settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LeidenParams {
    /// Modularity resolution γ (≥ 0).
    pub resolution: f64,
    /// Randomness of the refinement (the paper's θ, igraph's β), in edge-weight units; 0.01 in
    /// both references.
    pub beta: f64,
    /// Iterations; a negative value iterates until an iteration moves no node.
    pub n_iterations: i32,
    pub seed: u64,
}

impl Default for LeidenParams {
    fn default() -> Self {
        LeidenParams {
            resolution: 1.0,
            beta: 0.01,
            n_iterations: 2,
            seed: 0,
        }
    }
}

/// A Leiden partition.
#[derive(Clone, Debug, PartialEq)]
pub struct Partition {
    /// Cluster of every node, numbered 0.. by first appearance in node order.
    pub membership: Vec<u32>,
    pub n_clusters: usize,
    /// Iterations run (fewer than requested only for `n_iterations < 0`).
    pub iterations: usize,
    /// Whether the last iteration moved no node, so another would not change the partition.
    pub converged: bool,
}

/// Cap for `n_iterations < 0`. An iteration that moves a node raises the quality, so the loop
/// ends; the cap guards against moves driven by floating-point rounding alone.
pub const MAX_ITERATIONS: usize = 1000;

/// Run Leiden on `g`, starting from every node alone.
pub fn leiden(g: &Csr, p: &LeidenParams) -> Result<Partition, String> {
    let mut membership = vec![0u32; g.n()];
    let ws = &mut Workspace::default();
    let info = run(g, &g.strengths(), p, None, true, ws, &mut membership)?;
    Ok(info.with(membership))
}

/// Run Leiden on `g`, starting from `initial` (labels below the node count).
pub fn leiden_from(g: &Csr, p: &LeidenParams, initial: &[u32]) -> Result<Partition, String> {
    let mut membership = vec![0u32; g.n()];
    let ws = &mut Workspace::default();
    let info = run(
        g,
        &g.strengths(),
        p,
        Some(initial),
        true,
        ws,
        &mut membership,
    )?;
    Ok(info.with(membership))
}

/// Independent runs on one graph, in parallel; results in the order of `runs`. See
/// [`leiden_many_into`].
pub fn leiden_many(g: &Csr, runs: &[LeidenParams]) -> Result<Vec<Partition>, String> {
    let n = g.n();
    let mut out = vec![0u32; runs.len() * n];
    let infos = leiden_many_into(g, runs, &mut out)?;
    Ok(infos
        .into_iter()
        .enumerate()
        .map(|(i, info)| info.with(out[i * n..(i + 1) * n].to_vec()))
        .collect())
}

/// Independent runs on one graph, in parallel, each writing its membership into its row of `out`
/// (`runs.len()` rows of `g.n()` labels); returns the runs' summaries in the order of `runs`.
///
/// One task per thread takes the runs in turn and reuses one [`Workspace`] for all of them, so
/// memory is the graph, `out`, and one run's working set per thread (about 48 MB per thread on
/// Levine_32; run inside a smaller rayon pool to use less). With more threads than runs, the
/// spare threads share each run's refinement and aggregation; otherwise each run is sequential
/// inside. Either way each result is the same as from [`leiden`].
pub fn leiden_many_into(
    g: &Csr,
    runs: &[LeidenParams],
    out: &mut [u32],
) -> Result<Vec<RunInfo>, String> {
    many_into(g, runs, out, rayon::current_num_threads())
}

fn many_into(
    g: &Csr,
    runs: &[LeidenParams],
    out: &mut [u32],
    threads: usize,
) -> Result<Vec<RunInfo>, String> {
    let n = g.n();
    if out.len() != runs.len() * n {
        return Err(format!(
            "output holds {} labels for {} runs of {n} nodes",
            out.len(),
            runs.len()
        ));
    }
    if n == 0 {
        return runs.iter().map(run_empty).collect();
    }
    let nw = g.strengths();
    let rows: Vec<Mutex<&mut [u32]>> = out.chunks_mut(n).map(Mutex::new).collect();
    let next = AtomicUsize::new(0);
    let tasks = threads.clamp(1, runs.len().max(1));
    let inner = threads > runs.len(); // spare threads: parallel inside each run
    let done: Vec<Vec<(usize, Result<RunInfo, String>)>> = (0..tasks)
        .into_par_iter()
        .with_max_len(1)
        .map(|_| {
            let mut ws = Workspace::default();
            let mut mine = Vec::new();
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= runs.len() {
                    break mine;
                }
                let mut row = rows[i].lock().expect("each row is taken once");
                mine.push((i, run(g, &nw, &runs[i], None, inner, &mut ws, &mut row[..])));
            }
        })
        .collect();
    let mut infos: Vec<Option<Result<RunInfo, String>>> = vec![None; runs.len()];
    for (i, r) in done.into_iter().flatten() {
        infos[i] = Some(r);
    }
    infos
        .into_iter()
        .map(|r| r.expect("every run was taken"))
        .collect()
}

/// A run's summary; its membership is written to the caller's buffer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunInfo {
    pub n_clusters: usize,
    /// Iterations run (fewer than requested only for `n_iterations < 0`).
    pub iterations: usize,
    /// Whether the last iteration moved no node, so another would not change the partition.
    pub converged: bool,
}

impl RunInfo {
    fn with(self, membership: Vec<u32>) -> Partition {
        Partition {
            membership,
            n_clusters: self.n_clusters,
            iterations: self.iterations,
            converged: self.converged,
        }
    }
}

/// Modularity of `membership` at resolution γ: (1/2m) Σ_ij (A_ij − γ k_i k_j / 2m) δ(c_i, c_j).
pub fn modularity(g: &Csr, membership: &[u32], resolution: f64) -> Result<f64, String> {
    let n = g.n();
    check_labels(membership, n)?;
    let nw = g.strengths();
    let total: f64 = nw.iter().sum();
    if total == 0.0 {
        return Ok(0.0);
    }
    let mut k_c = vec![0f64; n];
    let mut internal = 0f64; // both directions of every internal edge: 2 × the internal weight
    for v in 0..n {
        let c = membership[v];
        k_c[c as usize] += nw[v];
        let (nb, ws) = g.row(v);
        for (&u, &x) in nb.iter().zip(ws) {
            if membership[u as usize] == c {
                internal += x;
            }
        }
    }
    let null: f64 = k_c.iter().map(|&k| k * k).sum();
    Ok(internal / total - resolution * null / (total * total))
}

fn check_labels(labels: &[u32], n: usize) -> Result<(), String> {
    if labels.len() != n {
        return Err(format!("{} labels for {n} nodes", labels.len()));
    }
    match labels.iter().find(|&&c| c as usize >= n) {
        Some(&c) => Err(format!("label {c} is not below the node count {n}")),
        None => Ok(()),
    }
}

fn check_params(p: &LeidenParams) -> Result<(), String> {
    if !(p.resolution.is_finite() && p.resolution >= 0.0) {
        return Err(format!(
            "resolution must be finite and ≥ 0, got {}",
            p.resolution
        ));
    }
    if !(p.beta.is_finite() && p.beta > 0.0) {
        return Err(format!("beta must be finite and > 0, got {}", p.beta));
    }
    if p.n_iterations == 0 {
        return Err("n_iterations must be positive, or negative to run until stable".into());
    }
    Ok(())
}

/// A run on a graph without nodes.
fn run_empty(p: &LeidenParams) -> Result<RunInfo, String> {
    check_params(p)?;
    Ok(RunInfo {
        n_clusters: 0,
        iterations: 0,
        converged: true,
    })
}

/// One run: writes the partition into `clu` (one label per node), numbered by first appearance.
#[allow(clippy::too_many_arguments)]
fn run(
    g: &Csr,
    nw: &[f64],
    p: &LeidenParams,
    initial: Option<&[u32]>,
    par: bool,
    ws: &mut Workspace,
    clu: &mut [u32],
) -> Result<RunInfo, String> {
    check_params(p)?;
    let n = g.n();
    debug_assert_eq!(clu.len(), n);
    match initial {
        None => clu.iter_mut().enumerate().for_each(|(v, c)| *c = v as u32),
        Some(m) => {
            check_labels(m, n)?;
            clu.copy_from_slice(m);
        }
    }
    let total: f64 = nw.iter().sum();
    let (mut iterations, mut converged) = (0, true);
    // Without edges modularity is undefined, and nothing moves
    if total > 0.0 {
        let gamma = p.resolution / total;
        let max_it = if p.n_iterations > 0 {
            p.n_iterations as usize
        } else {
            MAX_ITERATIONS
        };
        for it in 0..max_it {
            let stream = [p.seed, it as u64];
            let moved = one_iteration(g, nw, gamma, p.beta, clu, stream, par, ws);
            iterations += 1;
            converged = !moved;
            if p.n_iterations < 0 && converged {
                break;
            }
        }
    }
    let n_clusters = relabel_by_first_appearance(clu, &mut ws.map);
    Ok(RunInfo {
        n_clusters,
        iterations,
        converged,
    })
}

/// Buffers reused across levels, iterations and runs. They only grow, so a thread allocates its
/// memory once instead of at every level: allocating per level fragmented the heap when runs at
/// different resolutions followed each other (44 runs on Levine_32 on one thread peaked at 618 MB).
#[derive(Default)]
struct Workspace {
    clu: Vec<u32>,      // partition of the current level
    next_clu: Vec<u32>, // the next level's starting partition
    node_of: Vec<u32>,  // node of the current level holding each original node
    map: Vec<u32>,      // relabelling
    moving: MoveBufs,
    refining: RefineBufs,
    aggregating: AggregateBufs,
    spare: Vec<Level>, // aggregate graphs, reused
}

/// An aggregate graph and its node weights.
#[derive(Default)]
struct Level {
    g: Csr,
    nw: Vec<f64>,
}

#[derive(Default)]
struct MoveBufs {
    cw: Vec<f64>,      // node weight of each cluster
    count: Vec<u32>,   // nodes in each cluster
    unused: Vec<u32>,  // empty cluster ids, a stack
    order: Vec<u32>,   // the queue
    ewc: Vec<f64>,     // edge weight from the current node to each cluster
    cand: Vec<u32>,    // candidate clusters of the current node
    stable: Vec<bool>, // nodes not in the queue
}

/// Nodes grouped by label: `order[start[c]..start[c + 1]]` lists label c's nodes in ascending order.
#[derive(Default)]
struct Groups {
    start: Vec<usize>,
    fill: Vec<usize>,
    order: Vec<u32>,
}

#[derive(Default)]
struct RefineBufs {
    groups: Groups,
    pos: Vec<u32>,     // index of each node within its cluster
    local: Vec<u32>,   // sub-cluster of each node within its cluster, aligned with `groups.order`
    refined: Vec<u32>, // the refined partition
    counts: Vec<u32>,  // sub-clusters per cluster, then their offsets
    scratch: RefineScratch,
}

#[derive(Default)]
struct AggregateBufs {
    groups: Groups,
    acc: Vec<f64>, // edge weight from the current label to each label
    touched: Vec<u32>,
    stamp: Vec<u32>, // last row (plus one) that counted each label
}

/// `v` as `n` copies of `x`, reusing its allocation.
fn reset<T: Clone>(v: &mut Vec<T>, n: usize, x: T) {
    v.clear();
    v.resize(n, x);
}

/// Levels with fewer nodes run sequentially: below this, rayon's dispatch costs more than the
/// work (about 1 ms per run on a 500-node graph).
const PARALLEL_MIN_NODES: usize = 5000;

/// Rayon's minimum job length: 1 splits freely, `usize::MAX` keeps the work in one job on the
/// current thread (the sequential path of [`leiden_many_into`]). Either gives the same result.
fn split(par: bool) -> usize {
    if par {
        1
    } else {
        usize::MAX
    }
}

/// One Leiden iteration on the original graph, from the partition in `clu0`.
///
/// Returns whether local moving moved any node at any level.
#[allow(clippy::too_many_arguments)]
fn one_iteration(
    g0: &Csr,
    nw0: &[f64],
    gamma: f64,
    beta: f64,
    clu0: &mut [u32],
    stream: [u64; 2],
    par: bool,
    ws: &mut Workspace,
) -> bool {
    let par0 = par && g0.n() >= PARALLEL_MIN_NODES;
    ws.clu.clear();
    ws.clu.extend_from_slice(clu0);
    let mut current: Option<Level> = None; // the aggregate graph of the current level
    let mut aggregated = false; // whether `ws.node_of` is in use
    let mut moved = false;
    for level in 0u64.. {
        let (g, nw) = match &current {
            Some(l) => (&l.g, &l.nw[..]),
            None => (g0, nw0),
        };
        let par_level = par && g.n() >= PARALLEL_MIN_NODES;
        let mut rng = Rng::new(&[stream[0], stream[1], level, 0]);
        let (m, nc) = move_nodes(
            g,
            nw,
            gamma,
            &mut ws.clu,
            &mut rng,
            &mut ws.moving,
            &mut ws.map,
        );
        moved |= m;
        if nc == g.n() {
            break; // every node is alone: nothing left to aggregate
        }
        let stream3 = [stream[0], stream[1], level];
        let nr = refine(
            g,
            nw,
            gamma,
            beta,
            &ws.clu,
            nc,
            stream3,
            par_level,
            &mut ws.refining,
        );
        // Aggregate on the refined partition and start the next level from the unrefined one; if
        // the refinement merged nothing, aggregate on the unrefined partition instead.
        let refined = nr < g.n();
        let (labels, n_labels): (&[u32], usize) = if refined {
            (&ws.refining.refined, nr)
        } else {
            (&ws.clu, nc)
        };
        let mut next = ws.spare.pop().unwrap_or_default();
        aggregate(
            g,
            nw,
            labels,
            n_labels,
            par_level,
            &mut ws.aggregating,
            &mut next,
        );
        if aggregated {
            ws.node_of
                .par_iter_mut()
                .with_min_len(split(par0))
                .for_each(|x| *x = labels[*x as usize]);
        } else {
            ws.node_of.clear();
            ws.node_of.extend_from_slice(labels);
            aggregated = true;
        }
        if refined {
            reset(&mut ws.next_clu, nr, 0);
            for (&r, &c) in ws.refining.refined.iter().zip(&ws.clu) {
                ws.next_clu[r as usize] = c;
            }
        } else {
            ws.next_clu.clear();
            ws.next_clu.extend(0..nc as u32);
        }
        std::mem::swap(&mut ws.clu, &mut ws.next_clu);
        if let Some(old) = current.replace(next) {
            ws.spare.push(old);
        }
    }
    if aggregated {
        let clu = &ws.clu;
        clu0.par_iter_mut()
            .zip(ws.node_of.par_iter())
            .with_min_len(split(par0))
            .for_each(|(c, &x)| *c = clu[x as usize]);
    } else {
        clu0.copy_from_slice(&ws.clu);
    }
    if let Some(l) = current {
        ws.spare.push(l);
    }
    moved
}

/// Fast local moving (the paper's MoveNodesFast; Java `FastLocalMovingAlgorithm`).
///
/// Nodes are visited from a queue in random order; a node moves to the neighboring cluster, or an
/// empty one, with the largest gain if that gain is strictly above staying, and its stable
/// neighbors outside the new cluster are re-queued. On return the clusters are numbered 0..nc in
/// the order of their old numbers. Returns (any node moved, nc).
fn move_nodes(
    g: &Csr,
    nw: &[f64],
    gamma: f64,
    clu: &mut [u32],
    rng: &mut Rng,
    b: &mut MoveBufs,
    map: &mut Vec<u32>,
) -> (bool, usize) {
    let n = g.n();
    if n <= 1 {
        return (false, n);
    }
    let MoveBufs {
        cw,
        count,
        unused,
        order,
        ewc,
        cand,
        stable,
    } = b;
    reset(cw, n, 0.0);
    reset(count, n, 0);
    for (&c, &k) in clu.iter().zip(nw) {
        cw[c as usize] += k;
        count[c as usize] += 1;
    }
    // Stack of empty cluster ids, the smallest on top
    unused.clear();
    unused.extend((0..n as u32).rev().filter(|&c| count[c as usize] == 0));
    order.clear();
    order.extend(0..n as u32);
    rng.shuffle(order);
    reset(ewc, n, 0.0);
    cand.clear();
    reset(stable, n, false);
    let mut n_unstable = n;
    let mut i = 0usize;
    let mut moved = false;
    loop {
        let v = order[i] as usize;
        let cur = clu[v] as usize;
        let kv = nw[v];
        cw[cur] -= kv;
        count[cur] -= 1;
        if count[cur] == 0 {
            cw[cur] = 0.0; // no rounding residue: an empty cluster weighs exactly nothing
            unused.push(cur as u32);
        }
        cand.clear();
        cand.push(*unused.last().expect("v's cluster or another is empty"));
        let (nb, ws) = g.row(v);
        for (&u, &x) in nb.iter().zip(ws) {
            let l = clu[u as usize] as usize;
            if ewc[l] == 0.0 {
                cand.push(l as u32);
            }
            ewc[l] += x;
        }
        let mut best = cur;
        let mut best_gain = ewc[cur] - kv * cw[cur] * gamma;
        for &l in cand.iter() {
            let l = l as usize;
            let gain = ewc[l] - kv * cw[l] * gamma;
            if gain > best_gain {
                best = l;
                best_gain = gain;
            }
            ewc[l] = 0.0;
        }
        cw[best] += kv;
        count[best] += 1;
        if unused.last() == Some(&(best as u32)) {
            unused.pop();
        }
        stable[v] = true;
        n_unstable -= 1;
        if best != cur {
            clu[v] = best as u32;
            moved = true;
            for &u in nb {
                let u = u as usize;
                if stable[u] && clu[u] as usize != best {
                    stable[u] = false;
                    n_unstable += 1;
                    let p = i + n_unstable;
                    order[if p < n { p } else { p - n }] = u as u32;
                }
            }
        }
        if n_unstable == 0 {
            break;
        }
        i = if i + 1 < n { i + 1 } else { 0 };
    }
    let nc = compact(clu, n, map);
    (moved, nc)
}

/// Refinement (the paper's RefinePartition and MergeNodesSubset; Java `LocalMergingAlgorithm`).
///
/// Within each cluster every node starts alone. In random order, a node that is still alone and
/// well connected to the rest of its cluster may merge into a well-connected sub-cluster, chosen
/// with probability ∝ exp(gain/β) among the non-negative gains (staying alone has gain 0). The
/// clusters are refined in parallel, each with its own random stream. Leaves the refined
/// partition, numbered by (cluster, then sub-cluster), in `b.refined` and returns its size.
#[allow(clippy::too_many_arguments)]
fn refine(
    g: &Csr,
    nw: &[f64],
    gamma: f64,
    beta: f64,
    clu: &[u32],
    nc: usize,
    stream: [u64; 3],
    par: bool,
    b: &mut RefineBufs,
) -> usize {
    let n = g.n();
    let RefineBufs {
        groups,
        pos,
        local,
        refined,
        counts,
        scratch,
    } = b;
    group_by(clu, nc, groups);
    let (start, order) = (&groups.start, &groups.order);
    reset(pos, n, 0);
    for c in 0..nc {
        for (j, &v) in order[start[c]..start[c + 1]].iter().enumerate() {
            pos[v as usize] = j as u32;
        }
    }
    let pos = &*pos;
    reset(local, n, 0);
    let mut parts: Vec<&mut [u32]> = Vec::with_capacity(nc);
    let mut rest: &mut [u32] = local;
    for c in 0..nc {
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(start[c + 1] - start[c]);
        parts.push(head);
        rest = tail;
    }
    let refine_one = |s: &mut RefineScratch, c: usize, out: &mut [u32]| {
        let cluster = Cluster {
            id: c as u32,
            nodes: &order[start[c]..start[c + 1]],
            stream: [stream[0], stream[1], stream[2], c as u64],
        };
        refine_cluster(g, nw, gamma, beta, clu, pos, &cluster, out, s)
    };
    if par {
        parts
            .into_par_iter()
            .enumerate()
            .map_init(RefineScratch::default, |s, (c, out)| refine_one(s, c, out))
            .collect_into_vec(counts);
    } else {
        counts.clear();
        for (c, out) in parts.into_iter().enumerate() {
            counts.push(refine_one(scratch, c, out));
        }
    }
    // Sub-cluster counts become offsets
    let mut nr = 0usize;
    for k in counts.iter_mut() {
        let here = *k as usize;
        *k = nr as u32;
        nr += here;
    }
    let (offset, local) = (&*counts, &*local);
    (0..n)
        .into_par_iter()
        .with_min_len(split(par))
        .map(|v| {
            let c = clu[v] as usize;
            offset[c] + local[start[c] + pos[v] as usize]
        })
        .collect_into_vec(refined);
    nr
}

struct Cluster<'a> {
    id: u32,
    nodes: &'a [u32],
    stream: [u64; 4],
}

#[derive(Default)]
struct RefineScratch {
    cw: Vec<f64>,      // node weight of each sub-cluster
    ext: Vec<f64>,     // edge weight from each sub-cluster to the rest of the cluster
    single: Vec<bool>, // whether each sub-cluster is still a single node
    ewc: Vec<f64>,     // edge weight from the current node to each sub-cluster
    cand: Vec<u32>,
    gain: Vec<f64>,
    w_to: Vec<f64>,
    cum: Vec<f64>,
    perm: Vec<u32>,
    map: Vec<u32>,
}

#[allow(clippy::too_many_arguments)]
fn refine_cluster(
    g: &Csr,
    nw: &[f64],
    gamma: f64,
    beta: f64,
    clu: &[u32],
    pos: &[u32],
    cluster: &Cluster,
    out: &mut [u32],
    s: &mut RefineScratch,
) -> u32 {
    let nodes = cluster.nodes;
    let m = nodes.len();
    for (i, o) in out.iter_mut().enumerate() {
        *o = i as u32;
    }
    if m == 1 {
        return 1;
    }
    // Each node's weight, and its edge weight to the rest of its cluster. The rows are read in
    // place: a copy of the cluster's subgraph would be faster to scan but, for the large clusters
    // of low resolutions, nearly as large as the graph, once per thread.
    s.cw.clear();
    s.cw.extend(nodes.iter().map(|&v| nw[v as usize]));
    s.ext.clear();
    s.ext.extend(nodes.iter().map(|&v| {
        let (nb, ws) = g.row(v as usize);
        nb.iter()
            .zip(ws)
            .filter(|(&u, _)| clu[u as usize] == cluster.id)
            .map(|(_, &x)| x)
            .sum::<f64>()
    }));
    let total: f64 = s.cw.iter().sum();
    s.single.clear();
    s.single.resize(m, true);
    s.ewc.clear();
    s.ewc.resize(m, 0.0);
    s.perm.clear();
    s.perm.extend(0..m as u32);
    let mut rng = Rng::new(&cluster.stream);
    rng.shuffle(&mut s.perm);
    let mut merged = false;
    for idx in 0..m {
        let i = s.perm[idx] as usize;
        // Only a node still alone, and well connected to the rest of its cluster, may move
        if !s.single[i] || s.ext[i] < s.cw[i] * (total - s.cw[i]) * gamma {
            continue;
        }
        let kv = nw[nodes[i] as usize];
        let e_v = s.ext[i]; // the node's edge weight to the rest of its cluster
        s.cw[i] = 0.0;
        s.ext[i] = 0.0;
        s.cand.clear();
        s.cand.push(i as u32); // staying alone is always possible
        let (nb, ws) = g.row(nodes[i] as usize);
        for (&u, &x) in nb.iter().zip(ws) {
            if clu[u as usize] != cluster.id {
                continue;
            }
            let l = out[pos[u as usize] as usize] as usize;
            if s.ewc[l] == 0.0 {
                s.cand.push(l as u32);
            }
            s.ewc[l] += x;
        }
        // Gains of the well-connected candidates with a non-negative gain; the others get -inf
        s.gain.clear();
        s.w_to.clear();
        let mut max_gain = f64::NEG_INFINITY;
        let (mut n_ok, mut last_ok) = (0usize, 0usize);
        for (j, &l) in s.cand.iter().enumerate() {
            let l = l as usize;
            let mut gain = f64::NEG_INFINITY;
            if s.ext[l] >= s.cw[l] * (total - s.cw[l]) * gamma {
                let d = s.ewc[l] - kv * s.cw[l] * gamma;
                if d >= 0.0 {
                    gain = d;
                    n_ok += 1;
                    last_ok = j;
                }
            }
            max_gain = max_gain.max(gain);
            s.gain.push(gain);
            s.w_to.push(s.ewc[l]);
            s.ewc[l] = 0.0;
        }
        // Staying alone is eligible with gain 0, so there is at least one candidate. With more,
        // sample ∝ exp(gain/β), shifted by the largest gain so that nothing overflows.
        let pick = if n_ok == 1 {
            last_ok
        } else {
            s.cum.clear();
            let mut acc = 0.0;
            for &gain in &s.gain {
                if gain > f64::NEG_INFINITY {
                    acc += exp_nonpositive((gain - max_gain) / beta);
                }
                s.cum.push(acc);
            }
            let r = acc * rng.unif();
            // The first candidate whose share covers r; r < acc unless the product rounds up to
            // acc (probability ~2⁻⁵³), when the last eligible candidate is taken
            let j = s.cum.partition_point(|&x| x <= r);
            if j < s.cand.len() {
                j
            } else {
                last_ok
            }
        };
        let chosen = s.cand[pick] as usize;
        s.cw[chosen] += kv;
        // The node's edges into `chosen` become internal; its other edges in the cluster external
        s.ext[chosen] += e_v - 2.0 * s.w_to[pick];
        if chosen != i {
            out[i] = chosen as u32;
            s.single[chosen] = false;
            merged = true;
        }
    }
    if merged {
        compact(out, m, &mut s.map) as u32
    } else {
        m as u32
    }
}

/// The aggregate graph of `labels` into `out`: one node per label, edge weights summed between
/// labels, weight inside a label dropped (a node's own internal weight never changes a move's
/// gain). Rows come out in ascending neighbor order. Sequentially, rows are written straight into
/// `out`; in parallel, blocks of rows are built separately and then copied in.
fn aggregate(
    g: &Csr,
    nw: &[f64],
    labels: &[u32],
    nl: usize,
    par: bool,
    b: &mut AggregateBufs,
    out: &mut Level,
) {
    let AggregateBufs {
        groups,
        acc,
        touched,
        stamp,
    } = b;
    group_by(labels, nl, groups);
    let (start, order) = (&groups.start, &groups.order);
    (0..nl)
        .into_par_iter()
        .with_min_len(split(par))
        .map(|r| {
            order[start[r]..start[r + 1]]
                .iter()
                .map(|&v| nw[v as usize])
                .sum()
        })
        .collect_into_vec(&mut out.nw);
    let csr = &mut out.g;
    csr.indptr.clear();
    csr.indptr.push(0);
    csr.nbr.clear();
    csr.w.clear();
    if par {
        // Rows are independent, so how they are split into jobs never changes the result. At
        // most four jobs per thread, each with one dense accumulator over the labels.
        let n_jobs = (4 * rayon::current_num_threads()).clamp(1, nl.max(1));
        let blocks: Vec<(Vec<usize>, Vec<u32>, Vec<f64>)> = (0..n_jobs)
            .into_par_iter()
            .map(|j| {
                let (lo, hi) = (j * nl / n_jobs, (j + 1) * nl / n_jobs);
                let (mut acc, mut touched) = (vec![0f64; nl], Vec::new());
                let (mut ends, mut nbr, mut w) =
                    (Vec::with_capacity(hi - lo), Vec::new(), Vec::new());
                let rows = Rows {
                    g,
                    labels,
                    start,
                    order,
                };
                rows.build(lo..hi, &mut acc, &mut touched, &mut ends, &mut nbr, &mut w);
                (ends, nbr, w)
            })
            .collect();
        for (ends, nbr, w) in blocks {
            let base = csr.nbr.len();
            csr.indptr.extend(ends.iter().map(|e| base + e));
            csr.nbr.extend_from_slice(&nbr);
            csr.w.extend_from_slice(&w);
        }
    } else {
        let rows = Rows {
            g,
            labels,
            start,
            order,
        };
        // Count first, so the graph is allocated at its exact size: grown by doubling, each
        // thread's reused graphs kept up to twice the memory they need
        reset(stamp, nl, 0);
        let m = rows.count(0..nl, stamp);
        csr.indptr.reserve_exact(nl);
        csr.nbr.reserve_exact(m);
        csr.w.reserve_exact(m);
        reset(acc, nl, 0.0);
        touched.clear();
        rows.build(
            0..nl,
            acc,
            touched,
            &mut csr.indptr,
            &mut csr.nbr,
            &mut csr.w,
        );
        debug_assert_eq!(csr.nbr.len(), m);
    }
}

/// What building aggregate rows reads.
struct Rows<'a> {
    g: &'a Csr,
    labels: &'a [u32],
    start: &'a [usize],
    order: &'a [u32],
}

impl Rows<'_> {
    /// Number of entries the rows of labels `range` will hold. `stamp` must be zero.
    fn count(&self, range: std::ops::Range<usize>, stamp: &mut [u32]) -> usize {
        let mut m = 0;
        for r in range {
            let mark = r as u32 + 1;
            for &v in &self.order[self.start[r]..self.start[r + 1]] {
                for &u in self.g.row(v as usize).0 {
                    let q = self.labels[u as usize] as usize;
                    if q != r && stamp[q] != mark {
                        stamp[q] = mark;
                        m += 1;
                    }
                }
            }
        }
        m
    }

    /// Append the rows of labels `range`: neighbor labels to `nbr`, summed weights to `w`, and
    /// after each row the end offset `nbr.len()` to `ends`. `acc` must be zero, and is left zero.
    fn build(
        &self,
        range: std::ops::Range<usize>,
        acc: &mut [f64],
        touched: &mut Vec<u32>,
        ends: &mut Vec<usize>,
        nbr: &mut Vec<u32>,
        w: &mut Vec<f64>,
    ) {
        for r in range {
            for &v in &self.order[self.start[r]..self.start[r + 1]] {
                let (nb, ws) = self.g.row(v as usize);
                for (&u, &x) in nb.iter().zip(ws) {
                    let q = self.labels[u as usize];
                    if q as usize != r {
                        if acc[q as usize] == 0.0 {
                            touched.push(q);
                        }
                        acc[q as usize] += x; // duplicates summed in member and edge order
                    }
                }
            }
            touched.sort_unstable();
            for &q in touched.iter() {
                nbr.push(q);
                w.push(acc[q as usize]);
                acc[q as usize] = 0.0;
            }
            touched.clear();
            ends.push(nbr.len());
        }
    }
}

/// Group nodes by label into `gr` (`order[start[c]..start[c + 1]]`, ascending within a label).
fn group_by(labels: &[u32], nl: usize, gr: &mut Groups) {
    reset(&mut gr.start, nl + 1, 0);
    for &c in labels {
        gr.start[c as usize + 1] += 1;
    }
    for c in 0..nl {
        gr.start[c + 1] += gr.start[c];
    }
    gr.fill.clear();
    gr.fill.extend_from_slice(&gr.start[..nl]);
    reset(&mut gr.order, labels.len(), 0);
    for (v, &c) in labels.iter().enumerate() {
        gr.order[gr.fill[c as usize]] = v as u32;
        gr.fill[c as usize] += 1;
    }
}

/// Renumber labels in `0..n_ids` to 0..k, keeping their order; returns k.
fn compact(labels: &mut [u32], n_ids: usize, map: &mut Vec<u32>) -> usize {
    reset(map, n_ids, u32::MAX);
    for &l in labels.iter() {
        map[l as usize] = 0;
    }
    let mut k = 0u32;
    for x in map.iter_mut() {
        if *x == 0 {
            *x = k;
            k += 1;
        }
    }
    for l in labels.iter_mut() {
        *l = map[*l as usize];
    }
    k as usize
}

/// Renumber labels (all below `labels.len()`) 0.. by first appearance; returns the number of labels.
fn relabel_by_first_appearance(labels: &mut [u32], map: &mut Vec<u32>) -> usize {
    reset(map, labels.len(), u32::MAX);
    let mut k = 0u32;
    for l in labels.iter_mut() {
        let x = &mut map[*l as usize];
        if *x == u32::MAX {
            *x = k;
            k += 1;
        }
        *l = *x;
    }
    k as usize
}

/// 1/k! for k = 0..=13.
const INV_FACTORIAL: [f64; 14] = {
    let mut a = [1.0; 14];
    let mut k = 1;
    while k < 14 {
        a[k] = a[k - 1] / k as f64;
        k += 1;
    }
    a
};

/// e^x for x ≤ 0, with the same bits on every platform.
///
/// Range reduction x = k·ln2 + r with |r| ≤ ln2/2, then the Taylor polynomial of e^r to r^13 by
/// Horner's rule (relative truncation error below 1e-17). Results under e^-708 are returned as 0:
/// they are negligible next to the e^0 that every refinement draw contains.
fn exp_nonpositive(x: f64) -> f64 {
    debug_assert!(x <= 0.0 || x.is_nan());
    if x < -708.0 {
        return 0.0;
    }
    const LN2_HI: f64 = 6.931_471_803_691_238e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    let k = (x * std::f64::consts::LOG2_E).round();
    let r = (x - k * LN2_HI) - k * LN2_LO;
    let mut p = INV_FACTORIAL[13];
    for c in INV_FACTORIAL[..13].iter().rev() {
        p = p * r + c;
    }
    p * f64::from_bits(((k as i64 + 1023) as u64) << 52)
}

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256** (Blackman & Vigna), seeded from a list of words through SplitMix64.
#[derive(Clone, Debug)]
struct Rng([u64; 4]);

impl Rng {
    fn new(words: &[u64]) -> Rng {
        let mut h: u64 = 0x243F_6A88_85A3_08D3; // digits of π
        for &w in words {
            h = mix(h.wrapping_add(GOLDEN) ^ w);
        }
        let mut s = [0u64; 4];
        for lane in &mut s {
            h = h.wrapping_add(GOLDEN);
            *lane = mix(h);
        }
        if s == [0; 4] {
            s[0] = GOLDEN;
        }
        Rng(s)
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let out = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        out
    }

    /// Uniform in `0..n` (Lemire's multiply-shift with rejection, so unbiased).
    fn below(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0);
        let mut m = self.next_u64() as u128 * n as u128;
        if (m as u64) < n {
            let threshold = n.wrapping_neg() % n;
            while (m as u64) < threshold {
                m = self.next_u64() as u128 * n as u128;
            }
        }
        (m >> 64) as u64
    }

    /// Uniform in [0, 1).
    fn unif(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Fisher–Yates shuffle.
    fn shuffle(&mut self, a: &mut [u32]) {
        for i in (1..a.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            a.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csr(n: usize, edges: &[(u32, u32, f64)]) -> Csr {
        let rows: Vec<u32> = edges.iter().map(|e| e.0).collect();
        let cols: Vec<u32> = edges.iter().map(|e| e.1).collect();
        let w: Vec<f64> = edges.iter().map(|e| e.2).collect();
        Csr::from_edges(n, &rows, &cols, &w).unwrap()
    }

    /// `n_blocks` blocks of `size` nodes, each pair inside a block joined with probability `p_in`,
    /// plus `n_cross` random edges between blocks; weights uniform in (0.5, 1).
    fn planted(n_blocks: usize, size: usize, p_in: f64, n_cross: usize, seed: u64) -> Csr {
        let mut rng = Rng::new(&[seed]);
        let n = n_blocks * size;
        let mut pairs = std::collections::BTreeMap::new();
        for b in 0..n_blocks {
            for i in b * size..(b + 1) * size {
                for j in i + 1..(b + 1) * size {
                    if rng.unif() < p_in {
                        pairs.insert((i as u32, j as u32), 0.5 + 0.5 * rng.unif());
                    }
                }
            }
        }
        let mut added = 0;
        while added < n_cross {
            let (i, j) = (rng.below(n as u64) as u32, rng.below(n as u64) as u32);
            if i as usize / size != j as usize / size {
                pairs.insert((i.min(j), i.max(j)), 0.5 + 0.5 * rng.unif());
                added += 1;
            }
        }
        let edges: Vec<(u32, u32, f64)> = pairs.into_iter().map(|((i, j), w)| (i, j, w)).collect();
        csr(n, &edges)
    }

    fn params(resolution: f64, seed: u64) -> LeidenParams {
        LeidenParams {
            resolution,
            seed,
            ..LeidenParams::default()
        }
    }

    fn q(g: &Csr, m: &[u32], res: f64) -> f64 {
        modularity(g, m, res).unwrap()
    }

    fn clusters_are_connected(g: &Csr, membership: &[u32]) -> bool {
        let n = g.n();
        let mut seen = vec![false; n];
        let mut components = vec![0usize; n];
        for s in 0..n {
            if seen[s] {
                continue;
            }
            components[membership[s] as usize] += 1;
            let mut stack = vec![s];
            seen[s] = true;
            while let Some(v) = stack.pop() {
                for &u in g.row(v).0 {
                    let u = u as usize;
                    if !seen[u] && membership[u] == membership[v] {
                        seen[u] = true;
                        stack.push(u);
                    }
                }
            }
        }
        components.iter().all(|&c| c <= 1)
    }

    /// Largest gain, in the CPM form w − γ'·K·K', from merging two adjacent clusters.
    fn best_merge_gain(g: &Csr, membership: &[u32], res: f64) -> f64 {
        let nw = g.strengths();
        let gamma = res / nw.iter().sum::<f64>();
        let mut k_c = std::collections::HashMap::<u32, f64>::new();
        for (&c, &k) in membership.iter().zip(&nw) {
            *k_c.entry(c).or_default() += k;
        }
        let mut between = std::collections::HashMap::<(u32, u32), f64>::new();
        for v in 0..g.n() {
            let (nb, ws) = g.row(v);
            for (&u, &x) in nb.iter().zip(ws) {
                let (a, b) = (membership[v], membership[u as usize]);
                if a < b {
                    *between.entry((a, b)).or_default() += x;
                }
            }
        }
        between
            .iter()
            .map(|(&(a, b), &w)| w - gamma * k_c[&a] * k_c[&b])
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Largest gain from moving one node to another (or a new) cluster, over staying.
    fn best_node_move_gain(g: &Csr, membership: &[u32], res: f64) -> f64 {
        let nw = g.strengths();
        let gamma = res / nw.iter().sum::<f64>();
        let mut k_c = vec![0f64; g.n()];
        for (&c, &k) in membership.iter().zip(&nw) {
            k_c[c as usize] += k;
        }
        let mut best = f64::NEG_INFINITY;
        for v in 0..g.n() {
            let cur = membership[v];
            let mut to = std::collections::HashMap::<u32, f64>::new();
            let (nb, ws) = g.row(v);
            for (&u, &x) in nb.iter().zip(ws) {
                *to.entry(membership[u as usize]).or_default() += x;
            }
            let stay =
                to.get(&cur).copied().unwrap_or(0.0) - nw[v] * (k_c[cur as usize] - nw[v]) * gamma;
            let alone = 0.0;
            best = best.max(alone - stay);
            for (&c, &w) in &to {
                if c != cur {
                    best = best.max(w - nw[v] * k_c[c as usize] * gamma - stay);
                }
            }
        }
        best
    }

    #[test]
    fn csr_holds_both_directions() {
        let g = csr(4, &[(0, 1, 0.5), (0, 3, 0.25), (1, 2, 1.0)]);
        assert_eq!(g.indptr, vec![0, 2, 4, 5, 6]);
        assert_eq!(g.nbr, vec![1, 3, 0, 2, 1, 0]);
        assert_eq!(g.w, vec![0.5, 0.25, 0.5, 1.0, 1.0, 0.25]);
        assert_eq!(g.strengths(), vec![0.75, 1.5, 1.0, 0.25]);
        assert_eq!(g.n_edges(), 3);
    }

    #[test]
    fn csr_drops_zero_weights_and_rejects_bad_edges() {
        let g = csr(3, &[(0, 1, 0.0), (1, 2, 0.5)]);
        assert_eq!(g.indptr, vec![0, 0, 1, 2]);
        assert!(Csr::from_edges(2, &[0], &[0], &[1.0])
            .unwrap_err()
            .contains("self-loop"));
        assert!(Csr::from_edges(2, &[0], &[2], &[1.0])
            .unwrap_err()
            .contains("out of range"));
        assert!(Csr::from_edges(2, &[0], &[1], &[-1.0])
            .unwrap_err()
            .contains("weight"));
        assert!(Csr::from_edges(2, &[0], &[1], &[f64::NAN])
            .unwrap_err()
            .contains("weight"));
        assert!(Csr::from_edges(2, &[0, 1], &[1], &[1.0])
            .unwrap_err()
            .contains("length"));
    }

    #[test]
    fn two_triangles_joined_by_one_edge() {
        let g = csr(
            6,
            &[
                (0, 1, 1.0),
                (0, 2, 1.0),
                (1, 2, 1.0),
                (2, 3, 1.0),
                (3, 4, 1.0),
                (3, 5, 1.0),
                (4, 5, 1.0),
            ],
        );
        for seed in 0..20 {
            let p = leiden(&g, &params(1.0, seed)).unwrap();
            assert_eq!(p.membership, vec![0, 0, 0, 1, 1, 1], "seed {seed}");
        }
    }

    #[test]
    fn ring_of_cliques() {
        // 12 cliques of 6 nodes in a ring: at γ = 1 the modularity optimum is one per clique
        let (cliques, size) = (12u32, 6u32);
        let mut edges = Vec::new();
        for c in 0..cliques {
            for i in 0..size {
                for j in i + 1..size {
                    edges.push((c * size + i, c * size + j, 1.0));
                }
            }
            let (a, b) = (c * size, ((c + 1) % cliques) * size + 1);
            edges.push((a.min(b), a.max(b), 1.0));
        }
        let g = csr((cliques * size) as usize, &edges);
        let p = leiden(&g, &params(1.0, 7)).unwrap();
        assert_eq!(p.n_clusters, cliques as usize);
        for c in 0..cliques as usize {
            let block = &p.membership[c * size as usize..(c + 1) * size as usize];
            assert!(block.iter().all(|&x| x == block[0]));
        }
    }

    #[test]
    fn recovers_a_planted_partition() {
        let g = planted(8, 40, 0.3, 300, 3);
        let p = leiden(&g, &params(1.0, 1)).unwrap();
        let truth: Vec<u32> = (0..320).map(|v| (v / 40) as u32).collect();
        assert_eq!(p.membership, truth);
    }

    #[test]
    fn modularity_matches_the_definition() {
        // Two triangles and a bridge, with non-unit weights, against a dense evaluation of
        // (1/2m) Σ_ij (A_ij − γ k_i k_j / 2m) δ(c_i, c_j)
        let edges = [
            (0, 1, 0.3),
            (0, 2, 0.9),
            (1, 2, 0.4),
            (2, 3, 0.2),
            (3, 4, 0.7),
            (3, 5, 0.6),
            (4, 5, 0.8),
        ];
        let g = csr(6, &edges);
        let mut a = [[0f64; 6]; 6];
        for &(i, j, w) in &edges {
            a[i as usize][j as usize] = w;
            a[j as usize][i as usize] = w;
        }
        let k: Vec<f64> = a.iter().map(|r| r.iter().sum()).collect();
        let two_m: f64 = k.iter().sum();
        for (membership, res) in [
            ([0, 0, 0, 1, 1, 1], 1.0),
            ([0, 1, 0, 2, 2, 1], 0.7),
            ([0; 6], 1.3),
        ] {
            let mut dense = 0.0;
            for i in 0..6 {
                for j in 0..6 {
                    if membership[i] == membership[j] {
                        dense += a[i][j] - res * k[i] * k[j] / two_m;
                    }
                }
            }
            assert!((q(&g, &membership, res) - dense / two_m).abs() < 1e-14);
        }
        assert!(modularity(&g, &[0, 0, 0, 1, 1, 9], 1.0).is_err());
    }

    #[test]
    fn every_run_is_connected_and_gamma_separated() {
        // Leiden's guarantees after every iteration: each cluster is connected, and no two
        // adjacent clusters gain from merging
        for seed in 0..8 {
            let g = planted(12, 40, 0.1, 400, 100 + seed);
            for &res in &[0.25, 1.0, 3.0] {
                for n_iterations in [1, 2, 5] {
                    let p = leiden(
                        &g,
                        &LeidenParams {
                            n_iterations,
                            ..params(res, seed)
                        },
                    )
                    .unwrap();
                    assert!(
                        clusters_are_connected(&g, &p.membership),
                        "seed {seed}, γ {res}"
                    );
                    assert!(
                        best_merge_gain(&g, &p.membership, res) <= 1e-12,
                        "seed {seed}, γ {res}"
                    );
                }
            }
        }
    }

    #[test]
    fn quality_never_decreases_with_more_iterations() {
        // Iteration i draws from stream (seed, i), so k + 1 iterations extend k iterations
        let g = planted(12, 40, 0.1, 400, 21);
        for seed in 0..4 {
            let mut last = f64::NEG_INFINITY;
            for n_iterations in 1..=6 {
                let p = leiden(
                    &g,
                    &LeidenParams {
                        n_iterations,
                        ..params(1.0, seed)
                    },
                )
                .unwrap();
                let quality = q(&g, &p.membership, 1.0);
                assert!(
                    quality >= last - 1e-12,
                    "seed {seed}, {n_iterations} iterations"
                );
                last = quality;
            }
        }
    }

    #[test]
    fn until_stable_ends_node_optimal() {
        for seed in 0..4 {
            let g = planted(12, 40, 0.1, 400, 31 + seed);
            let p = leiden(
                &g,
                &LeidenParams {
                    n_iterations: -1,
                    ..params(1.0, seed)
                },
            )
            .unwrap();
            assert!(
                p.converged && p.iterations < 100,
                "{} iterations",
                p.iterations
            );
            // No single node gains by moving
            assert!(best_node_move_gain(&g, &p.membership, 1.0) <= 1e-12);
            // Another iteration from the result changes nothing
            let again = leiden_from(
                &g,
                &LeidenParams {
                    n_iterations: 1,
                    ..params(1.0, seed + 9)
                },
                &p.membership,
            )
            .unwrap();
            assert!(again.converged);
            assert_eq!(again.membership, p.membership);
        }
    }

    #[test]
    fn finds_the_optimum_of_small_graphs() {
        // Every partition of 8 nodes (4,140 of them) against the best of 10 seeds
        fn partitions(n: usize, prefix: &mut Vec<u32>, out: &mut Vec<Vec<u32>>) {
            if prefix.len() == n {
                out.push(prefix.clone());
                return;
            }
            let next = prefix.iter().max().map_or(0, |&m| m + 1);
            for c in 0..=next {
                prefix.push(c);
                partitions(n, prefix, out);
                prefix.pop();
            }
        }
        let mut all = Vec::new();
        partitions(8, &mut Vec::new(), &mut all);
        assert_eq!(all.len(), 4140);
        let mut rng = Rng::new(&[77]);
        let (mut graphs, mut optimal) = (0, 0);
        for _ in 0..40 {
            let mut edges = Vec::new();
            for i in 0..8u32 {
                for j in i + 1..8 {
                    if rng.unif() < 0.4 {
                        edges.push((i, j, 0.1 + 0.9 * rng.unif()));
                    }
                }
            }
            if edges.is_empty() {
                continue;
            }
            let g = csr(8, &edges);
            let best = all
                .iter()
                .map(|m| q(&g, m, 1.0))
                .fold(f64::NEG_INFINITY, f64::max);
            let found = (0..10)
                .map(|s| q(&g, &leiden(&g, &params(1.0, s)).unwrap().membership, 1.0))
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(found <= best + 1e-12);
            assert!(
                found >= best - 0.05 * best.abs() - 1e-12,
                "found {found}, optimum {best}"
            );
            graphs += 1;
            optimal += (found >= best - 1e-12) as usize;
        }
        assert!(
            optimal * 10 >= graphs * 9,
            "optimum on {optimal} of {graphs} graphs"
        );
    }

    #[test]
    fn scaling_the_weights_by_a_power_of_two_changes_nothing() {
        // Scaling every weight and β by 4 is exact in floating point and leaves every comparison
        // and every sampling probability unchanged
        let g = planted(10, 40, 0.1, 300, 41);
        let mut g4 = g.clone();
        g4.w.iter_mut().for_each(|x| *x *= 4.0);
        for seed in 0..4 {
            let p = leiden(&g, &params(1.0, seed)).unwrap();
            let p4 = leiden(
                &g4,
                &LeidenParams {
                    beta: 0.04,
                    ..params(1.0, seed)
                },
            )
            .unwrap();
            assert_eq!(p, p4, "seed {seed}");
        }
    }

    #[test]
    fn same_seed_same_partition_at_any_thread_count() {
        // Large enough for several aggregation blocks, with non-integer weights
        let g = planted(400, 25, 0.25, 6000, 9);
        let run = |threads: usize, seed: u64| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| leiden(&g, &params(1.0, seed)).unwrap())
        };
        for seed in 0..2 {
            let one = run(1, seed);
            for threads in [2, 3, 8] {
                assert_eq!(run(threads, seed), one, "seed {seed}, {threads} threads");
            }
        }
    }

    #[test]
    fn many_runs_are_the_same_at_any_thread_count() {
        let g = planted(20, 30, 0.2, 400, 6);
        let runs: Vec<LeidenParams> = (0..6).map(|s| params(0.5 + 0.3 * s as f64, s)).collect();
        let n = g.n();
        let mut base = vec![0u32; runs.len() * n];
        let infos = many_into(&g, &runs, &mut base, 1).unwrap();
        // 8 threads for 6 runs: the spare threads work inside the runs
        for threads in [2, 3, 8] {
            let mut out = vec![0u32; runs.len() * n];
            assert_eq!(many_into(&g, &runs, &mut out, threads).unwrap(), infos);
            assert_eq!(out, base, "{threads} threads");
        }
    }

    #[test]
    fn many_runs_match_single_runs() {
        let g = planted(6, 30, 0.2, 100, 5);
        let runs: Vec<LeidenParams> = [0.5, 1.0, 2.0]
            .iter()
            .flat_map(|&r| (0..3).map(move |s| params(r, s)))
            .collect();
        let many = leiden_many(&g, &runs).unwrap();
        for (p, r) in many.iter().zip(&runs) {
            assert_eq!(*p, leiden(&g, r).unwrap());
        }
    }

    #[test]
    fn isolated_nodes_and_edge_cases() {
        // Nodes 4 and 5 have no edges; they stay alone
        let g = csr(6, &[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0), (2, 3, 0.1)]);
        let p = leiden(&g, &params(1.0, 0)).unwrap();
        assert_ne!(p.membership[4], p.membership[5]);
        assert!(p.membership[..4]
            .iter()
            .all(|&c| c != p.membership[4] && c != p.membership[5]));
        // No edges at all, one node, no nodes
        assert_eq!(
            leiden(&csr(3, &[]), &params(1.0, 0)).unwrap().membership,
            vec![0, 1, 2]
        );
        assert_eq!(leiden(&csr(1, &[]), &params(1.0, 0)).unwrap().n_clusters, 1);
        assert_eq!(leiden(&csr(0, &[]), &params(1.0, 0)).unwrap().n_clusters, 0);
        // A single edge forms one cluster
        assert_eq!(
            leiden(&csr(2, &[(0, 1, 1.0)]), &params(1.0, 0))
                .unwrap()
                .n_clusters,
            1
        );
        // γ = 0 merges each connected component
        let p = leiden(
            &csr(5, &[(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0)]),
            &params(0.0, 0),
        )
        .unwrap();
        assert_eq!(p.membership, vec![0, 0, 0, 1, 1]);
    }

    #[test]
    fn rejects_bad_parameters_and_labels() {
        let g = csr(2, &[(0, 1, 1.0)]);
        assert!(leiden(&g, &params(-1.0, 0)).is_err());
        assert!(leiden(&g, &params(f64::NAN, 0)).is_err());
        assert!(leiden(
            &g,
            &LeidenParams {
                beta: 0.0,
                ..params(1.0, 0)
            }
        )
        .is_err());
        assert!(leiden(
            &g,
            &LeidenParams {
                n_iterations: 0,
                ..params(1.0, 0)
            }
        )
        .is_err());
        assert!(leiden_from(&g, &params(1.0, 0), &[0]).is_err());
        assert!(leiden_from(&g, &params(1.0, 0), &[0, 2]).is_err());
    }

    #[test]
    fn exp_matches_std() {
        let mut worst = 0f64;
        let mut x = 0.0;
        while x > -700.0 {
            let (a, b) = (exp_nonpositive(x), x.exp());
            worst = worst.max(((a - b) / b).abs());
            x -= 0.013_7;
        }
        assert!(worst < 1e-15, "max relative error {worst}");
        assert_eq!(exp_nonpositive(0.0), 1.0);
        assert_eq!(exp_nonpositive(-1000.0), 0.0);
    }

    #[test]
    fn rng_draws_are_uniform_and_shuffles_are_permutations() {
        let mut rng = Rng::new(&[42]);
        let mut hits = [0u32; 7];
        for _ in 0..70_000 {
            hits[rng.below(7) as usize] += 1;
        }
        assert!(
            hits.iter().all(|&h| (9_500..10_500).contains(&h)),
            "{hits:?}"
        );
        let mut a: Vec<u32> = (0..100).collect();
        rng.shuffle(&mut a);
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..100).collect::<Vec<u32>>());
        assert_ne!(a, sorted);
        assert!((0..1000).all(|_| (0.0..1.0).contains(&rng.unif())));
        assert_ne!(Rng::new(&[1, 2]).next_u64(), Rng::new(&[2, 1]).next_u64());
        assert_eq!(Rng::new(&[1, 2]).next_u64(), Rng::new(&[1, 2]).next_u64());
    }
}
