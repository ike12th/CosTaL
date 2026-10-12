//! Graph layer: the original `build_graph` symmetrization (docs/04 §0.0 item 3).
//!
//! The original uses `csr + csr.T − csr.multiply(csr.T)` (fuzzy union), **not** `(csr + csr.T) / 2`;
//! then it drops self-loops, keeps the upper triangle and removes edges that are exactly 0. We
//! follow it step by step, and the output matches the original scipy pipeline bit for bit.

use rayon::prelude::*;

/// Upper-triangular edge list in ascending (row, col) order — the same edge order as the
/// original scipy CSR.
#[derive(Clone, Debug, PartialEq)]
pub struct Edges {
    pub rows: Vec<u32>,
    pub cols: Vec<u32>,
    pub weights: Vec<f64>,
}

/// Fuzzy union `S + Sᵀ − S∘Sᵀ` of an n×k directed kNN with edge weights.
///
/// When a pair has edges in both directions the weight is `(a + b) − a·b`; with one direction
/// it is that direction's weight.
/// Addition and multiplication are commutative, so which direction comes first never changes a
/// single bit of the result.
pub fn fuzzy_union(n: usize, k: usize, idx: &[u32], w: &[f64]) -> Edges {
    assert!(idx.len() == n * k && w.len() == n * k);
    let mut e: Vec<(u64, f64)> = Vec::with_capacity(n * k);
    for (t, (&j, &wt)) in idx.iter().zip(w).enumerate() {
        let (i, j) = (t / k, j as usize);
        if i == j {
            continue;
        }
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        e.push((((lo as u64) << 32) | hi as u64, wt));
    }
    e.par_sort_unstable_by_key(|p| p.0);

    let mut out = Edges {
        rows: Vec::with_capacity(e.len()),
        cols: Vec::with_capacity(e.len()),
        weights: Vec::with_capacity(e.len()),
    };
    let mut t = 0;
    while t < e.len() {
        let (key, a) = e[t];
        let v = if t + 1 < e.len() && e[t + 1].0 == key {
            let b = e[t + 1].1;
            debug_assert!(
                t + 2 >= e.len() || e[t + 2].0 != key,
                "duplicate neighbor within one row"
            );
            t += 2;
            (a + b) - a * b
        } else {
            t += 1;
            a
        };
        if v != 0.0 {
            out.rows.push((key >> 32) as u32);
            out.cols.push(key as u32);
            out.weights.push(v);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzzy_union_combines_both_directions() {
        // 0→1 (0.5), 1→0 (0.4), 0→2 (0.3), 2→1 (0.0), 1→2 (0.2), 2→0 (0.6)
        let idx = [1, 2, 0, 2, 1, 0];
        let w = [0.5, 0.3, 0.4, 0.2, 0.0, 0.6];
        let g = fuzzy_union(3, 2, &idx, &w);
        assert_eq!(g.rows, vec![0, 0, 1]);
        assert_eq!(g.cols, vec![1, 2, 2]);
        assert_eq!(
            g.weights,
            vec![(0.5 + 0.4) - 0.5 * 0.4, (0.3 + 0.6) - 0.3 * 0.6, 0.2]
        );
    }

    #[test]
    fn drops_exact_zeros_and_keeps_one_sided_edges() {
        // 0→1 has weight 0 (one direction only) and must be dropped; one-way 1→2 is kept
        let idx = [1, 2, 0];
        let w = [0.0, 0.7, 0.0];
        let g = fuzzy_union(3, 1, &idx, &w);
        assert_eq!((g.rows, g.cols, g.weights), (vec![1], vec![2], vec![0.7]));
    }
}
