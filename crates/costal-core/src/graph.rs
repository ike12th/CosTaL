//! 图层：原版 `build_graph` 的对称化（docs/04 §0.0 第 3 条）。
//!
//! 原版是 `csr + csr.T − csr.multiply(csr.T)`（模糊并集），**不是** `(csr + csr.T) / 2`；
//! 然后去自环、取上三角、去掉恰为 0 的边。这里逐步照做，输出与原版 scipy 管线逐位一致。

use rayon::prelude::*;

/// 上三角边表，按（行, 列）升序——与原版 scipy CSR 的边序相同。
#[derive(Clone, Debug, PartialEq)]
pub struct Edges {
    pub rows: Vec<u32>,
    pub cols: Vec<u32>,
    pub weights: Vec<f64>,
}

/// 对 n×k 的有向 kNN 边权做模糊并集 `S + Sᵀ − S∘Sᵀ`。
///
/// 一对点两个方向都有边时权重为 `(a + b) − a·b`，只有一个方向时为该方向的权重。
/// 加法与乘法都满足交换律，所以哪个方向先出现不影响结果的每一位。
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
                "同一行里出现了重复邻居"
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
        // 0→1 (0.5)、1→0 (0.4)、0→2 (0.3)、2→1 (0.0)、1→2 (0.2)、2→0 (0.6)
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
        // 0→1 权重 0（单向）应被去掉；1→2 单向保留
        let idx = [1, 2, 0];
        let w = [0.0, 0.7, 0.0];
        let g = fuzzy_union(3, 1, &idx, &w);
        assert_eq!((g.rows, g.cols, g.weights), (vec![1], vec![2], vec![0.7]));
    }
}
