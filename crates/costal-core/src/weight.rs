//! Edge weights: pure transforms on the kNN output (docs/04 §6.1).
//!
//! tanimoto / cosine need only (cos, ‖a‖, ‖b‖), O(N·k); jaccard compares neighbors of
//! neighbors, O(N·k²).
//! Each follows the original CosTaL's order of operations, so in f64 they match the original
//! bit for bit.

use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weight {
    Tanimoto,
    Cosine,
    Jaccard,
}

impl Weight {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "tanimoto" => Some(Weight::Tanimoto),
            "cosine" => Some(Weight::Cosine),
            "jaccard" => Some(Weight::Jaccard),
            _ => None,
        }
    }
}

/// The original `tani_from_cos`: `d = a·b·cos`; 0 when `d == 0`, otherwise `d / (a² + b² − d)`.
#[inline]
pub fn tani_from_cos(a: f64, b: f64, cos: f64) -> f64 {
    let d = a * b * cos;
    if d == 0.0 {
        0.0
    } else {
        d / (a * a + b * b - d)
    }
}

/// Weight every edge of an n×k directed kNN by `weight`. All `idx` values must be in `[0, n)`,
/// with no duplicates within a row.
pub fn reweight(
    n: usize,
    k: usize,
    idx: &[u32],
    cos: &[f64],
    norms: &[f64],
    weight: Weight,
) -> Vec<f64> {
    assert!(idx.len() == n * k && cos.len() == n * k && norms.len() == n);
    match weight {
        Weight::Cosine => cos.to_vec(),
        Weight::Tanimoto => idx
            .par_iter()
            .zip(cos.par_iter())
            .enumerate()
            .map(|(e, (&j, &c))| tani_from_cos(norms[e / k], norms[j as usize], c))
            .collect(),
        Weight::Jaccard => jaccard(n, k, idx),
    }
}

/// The original `calc_jaccard`: edge (i, j) gets `s / (2k − s)` with s = |N(i) ∩ N(j)|;
/// neighbor sets exclude the row itself.
fn jaccard(n: usize, k: usize, idx: &[u32]) -> Vec<f64> {
    let mut sorted = idx.to_vec();
    sorted.par_chunks_mut(k).for_each(|row| row.sort_unstable());
    let mut out = vec![0.0f64; n * k];
    out.par_chunks_mut(k).enumerate().for_each(|(i, row_out)| {
        let a = &sorted[i * k..(i + 1) * k];
        for (t, &j) in idx[i * k..(i + 1) * k].iter().enumerate() {
            let b = &sorted[j as usize * k..(j as usize + 1) * k];
            let s = intersect_sorted(a, b) as f64;
            row_out[t] = s / (2.0 * k as f64 - s);
        }
    });
    out
}

fn intersect_sorted(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut c) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                c += 1;
                i += 1;
                j += 1;
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tanimoto_matches_closed_form() {
        // Same direction, different magnitude: cos = 1, T = ab / (a² + b² − ab)
        assert_eq!(tani_from_cos(1.0, 2.0, 1.0), 2.0 / 3.0);
        // Equal lengths degenerate to c / (2 − c)
        let c = 0.6;
        assert!((tani_from_cos(3.0, 3.0, c) - c / (2.0 - c)).abs() < 1e-15);
        assert_eq!(tani_from_cos(0.0, 2.0, 0.9), 0.0);
        // The lower bound for signed input is −1/3
        assert!((tani_from_cos(1.0, 1.0, -1.0) + 1.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn jaccard_counts_shared_neighbours() {
        // 0:{1,2} 1:{0,2} 2:{0,1} 3:{0,1}
        let idx = [1, 2, 0, 2, 0, 1, 0, 1];
        let w = reweight(4, 2, &idx, &[0.0; 8], &[1.0; 4], Weight::Jaccard);
        // Edge 0→1: N(0)∩N(1) = {2}, s=1 → 1/3; edge 3→0: N(3)∩N(0) = {1} → 1/3
        assert_eq!(w[0], 1.0 / 3.0);
        assert_eq!(w[6], 1.0 / 3.0);
        // Edge 2→0: N(2)={0,1}, N(0)={1,2}, intersection {1} → 1/3
        assert_eq!(w[4], 1.0 / 3.0);
    }
}
