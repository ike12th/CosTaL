//! CosTaL core.
//!
//! Pure Rust with no Python dependency. The actual algorithms (kNN graph / Tanimoto edge
//! weights / community detection) live here; `costal-py` only does bindings, no logic.
//!
//! - [`knn`]: exact kNN (fused blocked coarse selection + f64 rescoring, docs/03 §2, docs/06 §1)
//! - [`weight`]: edge weights (tanimoto / cosine / jaccard)
//! - [`graph`]: the original's fuzzy-union symmetrization
//! - [`leiden`]: Leiden community detection, deterministic at any thread count (docs/07 §5)
//! - [`matmul`], [`cache`]: tile matmul backends and cache detection (docs/06 §2–§3)

pub mod cache;
pub mod graph;
pub mod knn;
pub mod leiden;
pub mod matmul;
pub mod weight;

/// Version compiled into the binary, to confirm Python loaded the artifact of this build.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Skeleton self-check: returns a greeting that includes the version.
pub fn hello() -> String {
    format!("CosTaL core v{VERSION} is alive")
}

/// Row-wise L2 norms.
///
/// The point is not the function itself but proving that numpy arrays reach Rust without a copy
/// and results go back to Python. `‖·‖²` is later reused directly by Tanimoto edge weights
/// (see docs/01 §1.1: reusing a·b and ‖·‖² is what avoids a second pass over the edge list).
///
/// `x` is a row-major dense `n_rows × n_cols` matrix.
pub fn row_l2_norms(x: &[f32], n_rows: usize, n_cols: usize) -> Vec<f32> {
    assert_eq!(
        x.len(),
        n_rows * n_cols,
        "row_l2_norms: data length {} does not match shape {}x{}",
        x.len(),
        n_rows,
        n_cols
    );
    x.chunks_exact(n_cols.max(1))
        .take(n_rows)
        .map(|row| row.iter().map(|v| v * v).sum::<f32>().sqrt())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_carries_version() {
        assert!(hello().contains(VERSION));
    }

    #[test]
    fn l2_norms_of_unit_rows() {
        let x = [3.0f32, 4.0, 0.0, 1.0];
        let n = row_l2_norms(&x, 2, 2);
        assert!((n[0] - 5.0).abs() < 1e-6);
        assert!((n[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_norms_handles_zero_rows() {
        let x = [0.0f32; 6];
        assert_eq!(row_l2_norms(&x, 2, 3), vec![0.0, 0.0]);
    }
}
