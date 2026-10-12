//! Tile matmul: `out[m×n] = a[m×d] · b[n×d]ᵀ`, all row-major and tightly packed.
//!
//! macOS uses Accelerate (which uses AMX automatically); other platforms use the pure-Rust
//! `gemm` crate, with no bundled OpenBLAS/MKL (docs/06 §3). This only produces similarities for
//! f32 coarse selection; the final ranking comes from f64 rescoring, so switching backends
//! changes speed, never output (docs/06 §1).

/// Matmul backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Matmul {
    /// Apple Accelerate's `cblas_sgemm`; macOS only.
    Accelerate,
    /// The pure-Rust `gemm` crate; available on every platform.
    Gemm,
}

impl Matmul {
    /// Default backend for the current platform.
    pub fn platform_default() -> Self {
        if cfg!(target_os = "macos") {
            Matmul::Accelerate
        } else {
            Matmul::Gemm
        }
    }

    /// Whether this backend is available on the current platform.
    pub fn is_available(self) -> bool {
        match self {
            Matmul::Accelerate => cfg!(target_os = "macos"),
            Matmul::Gemm => true,
        }
    }

    /// Parse by name ("accelerate" / "gemm"), for the bindings.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "accelerate" => Some(Matmul::Accelerate),
            "gemm" => Some(Matmul::Gemm),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Matmul::Accelerate => "accelerate",
            Matmul::Gemm => "gemm",
        }
    }
}

/// `out[..m*n] = a · bᵀ`, where `a` is m×d, `b` is n×d, and `out` has row stride n.
///
/// The caller guarantees the backend is available (see [`Matmul::is_available`]) and the
/// slices are long enough.
pub fn tile_nt(
    backend: Matmul,
    a: &[f32],
    m: usize,
    b: &[f32],
    n: usize,
    d: usize,
    out: &mut [f32],
) {
    assert!(a.len() >= m * d && b.len() >= n * d && out.len() >= m * n);
    if m == 0 || n == 0 {
        return;
    }
    if d == 0 {
        out[..m * n].fill(0.0);
        return;
    }
    match backend {
        #[cfg(target_os = "macos")]
        // SAFETY: the assert above guarantees the lengths of all three buffers.
        Matmul::Accelerate => unsafe { accelerate::sgemm_nt(a, m, b, n, d, out) },
        #[cfg(not(target_os = "macos"))]
        Matmul::Accelerate => panic!("Accelerate is only available on macOS"),
        // SAFETY: as above; strides are given for row-major layout. gemm's convention is
        // dst = alpha·dst + beta·lhs·rhs, so alpha=0, beta=1, read_dst=false gives dst = lhs·rhs.
        Matmul::Gemm => unsafe {
            gemm::gemm(
                m,
                n,
                d,
                out.as_mut_ptr(),
                1,
                n as isize,
                false,
                a.as_ptr(),
                1,
                d as isize,
                // rhs = bᵀ (d×n): element (p, j) is b[j*d + p], column stride d, row stride 1
                b.as_ptr(),
                d as isize,
                1,
                0.0,
                1.0,
                false,
                false,
                false,
                gemm::Parallelism::None,
            )
        },
    }
}

#[cfg(target_os = "macos")]
mod accelerate {
    use std::os::raw::c_int;

    const ROW_MAJOR: c_int = 101;
    const NO_TRANS: c_int = 111;
    const TRANS: c_int = 112;

    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        #[allow(clippy::too_many_arguments)]
        fn cblas_sgemm(
            order: c_int,
            trans_a: c_int,
            trans_b: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f32,
            a: *const f32,
            lda: c_int,
            b: *const f32,
            ldb: c_int,
            beta: f32,
            c: *mut f32,
            ldc: c_int,
        );
    }

    /// SAFETY: the caller guarantees `a`, `b`, `out` hold at least m·d, n·d, m·n elements.
    pub unsafe fn sgemm_nt(a: &[f32], m: usize, b: &[f32], n: usize, d: usize, out: &mut [f32]) {
        cblas_sgemm(
            ROW_MAJOR,
            NO_TRANS,
            TRANS,
            m as c_int,
            n as c_int,
            d as c_int,
            1.0,
            a.as_ptr(),
            d as c_int,
            b.as_ptr(),
            d as c_int,
            0.0,
            out.as_mut_ptr(),
            n as c_int,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(a: &[f32], m: usize, b: &[f32], n: usize, d: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = (0..d).map(|p| a[i * d + p] * b[j * d + p]).sum();
            }
        }
        out
    }

    fn check(backend: Matmul) {
        let (m, n, d) = (37, 53, 19);
        let a: Vec<f32> = (0..m * d)
            .map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0)
            .collect();
        let b: Vec<f32> = (0..n * d)
            .map(|i| ((i * 104729) % 97) as f32 / 48.0 - 1.0)
            .collect();
        let mut out = vec![f32::NAN; m * n];
        tile_nt(backend, &a, m, &b, n, d, &mut out);
        for (got, want) in out.iter().zip(naive(&a, m, &b, n, d)) {
            assert!((got - want).abs() < 1e-4, "{backend:?}: {got} vs {want}");
        }
    }

    #[test]
    fn gemm_matches_naive() {
        check(Matmul::Gemm);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn accelerate_matches_naive() {
        check(Matmul::Accelerate);
    }
}
