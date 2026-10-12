//! 瓦片 matmul：`out[m×n] = a[m×d] · b[n×d]ᵀ`，全部行主序、紧密排列。
//!
//! macOS 走 Accelerate（自动用 AMX），其它平台走纯 Rust 的 `gemm` crate，
//! 不打包 OpenBLAS/MKL（docs/06 §3）。这里只产出 f32 粗选用的相似度；
//! 最终名次由 f64 精排决定，所以换后端只影响速度、不影响输出（docs/06 §1）。

/// matmul 后端。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Matmul {
    /// Apple Accelerate 的 `cblas_sgemm`，仅 macOS 可用。
    Accelerate,
    /// 纯 Rust 的 `gemm` crate，全平台可用。
    Gemm,
}

impl Matmul {
    /// 当前平台的默认后端。
    pub fn platform_default() -> Self {
        if cfg!(target_os = "macos") {
            Matmul::Accelerate
        } else {
            Matmul::Gemm
        }
    }

    /// 该后端在当前平台上是否可用。
    pub fn is_available(self) -> bool {
        match self {
            Matmul::Accelerate => cfg!(target_os = "macos"),
            Matmul::Gemm => true,
        }
    }

    /// 按名字解析（"accelerate" / "gemm"），供绑定层使用。
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

/// `out[..m*n] = a · bᵀ`。`a` 是 m×d，`b` 是 n×d，`out` 的行跨度为 n。
///
/// 调用方保证后端可用（见 [`Matmul::is_available`]）且切片长度足够。
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
        // SAFETY: 上面的 assert 保证了三个缓冲区的长度。
        Matmul::Accelerate => unsafe { accelerate::sgemm_nt(a, m, b, n, d, out) },
        #[cfg(not(target_os = "macos"))]
        Matmul::Accelerate => panic!("Accelerate 只在 macOS 上可用"),
        // SAFETY: 同上；步长按行主序给出。gemm 的约定是 dst = alpha·dst + beta·lhs·rhs，
        // 所以 alpha=0、beta=1、read_dst=false 即 dst = lhs·rhs。
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
                // rhs = bᵀ（d×n）：元素 (p, j) 在 b[j*d + p]，列步长 d、行步长 1
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

    /// SAFETY: 调用方保证 `a`、`b`、`out` 分别至少有 m·d、n·d、m·n 个元素。
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
