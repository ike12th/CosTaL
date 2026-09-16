//! CosTaL+ core.
//!
//! 纯 Rust、不依赖 Python。真正的算法（kNN 建图 / Tanimoto 边权 / 社区发现）
//! 都落在这里；`costal-py` 只做绑定，不放逻辑。
//!
//! 现阶段只有骨架级内容，用来打通 `cargo test` 与 `pip install -e .` 的链路。

/// 编译进二进制的版本号，用来确认 Python 侧加载的确实是这次构建的产物。
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// 骨架自检：返回一句带版本号的问候。
pub fn hello() -> String {
    format!("CosTaL+ core v{VERSION} is alive")
}

/// 按行计算 L2 范数。
///
/// 存在的意义不是这个函数本身，而是它证明了 numpy 数组能零拷贝进 Rust、
/// 结果能还给 Python。`‖·‖²` 后面会被 Tanimoto 边权直接复用
/// （见 docs/01 §1.1：a·b 与 ‖·‖² 复用是不二次遍历边表的前提）。
///
/// `x` 是行主序的 `n_rows × n_cols` 稠密矩阵。
pub fn row_l2_norms(x: &[f32], n_rows: usize, n_cols: usize) -> Vec<f32> {
    assert_eq!(
        x.len(),
        n_rows * n_cols,
        "row_l2_norms: 数据长度 {} 与形状 {}x{} 不符",
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
