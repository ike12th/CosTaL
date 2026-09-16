//! CosTaL+ 的 PyO3 绑定。这里只做类型转换和错误映射，算法在 `costal-core`。

use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// 骨架自检，返回一句带 core 版本号的问候。
#[pyfunction]
fn hello() -> String {
    costal_core::hello()
}

/// core crate 的版本号（与 wheel 版本一致时说明装的是这次构建的产物）。
#[pyfunction]
fn core_version() -> &'static str {
    costal_core::VERSION
}

/// 按行计算 L2 范数：`(n, d)` float32 数组 -> 长度 `n` 的 float32 数组。
#[pyfunction]
fn row_l2_norms<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let arr = x.as_array();
    let (n_rows, n_cols) = arr.dim();
    // 只有 C 连续时才能拿到扁平切片；非连续的先让 numpy 自己整理。
    let slice = arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("输入必须是 C 连续数组，先 np.ascontiguousarray"))?;
    let norms = py.detach(|| costal_core::row_l2_norms(slice, n_rows, n_cols));
    Ok(PyArray1::from_vec(py, norms))
}

#[pymodule]
fn _costal_plus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(row_l2_norms, m)?)?;
    Ok(())
}
