//! CosTaL+ 的 PyO3 绑定。这里只做类型转换和错误映射，算法在 `costal-core`。
//!
//! 下划线开头的函数是给 `costal_plus` 包内部用的底层入口；公开 API 在 Python 侧
//! （`python/costal_plus/_api.py`），参数校验与输入诊断也在那里。

use costal_core::knn::{self, Knn, KnnParams};
use costal_core::matmul::Matmul;
use costal_core::weight::{self, Weight};
use costal_core::{cache, graph};
use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

type KnnArrays<'py> = (
    Bound<'py, PyArray2<i32>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
);

type EdgeArrays<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
);

fn value_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

const NOT_CONTIGUOUS: &str = "输入必须是 C 连续数组，先 np.ascontiguousarray";

/// 在指定线程数的 rayon 线程池里跑；`None` 或 0 用全局池。
fn with_threads<T: Send>(n_threads: Option<usize>, f: impl FnOnce() -> T + Send) -> PyResult<T> {
    match n_threads {
        None | Some(0) => Ok(f()),
        Some(t) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .map_err(value_err)?;
            Ok(pool.install(f))
        }
    }
}

fn knn_to_py(py: Python<'_>, knn: Knn) -> PyResult<KnnArrays<'_>> {
    let idx: Vec<i32> = knn.idx.iter().map(|&j| j as i32).collect();
    Ok((
        Array2::from_shape_vec((knn.n, knn.k), idx)
            .map_err(value_err)?
            .into_pyarray(py),
        Array2::from_shape_vec((knn.n, knn.k), knn.cos)
            .map_err(value_err)?
            .into_pyarray(py),
        Array1::from_vec(knn.norms).into_pyarray(py),
    ))
}

/// 校验 (n, k) 的 int32 邻居矩阵，转成 u32。
fn neighbours(idx: &PyReadonlyArray2<'_, i32>) -> PyResult<(usize, usize, Vec<u32>)> {
    let (n, k) = idx.as_array().dim();
    let flat = idx.as_slice().map_err(|_| value_err(NOT_CONTIGUOUS))?;
    if let Some(&bad) = flat.iter().find(|&&j| j < 0 || j as usize >= n) {
        return Err(value_err(format!("邻居索引 {bad} 越界（n={n}）")));
    }
    Ok((n, k, flat.iter().map(|&j| j as u32).collect()))
}

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
    let (n_rows, n_cols) = x.as_array().dim();
    let slice = x.as_slice().map_err(|_| value_err(NOT_CONTIGUOUS))?;
    let norms = py.detach(|| costal_core::row_l2_norms(slice, n_rows, n_cols));
    Ok(PyArray1::from_vec(py, norms))
}

/// 精确 kNN（融合分块粗选 + f64 精排）。返回 (idx, cos, norms, stats)。
#[pyfunction]
#[pyo3(signature = (x, k, *, margin=None, row_block=None, col_block=None, matmul=None, sort_by_norm=true, n_threads=None))]
#[allow(clippy::too_many_arguments)]
fn _knn_exact<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    k: usize,
    margin: Option<usize>,
    row_block: Option<usize>,
    col_block: Option<usize>,
    matmul: Option<&str>,
    sort_by_norm: bool,
    n_threads: Option<usize>,
) -> PyResult<(KnnArrays<'py>, Bound<'py, PyDict>)> {
    let (n, d) = x.as_array().dim();
    if n > i32::MAX as usize {
        return Err(value_err(format!("n={n} 超出 int32 索引范围")));
    }
    let slice = x.as_slice().map_err(|_| value_err(NOT_CONTIGUOUS))?;
    let backend = match matmul {
        None => Matmul::platform_default(),
        Some(name) => Matmul::from_name(name)
            .ok_or_else(|| value_err(format!("未知 matmul 后端 {name:?}")))?,
    };
    let params = KnnParams {
        margin: margin.unwrap_or(k),
        row_block,
        col_block,
        matmul: backend,
        sort_by_norm,
        ..KnnParams::new(k)
    };
    let (result, st) = py
        .detach(|| {
            with_threads(n_threads, || {
                knn::knn_exact_with_stats(slice, n, d, &params)
            })
        })?
        .map_err(value_err)?;
    let stats = PyDict::new(py);
    stats.set_item("fallback_rows", st.fallback_rows)?;
    stats.set_item("row_block", st.row_block)?;
    stats.set_item("col_block", st.col_block)?;
    stats.set_item("matmul", backend.name())?;
    stats.set_item("margin", params.margin)?;
    Ok((knn_to_py(py, result)?, stats))
}

/// 黄金对照：不分块、不剪枝、全程 f64 的暴力 kNN（只用于测试与基准）。
#[pyfunction]
#[pyo3(signature = (x, k, *, n_threads=None))]
fn _knn_bruteforce<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    k: usize,
    n_threads: Option<usize>,
) -> PyResult<KnnArrays<'py>> {
    let (n, d) = x.as_array().dim();
    let slice = x.as_slice().map_err(|_| value_err(NOT_CONTIGUOUS))?;
    let result = py
        .detach(|| with_threads(n_threads, || knn::knn_bruteforce(slice, n, d, k)))?
        .map_err(value_err)?;
    knn_to_py(py, result)
}

/// 在一份 kNN 上算边权（"tanimoto" / "cosine" / "jaccard"）。
#[pyfunction]
fn _reweight<'py>(
    py: Python<'py>,
    idx: PyReadonlyArray2<'py, i32>,
    cos: PyReadonlyArray2<'py, f64>,
    norms: PyReadonlyArray1<'py, f64>,
    weight: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let w = Weight::from_name(weight).ok_or_else(|| value_err(format!("未知边权 {weight:?}")))?;
    let (n, k, idx) = neighbours(&idx)?;
    if cos.as_array().dim() != (n, k) || norms.as_array().len() != n {
        return Err(value_err("idx、cos、norms 的形状对不上"));
    }
    let cos = cos.as_slice().map_err(|_| value_err(NOT_CONTIGUOUS))?;
    let norms = norms.as_slice().map_err(|_| value_err(NOT_CONTIGUOUS))?;
    let out = py.detach(|| weight::reweight(n, k, &idx, cos, norms, w));
    Ok(Array2::from_shape_vec((n, k), out)
        .map_err(value_err)?
        .into_pyarray(py))
}

/// 原版 build_graph 的模糊并集对称化，返回上三角边表 (rows, cols, weights)。
#[pyfunction]
fn _fuzzy_union<'py>(
    py: Python<'py>,
    idx: PyReadonlyArray2<'py, i32>,
    w: PyReadonlyArray2<'py, f64>,
) -> PyResult<EdgeArrays<'py>> {
    let (n, k, idx) = neighbours(&idx)?;
    if w.as_array().dim() != (n, k) {
        return Err(value_err("idx 与 w 的形状对不上"));
    }
    let w = w.as_slice().map_err(|_| value_err(NOT_CONTIGUOUS))?;
    let e = py.detach(|| graph::fuzzy_union(n, k, &idx, w));
    let to_i64 = |v: Vec<u32>| Array1::from_iter(v.into_iter().map(i64::from)).into_pyarray(py);
    Ok((
        to_i64(e.rows),
        to_i64(e.cols),
        Array1::from_vec(e.weights).into_pyarray(py),
    ))
}

/// 本机每线程能独占的 L2 字节数；读不到返回 None（docs/06 §2.3）。
#[pyfunction]
fn l2_bytes_per_thread() -> Option<usize> {
    cache::l2_bytes_per_thread()
}

/// 瓦片预算（字节）：每线程 L2 的一半，读不到时用保底值。
#[pyfunction]
fn tile_budget_bytes() -> usize {
    cache::tile_budget_bytes()
}

/// 当前平台默认的 matmul 后端名。
#[pyfunction]
fn default_matmul() -> &'static str {
    Matmul::platform_default().name()
}

#[pymodule]
fn _costal_plus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(row_l2_norms, m)?)?;
    m.add_function(wrap_pyfunction!(_knn_exact, m)?)?;
    m.add_function(wrap_pyfunction!(_knn_bruteforce, m)?)?;
    m.add_function(wrap_pyfunction!(_reweight, m)?)?;
    m.add_function(wrap_pyfunction!(_fuzzy_union, m)?)?;
    m.add_function(wrap_pyfunction!(l2_bytes_per_thread, m)?)?;
    m.add_function(wrap_pyfunction!(tile_budget_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(default_matmul, m)?)?;
    Ok(())
}
