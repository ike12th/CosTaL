//! PyO3 bindings for CosTaL. Only type conversion and error mapping live here; the algorithms
//! are in `costal-core`.
//!
//! Functions starting with an underscore are low-level entry points for the `costal`
//! package itself; the public API is on the Python side (`python/costal/_api.py`), together
//! with argument validation and input diagnostics.

use costal_core::knn::{self, Knn, KnnParams, Select};
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
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f64>>,
);

fn value_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

const NOT_CONTIGUOUS: &str = "input must be a C-contiguous array; call np.ascontiguousarray first";

/// Run inside a rayon pool with the given thread count; `None` or 0 uses the global pool.
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
    // u32 → i32 reuses the allocation (same layout; n ≤ i32::MAX was checked on the way in)
    let idx: Vec<i32> = knn.idx.into_iter().map(|j| j as i32).collect();
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

/// Validate an (n, k) int32 neighbor matrix and view it as u32, without copying.
fn neighbours<'a>(idx: &'a PyReadonlyArray2<'_, i32>) -> PyResult<(usize, usize, &'a [u32])> {
    let (n, k) = idx.as_array().dim();
    let flat = idx.as_slice().map_err(|_| value_err(NOT_CONTIGUOUS))?;
    if let Some(&bad) = flat.iter().find(|&&j| j < 0 || j as usize >= n) {
        return Err(value_err(format!(
            "neighbor index {bad} out of range (n={n})"
        )));
    }
    // SAFETY: i32 and u32 have the same size and alignment, and every value was just checked to be
    // non-negative, so each one reads back as the same number.
    let view = unsafe { std::slice::from_raw_parts(flat.as_ptr().cast::<u32>(), flat.len()) };
    Ok((n, k, view))
}

/// Skeleton self-check: returns a greeting with the core version.
#[pyfunction]
fn hello() -> String {
    costal_core::hello()
}

/// Version of the core crate (matching the wheel version means this build is what got installed).
#[pyfunction]
fn core_version() -> &'static str {
    costal_core::VERSION
}

/// Row-wise L2 norms: `(n, d)` float32 array -> float32 array of length `n`.
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

fn parse_select(select: Option<&str>) -> PyResult<Select> {
    match select {
        None => Ok(Select::Cosine),
        Some(name) => {
            Select::from_name(name).ok_or_else(|| value_err(format!("unknown select {name:?}")))
        }
    }
}

/// Exact kNN (fused blocked coarse selection + f64 rescoring). Returns (idx, cos, norms, stats).
#[pyfunction]
#[pyo3(signature = (x, k, *, select=None, margin=None, row_block=None, col_block=None, matmul=None, sort_by_norm=true, prune=true, n_threads=None))]
#[allow(clippy::too_many_arguments)]
fn _knn_exact<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    k: usize,
    select: Option<&str>,
    margin: Option<usize>,
    row_block: Option<usize>,
    col_block: Option<usize>,
    matmul: Option<&str>,
    sort_by_norm: bool,
    prune: bool,
    n_threads: Option<usize>,
) -> PyResult<(KnnArrays<'py>, Bound<'py, PyDict>)> {
    let (n, d) = x.as_array().dim();
    if n > i32::MAX as usize {
        return Err(value_err(format!("n={n} exceeds the int32 index range")));
    }
    let slice = x.as_slice().map_err(|_| value_err(NOT_CONTIGUOUS))?;
    let backend = match matmul {
        None => Matmul::platform_default(),
        Some(name) => Matmul::from_name(name)
            .ok_or_else(|| value_err(format!("unknown matmul backend {name:?}")))?,
    };
    let params = KnnParams {
        select: parse_select(select)?,
        margin: margin.unwrap_or(k),
        row_block,
        col_block,
        matmul: backend,
        sort_by_norm,
        prune,
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
    stats.set_item("select", params.select.name())?;
    stats.set_item("blocks_skipped", st.blocks_skipped)?;
    stats.set_item("blocks_total", st.blocks_total)?;
    Ok((knn_to_py(py, result)?, stats))
}

/// Golden reference: brute-force kNN with no blocking, no pruning, f64 throughout (tests and
/// benchmarks only).
#[pyfunction]
#[pyo3(signature = (x, k, *, select=None, n_threads=None))]
fn _knn_bruteforce<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
    k: usize,
    select: Option<&str>,
    n_threads: Option<usize>,
) -> PyResult<KnnArrays<'py>> {
    let (n, d) = x.as_array().dim();
    let slice = x.as_slice().map_err(|_| value_err(NOT_CONTIGUOUS))?;
    let select = parse_select(select)?;
    let result = py
        .detach(|| {
            with_threads(n_threads, || {
                knn::knn_bruteforce_select(slice, n, d, k, select)
            })
        })?
        .map_err(value_err)?;
    knn_to_py(py, result)
}

/// Edge weights on a kNN ("tanimoto" / "cosine" / "jaccard").
#[pyfunction]
fn _reweight<'py>(
    py: Python<'py>,
    idx: PyReadonlyArray2<'py, i32>,
    cos: PyReadonlyArray2<'py, f64>,
    norms: PyReadonlyArray1<'py, f64>,
    weight: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let w = Weight::from_name(weight)
        .ok_or_else(|| value_err(format!("unknown edge weight {weight:?}")))?;
    let (n, k, idx) = neighbours(&idx)?;
    if cos.as_array().dim() != (n, k) || norms.as_array().len() != n {
        return Err(value_err("shapes of idx, cos and norms do not match"));
    }
    let cos = cos.as_slice().map_err(|_| value_err(NOT_CONTIGUOUS))?;
    let norms = norms.as_slice().map_err(|_| value_err(NOT_CONTIGUOUS))?;
    let out = py.detach(|| weight::reweight(n, k, idx, cos, norms, w));
    Ok(Array2::from_shape_vec((n, k), out)
        .map_err(value_err)?
        .into_pyarray(py))
}

/// The original build_graph's fuzzy-union symmetrization; returns the upper-triangular edge
/// list (rows, cols, weights).
#[pyfunction]
fn _fuzzy_union<'py>(
    py: Python<'py>,
    idx: PyReadonlyArray2<'py, i32>,
    w: PyReadonlyArray2<'py, f64>,
) -> PyResult<EdgeArrays<'py>> {
    let (n, k, idx) = neighbours(&idx)?;
    if w.as_array().dim() != (n, k) {
        return Err(value_err("shapes of idx and w do not match"));
    }
    let w = w.as_slice().map_err(|_| value_err(NOT_CONTIGUOUS))?;
    let e = py.detach(|| graph::fuzzy_union(n, k, idx, w));
    // u32 → i32 in place (indices < n ≤ i32::MAX)
    let to_i32 = |v: Vec<u32>| {
        Array1::from_vec(v.into_iter().map(|x| x as i32).collect::<Vec<i32>>()).into_pyarray(py)
    };
    Ok((
        to_i32(e.rows),
        to_i32(e.cols),
        Array1::from_vec(e.weights).into_pyarray(py),
    ))
}

/// L2 bytes each thread has to itself on this machine; None if unavailable (docs/06 §2.3).
#[pyfunction]
fn l2_bytes_per_thread() -> Option<usize> {
    cache::l2_bytes_per_thread()
}

/// Tile budget in bytes: half of the per-thread L2, or the fallback when unavailable.
#[pyfunction]
fn tile_budget_bytes() -> usize {
    cache::tile_budget_bytes()
}

/// Name of the default matmul backend on this platform.
#[pyfunction]
fn default_matmul() -> &'static str {
    Matmul::platform_default().name()
}

#[pymodule]
fn _costal(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
