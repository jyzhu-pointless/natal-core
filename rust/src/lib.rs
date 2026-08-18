//! Rust backend for NATAL Core.
//!
//! The crate exposes PyO3 sessions for age-structured, discrete-generation /
//! Wright-Fisher, and spatial multi-deme simulations.  NumPy arrays cross the
//! FFI edge as zero-copy views and are mutated in place by Rust; configuration
//! is snapshotted into plain Rust structs and CSR declarative hooks are
//! interpreted directly in the kernels.

mod config;
mod discrete;
mod discrete_session;
mod hooks;
mod lifecycle;
mod rng;
mod session;
mod spatial;
mod spatial_session;

use numpy::{
    PyArray4, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray4,
    PyReadwriteArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::discrete_session::DiscreteEngineSession;
use crate::session::EngineSession;
use crate::spatial_session::{HeterogeneousSpatialEngineSession, SpatialEngineSession};

/// Row-major flat index for ``(sex, age, ztype)``.
#[inline]
fn individual_index(sex: usize, age: usize, ztype: usize, n_ages: usize, n_ztypes: usize) -> usize {
    (sex * n_ages + age) * n_ztypes + ztype
}

/// Row-major flat index for ``(age, female_ztype, male_ztype)``.
#[inline]
fn sperm_index(age: usize, female_ztype: usize, male_ztype: usize, n_ztypes: usize) -> usize {
    (age * n_ztypes + female_ztype) * n_ztypes + male_ztype
}

/// Advance one age class for an age-structured deme.
///
/// Mirrors ``natal.engine.age_structured_simulator.run_aging``: every age
/// class shifts down one slot (oldest is dropped), then age 0 is zeroed for
/// both individual counts and stored sperm.
///
/// ## Parameters
/// - `individual_count`: Mutable array of shape `(2, n_ages, n_ztypes)`.
/// - `sperm_storage`: Mutable array of shape `(n_ages, n_ztypes, n_ztypes)`.
///
/// ## Returns
/// `Ok(())` after mutating the arrays in place.
///
/// ## Errors
/// Returns `PyValueError` if the sperm-storage shape is inconsistent.
#[pyfunction]
fn age_structured_aging(
    mut individual_count: PyReadwriteArray3<'_, f64>,
    mut sperm_storage: PyReadwriteArray3<'_, f64>,
) -> PyResult<()> {
    let ind_shape = individual_count.shape();
    let (n_sexes, n_ages, n_ztypes) = (ind_shape[0], ind_shape[1], ind_shape[2]);
    let sperm_shape = sperm_storage.shape();
    if sperm_shape != [n_ages, n_ztypes, n_ztypes] {
        return Err(PyValueError::new_err(format!(
            "sperm_storage shape must be ({n_ages}, {n_ztypes}, {n_ztypes}), got {sperm_shape:?}"
        )));
    }

    let ind = individual_count
        .as_slice_mut()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let sperm = sperm_storage
        .as_slice_mut()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    // Walk downward so age-1 slots are read before being overwritten.
    for age in (1..n_ages).rev() {
        let older = age - 1;
        for sex in 0..n_sexes {
            for ztype in 0..n_ztypes {
                ind[individual_index(sex, age, ztype, n_ages, n_ztypes)] =
                    ind[individual_index(sex, older, ztype, n_ages, n_ztypes)];
            }
        }
        for female_ztype in 0..n_ztypes {
            for male_ztype in 0..n_ztypes {
                sperm[sperm_index(age, female_ztype, male_ztype, n_ztypes)] =
                    sperm[sperm_index(older, female_ztype, male_ztype, n_ztypes)];
            }
        }
    }

    for sex in 0..n_sexes {
        for ztype in 0..n_ztypes {
            ind[individual_index(sex, 0, ztype, n_ages, n_ztypes)] = 0.0;
        }
    }
    for female_ztype in 0..n_ztypes {
        for male_ztype in 0..n_ztypes {
            sperm[sperm_index(0, female_ztype, male_ztype, n_ztypes)] = 0.0;
        }
    }

    Ok(())
}

/// Advance one age class for a discrete-generation deme.
///
/// Mirrors ``natal.engine.discrete_generation_simulator.run_discrete_aging``:
/// age-0 juveniles become age-1 adults for the two sex classes, then age 0 is
/// zeroed.
///
/// ## Parameters
/// - `individual_count`: Mutable array of shape `(n_sexes, n_ages, n_ztypes)`.
///
/// ## Returns
/// `Ok(())` after mutating the array in place.
///
/// ## Errors
/// Returns `PyValueError` if the array has fewer than two sexes or ages.
#[pyfunction]
fn discrete_aging(mut individual_count: PyReadwriteArray3<'_, f64>) -> PyResult<()> {
    let ind_shape = individual_count.shape();
    let (n_sexes, n_ages, n_ztypes) = (ind_shape[0], ind_shape[1], ind_shape[2]);
    if n_sexes < 2 || n_ages < 2 {
        return Err(PyValueError::new_err(format!(
            "individual_count shape must have n_sexes >= 2 and n_ages >= 2, got {individual_count:?}"
        )));
    }

    let ind = individual_count
        .as_slice_mut()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    for sex in 0..2 {
        for ztype in 0..n_ztypes {
            ind[individual_index(sex, 1, ztype, n_ages, n_ztypes)] =
                ind[individual_index(sex, 0, ztype, n_ages, n_ztypes)];
            ind[individual_index(sex, 0, ztype, n_ages, n_ztypes)] = 0.0;
        }
    }

    Ok(())
}

/// Deterministic adjacency migration exposed to Python.
///
/// ## Parameters
/// - `individual_count_all`: Stacked state array.
/// - `sperm_storage_all`: Stacked sperm array.
/// - `adjacency`: Dense adjacency matrix.
/// - `rate`: Per-age or global migration rate.
///
/// ## Returns
/// New stacked ``(individual_count, sperm_storage)`` arrays.
#[allow(clippy::type_complexity)] // PyO3 boundary returns two 4-D NumPy arrays.
#[pyfunction]
fn migrate_adjacency_deterministic<'py>(
    py: Python<'py>,
    individual_count_all: PyReadonlyArray4<'py, f64>,
    sperm_storage_all: PyReadonlyArray4<'py, f64>,
    adjacency: PyReadonlyArray2<'py, f64>,
    rate: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray4<f64>>, Bound<'py, PyArray4<f64>>)> {
    let ind_shape = individual_count_all.shape();
    if ind_shape.len() != 4 || ind_shape[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "individual_count_all must have shape (n_demes, 2, n_ages, n_ztypes), got {ind_shape:?}"
        )));
    }
    let n_demes = ind_shape[0];
    let n_ages = ind_shape[2];
    let n_ztypes = ind_shape[3];
    let sperm_shape = sperm_storage_all.shape();
    if sperm_shape != [n_demes, n_ages, n_ztypes, n_ztypes] {
        return Err(PyValueError::new_err(format!(
            "sperm_storage_all must have shape ({n_demes}, {n_ages}, {n_ztypes}, {n_ztypes}), got {sperm_shape:?}"
        )));
    }
    let adj_shape = adjacency.shape();
    if adj_shape != [n_demes, n_demes] {
        return Err(PyValueError::new_err(format!(
            "adjacency must have shape ({n_demes}, {n_demes}), got {adj_shape:?}"
        )));
    }
    let ind_in = individual_count_all
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let sperm_in = sperm_storage_all
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let adjacency_in = adjacency
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let rate_in = rate
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let (out_ind, out_sperm) = crate::spatial::migrate_adjacency_deterministic(
        ind_in,
        sperm_in,
        adjacency_in,
        rate_in,
        n_demes,
        n_ages,
        n_ztypes,
    )
    .map_err(PyRuntimeError::new_err)?;

    let ind_out = PyArray4::<f64>::zeros(py, [n_demes, 2, n_ages, n_ztypes], false);
    ind_out
        .readwrite()
        .as_slice_mut()
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .copy_from_slice(&out_ind);
    let sperm_out = PyArray4::<f64>::zeros(py, [n_demes, n_ages, n_ztypes, n_ztypes], false);
    sperm_out
        .readwrite()
        .as_slice_mut()
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .copy_from_slice(&out_sperm);
    Ok((ind_out, sperm_out))
}

/// Stochastic adjacency migration exposed to Python.
///
/// ## Parameters
/// - `individual_count_all`: Stacked state array.
/// - `sperm_storage_all`: Stacked sperm array.
/// - `adjacency`: Dense adjacency matrix.
/// - `rate`: Per-age or global migration rate.
/// - `seed`: RNG seed.
/// - `continuous_sampling`: Use continuous sampling.
///
/// ## Returns
/// New stacked arrays.
#[allow(clippy::type_complexity)] // PyO3 boundary returns two 4-D NumPy arrays.
#[allow(clippy::too_many_arguments)] // Signature mirrors the Python migration API.
#[pyfunction]
fn migrate_adjacency_stochastic<'py>(
    py: Python<'py>,
    individual_count_all: PyReadonlyArray4<'py, f64>,
    sperm_storage_all: PyReadonlyArray4<'py, f64>,
    adjacency: PyReadonlyArray2<'py, f64>,
    rate: PyReadonlyArray1<'py, f64>,
    seed: u64,
    continuous_sampling: bool,
) -> PyResult<(Bound<'py, PyArray4<f64>>, Bound<'py, PyArray4<f64>>)> {
    let ind_shape = individual_count_all.shape();
    if ind_shape.len() != 4 || ind_shape[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "individual_count_all must have shape (n_demes, 2, n_ages, n_ztypes), got {ind_shape:?}"
        )));
    }
    let n_demes = ind_shape[0];
    let n_ages = ind_shape[2];
    let n_ztypes = ind_shape[3];
    let sperm_shape = sperm_storage_all.shape();
    if sperm_shape != [n_demes, n_ages, n_ztypes, n_ztypes] {
        return Err(PyValueError::new_err(format!(
            "sperm_storage_all must have shape ({n_demes}, {n_ages}, {n_ztypes}, {n_ztypes}), got {sperm_shape:?}"
        )));
    }
    let adj_shape = adjacency.shape();
    if adj_shape != [n_demes, n_demes] {
        return Err(PyValueError::new_err(format!(
            "adjacency must have shape ({n_demes}, {n_demes}), got {adj_shape:?}"
        )));
    }
    let ind_in = individual_count_all
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let sperm_in = sperm_storage_all
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let adjacency_in = adjacency
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let rate_in = rate
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let (out_ind, out_sperm) = crate::spatial::migrate_adjacency_stochastic(
        ind_in,
        sperm_in,
        adjacency_in,
        rate_in,
        seed,
        continuous_sampling,
        n_demes,
        n_ages,
        n_ztypes,
    )
    .map_err(PyRuntimeError::new_err)?;

    let ind_out = PyArray4::<f64>::zeros(py, [n_demes, 2, n_ages, n_ztypes], false);
    ind_out
        .readwrite()
        .as_slice_mut()
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .copy_from_slice(&out_ind);
    let sperm_out = PyArray4::<f64>::zeros(py, [n_demes, n_ages, n_ztypes, n_ztypes], false);
    sperm_out
        .readwrite()
        .as_slice_mut()
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .copy_from_slice(&out_sperm);
    Ok((ind_out, sperm_out))
}

/// Deterministic kernel migration exposed to Python.
///
/// ## Parameters
/// - `individual_count_all`: Stacked state array.
/// - `sperm_storage_all`: Stacked sperm array.
/// - `migration_kernel`: Topology migration kernel.
/// - `topology_wrap`: Whether topology wraps.
/// - `kernel_include_center`: Whether the center cell is included.
/// - `rate`: Per-age or global migration rate.
///
/// ## Returns
/// New stacked arrays.
#[allow(clippy::type_complexity)] // PyO3 boundary returns two 4-D NumPy arrays.
#[allow(clippy::too_many_arguments)] // Signature mirrors the Python kernel migration API.
#[pyfunction]
fn migrate_kernel_deterministic<'py>(
    py: Python<'py>,
    individual_count_all: PyReadonlyArray4<'py, f64>,
    sperm_storage_all: PyReadonlyArray4<'py, f64>,
    migration_kernel: PyReadonlyArray2<'py, f64>,
    topology_wrap: bool,
    kernel_include_center: bool,
    rate: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray4<f64>>, Bound<'py, PyArray4<f64>>)> {
    let ind_shape = individual_count_all.shape();
    if ind_shape.len() != 4 || ind_shape[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "individual_count_all must have shape (n_demes, 2, n_ages, n_ztypes), got {ind_shape:?}"
        )));
    }
    let n_demes = ind_shape[0];
    let n_ages = ind_shape[2];
    let n_ztypes = ind_shape[3];
    let kernel_shape = migration_kernel.shape();
    let topology_rows = kernel_shape[0];
    let topology_cols = kernel_shape[1];
    if topology_rows * topology_cols != n_demes {
        return Err(PyValueError::new_err(format!(
            "migration_kernel topology {topology_rows}x{topology_cols} does not match {n_demes} demes"
        )));
    }
    let sperm_shape = sperm_storage_all.shape();
    if sperm_shape != [n_demes, n_ages, n_ztypes, n_ztypes] {
        return Err(PyValueError::new_err(format!(
            "sperm_storage_all must have shape ({n_demes}, {n_ages}, {n_ztypes}, {n_ztypes}), got {sperm_shape:?}"
        )));
    }
    let ind_in = individual_count_all
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let sperm_in = sperm_storage_all
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let kernel_in = migration_kernel
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let rate_in = rate
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let (out_ind, out_sperm) = crate::spatial::migrate_kernel_deterministic(
        ind_in,
        sperm_in,
        kernel_in,
        topology_rows,
        topology_cols,
        topology_wrap,
        kernel_include_center,
        rate_in,
        n_ages,
        n_ztypes,
    )
    .map_err(PyRuntimeError::new_err)?;
    let ind_out = PyArray4::<f64>::zeros(py, [n_demes, 2, n_ages, n_ztypes], false);
    ind_out
        .readwrite()
        .as_slice_mut()
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .copy_from_slice(&out_ind);
    let sperm_out = PyArray4::<f64>::zeros(py, [n_demes, n_ages, n_ztypes, n_ztypes], false);
    sperm_out
        .readwrite()
        .as_slice_mut()
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .copy_from_slice(&out_sperm);
    Ok((ind_out, sperm_out))
}

/// Stochastic kernel migration exposed to Python.
///
/// ## Parameters
/// - `individual_count_all`: Stacked state array.
/// - `sperm_storage_all`: Stacked sperm array.
/// - `migration_kernel`: Topology migration kernel.
/// - `topology_wrap`: Whether topology wraps.
/// - `kernel_include_center`: Whether the center cell is included.
/// - `rate`: Per-age or global migration rate.
/// - `seed`: RNG seed.
/// - `continuous_sampling`: Use continuous sampling.
///
/// ## Returns
/// New stacked arrays.
#[allow(clippy::type_complexity)] // PyO3 boundary returns two 4-D NumPy arrays.
#[allow(clippy::too_many_arguments)] // Signature mirrors the Python kernel migration API.
#[pyfunction]
fn migrate_kernel_stochastic<'py>(
    py: Python<'py>,
    individual_count_all: PyReadonlyArray4<'py, f64>,
    sperm_storage_all: PyReadonlyArray4<'py, f64>,
    migration_kernel: PyReadonlyArray2<'py, f64>,
    topology_wrap: bool,
    kernel_include_center: bool,
    rate: PyReadonlyArray1<'py, f64>,
    seed: u64,
    continuous_sampling: bool,
) -> PyResult<(Bound<'py, PyArray4<f64>>, Bound<'py, PyArray4<f64>>)> {
    let ind_shape = individual_count_all.shape();
    if ind_shape.len() != 4 || ind_shape[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "individual_count_all must have shape (n_demes, 2, n_ages, n_ztypes), got {ind_shape:?}"
        )));
    }
    let n_demes = ind_shape[0];
    let n_ages = ind_shape[2];
    let n_ztypes = ind_shape[3];
    let kernel_shape = migration_kernel.shape();
    let topology_rows = kernel_shape[0];
    let topology_cols = kernel_shape[1];
    if topology_rows * topology_cols != n_demes {
        return Err(PyValueError::new_err(format!(
            "migration_kernel topology {topology_rows}x{topology_cols} does not match {n_demes} demes"
        )));
    }
    let sperm_shape = sperm_storage_all.shape();
    if sperm_shape != [n_demes, n_ages, n_ztypes, n_ztypes] {
        return Err(PyValueError::new_err(format!(
            "sperm_storage_all must have shape ({n_demes}, {n_ages}, {n_ztypes}, {n_ztypes}), got {sperm_shape:?}"
        )));
    }
    let ind_in = individual_count_all
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let sperm_in = sperm_storage_all
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let kernel_in = migration_kernel
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let rate_in = rate
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let (out_ind, out_sperm) = crate::spatial::migrate_kernel_stochastic(
        ind_in,
        sperm_in,
        kernel_in,
        topology_rows,
        topology_cols,
        topology_wrap,
        kernel_include_center,
        rate_in,
        seed,
        continuous_sampling,
        n_ages,
        n_ztypes,
    )
    .map_err(PyRuntimeError::new_err)?;
    let ind_out = PyArray4::<f64>::zeros(py, [n_demes, 2, n_ages, n_ztypes], false);
    ind_out
        .readwrite()
        .as_slice_mut()
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .copy_from_slice(&out_ind);
    let sperm_out = PyArray4::<f64>::zeros(py, [n_demes, n_ages, n_ztypes, n_ztypes], false);
    sperm_out
        .readwrite()
        .as_slice_mut()
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .copy_from_slice(&out_sperm);
    Ok((ind_out, sperm_out))
}

/// PyO3 module entry point registering all Rust backend classes and functions.
///
/// ## Parameters
/// - `module`: PyO3 module being initialized.
///
/// ## Returns
/// ``Ok(())`` after registering all classes and functions.
#[pymodule]
fn _engine_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(age_structured_aging, module)?)?;
    module.add_function(wrap_pyfunction!(discrete_aging, module)?)?;
    module.add_class::<EngineSession>()?;
    module.add_class::<DiscreteEngineSession>()?;
    module.add_class::<SpatialEngineSession>()?;
    module.add_class::<HeterogeneousSpatialEngineSession>()?;
    module.add_function(wrap_pyfunction!(migrate_adjacency_deterministic, module)?)?;
    module.add_function(wrap_pyfunction!(migrate_adjacency_stochastic, module)?)?;
    module.add_function(wrap_pyfunction!(migrate_kernel_deterministic, module)?)?;
    module.add_function(wrap_pyfunction!(migrate_kernel_stochastic, module)?)?;
    Ok(())
}
