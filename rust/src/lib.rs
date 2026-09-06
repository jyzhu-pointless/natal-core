//! Rust backend for NATAL Core.
//!
//! The crate exposes PyO3 sessions for age-structured, discrete-generation /
//! Wright-Fisher, and spatial multi-deme simulations.  NumPy arrays cross the
//! FFI edge as zero-copy views and are mutated in place by Rust; configuration
//! is snapshotted into plain Rust structs and CSR declarative hooks are
//! interpreted directly in the kernels.

mod config;
mod contract;
pub mod curves;
mod discrete;
mod discrete_session;
/// Generated wire tables (parameters.jsonc is the single source; see
/// scripts/generate_param_tables.py — do not hand-edit).
mod eco_param_wire;
mod equilibrate;
mod hooks;
mod lifecycle;
mod offspring;
mod rng;
mod session;
mod spatial;
mod spatial_session;

use numpy::{
    PyArray4, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray4, PyReadwriteArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::contract::{Blueprint, Params};
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

/// Deterministic CSR migration exposed to Python.
///
/// ## Parameters
/// - `individual_count_all`: Stacked state array.
/// - `sperm_storage_all`: Stacked sperm array.
/// - `indptr`: CSR migration row pointer (``n_demes + 1``).
/// - `dest_idx`: CSR destination index per entry.
/// - `weights`: CSR normalized outbound weight per entry.
/// - `rate`: ``(n_demes, 2, n_ages)`` migration-rate column (flat).
///
/// ## Returns
/// New stacked ``(individual_count, sperm_storage)`` arrays.
#[allow(clippy::type_complexity)] // PyO3 boundary returns two 4-D NumPy arrays.
#[allow(clippy::too_many_arguments)] // Signature mirrors the Python migration API.
#[pyfunction]
fn migrate_csr_deterministic<'py>(
    py: Python<'py>,
    individual_count_all: PyReadonlyArray4<'py, f64>,
    sperm_storage_all: PyReadonlyArray4<'py, f64>,
    indptr: PyReadonlyArray1<'py, i64>,
    dest_idx: PyReadonlyArray1<'py, i64>,
    weights: PyReadonlyArray1<'py, f64>,
    rate: PyReadonlyArray1<'py, f64>,
    stay_after: bool,
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
    let ind_in = individual_count_all
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let sperm_in = sperm_storage_all
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let indptr_in = indptr
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let dest_in = dest_idx
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let weights_in = weights
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let rate_in = rate
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let (out_ind, out_sperm) = crate::spatial::migrate_csr_deterministic(
        ind_in, sperm_in, indptr_in, dest_in, weights_in, rate_in, stay_after, n_demes, n_ages,
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

/// Stochastic CSR migration exposed to Python.
///
/// ## Parameters
/// - `individual_count_all`: Stacked state array.
/// - `sperm_storage_all`: Stacked sperm array.
/// - `indptr`: CSR migration row pointer (``n_demes + 1``).
/// - `dest_idx`: CSR destination index per entry.
/// - `weights`: CSR normalized outbound weight per entry.
/// - `rate`: ``(n_demes, 2, n_ages)`` migration-rate column (flat).
/// - `seed`: RNG seed.
/// - `continuous_sampling`: Use continuous sampling.
///
/// ## Returns
/// New stacked arrays.
#[allow(clippy::type_complexity)] // PyO3 boundary returns two 4-D NumPy arrays.
#[allow(clippy::too_many_arguments)] // Signature mirrors the Python migration API.
#[pyfunction]
fn migrate_csr_stochastic<'py>(
    py: Python<'py>,
    individual_count_all: PyReadonlyArray4<'py, f64>,
    sperm_storage_all: PyReadonlyArray4<'py, f64>,
    indptr: PyReadonlyArray1<'py, i64>,
    dest_idx: PyReadonlyArray1<'py, i64>,
    weights: PyReadonlyArray1<'py, f64>,
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
    let ind_in = individual_count_all
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let sperm_in = sperm_storage_all
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let indptr_in = indptr
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let dest_in = dest_idx
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let weights_in = weights
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let rate_in = rate
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let (out_ind, out_sperm) = crate::spatial::migrate_csr_stochastic(
        ind_in,
        sperm_in,
        indptr_in,
        dest_in,
        weights_in,
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

/// Python-exposed equilibrium-metric calibration (parity with the
/// Python reference ``compute_equilibrium_metrics``).
///
/// ## Parameters
/// - `blueprint`: The frozen blueprint (dimensions).
/// - `params`: Current runtime parameters.
///
/// ## Returns
/// ``(expected_competition_strength, expected_survival_rate)``.
#[pyfunction]
fn equilibrium_metrics(
    blueprint: &Bound<'_, PyAny>,
    params: &Bound<'_, PyAny>,
) -> PyResult<(f64, f64)> {
    let bp = Blueprint::from_python(blueprint)?;
    // The Python parity helper evaluates the panmictic (single-column)
    // contract; length-1 columns reproduce the pre-columnization scalars.
    let pr = Params::from_python(params, 1)?;
    Ok(crate::equilibrate::equilibrium_metrics(&bp, &pr, 0))
}

#[pymodule]
fn _engine_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    use offspring::compute_offspring_tensor;

    module.add_function(wrap_pyfunction!(age_structured_aging, module)?)?;
    module.add_function(wrap_pyfunction!(discrete_aging, module)?)?;
    module.add_class::<EngineSession>()?;
    module.add_class::<DiscreteEngineSession>()?;
    module.add_class::<SpatialEngineSession>()?;
    module.add_class::<HeterogeneousSpatialEngineSession>()?;
    module.add_function(wrap_pyfunction!(equilibrium_metrics, module)?)?;
    module.add_function(wrap_pyfunction!(migrate_csr_deterministic, module)?)?;
    module.add_function(wrap_pyfunction!(migrate_csr_stochastic, module)?)?;
    module.add_function(wrap_pyfunction!(compute_offspring_tensor, module)?)?;
    Ok(())
}
