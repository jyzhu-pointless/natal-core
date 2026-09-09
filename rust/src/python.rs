//! Standalone Python function adapters.
//!
//! These are the module-level functions the extension exports: CSR migration
//! and equilibrium metric helpers. Session objects live in `sessions`.

use numpy::{
    PyArray4, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray4,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::model::blueprint::Blueprint;
use crate::model::ecology::EcologyParams;

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
pub(crate) fn migrate_csr_deterministic<'py>(
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

    let (out_ind, out_sperm) = crate::kernels::spatial::migrate_csr_deterministic(
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
pub(crate) fn migrate_csr_stochastic<'py>(
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

    let (out_ind, out_sperm) = crate::kernels::spatial::migrate_csr_stochastic(
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
pub(crate) fn equilibrium_metrics(
    blueprint: &Bound<'_, PyAny>,
    params: &Bound<'_, PyAny>,
) -> PyResult<(f64, f64)> {
    let bp = Blueprint::from_python(blueprint)?;
    // The Python parity helper evaluates the panmictic (single-column)
    // contract; length-1 columns reproduce the pre-columnization scalars.
    let pr = EcologyParams::from_python(params, 1)?;
    Ok(crate::kernels::equilibrium::equilibrium_metrics(
        &bp, &pr, 0,
    ))
}

/// Flat-signature equilibrium metrics for the Python sync channel.
///
/// Mirrors ``compute_equilibrium_metrics`` in the Python reference
/// (same statement order, bit-identical results) without materializing
/// a contract pair: the sensitive-parameter sync path runs on every
/// committed ecology write, so it must stay an O(n_ages) call.
#[pyfunction]
#[pyo3(signature = (
    carrying_capacity,
    eggs_per_female,
    sex_ratio,
    survival_rates,
    reproduction_rates,
    fertility,
    competition_weights,
    new_adult_age,
    n_ages,
    declared_distribution=None,
    external_expected_eggs=None,
))]
// The flat parameter table mirrors the Python reference signature
// one-to-one (plan 5.2 parity); a params struct would decouple the
// two spellings the single-source rule keeps aligned.
#[allow(clippy::too_many_arguments)]
pub(crate) fn equilibrium_metrics_flat(
    carrying_capacity: f64,
    eggs_per_female: f64,
    sex_ratio: f64,
    survival_rates: PyReadonlyArray2<'_, f64>,
    reproduction_rates: PyReadonlyArray1<'_, f64>,
    fertility: PyReadonlyArray1<'_, f64>,
    competition_weights: PyReadonlyArray1<'_, f64>,
    new_adult_age: usize,
    n_ages: usize,
    declared_distribution: Option<PyReadonlyArray2<'_, f64>>,
    external_expected_eggs: Option<f64>,
) -> PyResult<(f64, f64)> {
    let s_shape = survival_rates.shape();
    if s_shape != [2, n_ages] {
        return Err(PyValueError::new_err(format!(
            "survival_rates shape must be (2, {n_ages}), got {s_shape:?}"
        )));
    }
    for (name, arr, len) in [
        ("reproduction_rates", reproduction_rates.shape()[0], n_ages),
        ("fertility", fertility.shape()[0], n_ages),
        (
            "competition_weights",
            competition_weights.shape()[0],
            n_ages,
        ),
    ] {
        if arr != len {
            return Err(PyValueError::new_err(format!(
                "{name} must have length {len}, got {arr}"
            )));
        }
    }
    let s_view = survival_rates.as_array();
    let survival = s_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("survival_rates must be C-contiguous"))?;
    let r_view = reproduction_rates.as_array();
    let reproduce = r_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("reproduction_rates must be C-contiguous"))?;
    let f_view = fertility.as_array();
    let fert = f_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fertility must be C-contiguous"))?;
    let c_view = competition_weights.as_array();
    let comp = c_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("competition_weights must be C-contiguous"))?;
    let declared_storage: Vec<f64>;
    let declared: &[f64] = match declared_distribution {
        // The draft's derivation-mode sentinel is an empty (0, 0) array,
        // which lands here exactly like an explicit None.
        Some(arr) if arr.len() > 0 => {
            let d_shape = arr.shape();
            if d_shape != [2, n_ages] {
                return Err(PyValueError::new_err(format!(
                    "declared_distribution shape must be (2, {n_ages}), got {d_shape:?}"
                )));
            }
            let d_view = arr.as_array();
            let slice = d_view.as_slice().ok_or_else(|| {
                PyValueError::new_err("declared_distribution must be C-contiguous")
            })?;
            declared_storage = slice.to_vec();
            &declared_storage
        }
        _ => &[],
    };
    // The materialization convention stores "unused" as a negative value.
    let external = external_expected_eggs.unwrap_or(-1.0);
    Ok(crate::kernels::equilibrium::equilibrium_metrics_core(
        carrying_capacity,
        eggs_per_female,
        sex_ratio,
        survival,
        reproduce,
        fert,
        comp,
        declared,
        external,
        new_adult_age,
        n_ages,
    ))
}
