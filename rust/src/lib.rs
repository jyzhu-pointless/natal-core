//! Rust backend prototype for NATAL Core.
//!
//! This crate currently exposes the aging lifecycle kernel through PyO3 as a
//! proof of concept for the backend boundary: NumPy arrays cross the FFI edge
//! as zero-copy views and are mutated in place by Rust.

mod config;
mod hooks;
mod lifecycle;
mod rng;
mod session;

use numpy::{PyReadwriteArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::session::EngineSession;

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

#[pymodule]
fn _engine_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(age_structured_aging, module)?)?;
    module.add_function(wrap_pyfunction!(discrete_aging, module)?)?;
    module.add_class::<EngineSession>()?;
    Ok(())
}
