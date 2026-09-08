//! Numerical observation projection.
//!
//! `project` maps a D/S/A/Z population layout into explicit
//! group/deme/sex/age axes; `project_observation` exposes the same
//! reduction to Python for current-state snapshots.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Project D/S/A/Z input into group/deme/sex/age output with explicit axes.
pub fn project(
    ind: &[f64],
    mask: &[f64],
    dims: [usize; 4],
    selected: &[usize],
    collapse: bool,
    aggregate: bool,
) -> PyResult<Vec<f64>> {
    let [d, s, a, z] = dims;
    let plane = s * a * z;
    if plane == 0
        || ind.len() != d * plane
        || mask.len() % plane != 0
        || selected.is_empty()
        || selected.iter().any(|i| *i >= d)
    {
        return Err(PyValueError::new_err(
            "Observation dimensions or deme selection do not match the population layout",
        ));
    }
    let groups = mask.len() / plane;
    let out_d = if aggregate { 1 } else { selected.len() };
    let out_a = if collapse { 1 } else { a };
    let mut values = vec![0.0; groups * out_d * s * out_a];
    // Keep each reduction axis separate: genotype first, then age, then
    // deme. Recording and later projections therefore share the same
    // floating-point addition order as current-state observations.
    for group in 0..groups {
        for destination in 0..out_d {
            for sex in 0..s {
                for age_out in 0..out_a {
                    let mut result = 0.0;
                    let start_d = if aggregate { 0 } else { destination };
                    let end_d = if aggregate {
                        selected.len()
                    } else {
                        destination + 1
                    };
                    for &deme in &selected[start_d..end_d] {
                        let mut deme_total = 0.0;
                        let start_a = if collapse { 0 } else { age_out };
                        let end_a = if collapse { a } else { age_out + 1 };
                        for age in start_a..end_a {
                            let mut genotype_total = 0.0;
                            for genotype in 0..z {
                                let offset = (sex * a + age) * z + genotype;
                                genotype_total +=
                                    ind[deme * plane + offset] * mask[group * plane + offset];
                            }
                            deme_total += genotype_total;
                        }
                        result += deme_total;
                    }
                    values[((group * out_d + destination) * s + sex) * out_a + age_out] = result;
                }
            }
        }
    }
    Ok(values)
}

/// Apply the same native projection to a current-state snapshot.
#[pyfunction]
pub fn project_observation<'py>(
    py: Python<'py>,
    ind: PyReadonlyArray1<'py, f64>,
    mask: PyReadonlyArray1<'py, f64>,
    dimensions: [usize; 4],
    selected: Vec<usize>,
    collapse_age: bool,
    aggregate: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(PyArray1::from_vec(
        py,
        project(
            ind.as_slice()?,
            mask.as_slice()?,
            dimensions,
            &selected,
            collapse_age,
            aggregate,
        )?,
    ))
}
