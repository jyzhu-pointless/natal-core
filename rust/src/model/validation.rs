//! Shared domain checks for boundary data.
//!
//! Every check runs before any native mutation, so a rejected input never
//! leaves a half-updated contract behind.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::generated::ecology_parameters::ECO_PARAM_COLUMNS;

/// Validate externally supplied state values before an owner commits a boundary.
pub(crate) fn validate_state_values(ind: &[f64], sperm: &[f64], tick: i64) -> PyResult<()> {
    // Tick is an iteration counter; a negative value indicates a caller bug.
    if tick < 0 {
        return Err(PyValueError::new_err("tick must be nonnegative"));
    }
    // Individual and sperm-storage entries are counts: finite and nonnegative.
    if ind
        .iter()
        .chain(sperm)
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "state counts must be finite and nonnegative",
        ));
    }
    Ok(())
}

/// Check numerical domains before any native mutation.
pub(crate) fn validate_scalar_value(name: &str, value: f64) -> PyResult<()> {
    // Canonical ECO columns delegate to the generated per-id bound table.
    if let Some(id) = ECO_PARAM_COLUMNS.iter().position(|field| *field == name) {
        return crate::hooks::interpreter::validate_eco_param(id, value)
            .map_err(PyValueError::new_err);
    }
    // Field-specific domains: growth_mode is an integer enum 0..=4, while
    // external_expected_eggs uses -1 as the "unused" sentinel.
    let valid = value.is_finite()
        && match name {
            "growth_mode" => value.fract() == 0.0 && (0.0..=4.0).contains(&value),
            "external_expected_eggs" => value == -1.0 || value >= 0.0,
            _ => true,
        };
    if !valid {
        return Err(PyValueError::new_err(format!(
            "Invalid value for {name}: {value}"
        )));
    }
    Ok(())
}

/// Validate the intrinsic growth rate independently of the regulation mode.
pub(crate) fn validate_growth_contract(r: f64) -> PyResult<()> {
    if !r.is_finite() || r < 1.0 {
        return Err(PyValueError::new_err(
            "low_density_growth_rate must be finite and at least 1.0",
        ));
    }
    Ok(())
}

pub(crate) fn validate_tensor_values(name: &str, values: &[f64]) -> PyResult<()> {
    // Tensors hold counts/rates: reject NaN, infinity, and negative entries.
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(format!(
            "{name} requires finite nonnegative values"
        )));
    }
    Ok(())
}
