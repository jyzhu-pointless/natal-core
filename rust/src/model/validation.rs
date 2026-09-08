//! Shared domain checks for boundary data.
//!
//! Every check runs before any native mutation, so a rejected input never
//! leaves a half-updated contract behind.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::generated::ecology_parameters::ECO_PARAM_COLUMNS;

/// Validate externally supplied state values before an owner commits a boundary.
pub(crate) fn validate_state_values(ind: &[f64], sperm: &[f64], tick: i64) -> PyResult<()> {
    if tick < 0 {
        return Err(PyValueError::new_err("tick must be nonnegative"));
    }
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
    if let Some(id) = ECO_PARAM_COLUMNS.iter().position(|field| *field == name) {
        return crate::hooks::interpreter::validate_eco_param(id, value)
            .map_err(PyValueError::new_err);
    }
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

pub(crate) fn validate_tensor_values(name: &str, values: &[f64]) -> PyResult<()> {
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
