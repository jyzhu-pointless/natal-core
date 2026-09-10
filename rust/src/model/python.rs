//! Python object extraction helpers shared by the model types.
//!
//! Each helper names the field it reads, so a type error or a missing
//! attribute is reported against the contract field, not the raw Python
//! object.

use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;

/// Extract an int scalar, accepting Python ints and 0-d NumPy arrays.
pub(crate) fn extract_i64(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<i64> {
    let value = obj.getattr(name)?;
    if let Ok(scalar) = value.extract::<i64>() {
        return Ok(scalar);
    }
    value.call_method0("item")?.extract::<i64>()
}

/// Extract a float scalar, accepting Python floats and 0-d NumPy arrays.
pub(crate) fn extract_f64(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<f64> {
    let value = obj.getattr(name)?;
    if let Ok(scalar) = value.extract::<f64>() {
        return Ok(scalar);
    }
    value.call_method0("item")?.extract::<f64>()
}

/// Extract a bool scalar.
pub(crate) fn extract_bool(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    obj.getattr(name)?.extract::<bool>()
}

/// Extract a flat ``f64`` copy of any float64 NumPy array attribute.
///
/// Dimension-agnostic: blueprint/params arrays range from 1-D vectors to
/// the 3-D initial population; all are copied in row-major order.
pub(crate) fn extract_f64_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    let array = obj
        .getattr(name)?
        .extract::<PyReadonlyArrayDyn<'_, f64>>()?;
    Ok(array.as_slice()?.to_vec())
}

/// Extract a flat ``i64`` copy of an int64 NumPy array attribute.
pub(crate) fn extract_i64_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<i64>> {
    let array = obj
        .getattr(name)?
        .extract::<PyReadonlyArrayDyn<'_, i64>>()?;
    Ok(array.as_slice()?.to_vec())
}

/// Extract a flat ``bool`` copy of a bool NumPy array attribute.
pub(crate) fn extract_bool_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<bool>> {
    let array = obj
        .getattr(name)?
        .extract::<PyReadonlyArrayDyn<'_, bool>>()?;
    Ok(array.as_slice()?.to_vec())
}

/// Extract a tuple/sequence of strings attribute.
pub(crate) fn extract_string_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<String>> {
    obj.getattr(name)?.extract::<Vec<String>>()
}
