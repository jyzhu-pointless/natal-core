//! Session-owned custom values and their Python round trip.

use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt};
use std::collections::HashMap;

/// A session-owned custom value with its declared scalar or array type.
#[derive(Clone, Debug, PartialEq)]
pub enum CustomSlot {
    Bool(bool),
    Int(i64),
    Float(f64),
    Array { shape: Vec<usize>, values: Vec<f64> },
}

/// Validate and copy every custom value before publishing the dictionary.
pub(crate) fn custom_slots_from_python(
    dict: &Bound<'_, PyAny>,
) -> PyResult<HashMap<String, CustomSlot>> {
    let dict = dict.downcast::<PyDict>()?;
    // Copy every Python value into owned Rust storage so the session never holds a
    // borrowed Python reference.
    let mut slots = HashMap::new();
    for (key, value) in dict.iter() {
        let key = key.extract::<String>()?;
        // Order matters: Python bool is a subclass of int, so test bool first or
        // True/False would be stored as 1/0.
        let slot = if value.is_instance_of::<PyBool>() {
            CustomSlot::Bool(value.extract()?)
        } else if value.is_instance_of::<PyInt>() {
            CustomSlot::Int(value.extract()?)
        } else if value.is_instance_of::<PyFloat>() {
            CustomSlot::Float(value.extract()?)
        } else {
            // Anything else must be a float64 array (the only supported container).
            let array = value.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
            CustomSlot::Array {
                shape: array.shape().to_vec(),
                values: array.as_array().iter().copied().collect(),
            }
        };
        slots.insert(key, slot);
    }
    Ok(slots)
}

/// Return isolated Python values, preserving custom scalar types and array shapes.
pub(crate) fn custom_slots_to_python<'py>(
    py: Python<'py>,
    slots: &HashMap<String, CustomSlot>,
) -> PyResult<Bound<'py, PyDict>> {
    // Rebuild fresh Python values so no caller aliases Rust-owned storage.
    let dict = PyDict::new(py);
    for (name, slot) in slots {
        match slot {
            CustomSlot::Bool(value) => dict.set_item(name, value)?,
            CustomSlot::Int(value) => dict.set_item(name, value)?,
            CustomSlot::Float(value) => dict.set_item(name, value)?,
            CustomSlot::Array { shape, values } => {
                // Rebuild the original ndarray shape, not a flat vector; a shape
                // mismatch here is a bug in the stored slot, reported as ValueError.
                let array = numpy::ndarray::ArrayD::from_shape_vec(
                    numpy::ndarray::IxDyn(shape),
                    values.clone(),
                )
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
                dict.set_item(name, numpy::PyArray::from_owned_array(py, array))?;
            }
        }
    }
    Ok(dict)
}

pub(crate) fn extract_custom_slots(
    obj: &Bound<'_, PyAny>,
) -> PyResult<HashMap<String, CustomSlot>> {
    custom_slots_from_python(&obj.getattr("custom_slots")?)
}
