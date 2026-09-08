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
    let mut slots = HashMap::new();
    for (key, value) in dict.iter() {
        let key = key.extract::<String>()?;
        let slot = if value.is_instance_of::<PyBool>() {
            CustomSlot::Bool(value.extract()?)
        } else if value.is_instance_of::<PyInt>() {
            CustomSlot::Int(value.extract()?)
        } else if value.is_instance_of::<PyFloat>() {
            CustomSlot::Float(value.extract()?)
        } else {
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
    let dict = PyDict::new(py);
    for (name, slot) in slots {
        match slot {
            CustomSlot::Bool(value) => dict.set_item(name, value)?,
            CustomSlot::Int(value) => dict.set_item(name, value)?,
            CustomSlot::Float(value) => dict.set_item(name, value)?,
            CustomSlot::Array { shape, values } => {
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
