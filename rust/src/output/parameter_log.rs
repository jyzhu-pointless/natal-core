//! Parameter-change audit log and its Python value conversion.

use numpy::PyUntypedArrayMethods;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt};
use std::sync::{Arc, Mutex};

use crate::model::custom_fields::CustomSlot;

/// Public log surface retains its existing four-column shape.
pub type LogRow = (i64, String, f64, f64);

/// Shared log ownership lets the session capture exact commit positions.
pub type SharedLog = Arc<Mutex<Vec<LogEntry>>>;

/// A successful commit with provenance retained alongside the legacy projection.
#[derive(Clone)]
pub struct LogEntry {
    pub row: LogRow,
    pub event: String,
    pub deme: usize,
    pub values: Option<(Option<CustomSlot>, Option<CustomSlot>)>,
}

impl LogEntry {
    /// Translate the execution phase cursor into its named hook event.
    pub fn from_phase(row: LogRow, phase: usize, deme: usize) -> Self {
        let event = match phase {
            0 => "first",
            2 => "early",
            4 => "late",
            6 => "finish",
            _ => "update",
        };
        Self {
            row,
            event: event.to_owned(),
            deme,
            values: None,
        }
    }
}

/// Native append-only parameter timeline, truncated by checkpoint cursors.
#[pyclass]
#[derive(Clone, Default)]
pub struct ParameterLog {
    pub(crate) rows: SharedLog,
}

#[pymethods]
impl ParameterLog {
    #[new]
    fn new() -> Self {
        Self::default()
    }
    /// Append an actual successful parameter change.
    fn append(&self, row: LogRow) {
        if row.2 != row.3 {
            self.rows.lock().unwrap().push(LogEntry {
                row,
                event: "update".to_owned(),
                deme: 0,
                values: None,
            });
        }
    }
    /// Mark a callback transaction without copying the accumulated log.
    fn mark(&self) -> usize {
        self.rows.lock().unwrap().len()
    }
    /// Roll back only the entries added after a callback transaction mark.
    fn rollback(&self, position: usize) -> PyResult<()> {
        let mut rows = self.rows.lock().unwrap();
        if position > rows.len() {
            return Err(PyValueError::new_err(
                "Invalid parameter log transaction mark",
            ));
        }
        rows.truncate(position);
        Ok(())
    }
    /// Copy the current valid timeline.
    fn snapshot(&self) -> Vec<LogRow> {
        self.rows
            .lock()
            .unwrap()
            .iter()
            .filter(|entry| {
                entry.values.as_ref().is_none_or(|(old, new)| {
                    audit_scalar(old).is_some() && audit_scalar(new).is_some()
                })
            })
            .map(|entry| entry.row.clone())
            .collect()
    }
    /// Commit a change with the responsible event and spatial deme.
    fn append_detail(&self, row: LogRow, event: String, deme: usize) {
        if row.2 != row.3 {
            self.rows.lock().unwrap().push(LogEntry {
                row,
                event,
                deme,
                values: None,
            });
        }
    }
    /// Append typed scalar, tensor, or custom commits after native validation.
    #[allow(clippy::too_many_arguments)] // One atomic audit entry with explicit provenance and values.
    fn append_value(
        &self,
        tick: i64,
        name: String,
        old: &Bound<'_, PyAny>,
        new: &Bound<'_, PyAny>,
        event: String,
        deme: usize,
    ) -> PyResult<()> {
        let old = audit_value(old)?;
        let new = audit_value(new)?;
        if old == new {
            return Ok(());
        }
        let row = (
            tick,
            name,
            audit_scalar(&old).unwrap_or(0.0),
            audit_scalar(&new).unwrap_or(0.0),
        );
        self.rows.lock().unwrap().push(LogEntry {
            row,
            event,
            deme,
            values: Some((old, new)),
        });
        Ok(())
    }
    /// Export complete provenance; tensor values are independent native copies.
    #[allow(clippy::type_complexity)] // Public six-column audit schema includes two heterogeneous values.
    fn details(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<(i64, String, usize, String, Py<PyAny>, Py<PyAny>)>> {
        self.rows
            .lock()
            .unwrap()
            .iter()
            .map(|entry| {
                let (old, new) = match &entry.values {
                    Some((old, new)) => (audit_to_python(py, old)?, audit_to_python(py, new)?),
                    None => (
                        entry.row.2.into_pyobject(py)?.into_any().unbind(),
                        entry.row.3.into_pyobject(py)?.into_any().unbind(),
                    ),
                };
                Ok((
                    entry.row.0,
                    entry.event.clone(),
                    entry.deme,
                    entry.row.1.clone(),
                    old,
                    new,
                ))
            })
            .collect()
    }
    /// Remove all entries without changing the owning session.
    fn clear(&self) {
        self.rows.lock().unwrap().clear();
    }
}

/// Read an audit value without retaining Python objects or borrowed buffers.
fn audit_value(value: &Bound<'_, PyAny>) -> PyResult<Option<CustomSlot>> {
    if value.is_none() {
        return Ok(None);
    }
    let result = if value.is_instance_of::<PyBool>() {
        CustomSlot::Bool(value.extract()?)
    } else if value.is_instance_of::<PyInt>() {
        CustomSlot::Int(value.extract()?)
    } else if value.is_instance_of::<PyFloat>() {
        CustomSlot::Float(value.extract()?)
    } else {
        let array = value.extract::<numpy::PyReadonlyArrayDyn<'_, f64>>()?;
        CustomSlot::Array {
            shape: array.shape().to_vec(),
            values: array.as_array().iter().copied().collect(),
        }
    };
    Ok(Some(result))
}

/// Preserve the historical scalar-only log projection without fabricating tensors.
fn audit_scalar(value: &Option<CustomSlot>) -> Option<f64> {
    match value {
        Some(CustomSlot::Bool(value)) => Some(if *value { 1.0 } else { 0.0 }),
        Some(CustomSlot::Int(value)) => Some(*value as f64),
        Some(CustomSlot::Float(value)) => Some(*value),
        _ => None,
    }
}

/// Reuse the native typed-value serializer; every array owns a fresh buffer.
fn audit_to_python(py: Python<'_>, value: &Option<CustomSlot>) -> PyResult<Py<PyAny>> {
    let Some(value) = value else {
        return Ok(py.None());
    };
    let values = std::collections::HashMap::from([("value".to_owned(), value.clone())]);
    let dictionary = crate::model::custom_fields::custom_slots_to_python(py, &values)?;
    Ok(dictionary
        .get_item("value")?
        .expect("serializer includes the supplied key")
        .unbind())
}
