//! Native history storage and the shared numerical observation projection.
//!
//! Rows stay in an independently owned ring. Python receives copies, so an
//! exported array never aliases a row that a later run may evict or replace.
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::contract::CustomSlot;
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt};

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
/// Shared storage referenced by a session and its Python query adapter.
pub type SharedHistory = Arc<Mutex<HistoryData>>;

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
    let dictionary = crate::contract::custom_slots_to_python(py, &values)?;
    Ok(dictionary
        .get_item("value")?
        .expect("serializer includes the supplied key")
        .unbind())
}

/// All numerical history data and the execution layout live here.
pub struct HistoryData {
    pub rows: VecDeque<Vec<f64>>,
    pub width: usize,
    pub max_rows: Option<usize>,
    pub dimensions: [usize; 4],
    pub raw: bool,
    pub mask: Vec<f64>,
    pub selected: Vec<usize>,
    pub collapse_age: bool,
    pub aggregate: bool,
    pub log: SharedLog,
    pub cursors: VecDeque<(i64, Vec<usize>)>,
    pub extra_logs: Vec<SharedLog>,
    pub boundaries: VecDeque<(i64, usize, String)>,
}

impl HistoryData {
    /// Validate the overlap before changing either rows or retention metadata.
    pub fn append_row(&mut self, row: Vec<f64>, continuation: bool) -> PyResult<bool> {
        if row.len() != self.width || !row[0].is_finite() || row[0].fract() != 0.0 {
            return Err(PyValueError::new_err(
                "History row has an invalid width or tick",
            ));
        }
        if let Some(last) = self.rows.back() {
            if row[0] < last[0] {
                return Err(PyValueError::new_err(
                    "Kernel History starts before the latest recorded tick",
                ));
            }
            if row[0] == last[0] {
                if continuation && row == *last {
                    return Ok(false);
                }
                return Err(PyValueError::new_err(if continuation {
                    "Kernel History boundary payload does not match the latest recorded state"
                        .to_owned()
                } else {
                    format!("History already contains tick {}.", row[0] as i64)
                }));
            }
        }
        let tick = row[0] as i64;
        self.rows.push_back(row);
        self.boundaries.push_back((tick, 0, "Ready".to_owned()));
        let mut cursors = vec![self.log.lock().unwrap().len()];
        cursors.extend(self.extra_logs.iter().map(|log| log.lock().unwrap().len()));
        self.cursors.push_back((tick, cursors));
        self.evict();
        Ok(true)
    }
    /// Evict numerical rows and their matching log cursor together.
    pub fn evict(&mut self) {
        if let Some(limit) = self.max_rows {
            while self.rows.len() > limit {
                self.rows.pop_front();
                self.cursors.pop_front();
                self.boundaries.pop_front();
            }
        }
    }
    /// Record directly from the engine-owned arrays, projecting in Rust.
    pub fn record(
        &mut self,
        tick: i64,
        ind: &[f64],
        sperm: &[f64],
        continuation: bool,
    ) -> PyResult<bool> {
        let mut row = vec![tick as f64];
        if self.raw {
            row.extend_from_slice(ind);
            row.extend_from_slice(sperm);
        } else {
            row.extend(project(
                ind,
                &self.mask,
                self.dimensions,
                &self.selected,
                self.collapse_age,
                self.aggregate,
            )?);
        }
        self.append_row(row, continuation)
    }
    /// Drop future history and log entries at an exact retained boundary.
    pub fn restore_timeline(&mut self, tick: i64) -> PyResult<()> {
        let cursor = self
            .cursors
            .iter()
            .find(|(t, _)| *t == tick)
            .map(|(_, c)| c.clone())
            .ok_or_else(|| PyValueError::new_err(format!("Tick {tick} not found in history.")))?;
        self.rows.retain(|r| r[0] as i64 <= tick);
        self.cursors.retain(|(t, _)| *t <= tick);
        self.boundaries.retain(|(t, _, _)| *t <= tick);
        self.log.lock().unwrap().truncate(cursor[0]);
        for (log, position) in self.extra_logs.iter().zip(&cursor[1..]) {
            log.lock().unwrap().truncate(*position);
        }
        Ok(())
    }
}

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

/// Query adapter around shared Rust storage; no Python-owned numerical rows.
#[pyclass]
#[derive(Clone)]
pub struct HistoryStore {
    pub(crate) data: SharedHistory,
}

#[pymethods]
impl HistoryStore {
    #[new]
    #[pyo3(signature = (width, dimensions, raw, max_rows=None))]
    fn new(
        width: usize,
        dimensions: [usize; 4],
        raw: bool,
        max_rows: Option<usize>,
    ) -> PyResult<Self> {
        if width == 0 || max_rows == Some(0) {
            return Err(PyValueError::new_err(
                "row_size and max_rows must be positive",
            ));
        }
        Ok(Self {
            data: Arc::new(Mutex::new(HistoryData {
                rows: VecDeque::new(),
                width,
                max_rows,
                dimensions,
                raw,
                mask: Vec::new(),
                selected: (0..dimensions[0]).collect(),
                collapse_age: false,
                aggregate: false,
                log: Arc::default(),
                cursors: VecDeque::new(),
                extra_logs: Vec::new(),
                boundaries: VecDeque::new(),
            })),
        })
    }
    /// Attach the population's native parameter timeline before execution.
    fn bind_log(&self, log: PyRef<'_, ParameterLog>) {
        self.data.lock().unwrap().log = Arc::clone(&log.rows);
    }
    /// Attach all per-deme native log streams for exact checkpoint cursors.
    fn bind_logs(&self, py: Python<'_>, logs: Vec<Py<ParameterLog>>) {
        self.data.lock().unwrap().extra_logs = logs
            .iter()
            .map(|log| Arc::clone(&log.borrow(py).rows))
            .collect();
    }
    /// Install the already compiled observation selector.
    fn configure_observation(
        &self,
        mask: PyReadonlyArray1<'_, f64>,
        selected: Vec<usize>,
        collapse_age: bool,
        aggregate: bool,
    ) -> PyResult<()> {
        let mut data = self.data.lock().unwrap();
        let mask = mask.as_slice()?.to_vec();
        project(
            &vec![0.0; data.dimensions.iter().product()],
            &mask,
            data.dimensions,
            &selected,
            collapse_age,
            aggregate,
        )?;
        data.mask = mask;
        data.selected = selected;
        data.collapse_age = collapse_age;
        data.aggregate = aggregate;
        Ok(())
    }
    #[getter]
    fn max_rows(&self) -> Option<usize> {
        self.data.lock().unwrap().max_rows
    }
    #[setter]
    fn set_max_rows(&self, value: Option<usize>) -> PyResult<()> {
        if value == Some(0) {
            return Err(PyValueError::new_err("max_rows must be >= 1 or None"));
        }
        let mut data = self.data.lock().unwrap();
        data.max_rows = value;
        data.evict();
        Ok(())
    }
    fn __len__(&self) -> usize {
        self.data.lock().unwrap().rows.len()
    }
    /// Return only tick metadata without exporting numerical state.
    fn ticks(&self) -> Vec<i64> {
        self.data
            .lock()
            .unwrap()
            .rows
            .iter()
            .map(|r| r[0] as i64)
            .collect()
    }
    /// Describe partial snapshots separately from complete tick boundaries.
    fn boundaries(&self) -> Vec<(i64, usize, String)> {
        self.data
            .lock()
            .unwrap()
            .boundaries
            .iter()
            .cloned()
            .collect()
    }
    /// Append an externally imported batch with atomic validation.
    #[pyo3(signature = (rows, continuation=false))]
    fn append(&self, rows: PyReadonlyArray2<'_, f64>, continuation: bool) -> PyResult<()> {
        let shape = rows.shape();
        let mut data = self.data.lock().unwrap();
        if shape[1] != data.width {
            return Err(PyValueError::new_err(
                "History row width does not match schema",
            ));
        }
        let values = rows.as_array();
        // Validate the whole batch before modifying the ring.
        let mut previous = data.rows.back().map(|r| r[0]);
        let mut start = 0;
        for i in 0..shape[0] {
            let row = values.row(i);
            let tick = row[0];
            if !tick.is_finite() || tick.fract() != 0.0 {
                return Err(PyValueError::new_err(
                    "History tick must be a finite integer",
                ));
            }
            if let Some(last) = previous {
                if i == 0 && continuation && tick == last {
                    if row
                        .iter()
                        .copied()
                        .eq(data.rows.back().unwrap().iter().copied())
                    {
                        start = 1;
                        continue;
                    }
                    return Err(PyValueError::new_err(
                        "Kernel History boundary payload does not match the latest recorded state",
                    ));
                }
                if tick <= last {
                    return Err(PyValueError::new_err("History ticks must be strictly increasing and unique; batch starts before the latest tick"));
                }
            }
            previous = Some(tick);
        }
        for i in start..shape[0] {
            data.append_row(values.row(i).to_vec(), false)?;
        }
        Ok(())
    }
    /// Slice columns inside Rust; returned arrays always own their data.
    #[pyo3(signature = (start=0, end=None))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        end: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let data = self.data.lock().unwrap();
        let end = end.unwrap_or(data.width);
        if start > end || end > data.width {
            return Err(PyValueError::new_err(
                "History column slice is out of bounds",
            ));
        }
        let arr = PyArray2::zeros(py, [data.rows.len(), end - start], false);
        let mut view = arr.readwrite();
        let output = view.as_slice_mut()?;
        for (index, row) in data.rows.iter().enumerate() {
            output[index * (end - start)..(index + 1) * (end - start)]
                .copy_from_slice(&row[start..end]);
        }
        drop(view);
        Ok(arr)
    }
    /// Query a single exact record without copying other history rows.
    fn row<'py>(&self, py: Python<'py>, tick: i64) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let data = self.data.lock().unwrap();
        let row = data
            .rows
            .iter()
            .find(|r| r[0] as i64 == tick)
            .ok_or_else(|| PyValueError::new_err(format!("Tick {tick} not found in history.")))?;
        Ok(PyArray1::from_vec(py, row.clone()))
    }
    fn clear(&self) {
        let mut data = self.data.lock().unwrap();
        data.rows.clear();
        data.cursors.clear();
        data.boundaries.clear();
    }
    fn truncate(&self, tick: i64) -> PyResult<()> {
        let mut data = self.data.lock().unwrap();
        if !data.rows.iter().any(|r| r[0] as i64 <= tick) {
            return Err(PyValueError::new_err(format!(
                "No records with tick <= {tick} exist."
            )));
        }
        data.rows.retain(|r| r[0] as i64 <= tick);
        data.cursors.retain(|(t, _)| *t <= tick);
        data.boundaries.retain(|(t, _, _)| *t <= tick);
        Ok(())
    }
    fn restore_timeline(&self, tick: i64) -> PyResult<()> {
        self.data.lock().unwrap().restore_timeline(tick)
    }
    /// Recompute retained raw observations entirely within native storage.
    fn observe(&self, target: PyRef<'_, HistoryStore>) -> PyResult<()> {
        if Arc::ptr_eq(&self.data, &target.data) {
            return Err(PyValueError::new_err(
                "Observation destination must be independent",
            ));
        }
        let source = self.data.lock().unwrap();
        let mut output = target.data.lock().unwrap();
        if !source.raw {
            return Err(PyValueError::new_err(
                "observe() is only valid on raw-mode History",
            ));
        }
        // Imported rows have a caller-supplied schema. Validate the raw slice
        // before indexing so malformed metadata cannot poison either mutex.
        let raw_end = source
            .dimensions
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .and_then(|count| count.checked_add(1))
            .filter(|end| *end <= source.width)
            .ok_or_else(|| {
                PyValueError::new_err("Raw history row width does not match population dimensions")
            })?;
        for row in &source.rows {
            output.record(row[0] as i64, &row[1..raw_end], &[], false)?;
        }
        Ok(())
    }
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
