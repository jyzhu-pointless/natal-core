//! Ring storage for recorded history rows and the shared history handle.
//!
//! Rows stay in an independently owned ring.  Python receives copies, so an
//! exported array never aliases a row that a later run may evict or replace.

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::output::observation::project;
use crate::output::parameter_log::{ParameterLog, SharedLog};

/// Shared storage referenced by a session and its Python query adapter.
pub type SharedHistory = Arc<Mutex<HistoryData>>;

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
    fn empty(width: usize, dimensions: [usize; 4], raw: bool, max_rows: Option<usize>) -> Self {
        Self {
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
        }
    }

    /// Create an unbound store used by low-level session callers.
    pub(crate) fn transient(width: usize, dimensions: [usize; 4], raw: bool) -> SharedHistory {
        Arc::new(Mutex::new(Self::empty(width, dimensions, raw, None)))
    }

    /// Configure an unbound store from the session's compiled observation mask.
    pub(crate) fn configure_observation_slice(&mut self, mask: Vec<f64>) -> PyResult<()> {
        let selected = (0..self.dimensions[0]).collect::<Vec<_>>();
        let projected = project(
            &vec![0.0; self.dimensions.iter().product()],
            &mask,
            self.dimensions,
            &selected,
            false,
            false,
        )?;
        self.width = 1 + projected.len();
        self.mask = mask;
        self.selected = selected;
        Ok(())
    }

    /// Copy retained rows into the legacy low-level flat layout.
    pub(crate) fn flat_rows(&self) -> (Vec<f64>, usize) {
        let n_rows = self.rows.len();
        let flat = self
            .rows
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        (flat, n_rows)
    }

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
            data: Arc::new(Mutex::new(HistoryData::empty(
                width, dimensions, raw, max_rows,
            ))),
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
