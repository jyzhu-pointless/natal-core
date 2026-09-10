//! Event-local candidates: Python never borrows the live session or its RNG.
use std::cell::RefCell;
use std::collections::HashMap;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::rand_core::TryRng;
use rand::RngExt;
use rand_distr::{Distribution, Normal};

use crate::kernels::rng::SessionRng;
use crate::model::blueprint::Blueprint;
use crate::model::ecology::session_get_tensor;
use crate::model::ecology::session_tensor_write;
use crate::model::ecology::EcologyParams;
use crate::model::genetics::GeneticsTensors;

thread_local! {
    // Kernel error propagation remains independent of Python; only the original
    // exception is retained here until the enclosing PyO3 entry translates it.
    static CALLBACK_ERROR: RefCell<Option<PyErr>> = const { RefCell::new(None) };
}

/// Preserve Python exception identity across the numerical kernel boundary.
pub fn preserve_error(error: PyErr) -> String {
    CALLBACK_ERROR.with(|slot| *slot.borrow_mut() = Some(error));
    "python callback transaction failed".to_owned()
}

/// Translate a kernel error, recovering a callback's original exception.
pub fn map_error(message: String) -> PyErr {
    CALLBACK_ERROR
        .with(|slot| slot.borrow_mut().take())
        .unwrap_or_else(|| PyRuntimeError::new_err(message))
}

type OwnedStateArrays = (Py<PyArray1<f64>>, Py<PyArray1<f64>>);
type BoundStateArrays<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// Owned transaction for exactly one Python callback. Its RNG is a candidate
/// of the current Rust stream, committed with parameters and state on success.
#[pyclass]
pub struct HookTransaction {
    pub active: bool,
    pub parameters_changed: bool,
    pub blueprint: Blueprint,
    pub params: EcologyParams,
    pub genetics: GeneticsTensors,
    pub rng: SessionRng,
    pub state_ind: Vec<f64>,
    pub state_sperm: Vec<f64>,
    pub state_arrays: Option<OwnedStateArrays>,
}

impl HookTransaction {
    /// Validate and collect only a state candidate actually requested by Python.
    pub fn candidate_state(&self, py: Python<'_>) -> PyResult<Option<(Vec<f64>, Vec<f64>)>> {
        let Some((ind, sperm)) = &self.state_arrays else {
            return Ok(None);
        };
        let ind = ind.bind(py).readonly().as_slice()?.to_vec();
        let sperm = sperm.bind(py).readonly().as_slice()?.to_vec();
        if ind.len() != self.state_ind.len()
            || sperm.len() != self.state_sperm.len()
            || ind.iter().chain(&sperm).any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(PyValueError::new_err(
                "Hook state must preserve shape and contain finite nonnegative counts",
            ));
        }
        Ok(Some((ind, sperm)))
    }

    /// Reject retained handles once the callback has returned.
    pub fn ensure_active(&self) -> PyResult<()> {
        if self.active {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("Hook context has expired"))
        }
    }
}

#[pymethods]
impl HookTransaction {
    /// Materialize detached NumPy state only on the callback's first state read.
    fn state_arrays<'py>(&mut self, py: Python<'py>) -> PyResult<BoundStateArrays<'py>> {
        self.ensure_active()?;
        if self.state_arrays.is_none() {
            self.state_arrays = Some((
                PyArray1::from_slice(py, &self.state_ind).unbind(),
                PyArray1::from_slice(py, &self.state_sperm).unbind(),
            ));
        }
        let (ind, sperm) = self
            .state_arrays
            .as_ref()
            .expect("state arrays were materialized");
        Ok((ind.bind(py).clone(), sperm.bind(py).clone()))
    }

    /// Check the candidate before Python publishes its buffered audit metadata.
    fn validate_state(&self, py: Python<'_>) -> PyResult<()> {
        self.ensure_active()?;
        self.candidate_state(py).map(|_| ())
    }

    /// Read current candidate ecology, detached from the session.
    fn ecology<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.ensure_active()?;
        crate::sessions::ecology_snapshot::ecology_snapshot(py, &self.params)
    }

    /// Read isolated custom values from this event candidate.
    fn get_custom_slots<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.ensure_active()?;
        crate::model::custom_fields::custom_slots_to_python(py, &self.params.custom_slots[0])
    }

    /// Validate and replace the complete event custom-slot candidate.
    fn set_custom_slots(&mut self, source: &Bound<'_, PyAny>) -> PyResult<()> {
        self.ensure_active()?;
        self.params.custom_slots[0] =
            crate::model::custom_fields::custom_slots_from_python(source)?;
        self.parameters_changed = true;
        Ok(())
    }

    /// Read one scalar candidate value.
    fn get_scalar(&self, name: &str) -> PyResult<f64> {
        self.ensure_active()?;
        self.params.get_scalar(name)
    }

    /// Read one isolated tensor candidate.
    fn get_tensor<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.ensure_active()?;
        session_get_tensor(py, &self.params, &self.genetics, name)
    }

    /// Validate and apply one scalar transaction.
    fn apply(&mut self, writes: HashMap<String, f64>) -> PyResult<()> {
        self.ensure_active()?;
        self.params.apply(writes)?;
        self.parameters_changed = true;
        Ok(())
    }

    /// Replace one tensor candidate after validation.
    fn tensor_write(&mut self, name: &str, values: Vec<f64>) -> PyResult<()> {
        self.ensure_active()?;
        session_tensor_write(
            &self.blueprint,
            &mut self.params,
            &mut self.genetics,
            name,
            values,
        )?;
        self.parameters_changed = true;
        Ok(())
    }

    /// Import explicitly changed compiled fields into this candidate.
    fn refresh_params(&mut self, fields: Vec<String>, source: &Bound<'_, PyAny>) -> PyResult<()> {
        self.ensure_active()?;
        self.params.pull_fields(
            &self.blueprint,
            0,
            &fields,
            source,
            Some(&mut self.genetics),
        )?;
        self.parameters_changed = true;
        Ok(())
    }

    /// Sample using the event's candidate of the owning deme's Rust stream.
    fn sample(&mut self, kind: &str, a: f64, b: f64) -> PyResult<f64> {
        self.ensure_active()?;
        if !a.is_finite() || !b.is_finite() {
            return Err(PyValueError::new_err("sampling arguments must be finite"));
        }
        match kind {
            "uniform" if b >= a => {
                let word = self.rng.try_next_u64().expect("SessionRng is infallible");
                Ok(a + (b - a) * ((word >> 11) as f64 / 9_007_199_254_740_992.0))
            }
            "integers"
                if a.fract() == 0.0
                    && b.fract() == 0.0
                    && a >= -(1_i64 << 53) as f64
                    && b <= (1_i64 << 53) as f64
                    && b > a =>
            {
                Ok(self.rng.random_range(a as i64..b as i64) as f64)
            }
            "normal" if b >= 0.0 => {
                if b == 0.0 {
                    return Ok(a);
                }
                Ok(Normal::new(a, b)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .sample(&mut self.rng))
            }
            "binomial"
                if a >= 0.0
                    && a < i64::MAX as f64
                    && a.fract() == 0.0
                    && (0.0..=1.0).contains(&b) =>
            {
                Ok(crate::kernels::rng::binomial(&mut self.rng, a as i64, b))
            }
            _ => Err(PyValueError::new_err(
                "invalid distribution or sampling arguments",
            )),
        }
    }
}
