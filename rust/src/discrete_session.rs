//! PyO3 session object for the discrete-generation / Wright-Fisher backend.

use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray4, PyReadwriteArray3, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rand::rngs::SmallRng;

use crate::discrete::{self, DiscreteConfig};
use crate::hooks::HookProgram;
use crate::rng::new_rng;

/// Convert internal kernel error strings into ``PyRuntimeError``.
fn map_lifecycle_error(err: String) -> PyErr {
    PyRuntimeError::new_err(err)
}

/// PyO3-exported stateful session for discrete-generation / Wright-Fisher.
///
/// Owns a ``DiscreteConfig`` snapshot, RNG, and CSR hook program.
#[pyclass(name = "DiscreteEngineSession")]
pub struct DiscreteEngineSession {
    cfg: DiscreteConfig,
    rng: SmallRng,
    hooks: HookProgram,
}

#[pymethods]
impl DiscreteEngineSession {
    /// Create a session from a Python discrete config and an optional seed.
    ///
    /// ## Parameters
    /// - `config`: Python ``DiscretePopulationConfig``.
    /// - `seed`: RNG seed.
    ///
    /// ## Returns
    /// A new ``DiscreteEngineSession``.
    #[new]
    #[pyo3(signature = (config, seed=0))]
    fn new(config: &Bound<'_, PyAny>, seed: u64) -> PyResult<Self> {
        Ok(Self {
            cfg: DiscreteConfig::from_python(config)?,
            rng: new_rng(seed),
            hooks: HookProgram::default(),
        })
    }

    /// Replace the declarative CSR hook program used by ticks.
    fn set_hook_program(&mut self, program: &Bound<'_, PyAny>) -> PyResult<()> {
        self.hooks = HookProgram::from_python(program)?;
        Ok(())
    }

    /// Clear all declarative hooks.
    fn clear_hook_program(&mut self) {
        self.hooks = HookProgram::default();
    }

    /// Reseed the Rust RNG used by stochastic sampling.
    fn reseed(&mut self, seed: u64) {
        self.rng = new_rng(seed);
    }

    /// Run one discrete or Wright-Fisher tick in place.
    ///
    /// ## Parameters
    /// - `individual_count`: Mutable discrete state array.
    /// - `tick`: Current tick.
    /// - `wf`: If true, use the fused Wright-Fisher tick.
    ///
    /// ## Returns
    /// ``0`` or ``1`` (stop).
    fn tick(
        &mut self,
        mut individual_count: PyReadwriteArray3<'_, f64>,
        tick: i64,
        wf: bool,
    ) -> PyResult<i32> {
        // Discrete path runs the standard tick; WF path runs only the first
        // hook and then the fused Wright-Fisher update.
        let shape = individual_count.shape();
        if shape != [2, 2, self.cfg.n_ztypes] {
            return Err(PyValueError::new_err(format!(
                "individual_count shape must be [2, 2, {}], got {shape:?}",
                self.cfg.n_ztypes
            )));
        }
        let ind = individual_count
            .as_slice_mut()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        if wf {
            let result = self.hooks.execute_event(
                &mut self.rng,
                0,
                ind,
                &mut [],
                2,
                2,
                self.cfg.n_ztypes,
                tick,
                self.cfg.stochastic,
                self.cfg.continuous_sampling,
                -1,
            );
            if result != 0 {
                return Ok(result);
            }
            discrete::run_wf_tick(&mut self.rng, &self.cfg, ind).map_err(map_lifecycle_error)?;
            Ok(0)
        } else {
            discrete::run_tick(&mut self.rng, &self.cfg, &self.hooks, ind, tick)
                .map_err(map_lifecycle_error)
        }
    }

    /// Run up to ``n_ticks`` discrete/WF ticks inside Rust with optional recording.
    ///
    /// ## Returns
    /// ``(final_tick, history, was_stopped)``.
    #[allow(clippy::too_many_arguments)] // PyO3 boundary mirrors the Numba run_fn signature.
    #[pyo3(signature = (individual_count, tick, n_ticks, record_interval, wf, observation_mask=None))]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        mut individual_count: PyReadwriteArray3<'py, f64>,
        tick: i64,
        n_ticks: i64,
        record_interval: i64,
        wf: bool,
        observation_mask: Option<PyReadonlyArray4<'py, f64>>,
    ) -> PyResult<(i64, Bound<'py, PyArray2<f64>>, bool)> {
        // Run a batch of discrete or WF ticks inside Rust and return history.
        let shape = individual_count.shape();
        if shape != [2, 2, self.cfg.n_ztypes] {
            return Err(PyValueError::new_err(format!(
                "individual_count shape must be [2, 2, {}], got {shape:?}",
                self.cfg.n_ztypes
            )));
        }
        let ind = individual_count
            .as_slice_mut()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let mask_vec = match observation_mask {
            Some(mask) => Some(
                mask.as_slice()
                    .map_err(|err| PyValueError::new_err(err.to_string()))?
                    .to_vec(),
            ),
            None => None,
        };
        let (final_tick, flat_history, n_rows, was_stopped) = discrete::run_batch(
            &mut self.rng,
            &self.cfg,
            &self.hooks,
            ind,
            tick,
            n_ticks,
            record_interval,
            mask_vec.as_deref(),
            wf,
        )
        .map_err(map_lifecycle_error)?;
        let n_cols = if n_rows == 0 {
            0
        } else {
            flat_history.len() / n_rows
        };
        let history = PyArray2::<f64>::zeros(py, [n_rows, n_cols], false);
        history
            .readwrite()
            .as_slice_mut()
            .map_err(|err| PyValueError::new_err(err.to_string()))?
            .copy_from_slice(&flat_history);
        Ok((final_tick, history, was_stopped))
    }
}
