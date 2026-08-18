//! PyO3 sessions for homogeneous and heterogeneous spatial multi-deme runs.

use numpy::{PyReadonlyArray1, PyReadwriteArray4, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::config::SimConfig;
use crate::hooks::HookProgram;
use crate::spatial;

/// Convert internal kernel error strings into ``PyRuntimeError``.
fn map_lifecycle_error(err: String) -> PyErr {
    PyRuntimeError::new_err(err)
}

/// PyO3 session for homogeneous spatial multi-deme runs.
///
/// Holds one shared config, a hook program, and a base seed.
#[pyclass(name = "SpatialEngineSession")]
pub struct SpatialEngineSession {
    cfg: SimConfig,
    hooks: HookProgram,
    seed: u64,
}

#[pymethods]
impl SpatialEngineSession {
    /// Create a homogeneous spatial session from one Python config and a seed.
    ///
    /// ## Parameters
    /// - `config`: Python ``PopulationConfig``.
    /// - `seed`: Base RNG seed.
    ///
    /// ## Returns
    /// A new ``SpatialEngineSession``.
    #[new]
    #[pyo3(signature = (config, seed=0))]
    fn new(config: &Bound<'_, PyAny>, seed: u64) -> PyResult<Self> {
        Ok(Self {
            cfg: SimConfig::from_python(config)?,
            hooks: HookProgram::default(),
            seed,
        })
    }

    /// Replace the declarative CSR hook program used by deme ticks.
    fn set_hook_program(&mut self, program: &Bound<'_, PyAny>) -> PyResult<()> {
        self.hooks = HookProgram::from_python(program)?;
        Ok(())
    }

    /// Clear all declarative hooks.
    fn clear_hook_program(&mut self) {
        self.hooks = HookProgram::default();
    }

    /// Change the base seed used for per-deme RNG streams.
    fn reseed(&mut self, seed: u64) {
        self.seed = seed;
    }

    /// Run one tick for all demes in parallel and return the next tick value.
    ///
    /// ## Parameters
    /// - `individual_count_all`: Stacked state array.
    /// - `sperm_storage_all`: Stacked sperm array.
    /// - `tick`: Current tick.
    ///
    /// ## Returns
    /// ``tick + 1``.
    fn run(
        &mut self,
        mut individual_count_all: PyReadwriteArray4<'_, f64>,
        mut sperm_storage_all: PyReadwriteArray4<'_, f64>,
        tick: i64,
    ) -> PyResult<i64> {
        // Validate stacked state shape, then run all deme ticks in parallel.
        let ind_shape = individual_count_all.shape();
        if ind_shape.len() != 4 || ind_shape[1] != 2 {
            return Err(PyValueError::new_err(format!(
                "individual_count_all must have shape (n_demes, 2, n_ages, n_ztypes), got {ind_shape:?}"
            )));
        }
        let n_demes = ind_shape[0];
        let n_ages = ind_shape[2];
        let n_ztypes = ind_shape[3];
        if n_ages != self.cfg.n_ages || n_ztypes != self.cfg.n_ztypes {
            return Err(PyValueError::new_err(format!(
                "individual_count_all age/ztype dimensions ({n_ages}, {n_ztypes}) do not match config ({}, {})",
                self.cfg.n_ages, self.cfg.n_ztypes
            )));
        }
        let sperm_shape = sperm_storage_all.shape();
        if sperm_shape != [n_demes, n_ages, n_ztypes, n_ztypes] {
            return Err(PyValueError::new_err(format!(
                "sperm_storage_all must have shape ({n_demes}, {n_ages}, {n_ztypes}, {n_ztypes}), got {sperm_shape:?}"
            )));
        }
        let ind = individual_count_all
            .as_slice_mut()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let sperm = sperm_storage_all
            .as_slice_mut()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        spatial::run_spatial_tick(&self.cfg, &self.hooks, self.seed, ind, sperm, n_demes, tick)
            .map_err(map_lifecycle_error)?;
        Ok(tick + 1)
    }
}

/// PyO3 session for heterogeneous spatial multi-deme runs.
///
/// Holds a config bank, per-deme config ids, a hook program, and a base seed.
#[pyclass(name = "HeterogeneousSpatialEngineSession")]
pub struct HeterogeneousSpatialEngineSession {
    configs: Vec<SimConfig>,
    hooks: HookProgram,
    seed: u64,
    deme_config_ids: Vec<usize>,
}

#[pymethods]
impl HeterogeneousSpatialEngineSession {
    /// Create a heterogeneous spatial session from a config bank and deme config ids.
    ///
    /// ## Parameters
    /// - `config_bank`: List of Python configs, one per unique deme type.
    /// - `deme_config_ids`: Array mapping each deme to an index into `config_bank`.
    /// - `seed`: Base RNG seed.
    ///
    /// ## Returns
    /// A new `HeterogeneousSpatialEngineSession`.
    #[new]
    #[pyo3(signature = (config_bank, deme_config_ids, seed=0))]
    fn new(
        config_bank: Vec<Bound<'_, PyAny>>,
        deme_config_ids: PyReadonlyArray1<'_, i64>,
        seed: u64,
    ) -> PyResult<Self> {
        let mut configs = Vec::with_capacity(config_bank.len());
        for item in &config_bank {
            configs.push(SimConfig::from_python(item)?);
        }
        let ids = deme_config_ids
            .as_slice()
            .map_err(|err| PyValueError::new_err(err.to_string()))?
            .iter()
            .map(|&value| value as usize)
            .collect::<Vec<usize>>();
        Ok(Self {
            configs,
            hooks: HookProgram::default(),
            seed,
            deme_config_ids: ids,
        })
    }

    /// Replace the declarative CSR hook program used by deme ticks.
    ///
    /// ## Parameters
    /// - `program`: Python CSR `HookProgram`.
    fn set_hook_program(&mut self, program: &Bound<'_, PyAny>) -> PyResult<()> {
        self.hooks = HookProgram::from_python(program)?;
        Ok(())
    }

    /// Clear all declarative hooks.
    fn clear_hook_program(&mut self) {
        self.hooks = HookProgram::default();
    }

    /// Change the base seed used for per-deme RNG streams.
    ///
    /// ## Parameters
    /// - `seed`: New base seed.
    fn reseed(&mut self, seed: u64) {
        self.seed = seed;
    }

    /// Run one tick for all demes using their per-deme configs and return the next tick value.
    ///
    /// ## Parameters
    /// - `individual_count_all`: Stacked individual-count array.
    /// - `sperm_storage_all`: Stacked sperm-storage array.
    /// - `tick`: Current tick.
    ///
    /// ## Returns
    /// `tick + 1` after all deme ticks complete.
    fn run(
        &mut self,
        mut individual_count_all: PyReadwriteArray4<'_, f64>,
        mut sperm_storage_all: PyReadwriteArray4<'_, f64>,
        tick: i64,
    ) -> PyResult<i64> {
        let ind_shape = individual_count_all.shape();
        if ind_shape.len() != 4 || ind_shape[1] != 2 {
            return Err(PyValueError::new_err(format!(
                "individual_count_all must have shape (n_demes, 2, n_ages, n_ztypes), got {ind_shape:?}"
            )));
        }
        let n_demes = ind_shape[0];
        if n_demes != self.deme_config_ids.len() {
            return Err(PyValueError::new_err(format!(
                "individual_count_all has {n_demes} demes but config ids describe {}",
                self.deme_config_ids.len()
            )));
        }
        let n_ages = ind_shape[2];
        let n_ztypes = ind_shape[3];
        let sperm_shape = sperm_storage_all.shape();
        if sperm_shape != [n_demes, n_ages, n_ztypes, n_ztypes] {
            return Err(PyValueError::new_err(format!(
                "sperm_storage_all must have shape ({n_demes}, {n_ages}, {n_ztypes}, {n_ztypes}), got {sperm_shape:?}"
            )));
        }
        let ind = individual_count_all
            .as_slice_mut()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let sperm = sperm_storage_all
            .as_slice_mut()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        crate::spatial::run_spatial_tick_heterogeneous(
            &self.configs,
            &self.hooks,
            &self.deme_config_ids,
            self.seed,
            ind,
            sperm,
            tick,
        )
        .map_err(map_lifecycle_error)?;
        Ok(tick + 1)
    }
}
