//! PyO3 session object owning configuration copies, RNG state, and the
//! compiled CSR hook program.

use numpy::{PyArray2, PyReadonlyArray4, PyReadwriteArray3, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rand::rngs::SmallRng;

use crate::config::SimConfig;
use crate::hooks::HookProgram;
use crate::lifecycle;
use crate::rng::new_rng;

fn array_slices<'a>(
    ind: &'a mut PyReadwriteArray3<'_, f64>,
    sperm: &'a mut PyReadwriteArray3<'_, f64>,
    n_ages: usize,
    n_ztypes: usize,
) -> PyResult<(&'a mut [f64], &'a mut [f64])> {
    let ind_shape = ind.shape();
    if ind_shape != [2, n_ages, n_ztypes] {
        return Err(PyValueError::new_err(format!(
            "individual_count shape must be [2, {n_ages}, {n_ztypes}], got {ind_shape:?}"
        )));
    }
    let sperm_shape = sperm.shape();
    if sperm_shape != [n_ages, n_ztypes, n_ztypes] {
        return Err(PyValueError::new_err(format!(
            "sperm_storage shape must be [{n_ages}, {n_ztypes}, {n_ztypes}], got {sperm_shape:?}"
        )));
    }
    let ind_slice = ind
        .as_slice_mut()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let sperm_slice = sperm
        .as_slice_mut()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok((ind_slice, sperm_slice))
}

fn map_lifecycle_error(err: String) -> PyErr {
    PyRuntimeError::new_err(err)
}

#[pyclass(name = "EngineSession")]
pub struct EngineSession {
    cfg: SimConfig,
    rng: SmallRng,
    hooks: HookProgram,
}

#[pymethods]
impl EngineSession {
    #[new]
    #[pyo3(signature = (config, seed=0))]
    fn new(config: &Bound<'_, PyAny>, seed: u64) -> PyResult<Self> {
        let cfg = SimConfig::from_python(config)?;
        Ok(Self {
            cfg,
            rng: new_rng(seed),
            hooks: HookProgram::default(),
        })
    }

    /// Replace the declarative CSR hook program used by ``tick``.
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

    /// Run the reproduction stage in place.
    fn reproduction(
        &mut self,
        mut ind: PyReadwriteArray3<'_, f64>,
        mut sperm: PyReadwriteArray3<'_, f64>,
    ) -> PyResult<()> {
        let (ind_slice, sperm_slice) =
            array_slices(&mut ind, &mut sperm, self.cfg.n_ages, self.cfg.n_ztypes)?;
        lifecycle::reproduction(&mut self.rng, &self.cfg, ind_slice, sperm_slice)
            .map_err(map_lifecycle_error)
    }

    /// Run the survival stage in place.
    fn survival(
        &mut self,
        mut ind: PyReadwriteArray3<'_, f64>,
        mut sperm: PyReadwriteArray3<'_, f64>,
    ) -> PyResult<()> {
        let (ind_slice, sperm_slice) =
            array_slices(&mut ind, &mut sperm, self.cfg.n_ages, self.cfg.n_ztypes)?;
        lifecycle::survival(&mut self.rng, &self.cfg, ind_slice, sperm_slice)
            .map_err(map_lifecycle_error)
    }

    /// Run the aging stage in place.
    fn aging(
        &mut self,
        mut ind: PyReadwriteArray3<'_, f64>,
        mut sperm: PyReadwriteArray3<'_, f64>,
    ) -> PyResult<()> {
        let (ind_slice, sperm_slice) =
            array_slices(&mut ind, &mut sperm, self.cfg.n_ages, self.cfg.n_ztypes)?;
        lifecycle::aging(&self.cfg, ind_slice, sperm_slice);
        Ok(())
    }

    /// Run one complete age-structured tick with declarative hooks.
    ///
    /// Returns ``0`` (continue) or ``1`` (a hook requested a stop).  The tick
    /// value is *not* advanced here; the Python adapter owns tick bookkeeping
    /// exactly like ``natal.engine.lifecycle``.
    fn tick(
        &mut self,
        mut ind: PyReadwriteArray3<'_, f64>,
        mut sperm: PyReadwriteArray3<'_, f64>,
        tick: i64,
        deme_id: i64,
    ) -> PyResult<i32> {
        let (ind_slice, sperm_slice) =
            array_slices(&mut ind, &mut sperm, self.cfg.n_ages, self.cfg.n_ztypes)?;
        lifecycle::run_tick(
            &mut self.rng,
            &self.cfg,
            &self.hooks,
            ind_slice,
            sperm_slice,
            tick,
            deme_id,
        )
        .map_err(map_lifecycle_error)
    }

    /// Run up to ``n_ticks`` complete ticks inside Rust.
    ///
    /// The state arrays are mutated in place and the final tick value is
    /// returned.  When ``record_interval > 0``, flattened history rows are
    /// returned as a 2-D NumPy array whose row layout mirrors the Numba
    /// ``_run_loop_structured`` kernel.
    #[allow(clippy::too_many_arguments)] // PyO3 boundary mirrors the Numba run_fn signature.
    #[pyo3(signature = (individual_count, sperm_storage, tick, n_ticks, record_interval, observation_mask=None))]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        mut individual_count: PyReadwriteArray3<'py, f64>,
        mut sperm_storage: PyReadwriteArray3<'py, f64>,
        tick: i64,
        n_ticks: i64,
        record_interval: i64,
        observation_mask: Option<PyReadonlyArray4<'py, f64>>,
    ) -> PyResult<(i64, Bound<'py, PyArray2<f64>>, bool)> {
        let (ind_slice, sperm_slice) = array_slices(
            &mut individual_count,
            &mut sperm_storage,
            self.cfg.n_ages,
            self.cfg.n_ztypes,
        )?;
        let mask_vec = match observation_mask {
            Some(mask) => Some(
                mask.as_slice()
                    .map_err(|err| PyValueError::new_err(err.to_string()))?
                    .to_vec(),
            ),
            None => None,
        };

        let (final_tick, rows, was_stopped) = lifecycle::run_batch(
            &mut self.rng,
            &self.cfg,
            &self.hooks,
            ind_slice,
            sperm_slice,
            tick,
            n_ticks,
            record_interval,
            mask_vec.as_deref(),
        )
        .map_err(map_lifecycle_error)?;

        let history = PyArray2::<f64>::from_vec2(py, &rows)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok((final_tick, history, was_stopped))
    }
}

fn extract_i64_array(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<i64>> {
    use numpy::PyReadonlyArray1;
    let value = obj.getattr(name)?;
    if let Ok(array) = value.extract::<PyReadonlyArray1<'_, i64>>() {
        return Ok(array.as_slice()?.to_vec());
    }
    if let Ok(array) = value.extract::<PyReadonlyArray1<'_, i32>>() {
        return Ok(array.as_slice()?.iter().map(|&v| v as i64).collect());
    }
    Err(PyValueError::new_err(format!(
        "{name} must be an int32 or int64 array"
    )))
}

fn extract_f64_array(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    use numpy::PyReadonlyArray1;
    let array = obj.getattr(name)?.extract::<PyReadonlyArray1<'_, f64>>()?;
    Ok(array.as_slice()?.to_vec())
}

fn extract_bool_array(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<bool>> {
    use numpy::PyReadonlyArray1;
    let value = obj.getattr(name)?;
    if let Ok(array) = value.extract::<PyReadonlyArray1<'_, bool>>() {
        return Ok(array.as_slice()?.to_vec());
    }
    if let Ok(array) = value.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(array.as_slice()?.iter().map(|&v| v != 0.0).collect());
    }
    Err(PyValueError::new_err(format!(
        "{name} must be a bool or float64 array"
    )))
}

fn extract_i64_scalar(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<i64> {
    let value = obj.getattr(name)?;
    if let Ok(scalar) = value.extract::<i64>() {
        return Ok(scalar);
    }
    value.call_method0("item")?.extract::<i64>()
}

impl HookProgram {
    fn from_python(program: &Bound<'_, PyAny>) -> PyResult<Self> {
        let n_hooks = extract_i64_scalar(program, "n_hooks")?;
        Ok(Self {
            n_events: extract_i64_scalar(program, "n_events")?,
            n_hooks,
            hook_offsets: extract_i64_array(program, "hook_offsets")?,
            op_offsets: extract_i64_array(program, "op_offsets")?,
            op_types: extract_i64_array(program, "op_types_data")?,
            zidx_offsets: extract_i64_array(program, "zidx_offsets_data")?,
            zidx_data: extract_i64_array(program, "zidx_data")?,
            age_offsets: extract_i64_array(program, "age_offsets_data")?,
            age_data: extract_i64_array(program, "age_data")?,
            sex_masks: extract_bool_array(program, "sex_masks_data")?,
            params: extract_f64_array(program, "params_data")?,
            condition_offsets: extract_i64_array(program, "condition_offsets_data")?,
            condition_types: extract_i64_array(program, "condition_types_data")?,
            condition_params: extract_i64_array(program, "condition_params_data")?,
            deme_selector_types: extract_i64_array(program, "deme_selector_types")?,
            deme_selector_offsets: extract_i64_array(program, "deme_selector_offsets")?,
            deme_selector_data: extract_i64_array(program, "deme_selector_data")?,
        })
    }
}
