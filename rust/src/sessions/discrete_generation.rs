//! PyO3 session object for the discrete-generation / Wright-Fisher backend.

use crate::output::history::{HistoryStore, SharedHistory};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::hooks::interpreter::HookProgram;
use crate::kernels::discrete_generation::DiscreteGenerationConfig;
use crate::kernels::rng::{new_rng, SessionRng};
use crate::model::blueprint::Blueprint;
use crate::model::ecology::session_get_tensor;
use crate::model::ecology::session_tensor_write;
use crate::model::ecology::EcologyParams;
use crate::model::genetics::GeneticsTensors;
use crate::sessions::ecology_snapshot::{ecology_snapshot, restore_ecology};

/// Convert internal kernel error strings into ``PyRuntimeError``.
fn map_lifecycle_error(err: String) -> PyErr {
    crate::hooks::transaction::map_error(err)
}

/// Snapshot tuple for discrete sessions: ``(tick, ind_flat, rng_words, ecology)``.
type DiscreteSnapshot<'py> = (
    i64,
    Bound<'py, PyArray1<f64>>,
    Vec<u64>,
    Bound<'py, pyo3::types::PyDict>,
);

/// PyO3-exported stateful session for discrete-generation / Wright-Fisher.
///
/// Owns the frozen blueprint, the mutable params, the RNG, and a CSR hook
/// program.  A flat [`DiscreteGenerationConfig`] is assembled from the contracts at
/// every tick-batch entry point, so params writes take effect on the next
/// batch without rebuilding the session (the RNG keeps streaming).
#[pyclass(name = "DiscreteEngineSession")]
pub struct DiscreteGenerationSession {
    blueprint: Blueprint,
    params: EcologyParams,
    genetics: GeneticsTensors,
    rng: SessionRng,
    hooks: HookProgram,
    /// Audited set_param transitions accumulated across tick/run calls;
    /// drained by the Python adapter after each run (see ``AgeStructuredSession``).
    eco_journal: Vec<crate::hooks::interpreter::EcoJournalRow>,
    /// Record-aligned full checkpoints (plan 13.1 R3) — the discrete twin
    /// of ``AgeStructuredSession::checkpoints``.
    checkpoints: Vec<crate::kernels::age_structured::TickCheckpoint>,
    /// Native history shared with the Python read-only adapter.
    history_store: Option<SharedHistory>,
    /// Session-owned live state (plan S2): flattened counts plus the
    /// authoritative tick; runs and ticks operate on these directly.
    state_ind: Vec<f64>,
    state_tick: i64,
    execution: crate::sessions::status::ExecutionStatus,
    phase: usize,
}

#[pymethods]
impl DiscreteGenerationSession {
    /// Execute an explicit event on the same native state and RNG stream.
    #[pyo3(signature = (event, deme_id=0))]
    fn trigger_event(&mut self, event: usize, deme_id: i64) -> PyResult<i32> {
        if event >= 4 {
            return Err(PyValueError::new_err("unknown hook event"));
        }
        let mut values = self.params.eco_values_row(0);
        let mut ctx = Some(crate::kernels::age_structured::EcoCtx {
            bp: &self.blueprint,
            params: &mut self.params,
            genetics: &self.genetics,
            updated_genetics: None,
            phase: event * 2,
            deme: 0,
            tick: self.state_tick,
            journal: Vec::new(),
        });
        let mut result = self.hooks.execute_event(
            &mut self.rng,
            event as i64,
            &mut self.state_ind,
            &mut [],
            2,
            2,
            self.blueprint.n_ztypes,
            self.state_tick,
            self.blueprint.stochastic,
            self.blueprint.continuous_sampling,
            deme_id,
            &mut values,
        );
        let operation = (|| -> Result<(), String> {
            if result == 0 {
                result = self.hooks.fire_python_callbacks(
                    event,
                    &mut self.state_ind,
                    &mut [],
                    self.state_tick,
                    deme_id,
                    &mut self.rng,
                    &mut values,
                    &mut ctx,
                )?;
            }
            if let Some(context) = ctx.as_mut() {
                context.commit(&values)?;
            }
            Ok(())
        })();
        if let Some(context) = ctx.as_mut() {
            if let Some(shared) = &self.history_store {
                let store = shared.lock().unwrap();
                let mut log = store.log.lock().unwrap();
                for (tick, id, old, new, phase) in context.journal.drain(..) {
                    log.push(crate::output::parameter_log::LogEntry::from_phase(
                        (
                            tick,
                            crate::generated::ecology_parameters::ECO_PARAM_COLUMNS[id].to_owned(),
                            old,
                            new,
                        ),
                        phase,
                        0,
                    ));
                }
            } else {
                self.eco_journal.append(&mut context.journal);
            }
        }
        let genetics = ctx
            .as_mut()
            .and_then(|context| context.updated_genetics.take());
        drop(ctx);
        if let Some(genetics) = genetics {
            self.genetics = genetics;
        }
        if let Err(error) = operation {
            self.execution = crate::sessions::status::ExecutionStatus::Failed;
            return Err(map_lifecycle_error(error));
        }
        if result != 0 {
            self.execution = crate::sessions::status::ExecutionStatus::Stopped;
        }
        Ok(result)
    }

    /// Read native execution state and its within-tick phase cursor.
    fn execution_state(&self) -> (&str, usize) {
        (self.execution.name(), self.phase)
    }
    /// Mark an explicit user finish without discarding state or history.
    fn stop(&mut self) {
        self.execution = crate::sessions::status::ExecutionStatus::Stopped;
    }

    /// Create a session from the Python contract objects.
    ///
    /// ## Parameters
    /// - `blueprint`: A ``natal.contracts.Blueprint`` NamedTuple.
    /// - `params`: A ``natal.contracts.Params`` dataclass instance.
    /// - `seed`: RNG seed.
    ///
    /// ## Returns
    /// A new ``DiscreteGenerationSession`` owning copies of both contracts.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` when either contract is inconsistent or the
    /// blueprint is not discrete-shaped.
    #[new]
    #[pyo3(signature = (blueprint, params, seed=0))]
    fn from_parts(
        blueprint: &Bound<'_, PyAny>,
        params: &Bound<'_, PyAny>,
        seed: u64,
    ) -> PyResult<Self> {
        let bp = Blueprint::from_python(blueprint)?;
        // Panmictic sessions carry one deme: length-1 ecology columns and
        // the session-owned genetics tables split out of the contract.
        let pr = EcologyParams::from_python(params, 1)?;
        let genetics = GeneticsTensors::from_python(params)?;
        bp.validate()?;
        pr.validate(&bp)?;
        genetics.validate(&bp)?;
        // Validate the discrete normalization once at construction.
        DiscreteGenerationConfig::assemble(&bp, &pr, &genetics)?;
        // Seed the session-owned state from the blueprint's frozen initial
        // population; enable_rust_backend follows with set_state carrying
        // the live Python state.
        let state_ind = bp.initial_individual_count.to_vec();
        Ok(Self {
            blueprint: bp,
            params: pr,
            genetics,
            rng: new_rng(seed),
            hooks: HookProgram::default(),
            eco_journal: Vec::new(),
            checkpoints: Vec::new(),
            history_store: None,
            state_ind,
            state_tick: 0,
            execution: crate::sessions::status::ExecutionStatus::Ready,
            phase: 0,
        })
    }

    /// Pull exactly the named contract fields from the Python params object.
    ///
    /// ## Parameters
    /// - `fields`: Contract field names to pull.
    /// - `source`: A ``natal.contracts.Params`` carrying current values.
    ///
    /// ## Errors
    /// Returns ``PyKeyError``/``PyValueError`` on unknown names or size
    /// mismatch; nothing is written when any field fails.
    fn refresh_params(&mut self, fields: Vec<String>, source: &Bound<'_, PyAny>) -> PyResult<()> {
        let genetics = &mut self.genetics;
        self.params
            .pull_fields(&self.blueprint, 0, &fields, source, Some(genetics))
    }

    /// Batch scalar write straight into the owned params (hook write path).
    ///
    /// ## Errors
    /// Returns ``PyKeyError`` for unknown fields; atomic per call.
    fn apply(&mut self, writes: HashMap<String, f64>) -> PyResult<()> {
        self.params.apply(writes)
    }

    /// Whole-tensor contents write straight into the owned params.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` on size mismatch; previous contents are
    /// preserved on failure.
    fn tensor_write(&mut self, field: &str, values: Vec<f64>) -> PyResult<()> {
        session_tensor_write(
            &self.blueprint,
            &mut self.params,
            &mut self.genetics,
            field,
            values,
        )
    }

    /// Replace the custom dictionary atomically, retaining every declared type.
    fn set_custom_slots(&mut self, values: &Bound<'_, PyAny>) -> PyResult<()> {
        self.params.custom_slots[0] =
            crate::model::custom_fields::custom_slots_from_python(values)?;
        Ok(())
    }

    /// Return a detached custom dictionary.
    fn get_custom_slots<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        crate::model::custom_fields::custom_slots_to_python(py, &self.params.custom_slots[0])
    }

    /// Read a scalar param value from the owned params.
    ///
    /// ## Errors
    /// Returns ``PyKeyError`` for unknown or non-scalar fields.
    fn get_scalar(&self, name: &str) -> PyResult<f64> {
        self.params.get_scalar(name)
    }

    /// Read a copy of a tensor param from the owned params.
    ///
    /// ## Errors
    /// Returns ``PyKeyError`` for unknown or non-tensor fields.
    fn get_tensor<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        session_get_tensor(py, &self.params, &self.genetics, name)
    }

    /// Change execution switches while preserving the session's state and RNG.
    fn set_execution_flags(
        &mut self,
        stochastic: bool,
        continuous_sampling: bool,
        fixed_egg_count: bool,
        extreme_speed_mode: i64,
    ) -> PyResult<()> {
        if !(0..=3).contains(&extreme_speed_mode) {
            return Err(PyValueError::new_err(
                "extreme_speed_mode must be between 0 and 3",
            ));
        }
        self.blueprint.stochastic = stochastic;
        self.blueprint.continuous_sampling = continuous_sampling;
        self.blueprint.fixed_egg_count = fixed_egg_count;
        self.blueprint.extreme_speed_mode = extreme_speed_mode;
        Ok(())
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

    /// Register Python callbacks fired at the first/early/late event
    /// boundaries after the CSR hooks ran (see ``AgeStructuredSession``).
    #[pyo3(signature = (first, early, late, finish=None))]
    fn set_python_callbacks(
        &mut self,
        first: Vec<Py<PyAny>>,
        early: Vec<Py<PyAny>>,
        late: Vec<Py<PyAny>>,
        finish: Option<Vec<Py<PyAny>>>,
    ) {
        self.hooks.python_callbacks = vec![first, early, late, finish.unwrap_or_default()];
    }

    /// Clear all Python callbacks.
    fn clear_python_callbacks(&mut self) {
        self.hooks.python_callbacks = vec![Vec::new(), Vec::new(), Vec::new()];
    }

    /// Reseed the Rust RNG used by stochastic sampling.
    fn reseed(&mut self, seed: u64) {
        self.rng = new_rng(seed);
    }

    /// Drain the accumulated set_param audit journal.
    ///
    /// Returns and clears every ``(tick, param_id, old, new)`` transition
    /// committed by ``tick`` / ``run`` since the previous drain
    /// (change-only rows, commit order).
    ///
    /// ## Returns
    /// A list of ``(tick, param_id, old, new)`` tuples.
    fn drain_eco_journal(&mut self) -> Vec<(i64, usize, f64, f64)> {
        std::mem::take(&mut self.eco_journal)
            .into_iter()
            .map(|(tick, id, old, new, _)| (tick, id, old, new))
            .collect()
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
    #[pyo3(signature = (wf))]
    fn tick(&mut self, py: Python<'_>, wf: bool) -> PyResult<i32> {
        self.run(py, 1, 0, wf, None, 0)
            .map(|(_, _, stopped)| i32::from(stopped))
    }

    /// Run up to ``n_ticks`` discrete/WF ticks inside Rust with optional recording.
    ///
    /// ## Returns
    /// ``(final_tick, history, was_stopped)``.
    #[allow(clippy::too_many_arguments)] // PyO3 boundary mirrors the Numba run_fn signature.
    #[pyo3(signature = (n_ticks, record_interval, wf, observation_mask=None, checkpoint_every=0))]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        n_ticks: i64,
        record_interval: i64,
        wf: bool,
        observation_mask: Option<PyReadonlyArray4<'py, f64>>,
        checkpoint_every: i64,
    ) -> PyResult<(i64, Bound<'py, PyArray2<f64>>, bool)> {
        self.execution.begin()?;
        let outcome = self.run_inner(
            py,
            n_ticks,
            record_interval,
            wf,
            observation_mask,
            checkpoint_every,
        );
        // Earlier callbacks remain committed if a later callback fails.
        // Ecology already committed through EcoCtx; genetic overrides must
        // survive the context's unwinding before the wrapper marks Failed.
        for (_, _, genetics) in self
            .hooks
            .callback_commits
            .lock()
            .expect("callback queue poisoned")
            .drain(..)
        {
            if outcome.is_err() {
                self.genetics = genetics;
            }
        }
        self.execution = match &outcome {
            Ok(value) if value.2 => crate::sessions::status::ExecutionStatus::Stopped,
            Ok(_) => {
                self.phase = 0;
                crate::sessions::status::ExecutionStatus::Ready
            }
            Err(_) => crate::sessions::status::ExecutionStatus::Failed,
        };
        outcome
    }

    /// Capture a memory checkpoint of everything the session owns.
    ///
    /// Discrete populations have no sperm storage; see
    /// ``AgeStructuredSession.snapshot_state`` for the checkpoint semantics
    /// (RNG continuation via raw state words, ecology-only param rollback).
    ///
    /// ## Parameters
    /// - `individual_count`: Current discrete state array.
    /// - `tick`: Current tick value.
    ///
    /// ## Returns
    /// ``(tick, ind_flat, rng_words, ecology)``.
    /// Install a full live state (plan S2 state ownership; discrete twin).
    ///
    /// ## Errors
    /// Returns ``PyValueError`` when the vector has the wrong length.
    #[pyo3(signature = (ind_flat, tick))]
    fn set_state(&mut self, ind_flat: Vec<f64>, tick: i64) -> PyResult<()> {
        let cfg =
            DiscreteGenerationConfig::assemble(&self.blueprint, &self.params, &self.genetics)?;
        let want = 2 * 2 * cfg.n_ztypes;
        if ind_flat.len() != want {
            return Err(PyValueError::new_err(format!(
                "ind_flat must contain {want} values, got {}",
                ind_flat.len()
            )));
        }
        crate::model::validation::validate_state_values(&ind_flat, &[], tick)?;
        self.state_ind = ind_flat;
        self.state_tick = tick;
        self.execution = crate::sessions::status::ExecutionStatus::Ready;
        self.phase = 0;
        Ok(())
    }

    /// Return a point-in-time snapshot of the session-owned state.
    fn state_snapshot<'py>(&self, py: Python<'py>) -> (i64, Bound<'py, PyArray1<f64>>) {
        (self.state_tick, PyArray1::from_slice(py, &self.state_ind))
    }

    fn snapshot_state<'py>(&self, py: Python<'py>) -> PyResult<DiscreteSnapshot<'py>> {
        let ind_flat = PyArray1::from_slice(py, &self.state_ind);
        let rng_words = self.rng.state_words().to_vec();
        let ecology = ecology_snapshot(py, &self.params)?;
        Ok((self.state_tick, ind_flat, rng_words, ecology))
    }

    /// Restore a memory checkpoint produced by
    /// [`DiscreteGenerationSession::snapshot_state`].
    ///
    /// ## Parameters
    /// - `individual_count`: Live state array, overwritten.
    /// - `tick`: Tick value carried by the checkpoint.
    /// - `ind_flat`: Checkpoint individual counts.
    /// - `rng_words`: Four captured RNG state words.
    /// - `ecology`: Checkpoint ecology dict.
    ///
    /// ## Returns
    /// The restored tick value.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` on array size, RNG word count, or ecology
    /// field mismatch.
    #[pyo3(signature = (tick, ind_flat, rng_words, ecology))]
    fn restore_state(
        &mut self,
        tick: i64,
        ind_flat: numpy::PyReadonlyArray1<'_, f64>,
        rng_words: Vec<u64>,
        ecology: &Bound<'_, PyAny>,
    ) -> PyResult<i64> {
        if rng_words.len() != 4 {
            return Err(PyValueError::new_err(format!(
                "rng_words must contain 4 state words, got {}",
                rng_words.len()
            )));
        }
        let ind_src = ind_flat
            .as_slice()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        if ind_src.len() != self.state_ind.len() {
            return Err(PyValueError::new_err(
                "checkpoint array does not match the live state size",
            ));
        }
        crate::model::validation::validate_state_values(ind_src, &[], tick)?;
        let mut params = self.params.clone();
        restore_ecology(&mut params, &self.blueprint, ecology)?;
        self.state_ind.copy_from_slice(ind_src);
        self.state_tick = tick;
        self.execution = crate::sessions::status::ExecutionStatus::Ready;
        self.phase = 0;
        let mut words = [0_u64; 4];
        words.copy_from_slice(&rng_words);
        self.rng = SessionRng::from_state_words(words);
        self.params = params;
        Ok(tick)
    }

    /// Restore the newest record-aligned checkpoint for *tick* (discrete).
    ///
    /// Discrete twin of ``AgeStructuredSession::restore_from_checkpoint``: the
    /// state array is written back in place, the RNG continues from the
    /// captured words, and the ecology section is restored.  Returns
    /// ``Some((tick, ecology))`` or ``None`` when the tick has no
    /// checkpoint.
    #[pyo3(signature = (tick))]
    fn restore_from_checkpoint<'py>(
        &mut self,
        py: Python<'py>,
        tick: i64,
    ) -> PyResult<Option<(i64, Bound<'py, PyDict>)>> {
        let found = self
            .checkpoints
            .iter()
            .rev()
            .find(|cp| cp.tick == tick)
            .cloned();
        let Some(cp) = found else {
            return Ok(None);
        };
        if cp.ind.len() != self.state_ind.len() {
            return Err(PyValueError::new_err(
                "checkpoint arrays do not match the live state size",
            ));
        }
        if let Some(store) = &self.history_store {
            store.lock().unwrap().restore_timeline(tick)?;
        }
        self.params.custom_slots[0] = cp.custom_slots.clone();
        self.state_ind.copy_from_slice(&cp.ind);
        self.state_tick = cp.tick;
        self.execution = cp.execution;
        self.phase = cp.phase;
        self.rng = SessionRng::from_state_words(cp.rng_words);
        self.params
            .ecology_restore_words(&self.blueprint, &cp.eco_scalars, &cp.eco_vectors)?;
        Ok(Some((
            cp.tick,
            crate::sessions::ecology_snapshot::ecology_snapshot(py, &self.params)?,
        )))
    }

    /// Project the current engine-owned state without exporting raw arrays.
    fn observe_current<'py>(
        &self,
        py: Python<'py>,
        mask: numpy::PyReadonlyArray1<'py, f64>,
        selected: Vec<usize>,
        collapse_age: bool,
        aggregate: bool,
    ) -> PyResult<(i64, Bound<'py, PyArray1<f64>>)> {
        let values = crate::output::observation::project(
            &self.state_ind,
            mask.as_slice()?,
            [
                1,
                self.blueprint.n_sexes,
                self.blueprint.n_ages,
                self.blueprint.n_ztypes,
            ],
            &selected,
            collapse_age,
            aggregate,
        )?;
        Ok((self.state_tick, PyArray1::from_vec(py, values)))
    }

    /// Attach the same native history object used by the public query adapter.
    fn bind_history(&mut self, history: PyRef<'_, HistoryStore>) {
        self.history_store = Some(std::sync::Arc::clone(&history.data));
    }

    /// Record a manual stable boundary and its complete native checkpoint.
    fn record_history(&mut self, continuation: bool) -> PyResult<()> {
        let shared = self
            .history_store
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("History is not initialized"))?;
        let mut history = shared.lock().unwrap();
        let added = history.record(self.state_tick, &self.state_ind, &[], continuation)?;
        if added {
            if let Some(boundary) = history.boundaries.back_mut() {
                boundary.1 = self.phase;
                boundary.2 = self.execution.name().to_owned();
            }
        }
        if added && history.raw {
            let (eco_scalars, eco_vectors) = self.params.ecology_snapshot_words()?;
            self.checkpoints
                .push(crate::kernels::age_structured::TickCheckpoint {
                    execution: self.execution,
                    phase: self.phase,
                    tick: self.state_tick,
                    ind: self.state_ind.clone(),
                    sperm: Vec::new(),
                    rng_words: self.rng.state_words(),
                    eco_scalars,
                    eco_vectors,
                    custom_slots: self.params.custom_slots[0].clone(),
                });
        }
        if let Some(row) = history.rows.front() {
            self.checkpoints.retain(|cp| cp.tick >= row[0] as i64);
        }
        Ok(())
    }

    /// Drop every stored checkpoint (paired with ``clear_history``).
    fn clear_checkpoints(&mut self) {
        self.checkpoints.clear();
    }

    /// Drop checkpoints captured after *retain_until_tick*.
    /// Drop checkpoints older than *from_tick* (history eviction pair).
    fn retain_checkpoints_from(&mut self, from_tick: i64) {
        self.checkpoints
            .retain(|checkpoint| checkpoint.tick >= from_tick);
    }

    fn truncate_checkpoints(&mut self, retain_until_tick: i64) {
        self.checkpoints.retain(|cp| cp.tick <= retain_until_tick);
    }
}

impl DiscreteGenerationSession {
    /// Snapshot the canonical ECO param values (deme 0 column).
    ///
    /// ## Returns
    /// A length-``N_ECO_PARAMS`` array in ``ECO_PARAM_COLUMNS`` order.
    fn eco_values(&self) -> [f64; crate::hooks::interpreter::N_ECO_PARAMS] {
        let mut values = [0.0; crate::hooks::interpreter::N_ECO_PARAMS];
        for (id, slot) in values.iter_mut().enumerate() {
            *slot = self.params.eco_value(id, 0);
        }
        values
    }
}

impl DiscreteGenerationSession {
    fn run_inner<'py>(
        &mut self,
        py: Python<'py>,
        n_ticks: i64,
        record_interval: i64,
        wf: bool,
        observation_mask: Option<PyReadonlyArray4<'py, f64>>,
        checkpoint_every: i64,
    ) -> PyResult<(i64, Bound<'py, PyArray2<f64>>, bool)> {
        // Assemble the config from the owned contracts at the batch entry,
        // then run a batch of discrete or WF ticks directly on the
        // session-owned state (plan S2: control parameters only).
        let cfg =
            DiscreteGenerationConfig::assemble(&self.blueprint, &self.params, &self.genetics)?;
        let mask_vec = match observation_mask {
            Some(mask) => Some(
                mask.as_slice()
                    .map_err(|err| PyValueError::new_err(err.to_string()))?
                    .to_vec(),
            ),
            None => None,
        };
        let mut eco_values = self.eco_values();
        let mut eco_ctx = Some(crate::kernels::age_structured::EcoCtx {
            bp: &self.blueprint,
            params: &mut self.params,
            genetics: &self.genetics,
            updated_genetics: None,
            phase: 0,
            deme: 0,
            tick: self.state_tick,
            journal: Vec::new(),
        });
        if let Some(shared) = self.history_store.as_ref().map(std::sync::Arc::clone) {
            let mut current_tick = self.state_tick;
            let mut stopped = false;
            // Native retention occurs at each boundary. No array proportional
            // to the requested run duration is allocated or crosses into Python.
            for step in 0..=n_ticks.max(0) {
                {
                    let mut store = shared.lock().unwrap();
                    if let Some(ctx) = eco_ctx.as_mut() {
                        let mut log = store.log.lock().unwrap();
                        for &(tick, parameter, old, new, phase) in &ctx.journal {
                            log.push(crate::output::parameter_log::LogEntry::from_phase(
                                (
                                    tick,
                                    crate::generated::ecology_parameters::ECO_PARAM_COLUMNS
                                        [parameter]
                                        .to_owned(),
                                    old,
                                    new,
                                ),
                                phase,
                                0,
                            ));
                        }
                        ctx.journal.clear();
                    }
                    if !stopped && record_interval > 0 && current_tick % record_interval == 0 {
                        let added = store.record(current_tick, &self.state_ind, &[], true)?;
                        if added && store.raw {
                            crate::kernels::age_structured::capture_checkpoint(
                                &self.rng,
                                &self.state_ind,
                                &[],
                                current_tick,
                                &eco_ctx,
                                &mut self.checkpoints,
                            )
                            .map_err(map_lifecycle_error)?;
                        }
                        if let Some(row) = store.rows.front() {
                            self.checkpoints.retain(|cp| cp.tick >= row[0] as i64);
                        }
                    }
                }
                if step == n_ticks || stopped {
                    break;
                }
                let (tick, _, _, was_stopped) = crate::kernels::discrete_generation::run_batch(
                    &mut self.rng,
                    &cfg,
                    &self.hooks,
                    &mut self.state_ind,
                    current_tick,
                    1,
                    0,
                    None,
                    wf,
                    &mut eco_values,
                    &mut eco_ctx,
                    0,
                    &mut self.checkpoints,
                )
                .map_err(|err| {
                    self.state_tick = current_tick;
                    if let Some(ctx) = eco_ctx.as_ref() {
                        self.phase = ctx.phase;
                    }
                    map_lifecycle_error(err)
                })?;
                if let Some(ctx) = eco_ctx.as_ref() {
                    self.phase = ctx.phase;
                }
                current_tick = tick;
                stopped = was_stopped;
            }
            self.state_tick = current_tick;
            if let Some(ctx) = eco_ctx.as_mut() {
                if let Some(genetics) = ctx.updated_genetics.take() {
                    self.genetics = genetics;
                }
            }
            return Ok((
                current_tick,
                PyArray2::<f64>::zeros(py, [0, 0], false),
                stopped,
            ));
        }

        let (final_tick, flat_history, n_rows, was_stopped) =
            crate::kernels::discrete_generation::run_batch(
                &mut self.rng,
                &cfg,
                &self.hooks,
                &mut self.state_ind,
                self.state_tick,
                n_ticks,
                record_interval,
                mask_vec.as_deref(),
                wf,
                &mut eco_values,
                &mut eco_ctx,
                checkpoint_every,
                &mut self.checkpoints,
            )
            .map_err(map_lifecycle_error)?;
        self.state_tick = final_tick;
        if let Some(ctx) = eco_ctx.as_mut() {
            self.eco_journal.append(&mut ctx.journal);
            if let Some(genetics) = ctx.updated_genetics.take() {
                self.genetics = genetics;
            }
        }
        let n_cols = flat_history.len().checked_div(n_rows).unwrap_or(0);
        let history = PyArray2::<f64>::zeros(py, [n_rows, n_cols], false);
        history
            .readwrite()
            .as_slice_mut()
            .map_err(|err| PyValueError::new_err(err.to_string()))?
            .copy_from_slice(&flat_history);
        Ok((final_tick, history, was_stopped))
    }
}
