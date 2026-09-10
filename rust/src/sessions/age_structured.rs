//! PyO3 session object owning the contract, RNG state, and the compiled CSR
//! hook program.

use crate::output::history::{HistoryStore, SharedHistory};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::hooks::interpreter::HookProgram;
use crate::kernels::rng::{new_rng, SessionRng};
use crate::model::blueprint::Blueprint;
use crate::model::ecology::session_get_tensor;
use crate::model::ecology::session_tensor_write;
use crate::model::ecology::EcologyParams;
use crate::model::genetics::GeneticsTensors;
use crate::sessions::ecology_snapshot::{ecology_snapshot, restore_ecology};

/// Validate and return mutable slices for age-structured state arrays.
///
/// ## Parameters
/// - `ind`: PyO3 writable array for individual counts.
/// - `sperm`: PyO3 writable array for sperm storage.
/// - `n_ages`: Expected age count.
/// - `n_ztypes`: Expected zygote type count.
///
/// ## Returns
/// A pair of mutable flat slices.
///
/// ## Errors
/// Returns ``PyValueError`` if shapes do not match the config.
/// Convert internal kernel error strings into ``PyRuntimeError``.
///
/// ## Parameters
/// - `err`: Internal error message.
///
/// ## Returns
/// A ``PyRuntimeError`` for Python callers.
fn map_lifecycle_error(err: String) -> PyErr {
    crate::hooks::transaction::map_error(err)
}

/// Snapshot tuple: ``(tick, ind_flat, sperm_flat, rng_words, ecology)``.
pub type AgeSnapshot<'py> = (
    i64,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Vec<u64>,
    Bound<'py, PyDict>,
);

/// PyO3-exported stateful session for the age-structured Rust backend.
///
/// Owns the frozen blueprint, the mutable params, the RNG, and a CSR hook
/// program.  The lifecycle kernels read the owned contracts directly at
/// every stage boundary, so [`EcologyParams`] writes take effect within the
/// same tick without rebuilding the session (the RNG keeps streaming).
#[pyclass(name = "EngineSession")]
pub struct AgeStructuredSession {
    blueprint: Blueprint,
    params: EcologyParams,
    genetics: GeneticsTensors,
    rng: SessionRng,
    hooks: HookProgram,
    /// Audited set_param transitions accumulated across tick/run calls;
    /// drained by the Python adapter after each run so ``params_log`` and
    /// the draft stay synchronized with the session-owned columns.
    eco_journal: Vec<crate::hooks::interpreter::EcoJournalRow>,
    /// Record-aligned full checkpoints: state + RNG words +
    /// ecology, captured at every recorded tick of a raw-mode run.  The
    /// public ``restore_checkpoint`` restores from here.
    checkpoints: Vec<crate::kernels::age_structured::TickCheckpoint>,
    /// Native history shared with the Python read-only adapter.
    history_store: Option<SharedHistory>,
    /// Session-owned live state: flattened individual counts,
    /// flattened sperm storage, and the authoritative tick.  ``run`` and
    /// the stage methods operate on these directly — Python passes control
    /// parameters only and reads back snapshots.
    state_ind: Vec<f64>,
    state_sperm: Vec<f64>,
    state_tick: i64,
    execution: crate::sessions::status::ExecutionStatus,
    phase: usize,
}

#[pymethods]
impl AgeStructuredSession {
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
        let mut result = 0;
        let operation = (|| -> Result<(), String> {
            result = self.hooks.execute_event(
                &mut self.rng,
                event as i64,
                &mut self.state_ind,
                &mut self.state_sperm,
                2,
                self.blueprint.n_ages,
                self.blueprint.n_ztypes,
                self.state_tick,
                self.blueprint.stochastic,
                self.blueprint.continuous_sampling,
                deme_id,
                &mut values,
                &mut ctx,
            )?;
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

    /// Create an age-structured session from the Python contract objects.
    ///
    /// ## Parameters
    /// - `blueprint`: A ``natal.contracts.Blueprint`` NamedTuple.
    /// - `params`: A ``natal.contracts.Params`` dataclass instance.
    /// - `seed`: RNG seed.
    ///
    /// ## Returns
    /// A new ``AgeStructuredSession`` owning copies of both contracts.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` when either contract is inconsistent.
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
        // Seed the session-owned state from the blueprint's frozen
        // initial population.  enable_rust_backend immediately follows
        // with set_state carrying the live Python state, so this is
        // the safe default rather than the authority.
        let state_ind = bp.initial_individual_count.to_vec();
        let state_sperm = {
            let want = bp.n_ages * bp.n_ztypes * bp.n_ztypes;
            if bp.initial_sperm_storage.len() == want {
                bp.initial_sperm_storage.to_vec()
            } else {
                vec![0.0; want]
            }
        };
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
            state_sperm,
            state_tick: 0,
            execution: crate::sessions::status::ExecutionStatus::Ready,
            phase: 0,
        })
    }

    /// Pull exactly the named contract fields from the Python params object.
    ///
    /// Directed refresh: the session is not rebuilt and the RNG keeps
    /// streaming; only the listed scalars and tensors are overwritten.
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

    /// Replace the declarative CSR hook program used by ``tick``.
    ///
    /// ## Parameters
    /// - `program`: Python CSR ``HookProgram``.
    fn set_hook_program(&mut self, program: &Bound<'_, PyAny>) -> PyResult<()> {
        self.hooks = HookProgram::from_python(program)?;
        Ok(())
    }

    /// Clear all declarative hooks.
    fn clear_hook_program(&mut self) {
        self.hooks = HookProgram::default();
    }

    /// Register Python callbacks interleaved with the CSR hooks.
    ///
    /// The callbacks are fired inside ``execute_event`` at the callback
    /// slots of the event's cross-type priority order (the program's
    /// ``python_callback_slots`` column marks those slots).
    ///
    /// ## Parameters
    /// - `first`: Callables of the ``first`` event (priority order).
    /// - `early`: Callables of the ``early`` event.
    /// - `late`: Callables of the ``late`` event.
    ///
    /// ## Notes
    /// Each callable receives ``(ind, sperm, tick, deme_id)`` where the two
    /// arrays are fresh copies of the current state; a nonzero return value
    /// stops the run.
    #[pyo3(signature = (first, early, late, finish=None))]
    fn set_python_callbacks(
        &mut self,
        first: Vec<Py<PyAny>>,
        early: Vec<Py<PyAny>>,
        late: Vec<Py<PyAny>>,
        finish: Option<Vec<Py<PyAny>>>,
    ) {
        self.hooks
            .install_callback_lists(vec![first, early, late, finish.unwrap_or_default()]);
    }

    /// Clear all Python callbacks.
    fn clear_python_callbacks(&mut self) {
        self.hooks.clear_callbacks();
    }

    /// Reseed the Rust RNG used by stochastic sampling.
    ///
    /// ## Parameters
    /// - `seed`: New seed.
    fn reseed(&mut self, seed: u64) {
        self.rng = new_rng(seed);
    }

    /// Drain the accumulated set_param audit journal.
    ///
    /// Returns and clears every ``(tick, param_id, old, new)`` transition
    /// committed by ``tick`` / ``run`` since the previous drain.  Rows are
    /// change-only (same-value commits are not journaled) and ordered by
    /// commit time.
    ///
    /// ## Returns
    /// A list of ``(tick, param_id, old, new)`` tuples.
    fn drain_eco_journal(&mut self) -> Vec<(i64, usize, f64, f64)> {
        std::mem::take(&mut self.eco_journal)
            .into_iter()
            .map(|(tick, id, old, new, _)| (tick, id, old, new))
            .collect()
    }

    /// Run the reproduction stage in place on the session-owned state.
    fn reproduction(&mut self) -> PyResult<()> {
        crate::kernels::age_structured::reproduction(
            &mut self.rng,
            &self.blueprint,
            &self.params,
            &self.genetics,
            0,
            &mut self.state_ind,
            &mut self.state_sperm,
        )
        .map_err(map_lifecycle_error)
    }

    /// Run the survival stage in place.
    ///
    /// ## Parameters
    /// - `ind`: Mutable individual-count array.
    /// - `sperm`: Mutable sperm-storage array.
    fn survival(&mut self) -> PyResult<()> {
        crate::kernels::age_structured::survival(
            &mut self.rng,
            &self.blueprint,
            &self.params,
            &self.genetics,
            0,
            &mut self.state_ind,
            &mut self.state_sperm,
        )
        .map_err(map_lifecycle_error)
    }

    /// Run the aging stage in place.
    ///
    /// ## Parameters
    /// - `ind`: Mutable individual-count array.
    /// - `sperm`: Mutable sperm-storage array.
    fn aging(&mut self) -> PyResult<()> {
        crate::kernels::age_structured::aging(
            &self.blueprint,
            &mut self.state_ind,
            &mut self.state_sperm,
        );
        Ok(())
    }

    /// Run one complete age-structured tick with declarative hooks.
    ///
    /// Returns ``0`` (continue) or ``1`` (a hook requested a stop).  The tick
    /// value is *not* advanced here; the Python adapter owns tick bookkeeping
    /// exactly like ``natal.engine.lifecycle``.
    ///
    /// ## Parameters
    /// - `ind`: Mutable individual-count array.
    /// - `sperm`: Mutable sperm-storage array.
    /// - `tick`: Current tick.
    /// - `deme_id`: Current deme id.
    ///
    /// ## Returns
    /// ``0`` or ``1``.
    #[pyo3(signature = (deme_id))]
    fn tick(&mut self, deme_id: i64) -> PyResult<i32> {
        self.execution.begin()?;
        let tick = self.state_tick;
        let result = self.run_with_eco(tick, deme_id);
        self.execution = match &result {
            Ok(0) => {
                self.phase = 0;
                crate::sessions::status::ExecutionStatus::Ready
            }
            Ok(_) => crate::sessions::status::ExecutionStatus::Stopped,
            Err(_) => crate::sessions::status::ExecutionStatus::Failed,
        };
        if matches!(result, Ok(0)) {
            // Only a completed tick advances; a stop freezes the tick
            // exactly like the batch run() loop.
            self.state_tick = tick + 1;
        }
        result.map_err(map_lifecycle_error)
    }

    /// Run up to ``n_ticks`` complete ticks inside Rust.
    ///
    /// The state arrays are mutated in place and the final tick value is
    /// returned. When ``record_interval > 0`` and no history is bound,
    /// retained native rows are exported as the legacy 2-D NumPy layout.
    ///
    /// ## Returns
    /// ``(final_tick, history, was_stopped)``.
    #[pyo3(signature = (n_ticks, record_interval, observation_mask=None, checkpoint_every=0))]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        n_ticks: i64,
        record_interval: i64,
        observation_mask: Option<PyReadonlyArray4<'py, f64>>,
        checkpoint_every: i64,
    ) -> PyResult<(i64, Bound<'py, PyArray2<f64>>, bool)> {
        self.execution.begin()?;
        let outcome = self.run_inner(
            py,
            n_ticks,
            record_interval,
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

    /// Install a full live state.
    ///
    /// The session owns the counts, sperm storage, and tick; Python pushes
    /// a fresh state exactly when the population-level state changes
    /// outside the engine (construction with a live state,
    /// ``import_state``, backend refresh).
    ///
    /// ## Parameters
    /// - `ind_flat`: Flattened individual counts (2 * n_ages * n_ztypes).
    /// - `sperm_flat`: Flattened sperm storage (n_ages * n_ztypes^2).
    /// - `tick`: The authoritative tick.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` when either vector has the wrong length.
    #[pyo3(signature = (ind_flat, sperm_flat, tick))]
    fn set_state(&mut self, ind_flat: Vec<f64>, sperm_flat: Vec<f64>, tick: i64) -> PyResult<()> {
        let want_ind = 2 * self.blueprint.n_ages * self.blueprint.n_ztypes;
        let want_sperm = self.blueprint.n_ages * self.blueprint.n_ztypes * self.blueprint.n_ztypes;
        if ind_flat.len() != want_ind {
            return Err(PyValueError::new_err(format!(
                "ind_flat must contain {want_ind} values, got {}",
                ind_flat.len()
            )));
        }
        if sperm_flat.len() != want_sperm {
            return Err(PyValueError::new_err(format!(
                "sperm_flat must contain {want_sperm} values, got {}",
                sperm_flat.len()
            )));
        }
        crate::model::validation::validate_state_values(&ind_flat, &sperm_flat, tick)?;
        self.state_ind = ind_flat;
        self.state_sperm = sperm_flat;
        self.state_tick = tick;
        self.execution = crate::sessions::status::ExecutionStatus::Ready;
        self.phase = 0;
        Ok(())
    }

    /// Return a point-in-time snapshot of the session-owned state.
    ///
    /// ## Returns
    /// ``(tick, ind_flat, sperm_flat)`` — fresh copies; mutating them
    /// never reaches the session.
    fn state_snapshot<'py>(
        &self,
        py: Python<'py>,
    ) -> (i64, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        (
            self.state_tick,
            PyArray1::from_slice(py, &self.state_ind),
            PyArray1::from_slice(py, &self.state_sperm),
        )
    }

    /// Sum the live per-sex counts without exporting the state arrays.
    ///
    /// ## Returns
    /// ``(total, female, male)`` — bitwise identical to the Python-side
    /// ``individual_count.sum()`` reductions over the same state (the
    /// reduction replicates NumPy's pairwise summation order).
    fn counts(&self) -> (f64, f64, f64) {
        let plane = self.blueprint.n_ages * self.blueprint.n_ztypes;
        let female = crate::kernels::state_reduce::numpy_pairwise_sum(&self.state_ind[..plane]);
        let male =
            crate::kernels::state_reduce::numpy_pairwise_sum(&self.state_ind[plane..2 * plane]);
        (
            crate::kernels::state_reduce::numpy_pairwise_sum(&self.state_ind),
            female,
            male,
        )
    }

    /// Capture a memory checkpoint of everything the session owns.
    ///
    /// The state arrays are Python-owned, so they are passed in and returned
    /// as flat copies; the RNG state is captured as four raw state words so
    /// ``restore -> run`` *continues* the exact stream (continuation, not a
    /// reseed).  The ecology section of the params is copied; the genetics
    /// section is never rolled back (a checkpoint is a save, not an
    /// uninstallation of genetic mods).
    ///
    /// ## Parameters
    /// - `individual_count`: Current state array.
    /// - `sperm_storage`: Current sperm array.
    /// - `tick`: Current tick value.
    ///
    /// ## Returns
    /// ``(tick, ind_flat, sperm_flat, rng_words, ecology)``.
    fn snapshot_state<'py>(&self, py: Python<'py>) -> PyResult<AgeSnapshot<'py>> {
        let ind_flat = PyArray1::from_slice(py, &self.state_ind);
        let sperm_flat = PyArray1::from_slice(py, &self.state_sperm);
        let rng_words = self.rng.state_words().to_vec();
        let ecology = ecology_snapshot(py, &self.params)?;
        Ok((self.state_tick, ind_flat, sperm_flat, rng_words, ecology))
    }

    /// Restore a memory checkpoint produced by [`AgeStructuredSession::snapshot_state`].    ///
    /// Writes the state arrays back in place, rebuilds the RNG from the
    /// captured state words (exact continuation), and restores the ecology
    /// params section.  The genetics section is untouched.
    ///
    /// ## Parameters
    /// - `individual_count`: Live state array, overwritten.
    /// - `sperm_storage`: Live sperm array, overwritten.
    /// - `tick`: Tick value carried by the checkpoint.
    /// - `ind_flat`: Checkpoint individual counts.
    /// - `sperm_flat`: Checkpoint sperm storage.
    /// - `rng_words`: Four captured RNG state words.
    /// - `ecology`: Checkpoint ecology dict.
    ///
    /// ## Returns
    /// The restored tick value.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` on array size, RNG word count, or ecology
    /// field mismatch.
    #[pyo3(signature = (tick, ind_flat, sperm_flat, rng_words, ecology))]
    fn restore_state(
        &mut self,
        tick: i64,
        ind_flat: numpy::PyReadonlyArray1<'_, f64>,
        sperm_flat: numpy::PyReadonlyArray1<'_, f64>,
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
        let sperm_src = sperm_flat
            .as_slice()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        if ind_src.len() != self.state_ind.len() || sperm_src.len() != self.state_sperm.len() {
            return Err(PyValueError::new_err(
                "checkpoint arrays do not match the live state size",
            ));
        }
        crate::model::validation::validate_state_values(ind_src, sperm_src, tick)?;
        let mut params = self.params.clone();
        restore_ecology(&mut params, &self.blueprint, ecology)?;
        self.state_ind.copy_from_slice(ind_src);
        self.state_sperm.copy_from_slice(sperm_src);
        self.state_tick = tick;
        self.execution = crate::sessions::status::ExecutionStatus::Ready;
        self.phase = 0;
        let mut words = [0_u64; 4];
        words.copy_from_slice(&rng_words);
        self.rng = SessionRng::from_state_words(words);
        self.params = params;
        Ok(tick)
    }

    /// Restore the newest record-aligned checkpoint for *tick*.
    ///
    /// The session's checkpoint store (captured at every recorded tick of
    /// raw-mode runs) is rolled back in full: state arrays are written
    /// back in place, the RNG continues from the captured words, and the
    /// ecology section is restored through the validated channels.  The
    /// genetics section is untouched (a checkpoint is a save, not an
    /// uninstallation of genetic mods).
    ///
    /// ## Parameters
    /// - `individual_count`: Live state array, overwritten.
    /// - `sperm_storage`: Live sperm array, overwritten.
    /// - `tick`: The recorded tick to roll back to.
    ///
    /// ## Returns
    /// ``Some((tick, ecology))`` with the restored ecology dict (for the
    /// Python draft write-back), or ``None`` when no checkpoint exists at
    /// *tick*.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` on array size mismatch.
    #[pyo3(signature = (tick))]
    fn restore_from_checkpoint<'py>(
        &mut self,
        py: Python<'py>,
        tick: i64,
    ) -> PyResult<Option<(i64, Bound<'py, PyDict>)>> {
        // Clone out of the store first: the restore below needs &mut self
        // while the search borrows it immutably.
        let found = self
            .checkpoints
            .iter()
            .rev()
            .find(|cp| cp.tick == tick)
            .cloned();
        let Some(cp) = found else {
            return Ok(None);
        };
        if cp.ind.len() != self.state_ind.len() || cp.sperm.len() != self.state_sperm.len() {
            return Err(PyValueError::new_err(
                "checkpoint arrays do not match the live state size",
            ));
        }
        if let Some(store) = &self.history_store {
            store.lock().unwrap().restore_timeline(tick)?;
        }
        self.params.custom_slots[0] = cp.custom_slots.clone();
        self.state_ind.copy_from_slice(&cp.ind);
        self.state_sperm.copy_from_slice(&cp.sperm);
        self.state_tick = cp.tick;
        self.execution = cp.execution;
        self.phase = cp.phase;
        self.rng = SessionRng::from_state_words(cp.rng_words);
        self.params
            .ecology_restore_words(&self.blueprint, &cp.eco_scalars, &cp.eco_vectors)?;
        Ok(Some((cp.tick, ecology_snapshot(py, &self.params)?)))
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
        let added = history.record(
            self.state_tick,
            &self.state_ind,
            &self.state_sperm,
            continuation,
        )?;
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
                    sperm: self.state_sperm.to_vec(),
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
    ///
    /// Paired with the history truncate after a restore: reruns from the
    /// restored tick overwrite later ticks, so stale future checkpoints
    /// must not survive.
    /// Drop checkpoints older than *from_tick* (history eviction pair).
    ///
    /// When the recording plan evicts the oldest history rows, their
    /// checkpoints stop being restorable and must be dropped with them so
    /// the store stays bounded.
    fn retain_checkpoints_from(&mut self, from_tick: i64) {
        self.checkpoints
            .retain(|checkpoint| checkpoint.tick >= from_tick);
    }

    fn truncate_checkpoints(&mut self, retain_until_tick: i64) {
        self.checkpoints.retain(|cp| cp.tick <= retain_until_tick);
    }
}

impl AgeStructuredSession {
    /// Snapshot the canonical ECO param values for one deme column.
    ///
    /// ## Parameters
    /// - `deme`: Deme column to read.
    ///
    /// ## Returns
    /// A length-``N_ECO_PARAMS`` array in ``ECO_PARAM_COLUMNS`` order.
    fn eco_values(&self, deme: usize) -> [f64; crate::hooks::interpreter::N_ECO_PARAMS] {
        let mut values = [0.0; crate::hooks::interpreter::N_ECO_PARAMS];
        for (id, slot) in values.iter_mut().enumerate() {
            *slot = self.params.eco_value(id, deme);
        }
        values
    }

    /// Run one structured tick with a live ECO scratch and write-back.
    ///
    /// The EcoCtx commits set_param writes into the deme's ecology column
    /// at every event boundary; the lifecycle stages read the committed
    /// columns directly, so the same tick's later stages observe them — the
    /// same granularity as the Python executor.  The tick's journal rows are
    /// drained into the session audit trail.
    fn run_with_eco(&mut self, tick: i64, deme_id: i64) -> Result<i32, String> {
        // Split borrows explicitly: the kernel mutates the session-owned
        // state buffers while the EcoCtx borrows the contracts.
        let Self {
            rng,
            hooks,
            state_ind,
            state_sperm,
            blueprint,
            params,
            genetics,
            eco_journal,
            ..
        } = self;
        let deme = 0;
        let mut eco_values = params.eco_values_row(deme);
        let mut ctx = Some(crate::kernels::age_structured::EcoCtx {
            bp: blueprint,
            params,
            genetics,
            updated_genetics: None,
            phase: 0,
            deme,
            tick,
            journal: Vec::new(),
        });
        let result = crate::kernels::age_structured::run_tick(
            rng,
            blueprint,
            hooks,
            state_ind,
            state_sperm,
            tick,
            deme_id,
            &mut eco_values,
            &mut ctx,
            None,
        );
        if let Some(ctx) = ctx.as_mut() {
            eco_journal.append(&mut ctx.journal);
            self.phase = ctx.phase;
        }
        let updated = ctx.as_mut().and_then(|ctx| ctx.updated_genetics.take());
        drop(ctx);
        if let Some(updated) = updated {
            self.genetics = updated;
        }
        self.hooks
            .callback_commits
            .lock()
            .expect("callback queue poisoned")
            .clear();
        result
    }
}

/// Extract an int32/int64 1-D array into a ``Vec<i64>``.
///
/// ## Parameters
/// - `obj`: Python object.
/// - `name`: Attribute name.
///
/// ## Returns
/// A ``Vec<i64>`` copy.
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

/// Extract a float64 1-D array into a ``Vec<f64>``.
///
/// ## Parameters
/// - `obj`: Python object.
/// - `name`: Attribute name.
///
/// ## Returns
/// A ``Vec<f64>`` copy.
fn extract_f64_array(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    use numpy::PyReadonlyArray1;
    let array = obj.getattr(name)?.extract::<PyReadonlyArray1<'_, f64>>()?;
    Ok(array.as_slice()?.to_vec())
}

/// Extract a boolean/float64 1-D array into a ``Vec<bool>``.
///
/// ## Parameters
/// - `obj`: Python object.
/// - `name`: Attribute name.
///
/// ## Returns
/// A ``Vec<bool>`` copy.
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

/// Extract an int64 scalar, accepting 0-d NumPy arrays.
///
/// ## Parameters
/// - `obj`: Python object.
/// - `name`: Attribute name.
///
/// ## Returns
/// The scalar as ``i64``.
fn extract_i64_scalar(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<i64> {
    let value = obj.getattr(name)?;
    if let Ok(scalar) = value.extract::<i64>() {
        return Ok(scalar);
    }
    value.call_method0("item")?.extract::<i64>()
}

/// Extract a Python bool attribute.
///
/// ## Parameters
/// - `obj`: Python object.
/// - `name`: Attribute name.
///
/// ## Returns
/// The attribute as ``bool``.
fn extract_bool_scalar(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    obj.getattr(name)?.extract::<bool>()
}

impl HookProgram {
    /// Deserialize a Python CSR ``HookProgram`` into the Rust mirror.
    ///
    /// ## Parameters
    /// - `program`: Python ``HookProgram``.
    ///
    /// ## Returns
    /// A ``HookProgram`` with copied flat arrays.
    pub(crate) fn from_python(program: &Bound<'_, PyAny>) -> PyResult<Self> {
        let n_hooks = extract_i64_scalar(program, "n_hooks")?;
        let op_types = extract_i64_array(program, "op_types_data")?;
        let has_set_param = extract_bool_scalar(program, "has_set_param")?
            || op_types.contains(&crate::hooks::interpreter::OP_SET_PARAM_PUBLIC);
        // The callback slot column drives cross-type priority interleaving.
        // Reject non-empty programs without it: silently treating callback
        // slots as zero-op CSR slots would drop the callbacks entirely.
        let python_callback_slots = match extract_i64_array(program, "python_callback_slots") {
            Ok(slots) => slots,
            Err(_) if n_hooks == 0 => Vec::new(),
            Err(_) => {
                return Err(PyValueError::new_err(
                    "hook program is missing 'python_callback_slots'; rebuild the \
                     population program with the current natal version",
                ));
            }
        };
        if python_callback_slots.len() != n_hooks as usize {
            return Err(PyValueError::new_err(format!(
                "python_callback_slots has {} entries but n_hooks is {n_hooks}",
                python_callback_slots.len()
            )));
        }
        Ok(Self {
            n_events: extract_i64_scalar(program, "n_events")?,
            n_hooks,
            hook_offsets: extract_i64_array(program, "hook_offsets")?,
            op_offsets: extract_i64_array(program, "op_offsets")?,
            op_types,
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
            sp_param_ids: extract_i64_array(program, "sp_param_ids")?,
            sp_every: extract_i64_array(program, "sp_every")?,
            sp_start: extract_i64_array(program, "sp_start")?,
            rpn_offsets: extract_i64_array(program, "rpn_offsets")?,
            rpn_kinds: extract_i64_array(program, "rpn_kinds")?,
            rpn_payload: extract_i64_array(program, "rpn_payload")?,
            sp_literals: extract_f64_array(program, "sp_literals")?,
            convert_source_z: extract_i64_array(program, "convert_source_z")?,
            convert_target_z: extract_i64_array(program, "convert_target_z")?,
            python_callback_slots,
            has_set_param,
            ..Default::default()
        })
    }
}

impl AgeStructuredSession {
    fn run_inner<'py>(
        &mut self,
        py: Python<'py>,
        n_ticks: i64,
        record_interval: i64,
        observation_mask: Option<PyReadonlyArray4<'py, f64>>,
        checkpoint_every: i64,
    ) -> PyResult<(i64, Bound<'py, PyArray2<f64>>, bool)> {
        // Run the Rust batch loop directly on the session-owned state and
        // copy the flattened history into a NumPy 2-D array.  The lifecycle
        // kernels read the owned contracts at every stage boundary; Python
        // passes control parameters only (the session owns counts and tick).
        let mask_vec = match observation_mask {
            Some(mask) => Some(
                mask.as_slice()
                    .map_err(|err| PyValueError::new_err(err.to_string()))?
                    .to_vec(),
            ),
            None => None,
        };

        let mut eco_values = self.eco_values(0);

        // The borrowed EcoCtx commits set_param writes at every event
        // boundary; unbound callers drain its journal after the run.
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

        let bound = self.history_store.is_some();
        let shared = if let Some(shared) = self.history_store.as_ref().map(std::sync::Arc::clone) {
            shared
        } else {
            let dimensions = [1, 2, self.blueprint.n_ages, self.blueprint.n_ztypes];
            let width = if mask_vec.is_some() {
                1
            } else {
                1 + self.state_ind.len() + self.state_sperm.len()
            };
            let shared = crate::output::history::HistoryData::transient(
                width,
                dimensions,
                mask_vec.is_none(),
            );
            if let Some(mask) = mask_vec {
                shared.lock().unwrap().configure_observation_slice(mask)?;
            }
            shared
        };
        let mut current_tick = self.state_tick;
        let mut stopped = false;
        for step in 0..=n_ticks.max(0) {
            {
                let mut store = shared.lock().unwrap();
                if bound {
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
                }
                if !stopped && record_interval > 0 && current_tick % record_interval == 0 {
                    let added =
                        store.record(current_tick, &self.state_ind, &self.state_sperm, bound)?;
                    if added
                        && ((bound && store.raw)
                            || (!bound
                                && checkpoint_every > 0
                                && current_tick % checkpoint_every == 0))
                    {
                        crate::kernels::age_structured::capture_checkpoint(
                            &self.rng,
                            &self.state_ind,
                            &self.state_sperm,
                            current_tick,
                            &eco_ctx,
                            &mut self.checkpoints,
                        )
                        .map_err(map_lifecycle_error)?;
                    }
                    if bound {
                        if let Some(row) = store.rows.front() {
                            self.checkpoints.retain(|cp| cp.tick >= row[0] as i64);
                        }
                    }
                }
            }
            if step == n_ticks.max(0) || stopped {
                break;
            }
            let result = crate::kernels::age_structured::run_tick(
                &mut self.rng,
                &self.blueprint,
                &self.hooks,
                &mut self.state_ind,
                &mut self.state_sperm,
                current_tick,
                0,
                &mut eco_values,
                &mut eco_ctx,
                None,
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
            if result != 0 {
                stopped = true;
            } else {
                current_tick += 1;
            }
        }
        self.state_tick = current_tick;
        if let Some(ctx) = eco_ctx.as_mut() {
            if !bound {
                self.eco_journal.append(&mut ctx.journal);
            }
            if let Some(genetics) = ctx.updated_genetics.take() {
                self.genetics = genetics;
            }
        }
        if bound {
            return Ok((
                current_tick,
                PyArray2::<f64>::zeros(py, [0, 0], false),
                stopped,
            ));
        }
        let (flat_history, n_rows) = shared.lock().unwrap().flat_rows();
        let n_cols = flat_history.len().checked_div(n_rows).unwrap_or(0);
        let history = PyArray2::<f64>::zeros(py, [n_rows, n_cols], false);
        history
            .readwrite()
            .as_slice_mut()
            .map_err(|err| PyValueError::new_err(err.to_string()))?
            .copy_from_slice(&flat_history);
        Ok((current_tick, history, stopped))
    }
}
