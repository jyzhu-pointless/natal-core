//! PyO3 session object for the discrete-generation / Wright-Fisher backend.

use crate::output::history::{HistoryStore, SharedHistory};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::hooks::interpreter::HookProgram;
use crate::kernels::discrete_generation;
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
/// program.  The lifecycle kernels read the owned contracts directly at
/// every stage boundary, so params writes take effect within the same tick
/// without rebuilding the session (the RNG keeps streaming).
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
    /// Record-aligned full checkpoints — the discrete twin
    /// of ``AgeStructuredSession::checkpoints``.
    checkpoints: Vec<crate::kernels::age_structured::TickCheckpoint>,
    /// Native history shared with the Python read-only adapter.
    history_store: Option<SharedHistory>,
    /// Session-owned live state: flattened counts plus the
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
        // Event ids index the four fixed hook slots; reject unknown ids before
        // touching any session state.
        if event >= 4 {
            return Err(PyValueError::new_err("unknown hook event"));
        }
        // Live ECO scratch for set_param: hooks write scalars here and
        // ``commit`` folds them into the single discrete ecology column.
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
            // Discrete sessions carry no sperm storage, so the sperm argument
            // stays an empty slice; n_ages is fixed at 2 (juvenile, adult) and
            // the adult index is 1.
            result = self.hooks.execute_event(
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
                &mut ctx,
            )?;
            if let Some(context) = ctx.as_mut() {
                context.commit(&values)?;
            }
            Ok(())
        })();
        // Route audit rows: a bound HistoryStore owns the shared parameter
        // log, while unbound callers accumulate into ``eco_journal`` for
        // ``drain_eco_journal``.
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
        // A callback may have produced a validated genetics candidate; take it
        // out of the borrowed context and move it into the session.
        let genetics = ctx
            .as_mut()
            .and_then(|context| context.updated_genetics.take());
        drop(ctx);
        if let Some(genetics) = genetics {
            self.genetics = genetics;
        }
        // A failed event marks the session Failed; a nonzero hook result marks
        // it Stopped but keeps the mutated state and the advanced RNG.
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
        // Freeze the blueprint and split the contract into session-owned
        // pieces; all three are validated before any state is seeded.
        let bp = Blueprint::from_python(blueprint)?;
        // Panmictic sessions carry one deme: length-1 ecology columns and
        // the session-owned genetics tables split out of the contract.
        let pr = EcologyParams::from_python(params, 1)?;
        let genetics = GeneticsTensors::from_python(params)?;
        bp.validate()?;
        pr.validate(&bp)?;
        genetics.validate(&bp)?;
        // Validate the discrete normalization once at construction.
        discrete_generation::validate_discrete_shape(&bp)?;
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
        // ``extreme_speed_mode`` selects a fixed enum (0..=3); reject an
        // out-of-range value before mutating the frozen blueprint.
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

    /// Register Python callbacks interleaved with the CSR hooks by the
    /// event's cross-type priority order (see ``AgeStructuredSession``).
    #[pyo3(signature = (first, early, late, finish=None))]
    fn set_python_callbacks(
        &mut self,
        first: Vec<Py<PyAny>>,
        early: Vec<Py<PyAny>>,
        late: Vec<Py<PyAny>>,
        finish: Option<Vec<Py<PyAny>>>,
    ) {
        // Four callback lists in event order; ``None`` for finish installs an
        // empty finish slot rather than clearing the other lists.
        self.hooks
            .install_callback_lists(vec![first, early, late, finish.unwrap_or_default()]);
    }

    /// Clear all Python callbacks.
    fn clear_python_callbacks(&mut self) {
        self.hooks.clear_callbacks();
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
        // ``mem::take`` leaves the session store empty (drain semantics); the
        // phase element is dropped because the Python contract carries only
        // tick/param/old/new.
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
        // A single tick is a one-step batch with recording disabled; ``wf``
        // selects the fused Wright-Fisher update instead of the staged tick.
        self.run(py, 1, 0, wf, None, 0)
            .map(|(_, _, stopped)| i32::from(stopped))
    }

    /// Run up to ``n_ticks`` discrete/WF ticks inside Rust with optional recording.
    ///
    /// ## Returns
    /// ``(final_tick, history, was_stopped)``.
    #[allow(clippy::too_many_arguments)] // PyO3 boundary exposes the complete run control surface.
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
        // Batch status mirrors ``tick``: stop → Stopped, a clean batch → Ready
        // with the phase cursor reset, error → Failed.
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
    /// Install a full live state (session-owned state; discrete twin).
    ///
    /// ## Errors
    /// Returns ``PyValueError`` when the vector has the wrong length.
    #[pyo3(signature = (ind_flat, tick))]
    fn set_state(&mut self, ind_flat: Vec<f64>, tick: i64) -> PyResult<()> {
        // Discrete layout: [sex, age, ztype] with two sexes and two ages
        // (juvenile, adult), so the flat length is 2 * 2 * n_ztypes.
        let want = 2 * 2 * self.blueprint.n_ztypes;
        if ind_flat.len() != want {
            return Err(PyValueError::new_err(format!(
                "ind_flat must contain {want} values, got {}",
                ind_flat.len()
            )));
        }
        // Reject NaN/negative counts and negative ticks up front, then move the
        // buffer into the session and reset the lifecycle to Ready because
        // Python pushed a fresh, un-run state.  Sperm is empty for discrete.
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

    /// Sum the live per-sex counts without exporting the state arrays.
    ///
    /// ## Returns
    /// ``(total, female, male)`` — bitwise identical to the Python-side
    /// ``individual_count.sum()`` reductions over the same state (the
    /// reduction replicates NumPy's pairwise summation order).
    fn counts(&self) -> (f64, f64, f64) {
        // Flattened state puts the sex plane outermost: [female | male] blocks
        // of n_ages * n_ztypes, so the two sex halves are contiguous slices.
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

    /// Read the authoritative session tick without exporting state arrays.
    fn current_tick(&self) -> i64 {
        self.state_tick
    }

    fn snapshot_state<'py>(&self, py: Python<'py>) -> PyResult<DiscreteSnapshot<'py>> {
        let ind_flat = PyArray1::from_slice(py, &self.state_ind);
        // Four raw state words let ``restore_state`` continue the exact stream
        // (continuation), not reseed it.
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
        // A checkpoint always carries exactly the four Xoshiro256++ words.
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
        // Restore the ecology section into a clone so a malformed checkpoint
        // aborts before the live params are touched.
        let mut params = self.params.clone();
        restore_ecology(&mut params, &self.blueprint, ecology)?;
        // Overwrite the owned buffer in place, drop back to Ready, then
        // rebuild the generator from the captured words (exact continuation)
        // and commit the validated params last.
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
        // A checkpoint captured under a different blueprint shape cannot be
        // folded into this state, so refuse instead of truncating.
        if cp.ind.len() != self.state_ind.len() {
            return Err(PyValueError::new_err(
                "checkpoint arrays do not match the live state size",
            ));
        }
        // Rewind the bound history timeline (rows, cursors, and logs) to the
        // restored tick so a rerun can re-record the future ticks.
        if let Some(store) = &self.history_store {
            store.lock().unwrap().restore_timeline(tick)?;
        }
        // Custom slots travel with the checkpoint; the genetics section stays
        // out of scope (a checkpoint saves, it does not uninstall mods).
        self.params.custom_slots[0] = cp.custom_slots.clone();
        self.state_ind.copy_from_slice(&cp.ind);
        self.state_tick = cp.tick;
        self.execution = cp.execution;
        self.phase = cp.phase;
        // Bulk-restore the ecology from wire words that were captured from this
        // same blueprint; per-field validation still runs inside.
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
        // Dimensions follow the [sex, age, ztype] flat layout; ``project``
        // applies the compiled observation mask and returns an owned vector.
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
        // Share ownership of the row store with the Python HistoryStore; the
        // session only records into it, it never owns or frees the log.
        self.history_store = Some(std::sync::Arc::clone(&history.data));
    }

    /// Record a manual stable boundary and its complete native checkpoint.
    fn record_history(&mut self, continuation: bool) -> PyResult<()> {
        let shared = self
            .history_store
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("History is not initialized"))?;
        let mut history = shared.lock().unwrap();
        // ``record`` returns false when the exact boundary is already the
        // latest row (idempotent continuation); stamping then targets the
        // existing boundary with the current phase and status.
        let added = history.record(self.state_tick, &self.state_ind, &[], continuation)?;
        if added {
            if let Some(boundary) = history.boundaries.back_mut() {
                boundary.1 = self.phase;
                boundary.2 = self.execution.name().to_owned();
            }
        }
        // Raw-mode recording pairs every row with a full checkpoint (state,
        // RNG words, ecology, custom slots); discrete sessions leave the sperm
        // vector empty.
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
        // Restorable ticks must be a subset of retained history rows: drop
        // checkpoints whose row was evicted by the bounded store.
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
        // Keep checkpoints at or before the restored tick; a rerun overwrites
        // the later ticks, so their stale checkpoints must not survive.
        self.checkpoints.retain(|cp| cp.tick <= retain_until_tick);
    }
}

impl DiscreteGenerationSession {
    /// Snapshot the canonical ECO param values (deme 0 column).
    ///
    /// ## Returns
    /// A length-``N_ECO_PARAMS`` array in ``ECO_PARAM_COLUMNS`` order.
    fn eco_values(&self) -> [f64; crate::hooks::interpreter::N_ECO_PARAMS] {
        // Flatten the single deme's ecology into the canonical
        // ``ECO_PARAM_COLUMNS`` order the CSR interpreter indexes by id.
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
        // Run a batch of discrete or WF ticks directly on the session-owned
        // state (control parameters only).  The lifecycle kernels read the
        // owned contracts at every stage boundary.
        // Copy the mask into a Vec: the transient history store may outlive
        // the borrow of the Python array.
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
        // ``bound`` means the Python adapter already owns the row store (with
        // raw/observation mode configured); unbound calls get a transient
        // store sized for either full raw rows or a projected slice.
        let bound = self.history_store.is_some();
        let shared = if let Some(shared) = self.history_store.as_ref().map(std::sync::Arc::clone) {
            shared
        } else {
            let shared = crate::output::history::HistoryData::transient(
                if mask_vec.is_some() {
                    1
                } else {
                    1 + self.state_ind.len()
                },
                [1, 2, 2, self.blueprint.n_ztypes],
                mask_vec.is_none(),
            );
            if let Some(mask) = mask_vec {
                shared.lock().unwrap().configure_observation_slice(mask)?;
            }
            shared
        };
        // Record-then-step loop: the top of iteration ``step`` records the
        // current tick, so a run of n_ticks emits boundaries for the starting
        // tick through ``start + n_ticks`` (the pre-run state is always one of
        // them), and a hook stop freezes the tick.  ``max(0)`` clamps a
        // negative request to a no-op run.
        let mut current_tick = self.state_tick;
        let mut was_stopped = false;
        for step in 0..=n_ticks.max(0) {
            {
                let mut store = shared.lock().unwrap();
                // Bound mode flushes the previous tick's set_param journal into
                // the shared log under the store lock; unbound mode keeps the
                // rows in ``eco_journal`` and drains them after the run.
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
                // Record only while running and only on aligned ticks;
                // ``record`` also dedups the exact continuation boundary.
                if !was_stopped && record_interval > 0 && current_tick % record_interval == 0 {
                    let added = store.record(current_tick, &self.state_ind, &[], bound)?;
                    // Bound raw runs checkpoint every recorded tick; unbound
                    // runs checkpoint every ``checkpoint_every`` ticks (0
                    // disables).  ``capture_checkpoint`` ignores repeats.
                    if added
                        && ((bound && store.raw)
                            || (!bound
                                && checkpoint_every > 0
                                && current_tick % checkpoint_every == 0))
                    {
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
                    // Restorable checkpoints must track the bounded history:
                    // drop any whose row was evicted from the store.
                    if bound {
                        if let Some(row) = store.rows.front() {
                            self.checkpoints.retain(|cp| cp.tick >= row[0] as i64);
                        }
                    }
                }
            }
            // The terminal iteration only records; a hook stop also leaves the
            // state frozen at the stopping tick, already recorded above.
            if step == n_ticks.max(0) || was_stopped {
                break;
            }
            // ``wf`` selects the fused Wright-Fisher tick; otherwise the staged
            // six-boundary discrete tick runs (same order as the
            // age-structured engine, with an empty sperm slice).
            let result = if wf {
                (|| -> Result<i32, String> {
                    let hook_result = self.hooks.execute_event(
                        &mut self.rng,
                        0,
                        &mut self.state_ind,
                        &mut [],
                        2,
                        2,
                        self.blueprint.n_ztypes,
                        current_tick,
                        self.blueprint.stochastic,
                        self.blueprint.continuous_sampling,
                        0,
                        &mut eco_values,
                        &mut eco_ctx,
                    )?;
                    // Re-stamp the context for this tick and commit the first
                    // event's writes; the WF update reads the committed columns.
                    if let Some(ctx) = eco_ctx.as_mut() {
                        ctx.tick = current_tick;
                        ctx.commit(&eco_values)?;
                    }
                    if hook_result != 0 {
                        Ok(hook_result)
                    } else {
                        // The fused WF update reads the committed columns and
                        // candidate genetics through the live context.
                        let ctx = eco_ctx.as_ref().expect("wf tick lends an eco context");
                        // Prefer a callback-staged genetics candidate over the
                        // live table: it carries this event's validated edits.
                        let genetics = ctx.updated_genetics.as_ref().unwrap_or(ctx.genetics);
                        crate::kernels::discrete_generation::run_wf_tick(
                            &mut self.rng,
                            &self.blueprint,
                            // &mut EcologyParams coerces to the shared read.
                            ctx.params,
                            genetics,
                            ctx.deme,
                            &mut self.state_ind,
                        )
                        .map(|_| 0)
                    }
                })()
            } else {
                crate::kernels::discrete_generation::run_tick(
                    &mut self.rng,
                    &self.blueprint,
                    &self.hooks,
                    &mut self.state_ind,
                    current_tick,
                    0,
                    &mut eco_values,
                    &mut eco_ctx,
                    None,
                )
            }
            .map_err(|err| {
                self.state_tick = current_tick;
                if let Some(ctx) = eco_ctx.as_ref() {
                    self.phase = ctx.phase;
                }
                map_lifecycle_error(err)
            })?;
            // Publish the within-tick phase cursor so ``execution_state`` can
            // describe a partial (stopped/failed) tick.
            if let Some(ctx) = eco_ctx.as_ref() {
                self.phase = ctx.phase;
            }
            if result != 0 {
                was_stopped = true;
            } else {
                current_tick += 1;
            }
        }
        // Publish the final tick (frozen at the stopping tick when stopped),
        // drain the remaining journal for unbound callers, and adopt any
        // leftover genetics candidate from the last event.
        self.state_tick = current_tick;
        if let Some(ctx) = eco_ctx.as_mut() {
            if !bound {
                self.eco_journal.append(&mut ctx.journal);
            }
            if let Some(genetics) = ctx.updated_genetics.take() {
                self.genetics = genetics;
            }
        }
        // Bound callers read rows from the shared store, so the legacy 2-D
        // export is empty.
        if bound {
            return Ok((
                current_tick,
                PyArray2::<f64>::zeros(py, [0, 0], false),
                was_stopped,
            ));
        }
        // Materialize the retained rows as a dense [n_rows, n_cols] array;
        // ``checked_div`` yields 0 columns for an empty store.
        let (flat_history, n_rows) = shared.lock().unwrap().flat_rows();
        let n_cols = flat_history.len().checked_div(n_rows).unwrap_or(0);
        let history = PyArray2::<f64>::zeros(py, [n_rows, n_cols], false);
        history
            .readwrite()
            .as_slice_mut()
            .map_err(|err| PyValueError::new_err(err.to_string()))?
            .copy_from_slice(&flat_history);
        Ok((current_tick, history, was_stopped))
    }
}
