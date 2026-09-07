//! PyO3 session object owning the contract, RNG state, and the compiled CSR
//! hook program.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray4};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::config::SimConfig;
use crate::contract::{session_get_tensor, session_tensor_write, Blueprint, Params, TensorSet};
use crate::hooks::HookProgram;
use crate::lifecycle;
use crate::rng::{new_rng, SessionRng};

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
    PyRuntimeError::new_err(err)
}

/// Ecology scalar field names carried by a memory checkpoint —
/// generated from the jsonc wire order with ``growth_mode`` and
/// ``external_expected_eggs`` appended (plan 5.4: one source, no
/// hand-written copies).
use crate::eco_param_wire::ECOLOGY_SCALARS;

/// Ecology vector field names carried by a memory checkpoint — the
/// canonical definition lives next to ``Params`` in ``contract.rs``.
use crate::contract::ECOLOGY_VECTORS;

/// Copy the ecology section of *params* into a fresh Python dict.
///
/// The genetics section is deliberately excluded: a memory checkpoint is a
/// save, not an uninstallation of genetic mods.
pub(crate) fn ecology_snapshot<'py>(
    py: Python<'py>,
    params: &Params,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for name in ECOLOGY_SCALARS {
        dict.set_item(name, params.get_scalar(name)?)?;
    }
    for name in ECOLOGY_VECTORS {
        dict.set_item(name, params.get_tensor(py, name)?)?;
    }
    Ok(dict)
}

/// Write an ecology snapshot dict back into *params*.
///
/// ## Errors
/// Returns ``PyValueError`` when a vector has the wrong size; each
/// ``tensor_write`` validates before committing, so the previous contents
/// are preserved for the failing field.
pub(crate) fn restore_ecology(
    params: &mut Params,
    bp: &Blueprint,
    ecology: &Bound<'_, PyAny>,
) -> PyResult<()> {
    for name in ECOLOGY_SCALARS {
        let value: f64 = ecology.get_item(name)?.extract()?;
        params.apply(HashMap::from([(name.to_string(), value)]))?;
    }
    for name in ECOLOGY_VECTORS {
        let values: Vec<f64> = ecology
            .get_item(name)?
            .extract::<numpy::PyReadonlyArray1<'_, f64>>()?
            .as_slice()?
            .to_vec();
        params.tensor_write(bp, name, values)?;
    }
    Ok(())
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
/// program.  A flat [`SimConfig`] is assembled from the contracts at every
/// tick-batch entry point, so [`Params`] writes take effect on the next
/// batch without rebuilding the session (the RNG keeps streaming).
#[pyclass(name = "EngineSession")]
pub struct EngineSession {
    blueprint: Blueprint,
    params: Params,
    genetics: TensorSet,
    rng: SessionRng,
    hooks: HookProgram,
    /// Audited set_param transitions accumulated across tick/run calls;
    /// drained by the Python adapter after each run so ``params_log`` and
    /// the draft stay synchronized with the session-owned columns.
    eco_journal: Vec<crate::hooks::EcoJournalRow>,
    /// Record-aligned full checkpoints (plan 13.1 R3): state + RNG words +
    /// ecology, captured at every recorded tick of a raw-mode run.  The
    /// public ``restore_checkpoint`` restores from here.
    checkpoints: Vec<lifecycle::TickCheckpoint>,
    /// Session-owned live state (plan S2): flattened individual counts,
    /// flattened sperm storage, and the authoritative tick.  ``run`` and
    /// the stage methods operate on these directly — Python passes control
    /// parameters only and reads back snapshots.
    state_ind: Vec<f64>,
    state_sperm: Vec<f64>,
    state_tick: i64,
}

#[pymethods]
impl EngineSession {
    /// Create an age-structured session from the Python contract objects.
    ///
    /// ## Parameters
    /// - `blueprint`: A ``natal.contracts.Blueprint`` NamedTuple.
    /// - `params`: A ``natal.contracts.Params`` dataclass instance.
    /// - `seed`: RNG seed.
    ///
    /// ## Returns
    /// A new ``EngineSession`` owning copies of both contracts.
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
        let pr = Params::from_python(params, 1)?;
        let genetics = TensorSet::from_python(params)?;
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
            state_ind,
            state_sperm,
            state_tick: 0,
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

    /// Register Python callbacks fired at the first/early/late event
    /// boundaries after the CSR hooks ran.
    ///
    /// ## Parameters
    /// - `first`: Callables invoked after the ``first`` CSR event.
    /// - `early`: Callables invoked after the ``early`` CSR event.
    /// - `late`: Callables invoked after the ``late`` CSR event.
    ///
    /// ## Notes
    /// Each callable receives ``(ind, sperm, tick, deme_id)`` where the two
    /// arrays are fresh copies of the current state; a nonzero return value
    /// stops the run.
    fn set_python_callbacks(
        &mut self,
        first: Vec<Py<PyAny>>,
        early: Vec<Py<PyAny>>,
        late: Vec<Py<PyAny>>,
    ) {
        self.hooks.python_callbacks = vec![first, early, late];
    }

    /// Clear all Python callbacks.
    fn clear_python_callbacks(&mut self) {
        self.hooks.python_callbacks = vec![Vec::new(), Vec::new(), Vec::new()];
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
    fn drain_eco_journal(&mut self) -> Vec<crate::hooks::EcoJournalRow> {
        std::mem::take(&mut self.eco_journal)
    }

    /// Run the reproduction stage in place on the session-owned state.
    fn reproduction(&mut self) -> PyResult<()> {
        let cfg = SimConfig::assemble(&self.blueprint, &self.params, &self.genetics)?;
        lifecycle::reproduction(
            &mut self.rng,
            &cfg,
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
        let cfg = SimConfig::assemble(&self.blueprint, &self.params, &self.genetics)?;
        lifecycle::survival(
            &mut self.rng,
            &cfg,
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
        let cfg = SimConfig::assemble(&self.blueprint, &self.params, &self.genetics)?;
        lifecycle::aging(&cfg, &mut self.state_ind, &mut self.state_sperm);
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
        let cfg = SimConfig::assemble(&self.blueprint, &self.params, &self.genetics)?;
        let tick = self.state_tick;
        let result = self.run_with_eco(&cfg, tick, deme_id);
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
    /// returned.  When ``record_interval > 0``, flattened history rows are
    /// returned as a 2-D NumPy array whose row layout mirrors the Numba
    /// ``_run_loop_structured`` kernel.
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
        // Assemble the config from the owned contracts at the batch entry,
        // copy the observation mask if present, run the Rust batch loop
        // directly on the session-owned state, and copy the flattened
        // history into a NumPy 2-D array.  Python passes control
        // parameters only (plan S2: the session owns counts and tick).
        let cfg = SimConfig::assemble(&self.blueprint, &self.params, &self.genetics)?;
        let mask_vec = match observation_mask {
            Some(mask) => Some(
                mask.as_slice()
                    .map_err(|err| PyValueError::new_err(err.to_string()))?
                    .to_vec(),
            ),
            None => None,
        };

        let mut eco_values = self.eco_values(0);

        // Borrowed EcoCtx: run_batch commits set_param writes at every event
        // boundary; after the batch the journal is drained into the
        // session-owned audit trail for the Python adapter.
        let mut eco_ctx = Some(lifecycle::EcoCtx {
            bp: &self.blueprint,
            params: &mut self.params,
            genetics: &self.genetics,
            deme: 0,
            tick: self.state_tick,
            journal: Vec::new(),
        });

        let (final_tick, flat_history, n_rows, was_stopped) = lifecycle::run_batch(
            &mut self.rng,
            &cfg,
            &self.hooks,
            &mut self.state_ind,
            &mut self.state_sperm,
            self.state_tick,
            n_ticks,
            record_interval,
            mask_vec.as_deref(),
            &mut eco_values,
            &mut eco_ctx,
            checkpoint_every,
            &mut self.checkpoints,
        )
        .map_err(map_lifecycle_error)?;
        self.state_tick = final_tick;

        if let Some(ctx) = eco_ctx.as_mut() {
            self.eco_journal.append(&mut ctx.journal);
        }

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

    /// Install a full live state (plan S2 state ownership).
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
        let cfg = SimConfig::assemble(&self.blueprint, &self.params, &self.genetics)?;
        let want_ind = 2 * cfg.n_ages * cfg.n_ztypes;
        let want_sperm = cfg.n_ages * cfg.n_ztypes * cfg.n_ztypes;
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
        self.state_ind = ind_flat;
        self.state_sperm = sperm_flat;
        self.state_tick = tick;
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

    /// Restore a memory checkpoint produced by [`EngineSession::snapshot_state`].    ///
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
        self.state_ind.copy_from_slice(ind_src);
        self.state_sperm.copy_from_slice(sperm_src);
        self.state_tick = tick;
        let mut words = [0_u64; 4];
        words.copy_from_slice(&rng_words);
        self.rng = SessionRng::from_state_words(words);
        restore_ecology(&mut self.params, &self.blueprint, ecology)?;
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
        self.state_ind.copy_from_slice(&cp.ind);
        self.state_sperm.copy_from_slice(&cp.sperm);
        self.state_tick = cp.tick;
        self.rng = SessionRng::from_state_words(cp.rng_words);
        self.params
            .ecology_restore_words(&self.blueprint, &cp.eco_scalars, &cp.eco_vectors)?;
        Ok(Some((cp.tick, ecology_snapshot(py, &self.params)?)))
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
    fn truncate_checkpoints(&mut self, retain_until_tick: i64) {
        self.checkpoints.retain(|cp| cp.tick <= retain_until_tick);
    }
}

impl EngineSession {
    /// Snapshot the canonical ECO param values for one deme column.
    ///
    /// ## Parameters
    /// - `deme`: Deme column to read.
    ///
    /// ## Returns
    /// A length-``N_ECO_PARAMS`` array in ``ECO_PARAM_COLUMNS`` order.
    fn eco_values(&self, deme: usize) -> [f64; crate::hooks::N_ECO_PARAMS] {
        let mut values = [0.0; crate::hooks::N_ECO_PARAMS];
        for (id, slot) in values.iter_mut().enumerate() {
            *slot = self.params.eco_value(id, deme);
        }
        values
    }

    /// Run one structured tick with a live ECO scratch and write-back.
    ///
    /// The EcoCtx commits set_param writes into the deme's ecology column
    /// at every event boundary and re-assembles the config for the same
    /// tick's later stages — the same granularity as the Python executor.
    /// The tick's journal rows are drained into the session audit trail.
    fn run_with_eco(&mut self, cfg: &SimConfig, tick: i64, deme_id: i64) -> Result<i32, String> {
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
        let deme = deme_id.max(0) as usize;
        let mut eco_values = params.eco_values_row(deme);
        let mut ctx = Some(lifecycle::EcoCtx {
            bp: blueprint,
            params,
            genetics,
            deme,
            tick,
            journal: Vec::new(),
        });
        let result = lifecycle::run_tick(
            rng,
            cfg,
            hooks,
            state_ind,
            state_sperm,
            tick,
            deme_id,
            &mut eco_values,
            &mut ctx,
        );
        if let Some(ctx) = ctx.as_mut() {
            eco_journal.append(&mut ctx.journal);
        }
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
            || op_types
                .iter()
                .any(|&op| op == crate::hooks::OP_SET_PARAM_PUBLIC);
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
            has_set_param,
            ..Default::default()
        })
    }
}
