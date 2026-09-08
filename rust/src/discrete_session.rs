//! PyO3 session object for the discrete-generation / Wright-Fisher backend.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray4};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::contract::{session_get_tensor, session_tensor_write, Blueprint, Params, TensorSet};
use crate::discrete::{self, DiscreteConfig};
use crate::hooks::HookProgram;
use crate::rng::{new_rng, SessionRng};
use crate::session::{ecology_snapshot, restore_ecology};

/// Convert internal kernel error strings into ``PyRuntimeError``.
fn map_lifecycle_error(err: String) -> PyErr {
    PyRuntimeError::new_err(err)
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
/// program.  A flat [`DiscreteConfig`] is assembled from the contracts at
/// every tick-batch entry point, so params writes take effect on the next
/// batch without rebuilding the session (the RNG keeps streaming).
#[pyclass(name = "DiscreteEngineSession")]
pub struct DiscreteEngineSession {
    blueprint: Blueprint,
    params: Params,
    genetics: TensorSet,
    rng: SessionRng,
    hooks: HookProgram,
    /// Audited set_param transitions accumulated across tick/run calls;
    /// drained by the Python adapter after each run (see ``EngineSession``).
    eco_journal: Vec<crate::hooks::EcoJournalRow>,
    /// Record-aligned full checkpoints (plan 13.1 R3) — the discrete twin
    /// of ``EngineSession::checkpoints``.
    checkpoints: Vec<crate::lifecycle::TickCheckpoint>,
    /// Session-owned live state (plan S2): flattened counts plus the
    /// authoritative tick; runs and ticks operate on these directly.
    state_ind: Vec<f64>,
    state_tick: i64,
}

#[pymethods]
impl DiscreteEngineSession {
    /// Create a session from the Python contract objects.
    ///
    /// ## Parameters
    /// - `blueprint`: A ``natal.contracts.Blueprint`` NamedTuple.
    /// - `params`: A ``natal.contracts.Params`` dataclass instance.
    /// - `seed`: RNG seed.
    ///
    /// ## Returns
    /// A new ``DiscreteEngineSession`` owning copies of both contracts.
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
        let pr = Params::from_python(params, 1)?;
        let genetics = TensorSet::from_python(params)?;
        bp.validate()?;
        pr.validate(&bp)?;
        genetics.validate(&bp)?;
        // Validate the discrete normalization once at construction.
        DiscreteConfig::assemble(&bp, &pr, &genetics)?;
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
            state_ind,
            state_tick: 0,
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
    /// boundaries after the CSR hooks ran (see ``EngineSession``).
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
    fn drain_eco_journal(&mut self) -> Vec<crate::hooks::EcoJournalRow> {
        std::mem::take(&mut self.eco_journal)
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
    fn tick(&mut self, wf: bool) -> PyResult<i32> {
        let cfg = DiscreteConfig::assemble(&self.blueprint, &self.params, &self.genetics)?;
        // Discrete path runs the standard tick; WF path runs only the first
        // hook and then the fused Wright-Fisher update.  The kernel mutates
        // the session-owned state buffer directly (plan S2).
        let tick = self.state_tick;
        // Split borrows: hooks/rng/state on one side, the EcoCtx contract
        // borrow on the other.
        let Self {
            rng,
            hooks,
            state_ind,
            blueprint,
            params,
            genetics,
            eco_journal,
            ..
        } = self;
        let ind = state_ind.as_mut_slice();
        if wf {
            let mut eco_values = params.eco_values_row(0);
            let mut result = hooks.execute_event(
                rng,
                0,
                ind,
                &mut [],
                2,
                2,
                cfg.n_ztypes,
                tick,
                cfg.stochastic,
                cfg.continuous_sampling,
                0,
                &mut eco_values,
            );
            if result == 0 {
                result = hooks
                    .fire_python_callbacks(0, ind, &mut [], tick, 0)
                    .map_err(PyRuntimeError::new_err)?;
            }
            let mut ctx = crate::lifecycle::EcoCtx {
                bp: blueprint,
                params,
                genetics,
                deme: 0,
                tick,
                journal: Vec::new(),
            };
            ctx.commit(&eco_values).map_err(map_lifecycle_error)?;
            eco_journal.append(&mut ctx.journal);
            if result != 0 {
                return Ok(result);
            }
            discrete::run_wf_tick(rng, &cfg, ind).map_err(map_lifecycle_error)?;
            self.state_tick = tick + 1;
            Ok(0)
        } else {
            let mut eco_values = params.eco_values_row(0);
            let mut ctx = Some(crate::lifecycle::EcoCtx {
                bp: blueprint,
                params,
                genetics,
                deme: 0,
                tick,
                journal: Vec::new(),
            });
            let result =
                discrete::run_tick(rng, &cfg, hooks, ind, tick, 0, &mut eco_values, &mut ctx)
                    .map_err(map_lifecycle_error);
            if let Some(ctx) = ctx.as_mut() {
                eco_journal.append(&mut ctx.journal);
            }
            if matches!(result, Ok(0)) {
                // Only a completed tick advances; a stop freezes the tick
                // exactly like the batch loop and the WF branch above.
                self.state_tick = tick + 1;
            }
            result
        }
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
        // Assemble the config from the owned contracts at the batch entry,
        // then run a batch of discrete or WF ticks directly on the
        // session-owned state (plan S2: control parameters only).
        let cfg = DiscreteConfig::assemble(&self.blueprint, &self.params, &self.genetics)?;
        let mask_vec = match observation_mask {
            Some(mask) => Some(
                mask.as_slice()
                    .map_err(|err| PyValueError::new_err(err.to_string()))?
                    .to_vec(),
            ),
            None => None,
        };
        let mut eco_values = self.eco_values();
        let mut eco_ctx = Some(crate::lifecycle::EcoCtx {
            bp: &self.blueprint,
            params: &mut self.params,
            genetics: &self.genetics,
            deme: 0,
            tick: self.state_tick,
            journal: Vec::new(),
        });
        let (final_tick, flat_history, n_rows, was_stopped) = discrete::run_batch(
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

    /// Capture a memory checkpoint of everything the session owns.
    ///
    /// Discrete populations have no sperm storage; see
    /// ``EngineSession.snapshot_state`` for the checkpoint semantics
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
        let cfg = DiscreteConfig::assemble(&self.blueprint, &self.params, &self.genetics)?;
        let want = 2 * 2 * cfg.n_ztypes;
        if ind_flat.len() != want {
            return Err(PyValueError::new_err(format!(
                "ind_flat must contain {want} values, got {}",
                ind_flat.len()
            )));
        }
        self.state_ind = ind_flat;
        self.state_tick = tick;
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
    /// [`DiscreteEngineSession::snapshot_state`].
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
        self.state_ind.copy_from_slice(ind_src);
        self.state_tick = tick;
        let mut words = [0_u64; 4];
        words.copy_from_slice(&rng_words);
        self.rng = SessionRng::from_state_words(words);
        restore_ecology(&mut self.params, &self.blueprint, ecology)?;
        Ok(tick)
    }

    /// Restore the newest record-aligned checkpoint for *tick* (discrete).
    ///
    /// Discrete twin of ``EngineSession::restore_from_checkpoint``: the
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
        self.state_ind.copy_from_slice(&cp.ind);
        self.state_tick = cp.tick;
        self.rng = SessionRng::from_state_words(cp.rng_words);
        self.params
            .ecology_restore_words(&self.blueprint, &cp.eco_scalars, &cp.eco_vectors)?;
        Ok(Some((
            cp.tick,
            crate::session::ecology_snapshot(py, &self.params)?,
        )))
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

impl DiscreteEngineSession {
    /// Snapshot the canonical ECO param values (deme 0 column).
    ///
    /// ## Returns
    /// A length-``N_ECO_PARAMS`` array in ``ECO_PARAM_COLUMNS`` order.
    fn eco_values(&self) -> [f64; crate::hooks::N_ECO_PARAMS] {
        let mut values = [0.0; crate::hooks::N_ECO_PARAMS];
        for (id, slot) in values.iter_mut().enumerate() {
            *slot = self.params.eco_value(id, 0);
        }
        values
    }
}
