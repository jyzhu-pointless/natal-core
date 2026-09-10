//! PyO3 sessions for homogeneous and heterogeneous spatial multi-deme runs.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray4, PyUntypedArrayMethods};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;

use crate::hooks::interpreter::HookProgram;
use crate::kernels::discrete_generation;
use crate::kernels::rng::{new_rng, SessionRng};
use crate::model::blueprint::Blueprint;
use crate::model::ecology::EcologyParams;
use crate::model::genetics::is_genetics_tensor;
use crate::model::genetics::GeneticsTensors;
use crate::model::genetics::GENETICS_TENSORS;
use crate::output::history::{HistoryStore, SharedHistory};

/// Convert internal kernel error strings into ``PyRuntimeError``.
fn map_lifecycle_error(err: String) -> PyErr {
    crate::hooks::transaction::map_error(err)
}

/// PyO3 session for heterogeneous spatial multi-deme runs.
///
/// Variant bank: one shared blueprint, one columnized
/// ecology set (per-deme ``EcologyParams`` columns), a bank of shared genetics
/// [`GeneticsTensors`] variants, and a per-deme variant index.  Blueprint,
/// ecology columns, and genetics are stored exactly once each — no
/// per-deme contract clones.
///
/// Session ownership: the session also owns the stacked counts, sperm
/// storage, tick, and one persistent RNG stream per deme (``seed ^ deme``,
/// advancing across ticks instead of being rebuilt per tick).  ``run_tick``
/// takes control parameters only; lifecycle then migration consume the
/// same per-deme streams inside one call.  Python reads state back
/// through snapshots.
/// One restorable spatial boundary: the full owned runtime (state, sperm,
/// every per-deme RNG stream, and the ecology columns) plus its tick.
/// Captured at record-aligned ticks by the Python adapter; restoring
/// replaces the whole runtime atomically so `restore -> run` replays the
/// original stochastic trajectory.
pub struct SpatialTickCheckpoint {
    /// Lifecycle status and cursor retained by manual snapshots.
    pub execution: crate::sessions::status::ExecutionStatus,
    pub phase: usize,
    /// Tick the checkpoint was captured at.
    pub tick: i64,
    /// Flattened stacked individual counts.
    pub ind: Vec<f64>,
    /// Flattened stacked sperm storage.
    pub sperm: Vec<f64>,
    /// One 4-word Xoshiro256++ state per deme, in deme order.
    pub rng_words: Vec<[u64; 4]>,
    /// The complete ecology column set at capture time.
    pub ecology: EcologyParams,
}

#[pyclass(name = "HeterogeneousSpatialEngineSession")]
pub struct SpatialSession {
    blueprint: Blueprint,
    ecology: EcologyParams,
    variants: Vec<GeneticsTensors>,
    deme_variants: Vec<usize>,
    hooks: HookProgram,
    seed: u64,
    rngs: Vec<SessionRng>,
    state_ind: Vec<f64>,
    state_sperm: Vec<f64>,
    state_tick: i64,
    execution: crate::sessions::status::ExecutionStatus,
    phase: usize,
    /// Record-aligned restorable boundaries.
    checkpoints: Vec<SpatialTickCheckpoint>,
    /// Shared native numerical history and per-deme log cursors.
    history_store: Option<SharedHistory>,
    /// Discrete-generation demes tick the discrete lifecycle and carry no
    /// sperm plane (``state_sperm`` is a session-maintained zero sink the
    /// lifecycle never reads; migration's virgin bookkeeping sees zeros).
    discrete: bool,
    /// Deterministic-migration bookkeeping order mirrored from the
    /// Python-side frozen migration CSR (``stay_after_send``).
    stay_after_send: bool,
    /// Audited per-deme set_param transitions accumulated across run
    /// calls; drained by the Python adapter after each run.
    eco_journal: Vec<crate::kernels::spatial::SpatialEcoJournalRow>,
}

#[pymethods]
impl SpatialSession {
    /// Execute one explicit event against a managed deme's native state.
    fn trigger_deme_event(&mut self, deme: usize, event: usize) -> PyResult<i32> {
        if deme >= self.deme_variants.len() || event >= 4 {
            return Err(PyValueError::new_err("unknown deme or hook event"));
        }
        let mut params = self.ecology.single_deme(deme);
        let mut values = params.eco_values_row(0);
        let n_ages = self.blueprint.n_ages;
        let z = self.blueprint.n_ztypes;
        let ind_stride = 2 * n_ages * z;
        let sperm_stride = n_ages * z * z;
        let ind = &mut self.state_ind[deme * ind_stride..(deme + 1) * ind_stride];
        let sperm = &mut self.state_sperm[deme * sperm_stride..(deme + 1) * sperm_stride];
        let mut ctx = Some(crate::kernels::age_structured::EcoCtx {
            bp: &self.blueprint,
            params: &mut params,
            genetics: &self.variants[self.deme_variants[deme]],
            updated_genetics: None,
            phase: event * 2,
            deme: 0,
            tick: self.state_tick,
            journal: Vec::new(),
        });
        let mut result = 0;
        let outcome = (|| -> Result<(), String> {
            result = self.hooks.execute_event(
                &mut self.rngs[deme],
                event as i64,
                ind,
                sperm,
                2,
                n_ages,
                z,
                self.state_tick,
                self.blueprint.stochastic,
                self.blueprint.continuous_sampling,
                deme as i64,
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
                if let Some(log) = store.extra_logs.get(deme) {
                    for (tick, id, old, new, phase) in context.journal.drain(..) {
                        log.lock().unwrap().push(
                            crate::output::parameter_log::LogEntry::from_phase(
                                (
                                    tick,
                                    crate::generated::ecology_parameters::ECO_PARAM_COLUMNS[id]
                                        .to_owned(),
                                    old,
                                    new,
                                ),
                                phase,
                                deme,
                            ),
                        );
                    }
                }
            } else {
                self.eco_journal.extend(
                    context
                        .journal
                        .drain(..)
                        .map(|(tick, id, old, new, phase)| (deme, tick, id, old, new, phase)),
                );
            }
        }
        let genetics = ctx
            .as_mut()
            .and_then(|context| context.updated_genetics.take());
        drop(ctx);
        self.ecology.replace_deme(deme, &params);
        if let Some(genetics) = genetics {
            let variant = self
                .variants
                .iter()
                .position(|value| value == &genetics)
                .unwrap_or_else(|| {
                    self.variants.push(genetics);
                    self.variants.len() - 1
                });
            self.deme_variants[deme] = variant;
        }
        self.hooks
            .callback_commits
            .lock()
            .expect("callback queue poisoned")
            .clear();
        if let Err(error) = outcome {
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

    /// Create a heterogeneous spatial session from columnized contracts.
    ///
    /// ## Parameters
    /// - `blueprint`: A ``natal.contracts.Blueprint`` NamedTuple carrying
    ///   the spatial extent (``n_demes``) and migration CSR.
    /// - `ecology_columns`: Mapping of ecology field name to a flat column
    ///   array — scalar columns length ``n_demes``, vector columns
    ///   ``n_demes`` times their per-deme extent, ``growth_mode`` int64.
    /// - `tensor_bank`: List of ``{genetics tensor name: flat array}``
    ///   mappings, one per genetics variant.
    /// - `deme_variant_ids`: Int64 array mapping each deme to a bank index.
    /// - `individual_count_all`: Stacked initial state
    ///   ``(n_demes, 2, n_ages, n_ztypes)`` (the one-time build handoff).
    /// - `sperm_storage_all`: Stacked initial sperm
    ///   ``(n_demes, n_ages, n_ztypes, n_ztypes)``.
    /// - `tick`: The authoritative starting tick.
    /// - `stay_after_send`: Deterministic-migration bookkeeping order.
    /// - `seed`: Base RNG seed; deme *d* streams from ``seed ^ d``.
    ///
    /// ## Returns
    /// A new `SpatialSession`.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` when any contract piece is inconsistent, a
    /// deme variant id is out of range, or the stacked state does not
    /// match the blueprint dimensions, contains non-finite or negative
    /// values, or has a negative tick.
    #[new]
    #[pyo3(signature = (blueprint, ecology_columns, tensor_bank, deme_variant_ids, individual_count_all, sperm_storage_all, tick, model=String::from("age_structured"), stay_after_send=false, seed=0))]
    #[allow(clippy::too_many_arguments)] // One-time build handoff of the owned run data.
    fn from_parts(
        blueprint: &Bound<'_, PyAny>,
        ecology_columns: &Bound<'_, PyAny>,
        tensor_bank: &Bound<'_, PyAny>,
        deme_variant_ids: PyReadonlyArray1<'_, i64>,
        individual_count_all: PyReadonlyArray4<'_, f64>,
        sperm_storage_all: PyReadonlyArray4<'_, f64>,
        tick: i64,
        model: String,
        stay_after_send: bool,
        seed: u64,
    ) -> PyResult<Self> {
        validate_state_tick(tick)?;
        let bp = Blueprint::from_python(blueprint)?;
        bp.validate()?;
        let ecology = EcologyParams::from_columns(ecology_columns, bp.n_demes)?;
        ecology.validate(&bp)?;
        let mut variants = Vec::new();
        for entry in tensor_bank.try_iter()? {
            let tensors = GeneticsTensors::from_dict(&entry?)?;
            tensors.validate(&bp)?;
            variants.push(tensors);
        }
        if variants.is_empty() {
            return Err(PyValueError::new_err(
                "tensor_bank must contain at least one genetics variant",
            ));
        }
        let deme_variants = deme_variant_ids
            .as_slice()
            .map_err(|err| PyValueError::new_err(err.to_string()))?
            .iter()
            .map(|&value| value as usize)
            .collect::<Vec<usize>>();
        for (deme, &variant) in deme_variants.iter().enumerate() {
            if variant >= variants.len() {
                return Err(PyValueError::new_err(format!(
                    "deme {deme} variant id {variant} out of range for a bank of {} entries",
                    variants.len()
                )));
            }
        }
        let discrete = match model.as_str() {
            "age_structured" => false,
            "discrete_generation" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown spatial model {other:?} (expected \"age_structured\" or \"discrete_generation\")"
                )))
            }
        };
        let state_ind = validate_stacked_ind(individual_count_all, &bp, deme_variants.len())?;
        let state_sperm = validate_stacked_sperm(sperm_storage_all, &bp, deme_variants.len())?;
        // Validate the model's normalized shape once at construction (the
        // ecology columns and genetics variants were validated above).
        if discrete {
            discrete_generation::validate_discrete_shape(&bp)?;
        } else {
            if bp.n_ages == 0 || bp.n_ztypes == 0 {
                return Err(PyValueError::new_err(
                    "n_ages and n_ztypes must be positive",
                ));
            }
            if bp.new_adult_age == 0 || bp.new_adult_age > bp.n_ages {
                return Err(PyValueError::new_err(format!(
                    "new_adult_age must be in [1, {}], got {}",
                    bp.n_ages, bp.new_adult_age
                )));
            }
        }
        let rngs = (0..deme_variants.len())
            .map(|deme| new_rng(crate::kernels::rng::stream_seed(seed, deme as i64)))
            .collect();
        Ok(Self {
            blueprint: bp,
            ecology,
            variants,
            deme_variants,
            hooks: HookProgram::default(),
            seed,
            rngs,
            state_ind,
            state_sperm,
            state_tick: tick,
            execution: crate::sessions::status::ExecutionStatus::Ready,
            phase: 0,
            discrete,
            stay_after_send,
            eco_journal: Vec::new(),
            checkpoints: Vec::new(),
            history_store: None,
        })
    }

    /// Pull exactly the named ecology fields for one deme.
    ///
    /// ## Parameters
    /// - `deme`: Deme whose ecology column is written.
    /// - `fields`: Contract field names to pull.
    /// - `source`: A ``natal.contracts.Params`` carrying current values.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` for an out-of-range deme and
    /// ``PyKeyError``/``PyValueError`` on unknown names (including genetics
    /// tensor names, which belong to [`Self::refresh_variant_tensors`]) or
    /// size mismatch.
    fn refresh_deme_ecology(
        &mut self,
        deme: usize,
        fields: Vec<String>,
        source: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if deme >= self.ecology.n_demes {
            return Err(PyValueError::new_err(format!(
                "deme {deme} out of range for {} ecology columns",
                self.ecology.n_demes
            )));
        }
        self.ecology
            .pull_fields(&self.blueprint, deme, &fields, source, None)
    }

    /// Pull exactly the named genetics tensors for one bank variant.
    ///
    /// ## Parameters
    /// - `variant_id`: Index into the genetics variant bank.
    /// - `fields`: Genetics tensor names to pull.
    /// - `source`: A ``natal.contracts.Params`` carrying current values.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` for an out-of-range ``variant_id`` and
    /// ``PyKeyError``/``PyValueError`` on non-genetics names or size
    /// mismatch.
    fn refresh_variant_tensors(
        &mut self,
        variant_id: usize,
        fields: Vec<String>,
        source: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if variant_id >= self.variants.len() {
            return Err(PyValueError::new_err(format!(
                "variant_id {variant_id} out of range for a bank of {} entries",
                self.variants.len()
            )));
        }
        for field in &fields {
            if !is_genetics_tensor(field) {
                return Err(PyKeyError::new_err(format!(
                    "{field:?} is not a genetics tensor (expected one of {GENETICS_TENSORS:?})"
                )));
            }
        }
        self.variants[variant_id].pull_fields(&self.blueprint, &fields, source)
    }

    /// Validate and atomically merge an explicit compiled deme update.
    fn refresh_deme_parameters(
        &mut self,
        deme: usize,
        fields: Vec<String>,
        source: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if deme >= self.deme_variants.len() {
            return Err(PyValueError::new_err("deme out of range"));
        }
        let mut params = self.ecology.single_deme(deme);
        let mut genetics = self.variants[self.deme_variants[deme]].clone();
        params.pull_fields(&self.blueprint, 0, &fields, source, Some(&mut genetics))?;
        self.ecology.replace_deme(deme, &params);
        let variant = self
            .variants
            .iter()
            .position(|value| value == &genetics)
            .unwrap_or_else(|| {
                self.variants.push(genetics);
                self.variants.len() - 1
            });
        self.deme_variants[deme] = variant;
        Ok(())
    }

    /// Validate custom values before replacing the selected deme candidate.
    fn set_deme_custom_slots(&mut self, deme: usize, source: &Bound<'_, PyAny>) -> PyResult<()> {
        if deme >= self.deme_variants.len() {
            return Err(PyValueError::new_err("deme out of range"));
        }
        self.ecology.custom_slots[deme] =
            crate::model::custom_fields::custom_slots_from_python(source)?;
        Ok(())
    }

    /// Read one current scalar from the selected deme's native column.
    fn get_deme_scalar(&self, deme: usize, name: &str) -> PyResult<f64> {
        if deme >= self.deme_variants.len() {
            return Err(PyValueError::new_err("deme out of range"));
        }
        self.ecology.single_deme(deme).get_scalar(name)
    }

    /// Read a detached ecology or genetics tensor for one deme.
    fn get_deme_tensor<'py>(
        &self,
        py: Python<'py>,
        deme: usize,
        name: &str,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if deme >= self.deme_variants.len() {
            return Err(PyValueError::new_err("deme out of range"));
        }
        crate::model::ecology::session_get_tensor(
            py,
            &self.ecology.single_deme(deme),
            &self.variants[self.deme_variants[deme]],
            name,
        )
    }

    /// Read detached custom values for a managed deme.
    fn get_deme_custom_slots<'py>(
        &self,
        py: Python<'py>,
        deme: usize,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        if deme >= self.deme_variants.len() {
            return Err(PyValueError::new_err("deme out of range"));
        }
        crate::model::custom_fields::custom_slots_to_python(py, &self.ecology.custom_slots[deme])
    }

    /// Validate one deme's scalar candidates before replacing its column.
    fn apply_deme(
        &mut self,
        deme: usize,
        writes: std::collections::HashMap<String, f64>,
    ) -> PyResult<()> {
        if deme >= self.deme_variants.len() {
            return Err(PyValueError::new_err("deme out of range"));
        }
        let mut candidate = self.ecology.single_deme(deme);
        candidate.apply(writes)?;
        self.ecology.replace_deme(deme, &candidate);
        Ok(())
    }

    /// Commit a validated whole tensor, forking only changed genetic values.
    fn tensor_write_deme(&mut self, deme: usize, name: &str, values: Vec<f64>) -> PyResult<()> {
        if deme >= self.deme_variants.len() {
            return Err(PyValueError::new_err("deme out of range"));
        }
        let mut params = self.ecology.single_deme(deme);
        let mut genetics = self.variants[self.deme_variants[deme]].clone();
        crate::model::ecology::session_tensor_write(
            &self.blueprint,
            &mut params,
            &mut genetics,
            name,
            values,
        )?;
        self.ecology.replace_deme(deme, &params);
        let variant = self
            .variants
            .iter()
            .position(|value| value == &genetics)
            .unwrap_or_else(|| {
                self.variants.push(genetics);
                self.variants.len() - 1
            });
        self.deme_variants[deme] = variant;
        Ok(())
    }

    /// Number of genetics variants in the bank (read-only introspection).
    fn n_variants(&self) -> usize {
        self.variants.len()
    }

    /// Fork the genetics variant of one deme into a private bank entry.
    ///
    /// The deme's current variant table is cloned, appended to the bank,
    /// and the deme is re-pointed at the new entry; every other deme keeps
    /// sharing the original tables.  Returns the new variant id, which the
    /// caller can then write through [`Self::refresh_variant_tensors`].
    ///
    /// ## Parameters
    /// - `deme`: Deme whose genetics diverge.
    ///
    /// ## Returns
    /// The id of the freshly forked private variant.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` when the deme index is out of range.
    fn fork_variant(&mut self, deme: usize) -> PyResult<usize> {
        if deme >= self.deme_variants.len() {
            return Err(PyValueError::new_err(format!(
                "deme {deme} out of range for {} variant ids",
                self.deme_variants.len()
            )));
        }
        let current = self.deme_variants[deme];
        let cloned = self.variants[current].clone();
        self.variants.push(cloned);
        let new_id = self.variants.len() - 1;
        self.deme_variants[deme] = new_id;
        Ok(new_id)
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

    /// Register Python callables interleaved with the CSR hooks at the
    /// deme-tick event boundaries.
    ///
    /// Any callback-carrying program demotes the scheduler to a stable
    /// deme-order sequential loop so cross-deme callback order cannot
    /// depend on thread scheduling.  Each callable receives
    /// ``(ind, sperm, tick, deme_id)`` copies; a nonzero return stops the
    /// run at that boundary.
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

    /// Reseed every per-deme RNG stream from the new base seed.
    ///
    /// ## Parameters
    /// - `seed`: New base seed; deme *d* restarts from ``seed ^ d``.
    fn reseed(&mut self, seed: u64) {
        self.seed = seed;
        self.rngs = (0..self.deme_variants.len())
            .map(|deme| new_rng(crate::kernels::rng::stream_seed(seed, deme as i64)))
            .collect();
    }

    /// Drain the accumulated per-deme set_param audit journal.
    ///
    /// ## Returns
    /// A list of ``(deme, tick, param_id, old, new)`` tuples (change-only
    /// rows, commit order), cleared by the drain.
    fn drain_eco_journal(&mut self) -> Vec<(usize, i64, usize, f64, f64)> {
        std::mem::take(&mut self.eco_journal)
            .into_iter()
            .map(|(deme, tick, id, old, new, _)| (deme, tick, id, old, new))
            .collect()
    }

    /// Replace the live migration-rate column used by the migration stage.
    ///
    /// The rate column is runtime-mutable through the spatial params view;
    /// without this push a session-owned column would silently diverge
    /// from the Python-side contract array.
    ///
    /// ## Parameters
    /// - `values`: Flat ``(n_demes * 2 * n_ages)`` rate column, row-major
    ///   ``(n_demes, 2, n_ages)``.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` when the length does not match the
    /// spatial extent; the column is left untouched.
    fn set_migration_rate(&mut self, values: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let want = self.ecology.n_demes * 2 * self.blueprint.n_ages;
        let slice = values
            .as_slice()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        if slice.len() != want {
            return Err(PyValueError::new_err(format!(
                "migration_rate must hold (n_demes, 2, n_ages) = {} values, got {}",
                want,
                slice.len()
            )));
        }
        self.ecology.migration_rate = slice.to_vec();
        Ok(())
    }

    /// Run one complete tick for all demes inside Rust.
    ///
    /// Stage order per deme: lifecycle (first hook -> aging) consuming the
    /// deme's persistent RNG stream, then migration consuming the same
    /// stream — matching the frozen lifecycle-then-migration order.  The
    /// tick advances only when every hook let the tick complete; a stop
    /// keeps the modifications up to that boundary and freezes the tick.
    ///
    /// ## Returns
    /// The tick value after the call: ``tick + 1`` on a completed tick,
    /// the unchanged tick when a hook stopped the run.
    ///
    /// ## Errors
    /// Returns ``PyRuntimeError`` for shape mismatches or kernel errors,
    /// ``PyValueError`` when an in-run ``Op.set_param`` value fails the
    /// bounds gate.
    fn run_tick(&mut self) -> PyResult<i64> {
        self.execution.begin()?;
        let before_tick = self.state_tick;
        let journal_start = self.eco_journal.len();
        let outcome = self.run_inner();
        if let Some(shared) = &self.history_store {
            let store = shared.lock().unwrap();
            for (deme, tick, parameter, old, new, phase) in self.eco_journal.drain(journal_start..)
            {
                if let Some(log) = store.extra_logs.get(deme) {
                    log.lock()
                        .unwrap()
                        .push(crate::output::parameter_log::LogEntry::from_phase(
                            (
                                tick,
                                crate::generated::ecology_parameters::ECO_PARAM_COLUMNS[parameter]
                                    .to_owned(),
                                old,
                                new,
                            ),
                            phase,
                            deme,
                        ));
                }
            }
        }
        self.execution = match &outcome {
            Ok(value) if value == &before_tick => crate::sessions::status::ExecutionStatus::Stopped,
            Ok(_) => {
                self.phase = 0;
                crate::sessions::status::ExecutionStatus::Ready
            }
            Err(_) => crate::sessions::status::ExecutionStatus::Failed,
        };
        outcome
    }

    /// Advance and retain spatial boundaries entirely inside the session.
    fn run_steps(&mut self, n_ticks: i64, record_interval: i64) -> PyResult<(i64, bool)> {
        if n_ticks < 0 {
            return Err(PyValueError::new_err("n_steps must be >= 0"));
        }
        if record_interval > 0 && self.state_tick % record_interval == 0 {
            self.record_history(true)?;
        }
        for _ in 0..n_ticks {
            let previous = self.state_tick;
            self.run_tick()?;
            if self.state_tick == previous {
                return Ok((self.state_tick, true));
            }
            if record_interval > 0 && self.state_tick % record_interval == 0 {
                self.record_history(true)?;
            }
        }
        Ok((self.state_tick, false))
    }

    /// Capture one restorable boundary from the owned runtime.
    ///
    /// Called by the Python adapter at record-aligned ticks (raw history
    /// mode).  Storing clones the full state, every per-deme RNG stream,
    /// and the ecology columns; nothing is moved, so the current run is
    /// unaffected.
    fn capture_checkpoint(&mut self) -> i64 {
        self.checkpoints.push(SpatialTickCheckpoint {
            execution: self.execution,
            phase: self.phase,
            tick: self.state_tick,
            ind: self.state_ind.clone(),
            sperm: self.state_sperm.clone(),
            rng_words: self.rngs.iter().map(|rng| rng.state_words()).collect(),
            ecology: self.ecology.clone(),
        });
        self.state_tick
    }

    /// Restore the newest checkpoint at or before *tick*.
    ///
    /// Atomically replaces the owned state, all per-deme RNG streams, and
    /// the ecology columns from the checkpoint, rewinds the tick, and
    /// truncates stored checkpoints newer than the restored boundary.
    /// Nothing is changed when no checkpoint covers *tick*.
    ///
    /// ## Parameters
    /// - `tick`: Target tick; the newest checkpoint with
    ///   ``checkpoint.tick <= tick`` is restored.
    ///
    /// ## Returns
    /// The restored tick, or ``None`` when no checkpoint covers *tick*
    /// (the caller falls back to its own restore path); on ``None`` the
    /// runtime is untouched.
    fn restore_from_checkpoint(&mut self, tick: i64) -> PyResult<Option<i64>> {
        let Some(index) = self
            .checkpoints
            .iter()
            .rposition(|checkpoint| checkpoint.tick == tick)
        else {
            return Ok(None);
        };
        if let Some(store) = &self.history_store {
            store.lock().unwrap().restore_timeline(tick)?;
        }
        let checkpoint = &self.checkpoints[index];
        self.state_ind = checkpoint.ind.clone();
        self.state_sperm = checkpoint.sperm.clone();
        self.state_tick = checkpoint.tick;
        self.execution = checkpoint.execution;
        self.phase = checkpoint.phase;
        self.rngs = checkpoint
            .rng_words
            .iter()
            .map(|words| SessionRng::from_state_words(*words))
            .collect();
        self.ecology = checkpoint.ecology.clone();
        // Future checkpoints are invalid once the timeline rewinds.
        self.checkpoints.truncate(index + 1);
        Ok(Some(self.state_tick))
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
                self.blueprint.n_demes,
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

    /// Bind the native store owned by the public History adapter.
    fn bind_history(&mut self, history: PyRef<'_, HistoryStore>) {
        self.history_store = Some(std::sync::Arc::clone(&history.data));
    }

    /// Record directly from native stacked state without Python snapshots.
    fn record_history(&mut self, continuation: bool) -> PyResult<()> {
        let shared = self
            .history_store
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("History is not initialized"))?
            .clone();
        let (added, raw, earliest) = {
            let mut history = shared.lock().unwrap();
            let sperm = if self.discrete {
                &[][..]
            } else {
                &self.state_sperm[..]
            };
            let added = history.record(self.state_tick, &self.state_ind, sperm, continuation)?;
            if added {
                if let Some(boundary) = history.boundaries.back_mut() {
                    boundary.1 = self.phase;
                    boundary.2 = self.execution.name().to_owned();
                }
            }
            (
                added,
                history.raw,
                history.rows.front().map(|row| row[0] as i64),
            )
        };
        if added && raw {
            self.capture_checkpoint();
        }
        if let Some(tick) = earliest {
            self.checkpoints.retain(|cp| cp.tick >= tick);
        }
        Ok(())
    }

    /// Drop checkpoints older than *from_tick* (history eviction pair).
    fn retain_checkpoints_from(&mut self, from_tick: i64) {
        self.checkpoints
            .retain(|checkpoint| checkpoint.tick >= from_tick);
    }

    /// Truncate checkpoints newer than *retain_until_tick*.
    fn truncate_checkpoints(&mut self, retain_until_tick: i64) {
        self.checkpoints
            .retain(|checkpoint| checkpoint.tick <= retain_until_tick);
    }

    /// Clear all checkpoints (history cleared).
    fn clear_checkpoints(&mut self) {
        self.checkpoints.clear();
    }

    /// Export the current ecology columns for Python-side rollback.
    ///
    /// ## Returns
    /// A mapping of ecology column name to a flat copy of the column
    /// (scalar columns length ``n_demes``, vector columns ``n_demes``
    /// times their per-deme extent, ``growth_mode`` int64, plus the
    /// ``migration_rate`` column).
    fn ecology_columns_snapshot(&self, py: Python<'_>) -> PyResult<Vec<(String, Py<PyAny>)>> {
        use numpy::PyArray1 as PyArr;
        fn push_f64(
            columns: &mut Vec<(String, Py<PyAny>)>,
            name: &str,
            values: &[f64],
            py: Python<'_>,
        ) {
            let array = PyArr::from_slice(py, values);
            columns.push((
                name.to_string(),
                array.into_pyobject(py).unwrap().unbind().into_any(),
            ));
        }
        let mut columns: Vec<(String, Py<PyAny>)> = Vec::new();
        push_f64(
            &mut columns,
            "carrying_capacity",
            &self.ecology.carrying_capacity,
            py,
        );
        push_f64(
            &mut columns,
            "eggs_per_female",
            &self.ecology.eggs_per_female,
            py,
        );
        push_f64(&mut columns, "sex_ratio", &self.ecology.sex_ratio, py);
        push_f64(
            &mut columns,
            "sperm_displacement_rate",
            &self.ecology.sperm_displacement_rate,
            py,
        );
        push_f64(
            &mut columns,
            "low_density_growth_rate",
            &self.ecology.low_density_growth_rate,
            py,
        );
        let growth: Vec<i64> = self.ecology.growth_mode.clone();
        columns.push((
            "growth_mode".to_string(),
            PyArr::from_slice(py, &growth)
                .into_pyobject(py)
                .unwrap()
                .unbind()
                .into_any(),
        ));
        let declared: Vec<i64> = self
            .ecology
            .equilibrium_declared
            .iter()
            .map(|present| i64::from(*present))
            .collect();
        columns.push((
            "equilibrium_declared".to_string(),
            PyArr::from_slice(py, &declared)
                .into_pyobject(py)
                .unwrap()
                .unbind()
                .into_any(),
        ));
        push_f64(
            &mut columns,
            "external_expected_eggs",
            &self.ecology.external_expected_eggs,
            py,
        );
        for (name, values) in [
            ("survival_rates", &self.ecology.survival_rates),
            ("mating_rates", &self.ecology.mating_rates),
            ("reproduction_rates", &self.ecology.reproduction_rates),
            ("fertility", &self.ecology.fertility),
            ("competition_weights", &self.ecology.competition_weights),
            (
                "equilibrium_distribution",
                &self.ecology.equilibrium_distribution,
            ),
        ] {
            push_f64(&mut columns, name, values, py);
        }
        push_f64(
            &mut columns,
            "migration_rate",
            &self.ecology.migration_rate,
            py,
        );
        Ok(columns)
    }

    /// Snapshot the session-owned state for Python reads.
    ///
    /// ## Returns
    /// ``(tick, ind_flat, sperm_flat)`` — fresh copies; ``ind_flat`` is
    /// the stacked ``(n_demes, 2, n_ages, n_ztypes)`` counts flattened,
    /// ``sperm_flat`` the stacked sperm storage flattened.
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

    /// Install a full stacked live state (build/import handoff).
    ///
    /// ## Parameters
    /// - `individual_count_all`: Stacked ``(n_demes, 2, n_ages, n_ztypes)``.
    /// - `sperm_storage_all`: Stacked ``(n_demes, n_ages, n_ztypes, n_ztypes)``.
    /// - `tick`: The authoritative tick.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` on shape mismatches, non-finite or negative
    /// state values, or a negative tick.
    fn set_state(
        &mut self,
        individual_count_all: PyReadonlyArray4<'_, f64>,
        sperm_storage_all: PyReadonlyArray4<'_, f64>,
        tick: i64,
    ) -> PyResult<()> {
        validate_state_tick(tick)?;
        // Validate both planes before committing either: a failed import
        // must leave the owned state untouched (transactional boundary).
        let n_demes = self.deme_variants.len();
        let ind = validate_stacked_ind(individual_count_all, &self.blueprint, n_demes)?;
        let sperm = validate_stacked_sperm(sperm_storage_all, &self.blueprint, n_demes)?;
        self.state_ind = ind;
        self.state_sperm = sperm;
        self.state_tick = tick;
        self.execution = crate::sessions::status::ExecutionStatus::Ready;
        self.phase = 0;
        Ok(())
    }
}

/// Reject invalid clocks before any state import can commit.
fn validate_state_tick(tick: i64) -> PyResult<()> {
    if tick < 0 {
        return Err(PyValueError::new_err("state tick must be nonnegative"));
    }
    Ok(())
}

/// Validate a stacked initial-state array against the blueprint.
fn validate_stacked_ind(
    array: PyReadonlyArray4<'_, f64>,
    bp: &Blueprint,
    n_demes: usize,
) -> PyResult<Vec<f64>> {
    let shape = array.shape();
    if shape != [n_demes, 2, bp.n_ages, bp.n_ztypes] {
        return Err(PyValueError::new_err(format!(
            "individual_count_all must have shape ({n_demes}, 2, {}, {}), got {shape:?}",
            bp.n_ages, bp.n_ztypes
        )));
    }
    let values = array
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "individual_count_all must contain finite nonnegative values",
        ));
    }
    Ok(values.to_vec())
}

/// Validate a stacked sperm-storage array against the blueprint.
fn validate_stacked_sperm(
    array: PyReadonlyArray4<'_, f64>,
    bp: &Blueprint,
    n_demes: usize,
) -> PyResult<Vec<f64>> {
    let shape = array.shape();
    let want = [n_demes, bp.n_ages, bp.n_ztypes, bp.n_ztypes];
    if shape != want {
        return Err(PyValueError::new_err(format!(
            "sperm_storage_all must have shape {want:?}, got {shape:?}"
        )));
    }
    let values = array
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "sperm_storage_all must contain finite nonnegative values",
        ));
    }
    Ok(values.to_vec())
}

impl SpatialSession {
    fn run_inner(&mut self) -> PyResult<i64> {
        let n_demes = self.deme_variants.len();
        // Per-deme ECO scratch rows for OP_SET_PARAM; written back into
        // the ecology columns after all demes ticked.
        let mut eco_all = vec![0.0; n_demes * crate::hooks::interpreter::N_ECO_PARAMS];
        for deme in 0..n_demes {
            for id in 0..crate::hooks::interpreter::N_ECO_PARAMS {
                eco_all[deme * crate::hooks::interpreter::N_ECO_PARAMS + id] =
                    self.ecology.eco_value(id, deme);
            }
        }
        let tick = self.state_tick;
        // The lifecycle kernels read each deme's ecology column segment and
        // its shared genetics variant directly — no per-deme snapshot build.
        let code = if self.discrete {
            crate::kernels::spatial::run_spatial_tick_discrete(
                &self.hooks,
                &mut self.rngs,
                &mut self.state_ind,
                tick,
                &mut eco_all,
                &self.blueprint,
                &self.ecology,
                &self.variants,
                &self.deme_variants,
                &mut self.eco_journal,
            )
        } else {
            crate::kernels::spatial::run_spatial_tick_heterogeneous(
                &self.hooks,
                &mut self.rngs,
                &mut self.state_ind,
                &mut self.state_sperm,
                tick,
                &mut eco_all,
                &self.blueprint,
                &self.ecology,
                &self.variants,
                &self.deme_variants,
                &mut self.eco_journal,
            )
        };
        self.phase = self
            .hooks
            .phase_marks
            .lock()
            .expect("phase queue poisoned")
            .drain(..)
            .min()
            .unwrap_or(0);
        if code.is_ok() {
            for deme in 0..n_demes {
                for id in 0..crate::hooks::interpreter::N_ECO_PARAMS {
                    self.ecology.set_eco_value(
                        id,
                        deme,
                        eco_all[deme * crate::hooks::interpreter::N_ECO_PARAMS + id],
                    );
                }
            }
        }
        // Every callback committed to its local context before later stages;
        // merge complete candidates before migration consumes updated rates.
        for (deme, params, genetics) in self
            .hooks
            .callback_commits
            .lock()
            .expect("callback queue poisoned")
            .drain(..)
        {
            self.ecology.replace_deme(deme, &params);
            let variant = self
                .variants
                .iter()
                .position(|value| value == &genetics)
                .unwrap_or_else(|| {
                    self.variants.push(genetics);
                    self.variants.len() - 1
                });
            self.deme_variants[deme] = variant;
        }
        let code = code.map_err(map_lifecycle_error)?;
        if code != 0 {
            // A stop keeps the modifications up to the boundary and
            // freezes the tick.
            return Ok(self.state_tick);
        }
        // Migration stage: the frozen CSR and the live rate column, after
        // the lifecycle, on the same per-deme streams.  The zero-rate
        // identity mirrors the Python kernel's skip contract bitwise; the
        // discrete sperm plane is all-zero, so virgin bookkeeping sees
        // every female and no extra RNG draws are consumed.
        let all_zero = self.ecology.migration_rate.iter().all(|&rate| rate <= 0.0);
        if !all_zero {
            let (ind, sperm) = if self.blueprint.stochastic {
                crate::kernels::spatial::migrate_csr_stochastic_rngs(
                    &mut self.rngs,
                    &self.state_ind,
                    &self.state_sperm,
                    &self.blueprint.migration_indptr,
                    &self.blueprint.migration_dest_idx,
                    &self.blueprint.migration_weights,
                    &self.ecology.migration_rate,
                    self.blueprint.continuous_sampling,
                    n_demes,
                    self.blueprint.n_ages,
                    self.blueprint.n_ztypes,
                )
            } else {
                crate::kernels::spatial::migrate_csr_deterministic(
                    &self.state_ind,
                    &self.state_sperm,
                    &self.blueprint.migration_indptr,
                    &self.blueprint.migration_dest_idx,
                    &self.blueprint.migration_weights,
                    &self.ecology.migration_rate,
                    self.stay_after_send,
                    n_demes,
                    self.blueprint.n_ages,
                    self.blueprint.n_ztypes,
                )
            }
            .map_err(map_lifecycle_error)?;
            self.state_ind = ind;
            self.state_sperm = sperm;
        }
        self.state_tick += 1;
        Ok(self.state_tick)
    }
}
