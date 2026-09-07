//! PyO3 sessions for homogeneous and heterogeneous spatial multi-deme runs.

use numpy::{
    PyArray1, PyReadonlyArray1, PyReadonlyArray3, PyReadonlyArray4, PyReadwriteArray4,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::config::SimConfig;
use crate::contract::{is_genetics_tensor, Blueprint, Params, TensorSet, GENETICS_TENSORS};
use crate::hooks::HookProgram;
use crate::rng::{new_rng, SessionRng};
use crate::spatial;

/// Convert internal kernel error strings into ``PyRuntimeError``.
fn map_lifecycle_error(err: String) -> PyErr {
    PyRuntimeError::new_err(err)
}

/// PyO3 session for homogeneous spatial multi-deme runs.
///
/// Holds one shared contract pair (ecology columns tiled to the deme count,
/// one genetics set), a hook program, and a base seed.  The flat config is
/// assembled at each tick entry; per-deme RNG streams derive from
/// ``seed ^ deme_id``.
#[pyclass(name = "SpatialEngineSession")]
pub struct SpatialEngineSession {
    blueprint: Blueprint,
    params: Params,
    genetics: TensorSet,
    hooks: HookProgram,
    seed: u64,
    /// Audited per-deme set_param transitions accumulated across run
    /// calls; drained by the Python adapter after each run.
    eco_journal: Vec<crate::spatial::SpatialEcoJournalRow>,
}

#[pymethods]
impl SpatialEngineSession {
    /// Create a homogeneous spatial session from the Python contract objects.
    ///
    /// ## Parameters
    /// - `blueprint`: A ``natal.contracts.Blueprint`` NamedTuple.
    /// - `params`: A ``natal.contracts.Params`` dataclass instance.
    /// - `seed`: Base RNG seed.
    ///
    /// ## Returns
    /// A new ``SpatialEngineSession``.
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
        // Homogeneous demes share every value: the per-deme-0 contract
        // values are tiled into bp.n_demes columns so the migration-rate
        // column validates against its full spatial extent.
        let pr = Params::from_python(params, bp.n_demes)?;
        let genetics = TensorSet::from_python(params)?;
        bp.validate()?;
        pr.validate(&bp)?;
        genetics.validate(&bp)?;
        // Validate once at construction; assembly re-validates per tick.
        SimConfig::assemble(&bp, &pr, &genetics)?;
        Ok(Self {
            blueprint: bp,
            params: pr,
            genetics,
            hooks: HookProgram::default(),
            seed,
            eco_journal: Vec::new(),
        })
    }

    /// Pull exactly the named contract fields from the Python params object.
    ///
    /// ## Errors
    /// Returns ``PyKeyError``/``PyValueError`` on unknown names or size
    /// mismatch; nothing is written when any field fails.
    fn refresh_params(&mut self, fields: Vec<String>, source: &Bound<'_, PyAny>) -> PyResult<()> {
        let genetics = &mut self.genetics;
        self.params
            .pull_fields(&self.blueprint, 0, &fields, source, Some(genetics))
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

    /// Drain the accumulated per-deme set_param audit journal.
    ///
    /// ## Returns
    /// A list of ``(deme, tick, param_id, old, new)`` tuples (change-only
    /// rows, commit order), cleared by the drain.
    fn drain_eco_journal(&mut self) -> Vec<crate::spatial::SpatialEcoJournalRow> {
        std::mem::take(&mut self.eco_journal)
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
        // Homogeneous demes share one flat config assembled from column 0.
        let cfg = SimConfig::assemble(&self.blueprint, &self.params, &self.genetics)?;
        let n_demes = ind_shape[0];
        let n_ages = ind_shape[2];
        let n_ztypes = ind_shape[3];
        if n_ages != cfg.n_ages || n_ztypes != cfg.n_ztypes {
            return Err(PyValueError::new_err(format!(
                "individual_count_all age/ztype dimensions ({n_ages}, {n_ztypes}) do not match config ({}, {})",
                cfg.n_ages, cfg.n_ztypes
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
        // Per-deme ECO scratch rows for OP_SET_PARAM; written back into
        // the ecology columns after all demes ticked.
        let mut eco_all = vec![0.0; n_demes * crate::hooks::N_ECO_PARAMS];
        for deme in 0..n_demes {
            for id in 0..crate::hooks::N_ECO_PARAMS {
                eco_all[deme * crate::hooks::N_ECO_PARAMS + id] = self.params.eco_value(id, deme);
            }
        }
        spatial::run_spatial_tick(
            &cfg,
            &self.hooks,
            self.seed,
            ind,
            sperm,
            n_demes,
            tick,
            &mut eco_all,
            &self.blueprint,
            &self.params,
            &self.genetics,
            &mut self.eco_journal,
        )
        .map_err(map_lifecycle_error)?;
        for deme in 0..n_demes {
            for id in 0..crate::hooks::N_ECO_PARAMS {
                self.params.set_eco_value(
                    id,
                    deme,
                    eco_all[deme * crate::hooks::N_ECO_PARAMS + id],
                );
            }
        }
        Ok(tick + 1)
    }
}

/// PyO3 session for heterogeneous spatial multi-deme runs.
///
/// Slice-5 stage-2 variant bank: one shared blueprint, one columnized
/// ecology set (per-deme ``Params`` columns), a bank of shared genetics
/// [`TensorSet`] variants, and a per-deme variant index.  Blueprint,
/// ecology columns, and genetics are stored exactly once each — no
/// per-deme contract clones.
///
/// Plan S3 ownership: the session also owns the stacked counts, sperm
/// storage, tick, and one persistent RNG stream per deme (``seed ^ deme``,
/// advancing across ticks instead of being rebuilt per tick).  ``run_tick``
/// takes control parameters only; lifecycle then migration consume the
/// same per-deme streams inside one call.  Python reads state back
/// through snapshots.
#[pyclass(name = "HeterogeneousSpatialEngineSession")]
pub struct HeterogeneousSpatialEngineSession {
    blueprint: Blueprint,
    ecology: Params,
    variants: Vec<TensorSet>,
    deme_variants: Vec<usize>,
    hooks: HookProgram,
    seed: u64,
    rngs: Vec<SessionRng>,
    state_ind: Vec<f64>,
    state_sperm: Vec<f64>,
    state_tick: i64,
    /// Deterministic-migration bookkeeping order mirrored from the
    /// Python-side frozen migration CSR (``stay_after_send``).
    stay_after_send: bool,
    /// Audited per-deme set_param transitions accumulated across run
    /// calls; drained by the Python adapter after each run.
    eco_journal: Vec<crate::spatial::SpatialEcoJournalRow>,
}

#[pymethods]
impl HeterogeneousSpatialEngineSession {
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
    /// A new `HeterogeneousSpatialEngineSession`.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` when any contract piece is inconsistent, a
    /// deme variant id is out of range, or the stacked state does not
    /// match the blueprint dimensions.
    #[new]
    #[pyo3(signature = (blueprint, ecology_columns, tensor_bank, deme_variant_ids, individual_count_all, sperm_storage_all, tick, stay_after_send=false, seed=0))]
    #[allow(clippy::too_many_arguments)] // One-time build handoff of the owned run data.
    fn from_parts(
        blueprint: &Bound<'_, PyAny>,
        ecology_columns: &Bound<'_, PyAny>,
        tensor_bank: &Bound<'_, PyAny>,
        deme_variant_ids: PyReadonlyArray1<'_, i64>,
        individual_count_all: PyReadonlyArray4<'_, f64>,
        sperm_storage_all: PyReadonlyArray4<'_, f64>,
        tick: i64,
        stay_after_send: bool,
        seed: u64,
    ) -> PyResult<Self> {
        let bp = Blueprint::from_python(blueprint)?;
        bp.validate()?;
        let ecology = Params::from_columns(ecology_columns, bp.n_demes)?;
        ecology.validate(&bp)?;
        let mut variants = Vec::new();
        for entry in tensor_bank.try_iter()? {
            let tensors = TensorSet::from_dict(&entry?)?;
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
        let state_ind = validate_stacked_ind(individual_count_all, &bp, deme_variants.len())?;
        let state_sperm = validate_stacked_sperm(sperm_storage_all, &bp, deme_variants.len())?;
        let rngs = (0..deme_variants.len())
            .map(|deme| new_rng(crate::rng::stream_seed(seed, deme as i64)))
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
            stay_after_send,
            eco_journal: Vec::new(),
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

    /// Register Python callables fired at the deme-tick event boundaries.
    ///
    /// Any callback-carrying program demotes the scheduler to a stable
    /// deme-order sequential loop so cross-deme callback order cannot
    /// depend on thread scheduling.  Each callable receives
    /// ``(ind, sperm, tick, deme_id)`` copies; a nonzero return stops the
    /// run at that boundary.
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

    /// Reseed every per-deme RNG stream from the new base seed.
    ///
    /// ## Parameters
    /// - `seed`: New base seed; deme *d* restarts from ``seed ^ d``.
    fn reseed(&mut self, seed: u64) {
        self.seed = seed;
        self.rngs = (0..self.deme_variants.len())
            .map(|deme| new_rng(crate::rng::stream_seed(seed, deme as i64)))
            .collect();
    }

    /// Drain the accumulated per-deme set_param audit journal.
    ///
    /// ## Returns
    /// A list of ``(deme, tick, param_id, old, new)`` tuples (change-only
    /// rows, commit order), cleared by the drain.
    fn drain_eco_journal(&mut self) -> Vec<crate::spatial::SpatialEcoJournalRow> {
        std::mem::take(&mut self.eco_journal)
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
        let n_demes = self.deme_variants.len();
        // Assemble one flat config per deme at the tick entry: the deme's
        // ecology column segment plus its shared genetics variant.
        let mut configs = Vec::with_capacity(n_demes);
        for (deme, &variant) in self.deme_variants.iter().enumerate() {
            let tensors = self
                .variants
                .get(variant)
                .ok_or_else(|| PyValueError::new_err("variant id out of range"))?;
            configs.push(SimConfig::assemble_deme(
                &self.blueprint,
                &self.ecology,
                tensors,
                deme,
            )?);
        }
        // Per-deme ECO scratch rows for OP_SET_PARAM; written back into
        // the ecology columns after all demes ticked.
        let mut eco_all = vec![0.0; n_demes * crate::hooks::N_ECO_PARAMS];
        for deme in 0..n_demes {
            for id in 0..crate::hooks::N_ECO_PARAMS {
                eco_all[deme * crate::hooks::N_ECO_PARAMS + id] = self.ecology.eco_value(id, deme);
            }
        }
        let tick = self.state_tick;
        let code = crate::spatial::run_spatial_tick_heterogeneous(
            &configs,
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
        .map_err(map_lifecycle_error)?;
        for deme in 0..n_demes {
            for id in 0..crate::hooks::N_ECO_PARAMS {
                self.ecology.set_eco_value(
                    id,
                    deme,
                    eco_all[deme * crate::hooks::N_ECO_PARAMS + id],
                );
            }
        }
        if code != 0 {
            // A stop keeps the modifications up to the boundary and
            // freezes the tick (plan 7.4).
            return Ok(self.state_tick);
        }
        // Migration stage: the frozen CSR and the live rate column, after
        // the lifecycle, on the same per-deme streams.  The zero-rate
        // identity mirrors the Python kernel's skip contract bitwise.
        let all_zero = self.ecology.migration_rate.iter().all(|&rate| rate <= 0.0);
        if !all_zero {
            let (ind, sperm) = if self.blueprint.stochastic {
                crate::spatial::migrate_csr_stochastic_rngs(
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
                crate::spatial::migrate_csr_deterministic(
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
    /// Returns ``PyValueError`` on shape mismatches.
    fn set_state(
        &mut self,
        individual_count_all: PyReadonlyArray4<'_, f64>,
        sperm_storage_all: PyReadonlyArray4<'_, f64>,
        tick: i64,
    ) -> PyResult<()> {
        // Validate both planes before committing either: a failed import
        // must leave the owned state untouched (transactional boundary).
        let n_demes = self.deme_variants.len();
        let ind = validate_stacked_ind(individual_count_all, &self.blueprint, n_demes)?;
        let sperm = validate_stacked_sperm(sperm_storage_all, &self.blueprint, n_demes)?;
        self.state_ind = ind;
        self.state_sperm = sperm;
        self.state_tick = tick;
        Ok(())
    }

    /// Install one deme's state slice (per-deme import handoff).
    ///
    /// ## Parameters
    /// - `deme`: Deme index to overwrite.
    /// - `individual_count`: ``(2, n_ages, n_ztypes)`` counts.
    /// - `sperm_storage`: ``(n_ages, n_ztypes, n_ztypes)`` storage.
    /// - `tick`: The authoritative tick (all demes share the tick axis).
    ///
    /// ## Errors
    /// Returns ``PyValueError`` for an out-of-range deme or a shape
    /// mismatch.
    #[pyo3(signature = (deme, individual_count, sperm_storage, tick))]
    fn set_deme_state(
        &mut self,
        deme: usize,
        individual_count: PyReadonlyArray3<'_, f64>,
        sperm_storage: PyReadonlyArray3<'_, f64>,
        tick: i64,
    ) -> PyResult<()> {
        let bp = &self.blueprint;
        if deme >= self.deme_variants.len() {
            return Err(PyValueError::new_err(format!(
                "deme {deme} out of range for {} demes",
                self.deme_variants.len()
            )));
        }
        let shape = individual_count.shape();
        if shape != [2, bp.n_ages, bp.n_ztypes] {
            return Err(PyValueError::new_err(format!(
                "individual_count must have shape (2, {}, {}), got {shape:?}",
                bp.n_ages, bp.n_ztypes
            )));
        }
        let sperm_shape = sperm_storage.shape();
        if sperm_shape != [bp.n_ages, bp.n_ztypes, bp.n_ztypes] {
            return Err(PyValueError::new_err(format!(
                "sperm_storage must have shape ({}, {}, {}), got {sperm_shape:?}",
                bp.n_ages, bp.n_ztypes, bp.n_ztypes
            )));
        }
        let ind_stride = 2 * bp.n_ages * bp.n_ztypes;
        let sperm_stride = bp.n_ages * bp.n_ztypes * bp.n_ztypes;
        let ind_src = individual_count
            .as_slice()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let sperm_src = sperm_storage
            .as_slice()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.state_ind[deme * ind_stride..(deme + 1) * ind_stride].copy_from_slice(ind_src);
        self.state_sperm[deme * sperm_stride..(deme + 1) * sperm_stride].copy_from_slice(sperm_src);
        self.state_tick = tick;
        Ok(())
    }
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
    Ok(array
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .to_vec())
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
    Ok(array
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .to_vec())
}
