//! CSR declarative hook interpreter.
//!
//! The flat-array layout and opcode values mirror
//! ``natal.hooks.types.HookProgram`` and
//! ``natal.hooks.runtime.csr_kernel``.  Only declarative hooks are supported;
//! custom hook callables are handled by falling back to the Python/Numba path.
#![allow(clippy::needless_range_loop)] // Index loops mirror the CSR kernel for parity review.
#![allow(clippy::too_many_arguments)] // execute_event mirrors the HookProgram flat-array signature.

use crate::rng::SessionRng;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use crate::rng::{binomial, clamp01, continuous_binomial, EPS};

/// Result code indicating the simulation should continue.
pub const RESULT_CONTINUE: i32 = 0;
/// Result code indicating a hook requested an early stop.
pub const RESULT_STOP: i32 = 1;

// Opcode values mirror ``OpType`` in ``natal.hooks.types``.
const OP_SCALE: i64 = 0;
const OP_SET: i64 = 1;
const OP_ADD: i64 = 2;
const OP_SUBTRACT: i64 = 3;
const OP_KILL: i64 = 4;
const OP_SAMPLE: i64 = 5;
const OP_STOP_IF_ZERO: i64 = 6;
const OP_STOP_IF_BELOW: i64 = 7;
const OP_STOP_IF_ABOVE: i64 = 8;
const OP_STOP_IF_EXTINCTION: i64 = 9;
const OP_SET_PARAM: i64 = 10;
const OP_CONVERT: i64 = 11;

// RPN value-expression token kinds mirror ``natal.hooks.types`` (RPN_*).
const RPN_LITERAL: i64 = 0;
const RPN_PARAM: i64 = 1;
const RPN_ADD: i64 = 2;
const RPN_SUB: i64 = 3;
const RPN_MUL: i64 = 4;
const RPN_DIV: i64 = 5;

/// Number of fixed ecology params addressable by ``OP_SET_PARAM``.
/// Generated from ``src/natal/parameters.jsonc`` together with the
/// wire column names (plan 5.4: one source, no hand-written copies).
pub use crate::eco_param_wire::ECO_PARAM_BOUNDS;
pub use crate::eco_param_wire::N_ECO_PARAMS;

/// Public alias of the ``OP_SET_PARAM`` opcode for cross-module checks
/// (e.g. session deserialization flagging programs with param writes).
pub const OP_SET_PARAM_PUBLIC: i64 = 10;

/// One audited ``OP_SET_PARAM`` transition handed to the session:
/// ``(tick, param_id, old, new)``, recorded only when the committed value
/// actually changed.  Spatial sessions wrap rows with the deme id (see
/// ``spatial::SpatialEcoJournalRow``).
pub type EcoJournalRow = (i64, usize, f64, f64, usize);

// Validity bounds per ECO param id live in the generated
// ``eco_param_wire`` module (re-exported above): the jsonc is the single
// source and the pytest freshness test fails on drift, so an inf/nan or
// out-of-bounds RPN result can never silently enter the session columns
// with bounds the Python flush channel would treat differently.

/// Validate one ECO param value against the wire bounds table.
///
/// ## Parameters
/// - `id`: Index into ``ECO_PARAM_COLUMNS`` (clamped defensively).
/// - `value`: Candidate committed value.
///
/// ## Returns
/// ``Ok(())`` when finite and within bounds; otherwise an ``Err`` message
/// naming the parameter and its allowed interval (mirrors the Python
/// ``'<name>' requires a value in [lo, hi], got <v>`` wording).
pub fn validate_eco_param(id: usize, value: f64) -> Result<(), String> {
    let idx = id.min(N_ECO_PARAMS - 1);
    let (lo, hi) = ECO_PARAM_BOUNDS[idx];
    if value.is_finite() && lo <= value && value <= hi {
        return Ok(());
    }
    Err(format!(
        "'{}' requires a value in [{}, {}], got {}",
        crate::contract::ECO_PARAM_COLUMNS[idx],
        lo,
        hi,
        value
    ))
}

// Condition opcodes mirror ``natal.hooks.types`` (atomic 0..6, RPN 100+).
const COND_ALWAYS: i64 = 0;
const COND_TICK_EQ: i64 = 1;
const COND_TICK_MOD: i64 = 2;
const COND_TICK_GE: i64 = 3;
const COND_TICK_LT: i64 = 4;
const COND_TICK_LE: i64 = 5;
const COND_TICK_GT: i64 = 6;
const COND_OP_AND: i64 = 100;
const COND_OP_OR: i64 = 101;
const COND_OP_NOT: i64 = 102;

/// Rust mirror of the Python CSR ``HookProgram`` flat arrays.
///
/// The CSR layout is intentionally identical to ``natal.hooks.types.HookProgram``
/// so the Rust interpreter can consume the same serialized arrays without
/// reformatting on the Python side.
///
/// ## Fields
/// - `n_events`: Number of lifecycle events (usually 4).
/// - `n_hooks`: Total number of hooks across all events.
/// - `hook_offsets`: Event -> hook range offsets.
/// - `op_offsets`: Hook -> operation range offsets.
/// - `op_types`: Operation opcode per operation.
/// - `zidx_offsets` / `zidx_data`: Genotype index ranges per operation.
/// - `age_offsets` / `age_data`: Age index ranges per operation.
/// - `sex_masks`: Two boolean sex masks per operation (female, male).
/// - `params`: Numeric parameter per operation.
/// - `condition_offsets` / `condition_types` / `condition_params`: RPN conditions.
/// - `deme_selector_*`: Per-hook deme selectors for spatial runs.
#[derive(Default)]
pub struct HookProgram {
    pub n_events: i64,
    pub n_hooks: i64,
    pub hook_offsets: Vec<i64>,
    pub op_offsets: Vec<i64>,
    pub op_types: Vec<i64>,
    pub zidx_offsets: Vec<i64>,
    pub zidx_data: Vec<i64>,
    pub age_offsets: Vec<i64>,
    pub age_data: Vec<i64>,
    pub sex_masks: Vec<bool>,
    pub params: Vec<f64>,
    pub condition_offsets: Vec<i64>,
    pub condition_types: Vec<i64>,
    pub condition_params: Vec<i64>,
    pub deme_selector_types: Vec<i64>,
    pub deme_selector_offsets: Vec<i64>,
    pub deme_selector_data: Vec<i64>,
    // OP_SET_PARAM data area (per-op columns; -1 = not a set_param op).
    pub sp_param_ids: Vec<i64>,
    pub sp_every: Vec<i64>,
    pub sp_start: Vec<i64>,
    // RPN token stream (CSR via rpn_offsets) + shared literal pool.
    pub rpn_offsets: Vec<i64>,
    pub rpn_kinds: Vec<i64>,
    pub rpn_payload: Vec<i64>,
    pub sp_literals: Vec<f64>,
    // OP_CONVERT data area (single-ZType endpoints per op; -1 otherwise).
    pub convert_source_z: Vec<i64>,
    pub convert_target_z: Vec<i64>,
    /// True when any op is ``OP_SET_PARAM``; lets sessions rebuild their
    /// per-tick config so in-run parameter writes take effect on the next
    /// tick (matching the Python lifecycle granularity).
    pub has_set_param: bool,
    /// Optional Python callables per in-tick event (first, early, late),
    /// fired at the event boundary after the CSR hooks ran.  Empty lists
    /// keep the kernels callback-free; each callable receives
    /// ``(ind, sperm, tick, deme_id)`` and a nonzero return stops the run.
    /// Each callback receives private copies of the arrays, and the copies
    /// are written back after the call so hook state mutations take effect.
    pub python_callbacks: Vec<Vec<Py<PyAny>>>,
    /// Completed spatial callback candidates awaiting stable deme-order merge.
    /// Stage cursors from demes stopped or failed during this tick.
    pub phase_marks: std::sync::Mutex<Vec<usize>>,
    pub callback_commits:
        std::sync::Mutex<Vec<(usize, crate::contract::Params, crate::contract::TensorSet)>>,
}

impl HookProgram {
    /// Fire the Python callbacks registered for one event boundary.
    ///
    /// ## Parameters
    /// - `event`: Event index (0 first, 1 early, 2 late).
    /// - `ind`: Current individual-count flat slice.  Each callback receives
    ///   a private copy; mutations are written back into this slice after
    ///   the call so later callbacks and lifecycle stages observe them.
    /// - `sperm`: Current sperm-storage flat slice (same copy/write-back
    ///   semantics as *ind*; often empty for discrete models).
    /// - `tick`: Current tick.
    /// - `deme_id`: Current deme id.
    ///
    /// ## Returns
    /// ``Ok(0)`` to continue, ``Ok(nonzero)`` when a callback requested a
    /// stop, or an error string when a callback raised.
    ///
    /// ## Notes
    /// The GIL is already held inside every pymethod, so ``with_gil`` here
    /// is a cheap re-entry, not a lock acquisition from a foreign thread.
    /// The write-back keeps Python-callback semantics identical across the
    /// Rust, Numba, and reference backends: a hook's state writes are
    /// visible to subsequent hooks and lifecycle stages.
    pub fn fire_python_callbacks(
        &self,
        event: usize,
        ind: &mut [f64],
        sperm: &mut [f64],
        tick: i64,
        deme_id: i64,
        rng: &mut SessionRng,
        eco_values: &mut [f64],
        eco_ctx: &mut Option<crate::lifecycle::EcoCtx<'_>>,
    ) -> Result<i32, String> {
        let Some(callbacks) = self.python_callbacks.get(event) else {
            return Ok(0);
        };
        if callbacks.is_empty() {
            return Ok(0);
        }
        if let Some(ctx) = eco_ctx.as_mut() {
            ctx.commit(eco_values)?;
        }
        Python::with_gil(|py| -> PyResult<i32> {
            for callback in callbacks {
                let transactional = callback
                    .bind(py)
                    .getattr("__natal_transaction__")
                    .and_then(|value| value.extract::<bool>())
                    .unwrap_or(false);
                let arrays = if transactional {
                    None
                } else {
                    Some((
                        PyArray1::from_slice(py, ind),
                        PyArray1::from_slice(py, sperm),
                    ))
                };
                let transaction = if transactional {
                    let ctx = eco_ctx.as_ref().ok_or_else(|| {
                        pyo3::exceptions::PyRuntimeError::new_err(
                            "callback requires a session transaction",
                        )
                    })?;
                    Some(Py::new(
                        py,
                        crate::hook_transaction::HookTransaction {
                            active: true,
                            parameters_changed: false,
                            blueprint: ctx.bp.clone(),
                            params: ctx.params.clone(),
                            genetics: ctx
                                .updated_genetics
                                .as_ref()
                                .unwrap_or(ctx.genetics)
                                .clone(),
                            rng: rng.clone(),
                            state_ind: ind.to_vec(),
                            state_sperm: sperm.to_vec(),
                            state_arrays: None,
                        },
                    )?)
                } else {
                    None
                };
                let outcome = if let Some(tx) = transaction.as_ref() {
                    callback
                        .bind(py)
                        .call1((py.None(), py.None(), tick, deme_id, tx.clone_ref(py)))
                } else {
                    let (ind_arr, sperm_arr) = arrays.as_ref().expect("raw callback arrays exist");
                    callback
                        .bind(py)
                        .call1((ind_arr.clone(), sperm_arr.clone(), tick, deme_id))
                }
                .and_then(|value| value.extract::<i32>());
                // Invalidation happens even when Python raises. Retained samplers
                // cannot advance either the candidate or the live stream later.
                if let Some(tx) = transaction.as_ref() {
                    tx.borrow_mut(py).active = false;
                }
                let result = outcome?;
                let state_candidate = if let Some(tx) = transaction.as_ref() {
                    tx.borrow(py).candidate_state(py)?
                } else {
                    let (ind_arr, sperm_arr) = arrays.as_ref().expect("raw callback arrays exist");
                    let a = ind_arr.readonly().as_slice()?.to_vec();
                    let b = sperm_arr.readonly().as_slice()?.to_vec();
                    if a.len() != ind.len()
                        || b.len() != sperm.len()
                        || a.iter().chain(&b).any(|v| !v.is_finite() || *v < 0.0)
                    {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Hook state must preserve shape and contain finite nonnegative counts",
                        ));
                    }
                    Some((a, b))
                };
                if let Some(tx) = transaction.as_ref() {
                    let candidate = tx.borrow(py);
                    // Read-only and state-only callbacks have no parameter
                    // products to validate, clone, or publish into the bank.
                    if candidate.parameters_changed {
                        candidate.params.validate(&candidate.blueprint)?;
                        candidate.genetics.validate(&candidate.blueprint)?;
                        let ctx = eco_ctx.as_mut().expect("transaction requires context");
                        for (id, value) in eco_values.iter_mut().enumerate() {
                            *value = candidate.params.eco_value(id, ctx.deme);
                        }
                        *ctx.params = candidate.params.clone();
                        ctx.updated_genetics = Some(candidate.genetics.clone());
                        let mut commits = self
                            .callback_commits
                            .lock()
                            .expect("callback queue poisoned");
                        let update = (
                            deme_id as usize,
                            candidate.params.clone(),
                            candidate.genetics.clone(),
                        );
                        if let Some(previous) =
                            commits.iter_mut().find(|entry| entry.0 == deme_id as usize)
                        {
                            *previous = update;
                        } else {
                            commits.push(update);
                        }
                    }
                    *rng = candidate.rng.clone();
                }
                if let Some((ind_candidate, sperm_candidate)) = state_candidate {
                    ind.copy_from_slice(&ind_candidate);
                    sperm.copy_from_slice(&sperm_candidate);
                }
                if result != 0 {
                    return Ok(result);
                }
            }
            Ok(0)
        })
        .map_err(crate::hook_transaction::preserve_error)
    }
}

/// Evaluate one atomic tick condition.
///
/// Atomic conditions are ``always``, ``tick ==``, ``tick %``, and comparison
/// operators.  Values above ``COND_TICK_GT`` are not atomic and are handled by
/// the RPN evaluator.
///
/// ## Parameters
/// - `cond_type`: Condition opcode.
/// - `cond_param`: Numeric parameter.
/// - `tick`: Current simulation tick.
///
/// ## Returns
/// ``true`` when the atomic condition holds.
fn atomic_condition(cond_type: i64, cond_param: i64, tick: i64) -> bool {
    match cond_type {
        COND_ALWAYS => true,
        COND_TICK_EQ => tick == cond_param,
        COND_TICK_MOD => cond_param > 0 && tick % cond_param == 0,
        COND_TICK_GE => tick >= cond_param,
        COND_TICK_LT => tick < cond_param,
        COND_TICK_LE => tick <= cond_param,
        COND_TICK_GT => tick > cond_param,
        _ => cond_type < COND_OP_AND,
    }
}

/// Evaluate an RPN condition program.
///
/// The condition arrays form a stack program: atomic conditions push booleans
/// and ``AND`` / ``OR`` / ``NOT`` operators consume stack entries.
///
/// ## Parameters
/// - `cond_types`: Condition opcode array.
/// - `cond_params`: Condition parameter array.
/// - `cond_start`: Start index of this operation's condition.
/// - `cond_end`: End index (exclusive).
/// - `tick`: Current simulation tick.
///
/// ## Returns
/// ``true`` if the condition program evaluates to a single true value.
fn eval_condition(
    cond_types: &[i64],
    cond_params: &[i64],
    cond_start: usize,
    cond_end: usize,
    tick: i64,
) -> bool {
    // RPN evaluation: atomic tokens push 0/1; logical operators consume
    // the stack.  An empty program is treated as always-true.
    if cond_end <= cond_start {
        return true;
    }
    let mut stack: Vec<i32> = Vec::with_capacity(cond_end - cond_start + 1);
    for idx in cond_start..cond_end {
        let token_type = cond_types[idx];
        let token_param = cond_params[idx];
        if token_type <= COND_TICK_GT {
            stack.push(if atomic_condition(token_type, token_param, tick) {
                1
            } else {
                0
            });
            continue;
        }
        if token_type == COND_OP_NOT {
            if stack.is_empty() {
                return false;
            }
            let top = stack.len() - 1;
            stack[top] = if stack[top] == 0 { 1 } else { 0 };
            continue;
        }
        if token_type == COND_OP_AND {
            if stack.len() < 2 {
                return false;
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            stack.push(if lhs != 0 && rhs != 0 { 1 } else { 0 });
            continue;
        }
        if token_type == COND_OP_OR {
            if stack.len() < 2 {
                return false;
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            stack.push(if lhs != 0 || rhs != 0 { 1 } else { 0 });
            continue;
        }
        return false;
    }
    stack.len() == 1 && stack[0] != 0
}

/// Return whether a hook's deme selector matches the current ``deme_id``.
///
/// Selector types: 0 = all demes, 1 = single deme, 2 = inclusive range
/// ``[lo, hi)``, 3 = explicit list.
///
/// ## Parameters
/// - `program`: Hook program containing selector arrays.
/// - `hook_idx`: Index of the hook.
/// - `deme_id`: Current deme id.
///
/// ## Returns
/// ``true`` when the hook should run for this deme.
fn deme_matches(program: &HookProgram, hook_idx: usize, deme_id: i64) -> bool {
    // Selector 0 is global; 1 is a single deme; 2 is [lo, hi); 3 is a list.
    let sel_type = program.deme_selector_types[hook_idx];
    let start = program.deme_selector_offsets[hook_idx] as usize;
    let end = program.deme_selector_offsets[hook_idx + 1] as usize;
    match sel_type {
        0 => true,
        1 => program.deme_selector_data.get(start).copied() == Some(deme_id),
        2 => {
            if start + 1 < program.deme_selector_data.len() {
                let lo = program.deme_selector_data[start];
                let hi = program.deme_selector_data[start + 1];
                deme_id >= lo && deme_id < hi
            } else {
                false
            }
        }
        3 => {
            for idx in start..end {
                if program.deme_selector_data.get(idx).copied() == Some(deme_id) {
                    return true;
                }
            }
            false
        }
        _ => true,
    }
}

/// Sample survivors from ``n_base`` with deterministic or stochastic semantics.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `n_base`: Current count before mortality.
/// - `survival_prob`: Survival probability.
/// - `stochastic_flag`: Whether to sample stochastically.
/// - `dirichlet_flag`: Whether continuous/Dirichlet sampling is enabled.
///
/// ## Returns
/// The number of survivors as ``f64``.
fn sample_survivors(
    rng: &mut SessionRng,
    n_base: f64,
    survival_prob: f64,
    stochastic_flag: bool,
    dirichlet_flag: bool,
) -> f64 {
    // Deterministic path multiplies by probability.
    // Stochastic path uses discrete binomial or continuous binomial.
    if n_base <= 0.0 {
        return 0.0;
    }
    if stochastic_flag {
        if dirichlet_flag {
            return continuous_binomial(rng, n_base, survival_prob);
        }
        return binomial(rng, n_base.round() as i64, survival_prob);
    }
    n_base * survival_prob
}

/// Apply a target count to a male/non-sperm-storing state slot.
///
/// If the target is below the current count, survivors are sampled with
/// probability ``target / current``; otherwise the target is kept unchanged.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `current_count`: Current count.
/// - `target_count`: Desired count after operation.
/// - `stochastic_flag`: Whether to sample stochastically.
/// - `dirichlet_flag`: Whether continuous sampling is enabled.
///
/// ## Returns
/// The new count for the slot.
fn apply_target_without_sperm(
    rng: &mut SessionRng,
    current_count: f64,
    target_count: f64,
    stochastic_flag: bool,
    dirichlet_flag: bool,
) -> f64 {
    // If the target is not a reduction, keep the current value.
    // Otherwise sample survivors with probability target/current.
    let current_count = if stochastic_flag && !dirichlet_flag {
        current_count.round()
    } else {
        current_count
    };
    if target_count >= current_count {
        return target_count;
    }
    if current_count <= 0.0 {
        return 0.0;
    }
    let survival_prob = clamp01(target_count / current_count);
    sample_survivors(
        rng,
        current_count,
        survival_prob,
        stochastic_flag,
        dirichlet_flag,
    )
}

/// Apply a target count to a female slot while scaling stored sperm.
///
/// For female slots, reducing the count must also reduce stored sperm.  The
/// function scales each sperm category and separately samples surviving
/// virgins so the total remains consistent.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `current_count`: Current female count.
/// - `target_count`: Desired count after operation.
/// - `sperm_row`: Mutable sperm counts for this female genotype.
/// - `stochastic_flag`: Whether to sample stochastically.
/// - `dirichlet_flag`: Whether continuous sampling is enabled.
///
/// ## Returns
/// The new female count after applying mortality to sperm and virgins.
///
/// ## Panics
/// Panics if the state is inconsistent (`n_virgins < 0`).
fn apply_target_with_sperm(
    rng: &mut SessionRng,
    current_count: f64,
    target_count: f64,
    sperm_row: &mut [f64],
    stochastic_flag: bool,
    dirichlet_flag: bool,
) -> f64 {
    // Female reductions must also reduce stored sperm.
    // Scale each sperm category and sample surviving virgins independently.
    let current_count = if stochastic_flag && !dirichlet_flag {
        current_count.round()
    } else {
        current_count
    };
    if target_count >= current_count {
        return target_count;
    }
    if current_count <= 0.0 {
        for slot in sperm_row.iter_mut() {
            *slot = 0.0;
        }
        return 0.0;
    }
    let survival_prob = clamp01(target_count / current_count);
    if !stochastic_flag {
        for slot in sperm_row.iter_mut() {
            *slot *= survival_prob;
        }
        return target_count;
    }

    let total_sperm: f64 = sperm_row.iter().sum();
    let mut n_virgins_raw = current_count - total_sperm;
    if n_virgins_raw >= -EPS {
        n_virgins_raw = n_virgins_raw.max(0.0);
    }
    if n_virgins_raw < 0.0 {
        panic!(
            "Invalid state: n_virgins < 0 in apply_target_with_sperm: \
             n_virgins_raw={}, n_f_raw={}, total_sperm={}",
            n_virgins_raw, current_count, total_sperm
        );
    }
    let n_virgins = if dirichlet_flag {
        n_virgins_raw
    } else {
        n_virgins_raw.round()
    };

    let mut new_sperm_sum = 0.0;
    for gm_idx in 0..sperm_row.len() {
        let n_sperm = if dirichlet_flag {
            sperm_row[gm_idx]
        } else {
            sperm_row[gm_idx].round()
        };
        sperm_row[gm_idx] = sample_survivors(rng, n_sperm, survival_prob, true, dirichlet_flag);
        new_sperm_sum += sperm_row[gm_idx];
    }
    new_sperm_sum + sample_survivors(rng, n_virgins, survival_prob, true, dirichlet_flag)
}

/// Evaluate one ``OP_SET_PARAM`` RPN value program with a stack machine.
///
/// Mirrors ``_eval_rpn_value`` in the Python CSR kernel bit-for-bit:
/// operands push literals or the current ecology values; operators pop
/// two and push the result.  Division by zero follows explicit IEEE-754
/// semantics (``x/0`` = ±inf, ``0/0`` = nan) so the two interpreters and
/// the Numba kernel never diverge through error handling.
///
/// ## Parameters
/// - `program`: Hook program carrying the RPN token arrays.
/// - `rpn_start`: First token index of this op's program.
/// - `rpn_end`: One past the last token index.
/// - `eco_values`: Current ecology values indexed by ECO param id.
///
/// ## Returns
/// The evaluated expression value.
fn eval_rpn_value(
    program: &HookProgram,
    rpn_start: usize,
    rpn_end: usize,
    eco_values: &[f64],
) -> f64 {
    let mut stack: Vec<f64> = Vec::with_capacity(rpn_end - rpn_start + 1);
    for idx in rpn_start..rpn_end {
        let kind = program.rpn_kinds[idx];
        match kind {
            RPN_LITERAL => stack.push(program.sp_literals[program.rpn_payload[idx] as usize]),
            RPN_PARAM => stack.push(eco_values[program.rpn_payload[idx] as usize]),
            _ => {
                let rhs = stack.pop().unwrap_or(f64::NAN);
                let lhs = stack.pop().unwrap_or(f64::NAN);
                let value = match kind {
                    RPN_ADD => lhs + rhs,
                    RPN_SUB => lhs - rhs,
                    RPN_MUL => lhs * rhs,
                    RPN_DIV => {
                        // Explicit IEEE semantics (see docstring).
                        if rhs == 0.0 {
                            if lhs > 0.0 {
                                f64::INFINITY
                            } else if lhs < 0.0 {
                                f64::NEG_INFINITY
                            } else {
                                f64::NAN
                            }
                        } else {
                            lhs / rhs
                        }
                    }
                    _ => f64::NAN,
                };
                stack.push(value);
            }
        }
    }
    stack.first().copied().unwrap_or(f64::NAN)
}

/// Return how many of `n_base` individuals convert at `prob`.
///
/// Shares the sampling conventions of [`sample_survivors`] so
/// ``OP_CONVERT`` draws come from the same CSR sampling channel as the
/// Python kernel's ``_convert_count``.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `n_base`: Count eligible for conversion.
/// - `prob`: Per-individual conversion probability.
/// - `stochastic_flag`: Whether to sample stochastically.
/// - `dirichlet_flag`: Whether continuous sampling is enabled.
///
/// ## Returns
/// The converted count (continuous in Dirichlet mode).
fn convert_count(
    rng: &mut SessionRng,
    n_base: f64,
    prob: f64,
    stochastic_flag: bool,
    dirichlet_flag: bool,
) -> f64 {
    if n_base <= 0.0 {
        return 0.0;
    }
    if stochastic_flag {
        if dirichlet_flag {
            return continuous_binomial(rng, n_base, prob);
        }
        return binomial(rng, n_base.round() as i64, prob);
    }
    n_base * prob
}

impl HookProgram {
    /// Execute all CSR hooks for one lifecycle event in priority order.
    ///
    /// The method iterates hooks belonging to ``event_id``, filters by deme
    /// selector, evaluates each operation's condition, and applies mutating or
    /// stop-checking operations.  It mirrors ``natal.hooks.runtime.csr_kernel``.
    ///
    /// ## Parameters
    /// - `rng`: Random number generator for stochastic operations.
    /// - `event_id`: Lifecycle event index (first/early/late/finish).
    /// - `individual_count`: Mutable flat state array.
    /// - `sperm_storage`: Mutable flat sperm array (empty for discrete).
    /// - `n_sexes`: Number of sexes.
    /// - `n_ages`: Number of age classes.
    /// - `n_ztypes`: Number of zygote types.
    /// - `tick`: Current tick.
    /// - `stochastic`: Stochastic sampling flag.
    /// - `continuous_sampling`: Continuous sampling flag.
    /// - `deme_id`: Current deme id.
    /// - `eco_values`: Live scratch slice indexed by ``ECO_PARAM_NAMES``
    ///   position for ``OP_SET_PARAM``: the interpreter evaluates RPN
    ///   expressions against the current values and writes new values
    ///   back into this slice; the session owns the write-back into its
    ///   ecology columns.
    ///
    /// ## Returns
    /// ``RESULT_CONTINUE`` (0) or ``RESULT_STOP`` (1) if a stop operation triggered.
    #[allow(clippy::too_many_arguments)] // Mirrors the Python flat-array kernel signature.
    pub fn execute_event(
        &self,
        rng: &mut SessionRng,
        event_id: i64,
        individual_count: &mut [f64],
        sperm_storage: &mut [f64],
        n_sexes: usize,
        n_ages: usize,
        n_ztypes: usize,
        tick: i64,
        stochastic: bool,
        continuous_sampling: bool,
        deme_id: i64,
        eco_values: &mut [f64],
    ) -> i32 {
        // Iterate hooks in serialized order, respecting deme selectors.
        // For each operation: evaluate condition, apply mutation or stop check.
        // Stop operations short-circuit the whole event immediately.
        if event_id < 0 || event_id >= self.n_events || self.n_hooks == 0 {
            return RESULT_CONTINUE;
        }
        let hook_start = self.hook_offsets[event_id as usize] as usize;
        let hook_end = self.hook_offsets[event_id as usize + 1] as usize;

        for hook_idx in hook_start..hook_end {
            if hook_idx >= self.n_hooks as usize || !deme_matches(self, hook_idx, deme_id) {
                continue;
            }
            let op_start = self.op_offsets[hook_idx] as usize;
            let op_end = self.op_offsets[hook_idx + 1] as usize;

            for op_idx in op_start..op_end {
                let cond_start = self.condition_offsets[op_idx] as usize;
                let cond_end = self.condition_offsets[op_idx + 1] as usize;
                if !eval_condition(
                    &self.condition_types,
                    &self.condition_params,
                    cond_start,
                    cond_end,
                    tick,
                ) {
                    continue;
                }

                let op_type = self.op_types[op_idx];
                let param = self.params[op_idx];
                let zidx_start = self.zidx_offsets[op_idx] as usize;
                let zidx_end = self.zidx_offsets[op_idx + 1] as usize;
                let age_start = self.age_offsets[op_idx] as usize;
                let age_end = self.age_offsets[op_idx + 1] as usize;
                let sex_female = self.sex_masks[op_idx * 2];
                let sex_male = self.sex_masks[op_idx * 2 + 1];

                if op_type <= OP_SAMPLE {
                    for sex_idx in 0..n_sexes {
                        let selected = if sex_idx == 0 {
                            sex_female
                        } else if sex_idx == 1 {
                            sex_male
                        } else {
                            false
                        };
                        if !selected {
                            continue;
                        }
                        for age_ptr in age_start..age_end {
                            let age = self.age_data[age_ptr] as usize;
                            if age >= n_ages {
                                continue;
                            }
                            for zidx_ptr in zidx_start..zidx_end {
                                let zidx = self.zidx_data[zidx_ptr] as usize;
                                if zidx >= n_ztypes {
                                    continue;
                                }
                                let flat = (sex_idx * n_ages + age) * n_ztypes + zidx;
                                let current = individual_count[flat];
                                let target = match op_type {
                                    OP_SCALE => (current * param).max(0.0),
                                    OP_SET => param.max(0.0),
                                    OP_ADD => (current + param).max(0.0),
                                    OP_SUBTRACT => (current - param).max(0.0),
                                    OP_KILL => (current * (1.0 - param)).max(0.0),
                                    OP_SAMPLE => current.min(param.max(0.0)),
                                    _ => current,
                                };

                                individual_count[flat] = if sex_idx == 0
                                    && !sperm_storage.is_empty()
                                {
                                    // Models without a sperm dimension (discrete
                                    // generation) pass an empty slice; virgin
                                    // females there have no storage to displace.
                                    let row = &mut sperm_storage[(age * n_ztypes + zidx) * n_ztypes
                                        ..(age * n_ztypes + zidx + 1) * n_ztypes];
                                    apply_target_with_sperm(
                                        rng,
                                        current,
                                        target,
                                        row,
                                        stochastic,
                                        continuous_sampling,
                                    )
                                } else {
                                    apply_target_without_sperm(
                                        rng,
                                        current,
                                        target,
                                        stochastic,
                                        continuous_sampling,
                                    )
                                };
                            }
                        }
                    }
                } else if op_type == OP_SET_PARAM {
                    // Schedule check + RPN evaluation + eco write.  The
                    // interpreter only computes; the session owns the
                    // write-back into its ecology columns.
                    let start_tick = self.sp_start[op_idx];
                    let every_ticks = self.sp_every[op_idx];
                    if tick >= start_tick && (tick - start_tick) % every_ticks == 0 {
                        let rpn_start = self.rpn_offsets[op_idx] as usize;
                        let rpn_end = self.rpn_offsets[op_idx + 1] as usize;
                        let value = eval_rpn_value(self, rpn_start, rpn_end, eco_values);
                        let param_id = self.sp_param_ids[op_idx];
                        if param_id >= 0 && (param_id as usize) < eco_values.len() {
                            eco_values[param_id as usize] = value;
                        }
                    }
                } else if op_type == OP_CONVERT {
                    // One-to-one probabilistic ZType migration: males move
                    // plain counts; females move the virgin part and every
                    // sperm bucket atomically (female label follows the
                    // row, male axis untouched).  Every moved unit is
                    // subtracted from the source and added to the target,
                    // so totals are conserved by construction.
                    let src_z = self.convert_source_z[op_idx] as usize;
                    let dst_z = self.convert_target_z[op_idx] as usize;
                    let prob = param;
                    if src_z < n_ztypes && dst_z < n_ztypes {
                        let has_sperm = !sperm_storage.is_empty();
                        for age in 0..n_ages {
                            let male_flat = (n_ages + age) * n_ztypes + src_z;
                            let moved_male = convert_count(
                                rng,
                                individual_count[male_flat],
                                prob,
                                stochastic,
                                continuous_sampling,
                            );
                            individual_count[male_flat] -= moved_male;
                            individual_count[(n_ages + age) * n_ztypes + dst_z] += moved_male;

                            if has_sperm {
                                // Virgin count is fixed before the bucket
                                // loop: the loop moves buckets out of the
                                // source row, so the pre-loop row sum is
                                // the mated total.
                                let mut sperm_row_sum = 0.0;
                                for mz in 0..n_ztypes {
                                    sperm_row_sum +=
                                        sperm_storage[(age * n_ztypes + src_z) * n_ztypes + mz];
                                }
                                let mut moved_mated = 0.0;
                                for mz in 0..n_ztypes {
                                    let bucket_flat = (age * n_ztypes + src_z) * n_ztypes + mz;
                                    let moved_bucket = convert_count(
                                        rng,
                                        sperm_storage[bucket_flat],
                                        prob,
                                        stochastic,
                                        continuous_sampling,
                                    );
                                    sperm_storage[bucket_flat] -= moved_bucket;
                                    sperm_storage[(age * n_ztypes + dst_z) * n_ztypes + mz] +=
                                        moved_bucket;
                                    moved_mated += moved_bucket;
                                }
                                let virgins = (individual_count[age * n_ztypes + src_z]
                                    - sperm_row_sum)
                                    .max(0.0);
                                let moved_virgin = convert_count(
                                    rng,
                                    virgins,
                                    prob,
                                    stochastic,
                                    continuous_sampling,
                                );
                                individual_count[age * n_ztypes + src_z] -=
                                    moved_mated + moved_virgin;
                                individual_count[age * n_ztypes + dst_z] +=
                                    moved_mated + moved_virgin;
                            } else {
                                let female_flat = age * n_ztypes + src_z;
                                let moved_female = convert_count(
                                    rng,
                                    individual_count[female_flat],
                                    prob,
                                    stochastic,
                                    continuous_sampling,
                                );
                                individual_count[female_flat] -= moved_female;
                                individual_count[age * n_ztypes + dst_z] += moved_female;
                            }
                        }
                    }
                }

                if (OP_STOP_IF_ZERO..=OP_STOP_IF_ABOVE).contains(&op_type) {
                    let mut selected_total = 0.0;
                    for sex_idx in 0..n_sexes {
                        let selected = if sex_idx == 0 {
                            sex_female
                        } else if sex_idx == 1 {
                            sex_male
                        } else {
                            false
                        };
                        if !selected {
                            continue;
                        }
                        for age_ptr in age_start..age_end {
                            let age = self.age_data[age_ptr] as usize;
                            if age >= n_ages {
                                continue;
                            }
                            for zidx_ptr in zidx_start..zidx_end {
                                let zidx = self.zidx_data[zidx_ptr] as usize;
                                if zidx >= n_ztypes {
                                    continue;
                                }
                                selected_total +=
                                    individual_count[(sex_idx * n_ages + age) * n_ztypes + zidx];
                            }
                        }
                    }
                    if op_type == OP_STOP_IF_ZERO && selected_total <= 0.0 {
                        return RESULT_STOP;
                    }
                    if op_type == OP_STOP_IF_BELOW && selected_total < param {
                        return RESULT_STOP;
                    }
                    if op_type == OP_STOP_IF_ABOVE && selected_total > param {
                        return RESULT_STOP;
                    }
                } else if op_type == OP_STOP_IF_EXTINCTION
                    && individual_count.iter().sum::<f64>() <= 0.0
                {
                    return RESULT_STOP;
                }
            }
        }
        RESULT_CONTINUE
    }
}

#[cfg(test)]
mod setparam_convert_tests {
    //! Unit tests for the OP_SET_PARAM RPN interpreter and the
    //! OP_CONVERT migration, mirroring the Python-side invariants.

    use super::*;

    /// Build a minimal one-hook program with the given op payload.
    fn single_op_program(op_type: i64, param: f64) -> HookProgram {
        // One hook on event 1, one op with all-deme selectors, a dummy
        // ztype/age span, and an always-true condition.
        HookProgram {
            n_events: 4,
            n_hooks: 1,
            hook_offsets: vec![0, 0, 1, 1, 1],
            op_offsets: vec![0, 1],
            op_types: vec![op_type],
            params: vec![param],
            zidx_offsets: vec![0, 1],
            zidx_data: vec![0],
            age_offsets: vec![0, 1],
            age_data: vec![0],
            sex_masks: vec![false, false],
            condition_offsets: vec![0, 1],
            condition_types: vec![COND_ALWAYS],
            condition_params: vec![0],
            deme_selector_types: vec![0],
            deme_selector_offsets: vec![0, 0],
            ..HookProgram::default()
        }
    }

    fn seeded() -> SessionRng {
        crate::rng::new_rng(42)
    }

    /// The generated wire tables are internally consistent: one bounds
    /// row per column, and every row is a real interval.  Value-level
    /// freshness against ``parameters.jsonc`` is enforced by the Python
    /// generator's ``--check`` test — no second hand-written copy here.
    #[test]
    fn eco_param_wire_tables_are_consistent() {
        assert_eq!(
            ECO_PARAM_BOUNDS.len(),
            crate::contract::ECO_PARAM_COLUMNS.len()
        );
        // Compile-time completeness: the generated wire table must never
        // be empty (a const assert avoids clippy's const_is_empty lint on
        // the runtime form).
        const _: () = assert!(!crate::contract::ECO_PARAM_COLUMNS.is_empty());
        for (id, (lo, hi)) in ECO_PARAM_BOUNDS.iter().enumerate() {
            assert!(lo <= hi, "bounds row {id} is not an interval");
        }
    }

    /// Validation accepts boundary values and rejects non-finite /
    /// out-of-bounds ones with a message naming the parameter.
    #[test]
    fn validate_eco_param_accepts_bounds_and_rejects_inf_nan() {
        assert!(validate_eco_param(0, 0.0).is_ok());
        assert!(validate_eco_param(0, 1e12).is_ok());
        assert!(validate_eco_param(2, 1.0).is_ok());
        let inf_err = validate_eco_param(0, f64::INFINITY).unwrap_err();
        assert!(
            inf_err.contains("carrying_capacity") && inf_err.contains("inf"),
            "message names the parameter and the offending value: {inf_err}"
        );
        assert!(validate_eco_param(1, f64::NAN).is_err());
        assert!(validate_eco_param(2, 1.5).is_err());
        assert!(validate_eco_param(3, -0.1).is_err());
        assert!(validate_eco_param(4, 2e6).is_err());
    }

    #[test]
    fn rpn_evaluates_expressions_against_current_values() {
        // "K * 0.95" with K = 200 -> 190.  Tokens: [param0, lit0, mul].
        let mut program = single_op_program(OP_SET_PARAM, 0.0);
        program.sp_param_ids = vec![0];
        program.sp_every = vec![1];
        program.sp_start = vec![0];
        program.rpn_offsets = vec![0, 3];
        program.rpn_kinds = vec![RPN_PARAM, RPN_LITERAL, RPN_MUL];
        program.rpn_payload = vec![0, 0, 0];
        program.sp_literals = vec![0.95];

        let mut eco = [200.0, 0.0, 0.0, 0.0, 0.0];
        let mut rng = seeded();
        let mut ind = vec![0.0; 2];
        let result = program.execute_event(
            &mut rng,
            1,
            &mut ind,
            &mut [],
            2,
            1,
            1,
            5,
            false,
            false,
            0,
            &mut eco,
        );
        assert_eq!(result, RESULT_CONTINUE);
        assert!((eco[0] - 190.0).abs() < 1e-12, "K * 0.95 with K=200 -> 190");
    }

    #[test]
    fn rpn_division_by_zero_follows_ieee_semantics() {
        // "1 / (K - K)" with K = 50 -> 1 / 0 -> +inf.
        let mut program = single_op_program(OP_SET_PARAM, 0.0);
        program.sp_param_ids = vec![0];
        program.sp_every = vec![1];
        program.sp_start = vec![0];
        program.rpn_offsets = vec![0, 5];
        program.rpn_kinds = vec![RPN_LITERAL, RPN_PARAM, RPN_PARAM, RPN_SUB, RPN_DIV];
        program.rpn_payload = vec![0, 0, 0, 0, 0];
        program.sp_literals = vec![1.0];

        let mut eco = [50.0, 0.0, 0.0, 0.0, 0.0];
        let mut rng = seeded();
        let mut ind = vec![0.0; 2];
        program.execute_event(
            &mut rng,
            1,
            &mut ind,
            &mut [],
            2,
            1,
            1,
            0,
            false,
            false,
            0,
            &mut eco,
        );
        assert!(
            eco[0].is_infinite() && eco[0] > 0.0,
            "1/0 -> +inf, got {}",
            eco[0]
        );
    }

    #[test]
    fn set_param_respects_every_and_start_schedule() {
        // every=10, start=5: fires at ticks 5, 15, 25 — not 0..4 or 10.
        let mut program = single_op_program(OP_SET_PARAM, 0.0);
        program.sp_param_ids = vec![0];
        program.sp_every = vec![10];
        program.sp_start = vec![5];
        program.rpn_offsets = vec![0, 1];
        program.rpn_kinds = vec![RPN_LITERAL];
        program.rpn_payload = vec![0];
        program.sp_literals = vec![-1.0];

        for tick in [0_i64, 4, 5, 10, 14, 15, 24, 25] {
            let mut eco = [7.0, 0.0, 0.0, 0.0, 0.0];
            let mut rng = seeded();
            let mut ind = vec![0.0; 2];
            program.execute_event(
                &mut rng,
                1,
                &mut ind,
                &mut [],
                2,
                1,
                1,
                tick,
                false,
                false,
                0,
                &mut eco,
            );
            let expected = if tick >= 5 && (tick - 5) % 10 == 0 {
                -1.0
            } else {
                7.0
            };
            assert_eq!(eco[0], expected, "tick {tick}");
        }
    }

    #[test]
    fn deterministic_convert_conserves_totals_with_sperm() {
        // 2 ages, 3 ztypes, sperm (age, female_z, male_z).
        let mut program = single_op_program(OP_CONVERT, 0.25);
        program.convert_source_z = vec![0];
        program.convert_target_z = vec![1];

        let n_ages = 2;
        let n_ztypes = 3;
        let mut ind = vec![0.0; 2 * n_ages * n_ztypes];
        let mut sperm = vec![0.0; n_ages * n_ztypes * n_ztypes];
        // Female A|A: 20 per age (6 mated via sperm buckets, 14 virgin).
        // Male A|A: 15 per age.
        for age in 0..n_ages {
            ind[age * n_ztypes] = 20.0;
            ind[(n_ages + age) * n_ztypes] = 15.0;
            sperm[age * n_ztypes * n_ztypes + 1] = 3.0;
            sperm[age * n_ztypes * n_ztypes + 2] = 3.0;
        }
        let ind_before = ind.clone();
        let sperm_before = sperm.clone();
        let total_before: f64 = ind.iter().sum::<f64>() + sperm.iter().sum::<f64>();

        let mut rng = seeded();
        let mut eco = [0.0; N_ECO_PARAMS];
        let result = program.execute_event(
            &mut rng, 1, &mut ind, &mut sperm, 2, n_ages, n_ztypes, 0, false, false, 0, &mut eco,
        );
        assert_eq!(result, RESULT_CONTINUE);

        let total_after: f64 = ind.iter().sum::<f64>() + sperm.iter().sum::<f64>();
        assert!(
            (total_before - total_after).abs() < 1e-12,
            "total conserved"
        );

        // Males: 15 -> 11.25 per age (25 % deterministic migration).
        for age in 0..n_ages {
            assert!(
                (ind[(n_ages + age) * n_ztypes] - 11.25).abs() < 1e-12,
                "male deterministic migration"
            );
            assert!(
                (ind[(n_ages + age) * n_ztypes + 1] - 3.75).abs() < 1e-12,
                "male target gains exactly the moved count"
            );
            // Females: virgins 14 and each 3-count sperm bucket migrate at
            // 25 % -> 0.75 + 0.75 + 3.5 = 5.0 moved, 15.0 remain.
            assert!(
                (ind[age * n_ztypes] - 15.0).abs() < 1e-12,
                "female source after migration"
            );
            assert!(
                (ind[age * n_ztypes + 1] - 5.0).abs() < 1e-12,
                "female target after migration"
            );
        }
        // Per-bucket atomicity: source bucket + target bucket unchanged sum.
        for age in 0..n_ages {
            for mz in 0..n_ztypes {
                let src = age * n_ztypes * n_ztypes + mz;
                let dst = (age * n_ztypes + 1) * n_ztypes + mz;
                assert!(
                    (sperm[src] + sperm[dst] - (sperm_before[src] + sperm_before[dst])).abs()
                        < 1e-12,
                    "bucket ({age}, 0, {mz}) conserved"
                );
            }
        }
        // The male axis of every sperm row is untouched.
        for mz in 0..n_ztypes {
            let col_before: f64 = (0..n_ages)
                .map(|age| sperm_before[age * n_ztypes * n_ztypes + mz])
                .sum();
            let col_after: f64 = (0..n_ages)
                .map(|age| {
                    sperm[age * n_ztypes * n_ztypes + mz]
                        + sperm[(age * n_ztypes + 1) * n_ztypes + mz]
                        + sperm[(age * n_ztypes + 2) * n_ztypes + mz]
                })
                .sum();
            assert!(
                (col_before - col_after).abs() < 1e-12,
                "male sperm axis {mz} frozen"
            );
        }
        let _ = ind_before;
    }

    #[test]
    fn stochastic_convert_conservs_expectation() {
        let mut program = single_op_program(OP_CONVERT, 0.5);
        program.convert_source_z = vec![0];
        program.convert_target_z = vec![2];

        let n_ztypes = 3;
        let trials = 400;
        let mut moved_sum = 0.0;
        // Distinct seed per trial: re-using one seed would repeat the
        // same draw and make the mean a single sample.
        for trial in 0..trials {
            let mut ind = vec![0.0; 2 * n_ztypes];
            ind[0] = 100.0; // female A|A, age 0, no sperm
            let mut rng = crate::rng::new_rng(1_000 + trial as u64);
            let mut eco = [0.0; N_ECO_PARAMS];
            program.execute_event(
                &mut rng,
                1,
                &mut ind,
                &mut [],
                2,
                1,
                n_ztypes,
                0,
                true,
                false,
                0,
                &mut eco,
            );
            moved_sum += ind[2];
        }
        let mean = moved_sum / trials as f64;
        // 100 females * 0.5 = 50 expected; 3-sigma of Binomial(100, .5)
        // averaged over 400 trials is ~0.06 * 5 = loose bound 2.5.
        assert!(
            (mean - 50.0).abs() < 2.5,
            "expected 50 moved on average, got {mean}"
        );
    }
}
