//! CSR declarative hook interpreter.
//!
//! The flat-array layout and opcode values mirror
//! ``natal.hooks.types.HookProgram`` and
//! ``natal.hooks.runtime.csr_kernel``.  Only declarative hooks are supported;
//! custom hook callables are handled by falling back to the Python/Numba path.
#![allow(clippy::needless_range_loop)] // Index loops mirror the CSR kernel for parity review.
#![allow(clippy::too_many_arguments)] // execute_event mirrors the HookProgram flat-array signature.

use rand::rngs::SmallRng;

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
    rng: &mut SmallRng,
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
    rng: &mut SmallRng,
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
    rng: &mut SmallRng,
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
    ///
    /// ## Returns
    /// ``RESULT_CONTINUE`` (0) or ``RESULT_STOP`` (1) if a stop operation triggered.
    pub fn execute_event(
        &self,
        rng: &mut SmallRng,
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

                                individual_count[flat] = if sex_idx == 0 {
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
