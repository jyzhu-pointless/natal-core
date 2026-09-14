//! Spatial multi-deme lifecycle and migration kernels.
//!
//! The module schedules one tick for every deme over a stacked state —
//! age-structured and discrete-generation lifecycles share the same D-axis
//! scheduler, RNG-bank discipline, and migration stage — and implements
//! both adjacency-based and topology-kernel migration in deterministic and
//! stochastic forms.
//!
//! Per-deme RNG streams live in the session's persistent bank (seeded
//! ``seed ^ deme_id`` and advancing across ticks); the standalone one-shot
//! migration entries rebuild streams from a base seed for their
//! reproducibility contract.

#![allow(clippy::needless_range_loop)] // Index loops mirror the Python reference for parity review.
#![allow(clippy::too_many_arguments)] // Migration helpers pass parallel state/layout channels.

use rayon::prelude::*;

use crate::hooks::interpreter::HookProgram;
use crate::kernels::age_structured;
use crate::kernels::discrete_generation;
use crate::kernels::rng::{new_rng, SessionRng};
use crate::model::blueprint::Blueprint;
use crate::model::ecology::EcologyParams;
use crate::model::genetics::GeneticsTensors;

/// One audited spatial set_param transition: ``(deme, tick, param_id, old,
/// new)`` — the per-deme wrapper around
/// [`crate::hooks::interpreter::EcoJournalRow`], because spatial EcoCtx instances are
/// per-deme locals whose journals must carry the owning deme id.
pub type SpatialEcoJournalRow = (usize, i64, usize, f64, f64, usize);

/// Cut this deme's local ecology copy when the program writes params.
///
/// When the program carries set_param ops, every parallel deme ticks
/// against a private single-deme copy of its ecology column (parallel
/// demes cannot share ``&mut EcologyParams``).  ``commit`` journals and writes
/// the local column, ``assemble`` re-reads it, so later stages of the
/// **same tick** observe the write — the granularity the Python per-deme
/// lifecycle has always had.  Programs without set_param get ``None``
/// (zero overhead, identical numerics).
fn local_params(hooks: &HookProgram, params: &EcologyParams, deme: usize) -> Option<EcologyParams> {
    // A private column is cut when the program might write this deme's
    // ecology: any installed hook, a set_param op, or a Python callback.
    // Hook-free programs get None and read the shared columns directly.
    if hooks.n_hooks > 0
        || hooks.has_set_param
        || hooks
            .python_callbacks
            .iter()
            .any(|callbacks| !callbacks.is_empty())
    {
        Some(params.single_deme(deme))
    } else {
        None
    }
}

/// Shared D-axis scheduler: split the stacked state into per-deme chunks
/// and run every deme's tick, either on the rayon pool (callback-free
/// programs — deme streams are disjoint so scheduling cannot change the
/// assignment) or in stable deme order (callback-carrying programs fire on
/// the GIL).  This is the ONE scheduler for every spatial lifecycle.
///
/// ## Parameters
/// - `hooks`: CSR hook program (its callback inventory picks the mode).
/// - `rngs`: Per-deme persistent RNG streams (advanced in place).
/// - `ind_all` / `sperm_all`: Stacked state slices.
/// - `eco_all`: Per-deme ECO scratch rows for OP_SET_PARAM.
/// - `n_demes` / `ind_stride` / `sperm_stride`: Layout of the stacks.
/// - `journal`: Session audit sink, extended in deme-then-commit order.
/// - `tick_deme`: Per-deme body ``(deme_id, rng, ind, sperm, eco)``.
///
/// ## Returns
/// ``Ok(0)`` when every deme completed the tick, ``Ok(1)`` when a hook
/// stopped (the caller keeps the boundary state and freezes the tick), or
/// an error string.
#[allow(clippy::too_many_arguments)] // Generic scheduler boundary.
pub fn schedule_deme_ticks<F>(
    hooks: &HookProgram,
    rngs: &mut [SessionRng],
    ind_all: &mut [f64],
    sperm_all: &mut [f64],
    eco_all: &mut [f64],
    n_demes: usize,
    ind_stride: usize,
    sperm_stride: usize,
    journal: &mut Vec<SpatialEcoJournalRow>,
    tick_deme: F,
) -> Result<i32, String>
where
    F: Fn(
            usize,
            &mut SessionRng,
            &mut [f64],
            &mut [f64],
            &mut [f64],
        ) -> (Result<i32, String>, Vec<SpatialEcoJournalRow>)
        + Sync,
{
    // Split the stacked planes into per-deme contiguous chunks; chunk d is
    // deme d's slice, so no two deme bodies can alias the same memory.
    let mut ind_chunks: Vec<&mut [f64]> = ind_all.chunks_mut(ind_stride).collect();
    let mut sperm_chunks: Vec<&mut [f64]> = sperm_all.chunks_mut(sperm_stride).collect();
    let mut eco_chunks: Vec<&mut [f64]> = eco_all
        .chunks_mut(crate::hooks::interpreter::N_ECO_PARAMS)
        .collect();
    // Fail loudly on a layout mismatch rather than ticking a partial set.
    if ind_chunks.len() != n_demes || sperm_chunks.len() != n_demes || rngs.len() != n_demes {
        return Err(format!(
            "stacked state length mismatch: expected {n_demes} demes, got ind={} sperm={} rngs={}",
            ind_chunks.len(),
            sperm_chunks.len(),
            rngs.len()
        ));
    }
    // Python callbacks run on the GIL: firing them from rayon workers
    // would race the interpreter and make cross-deme callback order
    // nondeterministic, so any callback-carrying program demotes the
    // scheduler to a stable deme-order sequential loop.
    let sequential = hooks
        .python_callbacks
        .iter()
        .any(|callbacks| !callbacks.is_empty());

    // Same tick body in both arms; only the scheduler differs. Demes are
    // independent (disjoint state, own RNG stream, own eco row), so the
    // parallel arm cannot change any deme's draws.
    let results: Vec<(Result<i32, String>, Vec<SpatialEcoJournalRow>)> = if sequential {
        ind_chunks
            .iter_mut()
            .zip(sperm_chunks.iter_mut())
            .zip(eco_chunks.iter_mut())
            .zip(rngs.iter_mut())
            .enumerate()
            .map(|(deme_id, (((ind, sperm), eco), rng))| tick_deme(deme_id, rng, ind, sperm, eco))
            .collect()
    } else {
        ind_chunks
            .par_iter_mut()
            .zip(sperm_chunks.par_iter_mut())
            .zip(eco_chunks.par_iter_mut())
            .zip(rngs.par_iter_mut())
            .enumerate()
            .map(|(deme_id, (((ind, sperm), eco), rng))| tick_deme(deme_id, rng, ind, sperm, eco))
            .collect()
    };

    // Merge in deme order (rayon's collect preserves iterator order), so
    // journal rows stay deterministic even when ticks ran in parallel.
    let mut stopped = false;
    for (result, rows) in results {
        journal.extend(rows);
        let code = result?;
        if code != 0 {
            // Keep sweeping the remaining results so every journal row
            // lands, then report the stop to the caller.
            stopped = true;
        }
    }
    Ok(if stopped { 1 } else { 0 })
}

/// Tick one heterogeneous deme: lifecycle plus per-deme hook context.
///
/// Shared body of the parallel and sequential heterogeneous schedulers.
/// The local ecology copy (cut only when the program writes params) makes
/// same-tick `set_param` writes visible to later stages of this deme; the
/// lifecycle otherwise reads the session's columns at this deme directly.
///
/// ## Returns
/// ``(result, journal_rows)`` for this deme; ``result`` is ``Ok(0)`` to
/// continue, ``Ok(1)`` when a hook stopped, or an error string.
#[allow(clippy::too_many_arguments)] // Per-deme boundary mirrors the panmictic tick API.
fn tick_hetero_deme(
    deme_id: usize,
    bp: &Blueprint,
    hooks: &HookProgram,
    rng: &mut SessionRng,
    ind: &mut [f64],
    sperm: &mut [f64],
    eco: &mut [f64],
    tick: i64,
    params: &EcologyParams,
    genetics: &GeneticsTensors,
) -> (Result<i32, String>, Vec<SpatialEcoJournalRow>) {
    let mut local = local_params(hooks, params, deme_id);
    let mut ctx = local.as_mut().map(|local_params| age_structured::EcoCtx {
        bp,
        params: local_params,
        genetics,
        updated_genetics: None,
        phase: 0,
        // The local copy has exactly one column: writes target index 0.
        deme: 0,
        tick,
        journal: Vec::new(),
    });
    // Without a context (hook-free program) the lifecycle reads this deme's
    // session column segment directly.
    let columns = if ctx.is_some() {
        None
    } else {
        Some((params, genetics, deme_id))
    };
    // Run this deme's full lifecycle on its own chunk; callback candidates
    // and stop marks are queued for the session to merge after the batch.
    let result = age_structured::run_tick(
        rng,
        bp,
        hooks,
        ind,
        sperm,
        tick,
        deme_id as i64,
        eco,
        &mut ctx,
        columns,
    );
    let rows = ctx.map(|ctx| {
        if let Some(genetics) = ctx.updated_genetics.as_ref() {
            let mut commits = hooks
                .callback_commits
                .lock()
                .expect("callback queue poisoned");
            let update = (deme_id, ctx.params.clone(), genetics.clone());
            // Only a successful callback sets updated_genetics, and it enqueues
            // the same deme atomically. Preserve later declarative ecology writes.
            let previous = commits
                .iter_mut()
                .find(|entry| entry.0 == deme_id)
                .expect("successful callback has a queued deme candidate");
            *previous = update;
        }
        // Record where this deme stopped; the session takes the minimum
        // mark so a partial tick reports the earliest stage among demes.
        if !matches!(result, Ok(0)) {
            hooks
                .phase_marks
                .lock()
                .expect("phase queue poisoned")
                .push(ctx.phase);
        }
        // The local copy's final values are the deme's tick result:
        // reflect them into the eco scratch row the session reads
        // for its column write-back (multi-event writes included).
        for id in 0..crate::hooks::interpreter::N_ECO_PARAMS {
            eco[id] = ctx.params.eco_value(id, 0);
        }
        ctx.journal
            .into_iter()
            .map(|(t, id, old, new, phase)| (deme_id, t, id, old, new, phase))
            .collect::<Vec<SpatialEcoJournalRow>>()
    });
    (result, rows.unwrap_or_default())
}

/// Run one tick for every deme over the shared contracts.
///
/// Every deme consumes its own ecology column entry plus its shared genetics
/// variant (``deme_variants[d]`` indexes the bank) and its own persistent RNG
/// stream from *rngs*; the stream advances across ticks instead of being
/// rebuilt per tick.
///
/// ## Parameters
/// - `hooks`: CSR hook program.
/// - `rngs`: Per-deme persistent RNG streams (advanced in place).
/// - `ind_all` / `sperm_all`: Stacked state slices.
/// - `tick`: Current tick.
/// - `eco_all`: Per-deme ECO scratch rows for OP_SET_PARAM.
/// - `bp`: Blueprint providing dimensions and sampling flags.
/// - `params`: Columnized session ecology; each deme reads its segment.
/// - `variants`: Genetics variant bank shared across demes.
/// - `deme_variants`: Per-deme index into the variant bank.
/// - `journal`: Session audit sink, extended with this tick's per-deme
///   set_param transitions in deme order.
///
/// ## Returns
/// ``Ok(0)`` when every deme completed the tick, ``Ok(1)`` when a hook
/// stopped (state keeps the modifications up to that boundary; the
/// caller freezes the tick), or an error string.
#[allow(clippy::too_many_arguments)] // Session boundary mirror of the panmictic tick API.
pub fn run_spatial_tick_heterogeneous(
    hooks: &HookProgram,
    rngs: &mut [SessionRng],
    ind_all: &mut [f64],
    sperm_all: &mut [f64],
    tick: i64,
    eco_all: &mut [f64],
    bp: &Blueprint,
    params: &EcologyParams,
    variants: &[GeneticsTensors],
    deme_variants: &[usize],
    journal: &mut Vec<SpatialEcoJournalRow>,
) -> Result<i32, String> {
    let n_demes = deme_variants.len();
    let n_ages = bp.n_ages;
    let n_ztypes = bp.n_ztypes;
    // Zero demes and mismatched column/stream counts are contract errors.
    if n_demes == 0 {
        return Err("heterogeneous spatial run requires at least one deme".to_string());
    }
    if params.n_demes != n_demes || rngs.len() != n_demes {
        return Err(format!(
            "heterogeneous spatial run requires {n_demes} ecology columns, variant ids, and RNG streams, got {}, {}, and {}",
            params.n_demes,
            variants.len(),
            rngs.len()
        ));
    }
    // Strides are the flattened deme planes: (sexes, ages, ztypes) for the
    // individual counts and (ages, female_ztype, male_ztype) for sperm.
    schedule_deme_ticks(
        hooks,
        rngs,
        ind_all,
        sperm_all,
        eco_all,
        n_demes,
        2 * n_ages * n_ztypes,
        n_ages * n_ztypes * n_ztypes,
        journal,
        |deme_id, rng, ind, sperm, eco| {
            let genetics = &variants[deme_variants[deme_id]];
            tick_hetero_deme(
                deme_id, bp, hooks, rng, ind, sperm, eco, tick, params, genetics,
            )
        },
    )
}

/// Tick one discrete-generation deme: lifecycle plus per-deme hook context.
///
/// Discrete twin of [`tick_hetero_deme`]; the sperm plane is a
/// session-maintained zero sink the discrete lifecycle never touches.
#[allow(clippy::too_many_arguments)] // Per-deme boundary mirror.
fn tick_discrete_deme(
    deme_id: usize,
    bp: &Blueprint,
    hooks: &HookProgram,
    rng: &mut SessionRng,
    ind: &mut [f64],
    eco: &mut [f64],
    tick: i64,
    params: &EcologyParams,
    genetics: &GeneticsTensors,
) -> (Result<i32, String>, Vec<SpatialEcoJournalRow>) {
    let mut local = local_params(hooks, params, deme_id);
    let mut ctx = local.as_mut().map(|local_params| age_structured::EcoCtx {
        bp,
        params: local_params,
        genetics,
        updated_genetics: None,
        phase: 0,
        // The local copy has exactly one column: writes target index 0.
        deme: 0,
        tick,
        journal: Vec::new(),
    });
    // Without a context (hook-free program) the lifecycle reads this deme's
    // session column segment directly.
    let columns = if ctx.is_some() {
        None
    } else {
        Some((params, genetics, deme_id))
    };
    let result = discrete_generation::run_tick(
        rng,
        bp,
        hooks,
        ind,
        tick,
        deme_id as i64,
        eco,
        &mut ctx,
        columns,
    );
    let rows = ctx.map(|ctx| {
        if let Some(genetics) = ctx.updated_genetics.as_ref() {
            let mut commits = hooks
                .callback_commits
                .lock()
                .expect("callback queue poisoned");
            let update = (deme_id, ctx.params.clone(), genetics.clone());
            // Only a successful callback sets updated_genetics, and it enqueues
            // the same deme atomically. Preserve later declarative ecology writes.
            let previous = commits
                .iter_mut()
                .find(|entry| entry.0 == deme_id)
                .expect("successful callback has a queued deme candidate");
            *previous = update;
        }
        // Record where this deme stopped; the session takes the minimum
        // mark so a partial tick reports the earliest stage among demes.
        if !matches!(result, Ok(0)) {
            hooks
                .phase_marks
                .lock()
                .expect("phase queue poisoned")
                .push(ctx.phase);
        }
        for id in 0..crate::hooks::interpreter::N_ECO_PARAMS {
            eco[id] = ctx.params.eco_value(id, 0);
        }
        ctx.journal
            .into_iter()
            .map(|(t, id, old, new, phase)| (deme_id, t, id, old, new, phase))
            .collect::<Vec<SpatialEcoJournalRow>>()
    });
    (result, rows.unwrap_or_default())
}

/// Run one discrete-generation tick for every deme over the variant bank.
///
/// Same scheduler, RNG-bank discipline, and stop contract as the
/// age-structured kernel; the discrete lifecycle carries no sperm plane.
///
/// ## Parameters
/// - `hooks`, `rngs`, `ind_all`, `tick`, `eco_all`, `bp`, `params`,
///   `variants`, `deme_variants`, `journal`: see
///   [`run_spatial_tick_heterogeneous`].
///
/// ## Returns
/// ``Ok(0)`` continue / ``Ok(1)`` stopped / error string.
#[allow(clippy::too_many_arguments)] // Session boundary mirror of the age kernel.
pub fn run_spatial_tick_discrete(
    hooks: &HookProgram,
    rngs: &mut [SessionRng],
    ind_all: &mut [f64],
    tick: i64,
    eco_all: &mut [f64],
    bp: &Blueprint,
    params: &EcologyParams,
    variants: &[GeneticsTensors],
    deme_variants: &[usize],
    journal: &mut Vec<SpatialEcoJournalRow>,
) -> Result<i32, String> {
    let n_demes = deme_variants.len();
    let n_ztypes = bp.n_ztypes;
    // Discrete canonicalization: 2 sexes x 2 ages.
    let ind_stride = 2 * 2 * n_ztypes;
    if n_demes == 0 {
        return Err("discrete spatial run requires at least one deme".to_string());
    }
    if params.n_demes != n_demes || rngs.len() != n_demes {
        return Err(format!(
            "discrete spatial run requires {n_demes} ecology columns, variant ids, and RNG streams, got {}, {}, and {}",
            params.n_demes,
            variants.len(),
            rngs.len()
        ));
    }
    // A zero sperm plane satisfies the scheduler's chunking contract; the
    // discrete lifecycle never reads it.
    let mut sink = vec![0.0f64; ind_all.len().max(1)];
    schedule_deme_ticks(
        hooks,
        rngs,
        ind_all,
        &mut sink,
        eco_all,
        n_demes,
        ind_stride,
        ind_stride,
        journal,
        |deme_id, rng, ind, _sperm_sink, eco| {
            let genetics = &variants[deme_variants[deme_id]];
            tick_discrete_deme(deme_id, bp, hooks, rng, ind, eco, tick, params, genetics)
        },
    )
}

/// Deterministically move individuals and stored sperm along a CSR routing table.
///
/// Outbound counts are ``value * rate`` and are distributed by the frozen
/// CSR weights folded onto the Blueprint at build time.  Female virgins and
/// stored sperm are migrated separately from males.
///
/// ## Returns
/// ``(out_ind, out_sperm)`` new stacked arrays.
pub fn migrate_csr_deterministic(
    ind_all: &[f64],
    sperm_all: &[f64],
    indptr: &[i64],
    dest_idx: &[i64],
    weights: &[f64],
    rate: &[f64],
    // Retained for the frozen CSR/session contract; it no longer changes the
    // deterministic result (both orders send first and keep the residual).
    _stay_after: bool,
    n_demes: usize,
    n_ages: usize,
    n_ztypes: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    // For each source deme, compute outbound individuals/sperm using the
    // per-deme/per-sex/per-age migration rate, then distribute them along
    // the CSR entries in stored order.  Males and stored sperm are handled
    // separately from female virgins.
    // Individual plane layout is (deme, sex, age, ztype); the CSR row
    // pointers and the rate column must both cover the declared extent.
    let ind_stride = 2 * n_ages * n_ztypes;
    if indptr.len() != n_demes + 1 {
        return Err(format!(
            "migration indptr length {} does not match n_demes + 1 ({})",
            indptr.len(),
            n_demes + 1
        ));
    }
    if rate.len() != n_demes * 2 * n_ages {
        return Err(format!(
            "migration rate length {} does not match (n_demes, 2, n_ages) = {}",
            rate.len(),
            n_demes * 2 * n_ages
        ));
    }
    // Out-of-place outputs: every source scatters into destinations, so
    // accumulating into fresh zeroed arrays is required (and matches the
    // Python kernel's thread-local buffers merged by addition).
    let mut out_ind = vec![0.0; ind_all.len()];
    let mut out_sperm = vec![0.0; sperm_all.len()];

    // Sources are independent; stable deme order fixes the float addition
    // order, so the result is bit-reproducible.
    for src in 0..n_demes {
        // CSR row slice: this source's destinations and outbound weights.
        let row_start = indptr[src] as usize;
        let row_end = indptr[src + 1] as usize;

        for age in 0..n_ages {
            // Female virgins (sex 0) and their stored sperm move at the
            // female rate so the virgin/stored bookkeeping stays consistent.
            let female_rate = rate[src * 2 * n_ages + age];
            for female_ztype in 0..n_ztypes {
                // Virgin females = the female tally minus the sperm they
                // already carry (stored sperm sums over the male z index);
                // only virgins are candidates for dispersal.
                let mut stored_total = 0.0;
                for male_ztype in 0..n_ztypes {
                    stored_total += sperm_all[(src * n_ages + age) * n_ztypes * n_ztypes
                        + female_ztype * n_ztypes
                        + male_ztype];
                }
                let female_total = ind_all[src * ind_stride + age * n_ztypes + female_ztype];
                let mut virgin_count = female_total - stored_total;
                // Float subtraction can leave a tiny negative drift; clamp
                // only within 1e-9, anything larger is a real inconsistency.
                if virgin_count < 0.0 && virgin_count.abs() < 1e-9 {
                    virgin_count = 0.0;
                }

                // Deterministic outbound mass = count x rate; the loop below
                // only splits that mass across the CSR destinations.
                let outbound = virgin_count * female_rate;
                let src_ind_idx = src * ind_stride + age * n_ztypes + female_ztype;
                // Send first, then keep the `count - moved` residual at the
                // source.  This conserves total mass for any row sum; the old
                // adjacency order (park `count - outbound` first) under-counted
                // sub-stochastic rows, so it is gone.
                if row_start == row_end {
                    // Empty CSR row (isolated deme): nothing leaves, matching
                    // the Python reference's keep-all branch.
                    out_ind[src_ind_idx] += virgin_count;
                } else {
                    let mut moved_total = 0.0;
                    for entry in row_start..row_end {
                        let dst = dest_idx[entry] as usize;
                        let moved = outbound * weights[entry];
                        out_ind[dst * ind_stride + age * n_ztypes + female_ztype] += moved;
                        moved_total += moved;
                    }
                    out_ind[src_ind_idx] += virgin_count - moved_total;
                }

                // Stored sperm travels with its carrier female at the female
                // rate; each moved amount is added both to the destination's
                // sperm slot and to that destination's female tally, keeping
                // the virgin residual consistent at both ends.
                for male_ztype in 0..n_ztypes {
                    let sperm_idx = (src * n_ages + age) * n_ztypes * n_ztypes
                        + female_ztype * n_ztypes
                        + male_ztype;
                    let value = sperm_all[sperm_idx];
                    let outbound_sperm = value * female_rate;
                    if row_start == row_end {
                        // Empty CSR row: keep everything at the source.
                        out_sperm[sperm_idx] += value;
                        out_ind[src_ind_idx] += value;
                    } else {
                        let mut moved_total = 0.0;
                        for entry in row_start..row_end {
                            let dst = dest_idx[entry] as usize;
                            let moved = outbound_sperm * weights[entry];
                            let dst_sperm_idx = (dst * n_ages + age) * n_ztypes * n_ztypes
                                + female_ztype * n_ztypes
                                + male_ztype;
                            out_sperm[dst_sperm_idx] += moved;
                            out_ind[dst * ind_stride + age * n_ztypes + female_ztype] += moved;
                            moved_total += moved;
                        }
                        out_sperm[sperm_idx] += value - moved_total;
                        out_ind[src_ind_idx] += value - moved_total;
                    }
                }
            }
        }

        // Males (sex 1) migrate at their own rate column.
        for age in 0..n_ages {
            let male_rate = rate[src * 2 * n_ages + n_ages + age];
            for ztype in 0..n_ztypes {
                let src_idx = src * ind_stride + (n_ages + age) * n_ztypes + ztype;
                let value = ind_all[src_idx];
                let outbound = value * male_rate;
                if row_start == row_end {
                    // Empty CSR row: keep everything at the source.
                    out_ind[src_idx] += value;
                } else {
                    let mut moved_total = 0.0;
                    for entry in row_start..row_end {
                        let dst = dest_idx[entry] as usize;
                        let moved = outbound * weights[entry];
                        out_ind[dst * ind_stride + (n_ages + age) * n_ztypes + ztype] += moved;
                        moved_total += moved;
                    }
                    out_ind[src_idx] += value - moved_total;
                }
            }
        }
    }
    Ok((out_ind, out_sperm))
}

/// Stochastically move individuals and stored sperm along a CSR routing table.
///
/// Outbound counts are sampled with binomial/continuous-binomial and then
/// multinomially distributed among the CSR destinations.  Each source deme
/// consumes its own RNG stream from *rngs* (the session's persistent
/// per-deme bank), so the stream advances across ticks instead of being
/// rebuilt per call.
///
/// ## Returns
/// ``(out_ind, out_sperm)`` new stacked arrays.
pub fn migrate_csr_stochastic_rngs(
    rngs: &mut [SessionRng],
    ind_all: &[f64],
    sperm_all: &[f64],
    indptr: &[i64],
    dest_idx: &[i64],
    weights: &[f64],
    rate: &[f64],
    continuous_sampling: bool,
    n_demes: usize,
    n_ages: usize,
    n_ztypes: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    // Stochastic variant of CSR migration: sample outbound counts and
    // multinomially distribute them among the destinations.  Sources are
    // visited in stable deme order; each consumes only its own stream, so
    // thread count and scheduling cannot change the assignment.
    // Same layout/contract checks as the deterministic twin.
    let ind_stride = 2 * n_ages * n_ztypes;
    if indptr.len() != n_demes + 1 {
        return Err(format!(
            "migration indptr length {} does not match n_demes + 1 ({})",
            indptr.len(),
            n_demes + 1
        ));
    }
    if rate.len() != n_demes * 2 * n_ages {
        return Err(format!(
            "migration rate length {} does not match (n_demes, 2, n_ages) = {}",
            rate.len(),
            n_demes * 2 * n_ages
        ));
    }
    // One stream per source deme: source s consumes only rngs[s], so the
    // draw assignment cannot depend on scheduling.
    if rngs.len() != n_demes {
        return Err(format!(
            "migration requires {n_demes} per-deme RNG streams, got {}",
            rngs.len()
        ));
    }
    // Size the reusable scratch buffers to the widest CSR row so the hot
    // loop allocates nothing.
    let mut max_row = 1usize;
    for pair in indptr.windows(2) {
        let len = (pair[1] - pair[0]).max(0) as usize;
        if len > max_row {
            max_row = len;
        }
    }
    let mut out_ind = vec![0.0; ind_all.len()];
    let mut out_sperm = vec![0.0; sperm_all.len()];
    // `distributed` and `probs` are reused per bucket; each call rewrites
    // them fully, so no stale destination value can leak.
    let mut distributed = vec![0.0; max_row];
    let mut probs = vec![0.0; max_row];

    // Stable deme order plus per-deme streams: thread count cannot shift a
    // draw. Within a bucket, outbound is sampled before destinations.
    for src in 0..n_demes {
        let rng = &mut rngs[src];
        // CSR row slice: this source's destinations and outbound weights.
        let row_start = indptr[src] as usize;
        let row_end = indptr[src + 1] as usize;
        let row_len = row_end - row_start;
        for age in 0..n_ages {
            // Female virgins (sex 0) and their stored sperm move at the
            // female rate so the virgin/stored bookkeeping stays consistent.
            let female_rate = rate[src * 2 * n_ages + age];
            for female_ztype in 0..n_ztypes {
                let mut stored_total = 0.0;
                for male_ztype in 0..n_ztypes {
                    stored_total += sperm_all[(src * n_ages + age) * n_ztypes * n_ztypes
                        + female_ztype * n_ztypes
                        + male_ztype];
                }
                let female_total = ind_all[src * ind_stride + age * n_ztypes + female_ztype];
                // Virgin females = female tally minus stored sperm; clamp a
                // tiny negative float drift within 1e-9 so no negative mass
                // can migrate.
                let mut virgin_count = female_total - stored_total;
                if virgin_count < 0.0 && virgin_count.abs() < 1e-9 {
                    virgin_count = 0.0;
                }

                // Draw the outbound count first, then the destinations, on
                // this deme's stream; that pair order is part of the
                // trajectory contract.
                let outbound = sample_outbound(rng, virgin_count, female_rate, continuous_sampling);
                let moved_total = distribute_csr_outbound(
                    rng,
                    outbound,
                    weights,
                    row_start,
                    row_len,
                    continuous_sampling,
                    &mut distributed,
                    &mut probs,
                );
                for pos in 0..row_len {
                    let dst = dest_idx[row_start + pos] as usize;
                    out_ind[dst * ind_stride + age * n_ztypes + female_ztype] += distributed[pos];
                }
                out_ind[src * ind_stride + age * n_ztypes + female_ztype] +=
                    virgin_count - moved_total;

                for male_ztype in 0..n_ztypes {
                    let sperm_idx = (src * n_ages + age) * n_ztypes * n_ztypes
                        + female_ztype * n_ztypes
                        + male_ztype;
                    // Each stored-sperm bucket draws outbound then
                    // destinations; moved mass is mirrored into the female
                    // tally at both ends.
                    let value = sperm_all[sperm_idx];
                    let outbound_sperm =
                        sample_outbound(rng, value, female_rate, continuous_sampling);
                    let moved_sperm_total = distribute_csr_outbound(
                        rng,
                        outbound_sperm,
                        weights,
                        row_start,
                        row_len,
                        continuous_sampling,
                        &mut distributed,
                        &mut probs,
                    );
                    for pos in 0..row_len {
                        let dst = dest_idx[row_start + pos] as usize;
                        let moved = distributed[pos];
                        let dst_sperm_idx = (dst * n_ages + age) * n_ztypes * n_ztypes
                            + female_ztype * n_ztypes
                            + male_ztype;
                        out_sperm[dst_sperm_idx] += moved;
                        out_ind[dst * ind_stride + age * n_ztypes + female_ztype] += moved;
                    }
                    out_sperm[sperm_idx] += value - moved_sperm_total;
                    out_ind[src * ind_stride + age * n_ztypes + female_ztype] +=
                        value - moved_sperm_total;
                }
            }
        }

        // Males (sex 1) migrate at their own rate column.
        for age in 0..n_ages {
            let male_rate = rate[src * 2 * n_ages + n_ages + age];
            for ztype in 0..n_ztypes {
                let src_idx = src * ind_stride + (n_ages + age) * n_ztypes + ztype;
                let value = ind_all[src_idx];
                let outbound = sample_outbound(rng, value, male_rate, continuous_sampling);
                let moved_total = distribute_csr_outbound(
                    rng,
                    outbound,
                    weights,
                    row_start,
                    row_len,
                    continuous_sampling,
                    &mut distributed,
                    &mut probs,
                );
                for pos in 0..row_len {
                    let dst = dest_idx[row_start + pos] as usize;
                    out_ind[dst * ind_stride + (n_ages + age) * n_ztypes + ztype] +=
                        distributed[pos];
                }
                out_ind[src_idx] += value - moved_total;
            }
        }
    }
    Ok((out_ind, out_sperm))
}

/// Stochastically move individuals and stored sperm along a CSR routing table.
///
/// One-shot seed form of [`migrate_csr_stochastic_rngs`]: every source deme
/// gets a fresh ``seed ^ src`` stream, matching the standalone sampling
/// entry point's historical reproducibility contract.
///
/// ## Returns
/// ``(out_ind, out_sperm)`` new stacked arrays.
#[allow(clippy::too_many_arguments)] // Standalone mirror of the session variant.
pub fn migrate_csr_stochastic(
    ind_all: &[f64],
    sperm_all: &[f64],
    indptr: &[i64],
    dest_idx: &[i64],
    weights: &[f64],
    rate: &[f64],
    seed: u64,
    continuous_sampling: bool,
    n_demes: usize,
    n_ages: usize,
    n_ztypes: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    // One-shot reproducibility contract: rebuild a fresh `seed ^ deme`
    // stream per source instead of reusing a session's persistent bank.
    let mut rngs: Vec<SessionRng> = (0..n_demes)
        .map(|deme| new_rng(crate::kernels::rng::stream_seed(seed, deme as i64)))
        .collect();
    migrate_csr_stochastic_rngs(
        &mut rngs,
        ind_all,
        sperm_all,
        indptr,
        dest_idx,
        weights,
        rate,
        continuous_sampling,
        n_demes,
        n_ages,
        n_ztypes,
    )
}

/// Sample how many individuals or stored sperm leave a source deme.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `value`: Source count.
/// - `rate`: Migration rate.
/// - `continuous_sampling`: Use continuous sampling.
///
/// ## Returns
/// The outbound count.
fn sample_outbound(rng: &mut SessionRng, value: f64, rate: f64, continuous_sampling: bool) -> f64 {
    // Sample how many individuals leave: deterministic rate, continuous
    // binomial, or discrete binomial.
    // Guard before any draw: an empty bucket returns without consuming
    // randomness, which keeps the stream aligned for later buckets.
    if value <= 0.0 || rate <= 0.0 {
        return 0.0;
    }
    // A rate at or above one moves the whole bucket; likewise no draw.
    if rate >= 1.0 {
        return value;
    }
    // Continuous sampling keeps fractional counts; the discrete binomial
    // rounds the count to an integer first.
    if continuous_sampling {
        return crate::kernels::rng::continuous_binomial(rng, value, rate);
    }
    crate::kernels::rng::binomial(rng, value.round() as i64, rate)
}

/// Distribute outbound migrants among the CSR destinations of one source row.
///
/// The row weights are normalized before the multinomial samplers see them:
/// the discrete multinomial sampler needs a probability vector, while a CSR row
/// need not sum to one.  Builder-folded rows always sum to one — the adjacency
/// path row-normalizes relative outbound weights and the kernel path
/// renormalizes over valid neighbors — so this normalization is the identity
/// for every public build path; it only does work for a raw hand-built CSR.
///
/// ## Returns
/// The total mass assigned to destinations.
fn distribute_csr_outbound(
    rng: &mut SessionRng,
    outbound: f64,
    weights: &[f64],
    row_start: usize,
    row_len: usize,
    continuous_sampling: bool,
    distributed: &mut [f64],
    probs: &mut [f64],
) -> f64 {
    // Clear scratch buffers, collect the row weights, normalize, sample.
    // Reset the scratch row before the early exits so the caller can read
    // it even when nothing is distributed.
    for slot in distributed.iter_mut() {
        *slot = 0.0;
    }
    // No mass or no destinations: nothing to draw.
    if outbound <= 0.0 || row_len == 0 {
        return 0.0;
    }
    let mut total = 0.0;
    for pos in 0..row_len {
        let weight = weights[row_start + pos];
        probs[pos] = weight;
        total += weight;
    }
    // All-zero weights carry no destination probability.
    if total <= 0.0 {
        return 0.0;
    }
    // Normalize into a probability vector: the discrete multinomial sampler
    // requires one, while folded rows (raw adjacency values) need not sum to
    // one. A single reciprocal multiply preserves the fold's float order.
    let inv_total = 1.0 / total;
    for pos in 0..row_len {
        probs[pos] *= inv_total;
    }
    // Continuous keeps fractional outbound mass; discrete rounds the count.
    if continuous_sampling {
        crate::kernels::rng::continuous_multinomial(rng, outbound, &probs[..row_len], distributed);
    } else {
        crate::kernels::rng::multinomial(
            rng,
            outbound.round() as i64,
            &probs[..row_len],
            distributed,
        );
    }
    distributed[..row_len].iter().sum()
}
