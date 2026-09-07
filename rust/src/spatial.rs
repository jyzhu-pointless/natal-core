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
#![allow(clippy::too_many_arguments)] // Migration helpers mirror the Numba kernel signatures.

use rayon::prelude::*;

use crate::config::SimConfig;
use crate::contract::{Blueprint, Params, TensorSet};
use crate::discrete;
use crate::discrete::DiscreteConfig;
use crate::hooks::HookProgram;
use crate::lifecycle;
use crate::rng::{new_rng, SessionRng};

/// One audited spatial set_param transition: ``(deme, tick, param_id, old,
/// new)`` — the per-deme wrapper around
/// [`crate::hooks::EcoJournalRow`], because spatial EcoCtx instances are
/// per-deme locals whose journals must carry the owning deme id.
pub type SpatialEcoJournalRow = (usize, i64, usize, f64, f64);

/// Cut this deme's local ecology copy when the program writes params.
///
/// When the program carries set_param ops, every parallel deme ticks
/// against a private single-deme copy of its ecology column (parallel
/// demes cannot share ``&mut Params``).  ``commit`` journals and writes
/// the local column, ``assemble`` re-reads it, so later stages of the
/// **same tick** observe the write — the granularity the Python per-deme
/// lifecycle has always had.  Programs without set_param get ``None``
/// (zero overhead, identical numerics).
fn local_params(hooks: &HookProgram, params: &Params, deme: usize) -> Option<Params> {
    if hooks.has_set_param {
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
    let mut ind_chunks: Vec<&mut [f64]> = ind_all.chunks_mut(ind_stride).collect();
    let mut sperm_chunks: Vec<&mut [f64]> = sperm_all.chunks_mut(sperm_stride).collect();
    let mut eco_chunks: Vec<&mut [f64]> = eco_all.chunks_mut(crate::hooks::N_ECO_PARAMS).collect();
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
/// same-tick `set_param` writes visible to later stages of this deme.
///
/// ## Returns
/// ``(result, journal_rows)`` for this deme; ``result`` is ``Ok(0)`` to
/// continue, ``Ok(1)`` when a hook stopped, or an error string.
#[allow(clippy::too_many_arguments)] // Per-deme boundary mirrors the panmictic tick API.
fn tick_hetero_deme(
    deme_id: usize,
    cfg: &SimConfig,
    hooks: &HookProgram,
    rng: &mut SessionRng,
    ind: &mut [f64],
    sperm: &mut [f64],
    eco: &mut [f64],
    tick: i64,
    bp: &Blueprint,
    params: &Params,
    genetics: &TensorSet,
) -> (Result<i32, String>, Vec<SpatialEcoJournalRow>) {
    let mut local = local_params(hooks, params, deme_id);
    let mut ctx = local.as_mut().map(|local_params| lifecycle::EcoCtx {
        bp,
        params: local_params,
        genetics,
        // The local copy has exactly one column: writes target index 0.
        deme: 0,
        tick,
        journal: Vec::new(),
    });
    let result = lifecycle::run_tick(
        rng,
        cfg,
        hooks,
        ind,
        sperm,
        tick,
        deme_id as i64,
        eco,
        &mut ctx,
    );
    let rows = ctx.map(|ctx| {
        // The local copy's final values are the deme's tick result:
        // reflect them into the eco scratch row the session reads
        // for its column write-back (multi-event writes included).
        for id in 0..crate::hooks::N_ECO_PARAMS {
            eco[id] = ctx.params.eco_value(id, 0);
        }
        ctx.journal
            .into_iter()
            .map(|(t, id, old, new)| (deme_id, t, id, old, new))
            .collect::<Vec<SpatialEcoJournalRow>>()
    });
    (result, rows.unwrap_or_default())
}

/// Run one tick for every deme with one flat config per deme.
///
/// The configs slice is index-aligned with the demes: deme *d* consumes
/// ``configs[d]`` (its own ecology column entry plus its shared genetics
/// variant, already assembled by the session).  Each deme consumes its
/// own persistent RNG stream from *rngs*; the stream advances across
/// ticks instead of being rebuilt per tick.
///
/// ## Parameters
/// - `configs`: Per-deme configs (length equals the deme count).
/// - `hooks`: CSR hook program.
/// - `rngs`: Per-deme persistent RNG streams (advanced in place).
/// - `ind_all`: Stacked individual-count slice.
/// - `sperm_all`: Stacked sperm-storage slice.
/// - `tick`: Current tick.
/// - `bp`: Blueprint backing config re-assembly (set_param programs).
/// - `params`: Columnized session ecology the local copies are cut from.
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
    configs: &[SimConfig],
    hooks: &HookProgram,
    rngs: &mut [SessionRng],
    ind_all: &mut [f64],
    sperm_all: &mut [f64],
    tick: i64,
    eco_all: &mut [f64],
    bp: &Blueprint,
    params: &Params,
    variants: &[TensorSet],
    deme_variants: &[usize],
    journal: &mut Vec<SpatialEcoJournalRow>,
) -> Result<i32, String> {
    if configs.is_empty() {
        return Err(
            "heterogeneous spatial run requires at least one config and one deme".to_string(),
        );
    }
    let n_demes = configs.len();
    let n_ages = configs[0].n_ages;
    let n_ztypes = configs[0].n_ztypes;
    if params.n_demes != n_demes || deme_variants.len() != n_demes || rngs.len() != n_demes {
        return Err(format!(
            "heterogeneous spatial run requires {n_demes} ecology columns, variant ids, and RNG streams, got {}, {}, and {}",
            params.n_demes,
            deme_variants.len(),
            rngs.len()
        ));
    }
    // Capture per-deme config references up front so the scheduler body
    // only carries the deme id and the disjoint mutable slices.
    let deme_configs: Vec<&SimConfig> = configs.iter().collect();
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
            let cfg = deme_configs[deme_id];
            if cfg.n_ages != n_ages || cfg.n_ztypes != n_ztypes {
                return (
                    Err(format!(
                        "config for deme {deme_id} dimensions do not match the stacked state"
                    )),
                    Vec::new(),
                );
            }
            let genetics = &variants[deme_variants[deme_id]];
            tick_hetero_deme(
                deme_id, cfg, hooks, rng, ind, sperm, eco, tick, bp, params, genetics,
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
    cfg: &DiscreteConfig,
    hooks: &HookProgram,
    rng: &mut SessionRng,
    ind: &mut [f64],
    eco: &mut [f64],
    tick: i64,
    bp: &Blueprint,
    params: &Params,
    genetics: &TensorSet,
) -> (Result<i32, String>, Vec<SpatialEcoJournalRow>) {
    let mut local = local_params(hooks, params, deme_id);
    let mut ctx = local.as_mut().map(|local_params| lifecycle::EcoCtx {
        bp,
        params: local_params,
        genetics,
        // The local copy has exactly one column: writes target index 0.
        deme: 0,
        tick,
        journal: Vec::new(),
    });
    let result = discrete::run_tick(rng, cfg, hooks, ind, tick, deme_id as i64, eco, &mut ctx);
    let rows = ctx.map(|ctx| {
        for id in 0..crate::hooks::N_ECO_PARAMS {
            eco[id] = ctx.params.eco_value(id, 0);
        }
        ctx.journal
            .into_iter()
            .map(|(t, id, old, new)| (deme_id, t, id, old, new))
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
/// - `configs`: Per-deme discrete configs (index-aligned with the demes).
/// - `hooks`, `rngs`, `ind_all`, `tick`, `eco_all`, `bp`, `params`,
///   `variants`, `deme_variants`, `journal`: see
///   [`run_spatial_tick_heterogeneous`].
///
/// ## Returns
/// ``Ok(0)`` continue / ``Ok(1)`` stopped / error string.
#[allow(clippy::too_many_arguments)] // Session boundary mirror of the age kernel.
pub fn run_spatial_tick_discrete(
    configs: &[DiscreteConfig],
    hooks: &HookProgram,
    rngs: &mut [SessionRng],
    ind_all: &mut [f64],
    tick: i64,
    eco_all: &mut [f64],
    bp: &Blueprint,
    params: &Params,
    variants: &[TensorSet],
    deme_variants: &[usize],
    journal: &mut Vec<SpatialEcoJournalRow>,
) -> Result<i32, String> {
    if configs.is_empty() {
        return Err("discrete spatial run requires at least one config and one deme".to_string());
    }
    let n_demes = configs.len();
    let n_ztypes = configs[0].n_ztypes;
    // Discrete canonicalization: 2 sexes x 2 ages.
    let ind_stride = 2 * 2 * n_ztypes;
    if params.n_demes != n_demes || deme_variants.len() != n_demes || rngs.len() != n_demes {
        return Err(format!(
            "discrete spatial run requires {n_demes} ecology columns, variant ids, and RNG streams, got {}, {}, and {}",
            params.n_demes,
            deme_variants.len(),
            rngs.len()
        ));
    }
    // A zero sperm plane satisfies the scheduler's chunking contract; the
    // discrete lifecycle never reads it.
    let mut sink = vec![0.0f64; ind_all.len().max(1)];
    let deme_configs: Vec<&DiscreteConfig> = configs.iter().collect();
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
            let cfg = deme_configs[deme_id];
            let genetics = &variants[deme_variants[deme_id]];
            tick_discrete_deme(
                deme_id, cfg, hooks, rng, ind, eco, tick, bp, params, genetics,
            )
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
    stay_after: bool,
    n_demes: usize,
    n_ages: usize,
    n_ztypes: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    // For each source deme, compute outbound individuals/sperm using the
    // per-deme/per-sex/per-age migration rate, then distribute them along
    // the CSR entries in stored order.  Males and stored sperm are handled
    // separately from female virgins.
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
    let mut out_ind = vec![0.0; ind_all.len()];
    let mut out_sperm = vec![0.0; sperm_all.len()];

    for src in 0..n_demes {
        let row_start = indptr[src] as usize;
        let row_end = indptr[src + 1] as usize;

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
                let mut virgin_count = female_total - stored_total;
                if virgin_count < 0.0 && virgin_count.abs() < 1e-9 {
                    virgin_count = 0.0;
                }

                let outbound = virgin_count * female_rate;
                let src_ind_idx = src * ind_stride + age * n_ztypes + female_ztype;
                if stay_after {
                    let mut moved_total = 0.0;
                    for entry in row_start..row_end {
                        let dst = dest_idx[entry] as usize;
                        let moved = outbound * weights[entry];
                        out_ind[dst * ind_stride + age * n_ztypes + female_ztype] += moved;
                        moved_total += moved;
                    }
                    out_ind[src_ind_idx] += virgin_count - moved_total;
                } else if row_start == row_end {
                    // Empty CSR row (isolated deme): nothing leaves, matching
                    // the Python reference's keep-all branch.
                    out_ind[src_ind_idx] += virgin_count;
                } else {
                    let stay = virgin_count - outbound;
                    out_ind[src_ind_idx] += stay;
                    for entry in row_start..row_end {
                        let dst = dest_idx[entry] as usize;
                        let prob = weights[entry];
                        out_ind[dst * ind_stride + age * n_ztypes + female_ztype] +=
                            outbound * prob;
                    }
                }

                for male_ztype in 0..n_ztypes {
                    let sperm_idx = (src * n_ages + age) * n_ztypes * n_ztypes
                        + female_ztype * n_ztypes
                        + male_ztype;
                    let value = sperm_all[sperm_idx];
                    let outbound_sperm = value * female_rate;
                    if stay_after {
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
                    } else if row_start == row_end {
                        // Empty CSR row: keep everything at the source.
                        out_sperm[sperm_idx] += value;
                        out_ind[src_ind_idx] += value;
                    } else {
                        let stay_sperm = value - outbound_sperm;
                        out_sperm[sperm_idx] += stay_sperm;
                        out_ind[src_ind_idx] += stay_sperm;
                        for entry in row_start..row_end {
                            let dst = dest_idx[entry] as usize;
                            let prob = weights[entry];
                            let moved = outbound_sperm * prob;
                            let dst_sperm_idx = (dst * n_ages + age) * n_ztypes * n_ztypes
                                + female_ztype * n_ztypes
                                + male_ztype;
                            out_sperm[dst_sperm_idx] += moved;
                            out_ind[dst * ind_stride + age * n_ztypes + female_ztype] += moved;
                        }
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
                if stay_after {
                    let mut moved_total = 0.0;
                    for entry in row_start..row_end {
                        let dst = dest_idx[entry] as usize;
                        let moved = outbound * weights[entry];
                        out_ind[dst * ind_stride + (n_ages + age) * n_ztypes + ztype] += moved;
                        moved_total += moved;
                    }
                    out_ind[src_idx] += value - moved_total;
                } else if row_start == row_end {
                    // Empty CSR row: keep everything at the source.
                    out_ind[src_idx] += value;
                } else {
                    let stay = value - outbound;
                    out_ind[src_idx] += stay;
                    for entry in row_start..row_end {
                        let dst = dest_idx[entry] as usize;
                        let prob = weights[entry];
                        out_ind[dst * ind_stride + (n_ages + age) * n_ztypes + ztype] +=
                            outbound * prob;
                    }
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
    if rngs.len() != n_demes {
        return Err(format!(
            "migration requires {n_demes} per-deme RNG streams, got {}",
            rngs.len()
        ));
    }
    let mut max_row = 1usize;
    for pair in indptr.windows(2) {
        let len = (pair[1] - pair[0]).max(0) as usize;
        if len > max_row {
            max_row = len;
        }
    }
    let mut out_ind = vec![0.0; ind_all.len()];
    let mut out_sperm = vec![0.0; sperm_all.len()];
    let mut distributed = vec![0.0; max_row];
    let mut probs = vec![0.0; max_row];

    for src in 0..n_demes {
        let rng = &mut rngs[src];
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
                let mut virgin_count = female_total - stored_total;
                if virgin_count < 0.0 && virgin_count.abs() < 1e-9 {
                    virgin_count = 0.0;
                }

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
    let mut rngs: Vec<SessionRng> = (0..n_demes)
        .map(|deme| new_rng(crate::rng::stream_seed(seed, deme as i64)))
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
    if value <= 0.0 || rate <= 0.0 {
        return 0.0;
    }
    if rate >= 1.0 {
        return value;
    }
    if continuous_sampling {
        return crate::rng::continuous_binomial(rng, value, rate);
    }
    crate::rng::binomial(rng, value.round() as i64, rate)
}

/// Distribute outbound migrants among the CSR destinations of one source row.
///
/// The row weights are normalized before the multinomial samplers see them:
/// kernel-mode CSR rows may intentionally sum to less than one (boundary
/// demes keep mass at the source), and the discrete multinomial sampler
/// assumes a probability vector.
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
    for slot in distributed.iter_mut() {
        *slot = 0.0;
    }
    if outbound <= 0.0 || row_len == 0 {
        return 0.0;
    }
    let mut total = 0.0;
    for pos in 0..row_len {
        let weight = weights[row_start + pos];
        probs[pos] = weight;
        total += weight;
    }
    if total <= 0.0 {
        return 0.0;
    }
    let inv_total = 1.0 / total;
    for pos in 0..row_len {
        probs[pos] *= inv_total;
    }
    if continuous_sampling {
        crate::rng::continuous_multinomial(rng, outbound, &probs[..row_len], distributed);
    } else {
        crate::rng::multinomial(rng, outbound.round() as i64, &probs[..row_len], distributed);
    }
    distributed[..row_len].iter().sum()
}
