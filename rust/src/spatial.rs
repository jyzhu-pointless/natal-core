//! Spatial multi-deme lifecycle and migration kernels.
//!
//! The module runs the per-deme age-structured lifecycle over a stacked state,
//! supports heterogeneous config banks, and implements both adjacency-based
//! and topology-kernel migration in deterministic and stochastic forms.

#![allow(clippy::needless_range_loop)] // Index loops mirror the Python reference for parity review.
#![allow(clippy::too_many_arguments)] // Migration helpers mirror the Numba kernel signatures.

use rand::rngs::SmallRng;
use rayon::prelude::*;

use crate::config::SimConfig;
use crate::hooks::HookProgram;
use crate::lifecycle;
use crate::rng::new_rng;

/// Run one age-structured tick for every deme in parallel (homogeneous config).
///
/// The stacked state is split into per-deme chunks; each deme gets an
/// independent RNG derived from ``seed + deme_id``.
///
/// ## Parameters
/// - `cfg`: Shared simulation config.
/// - `hooks`: CSR hook program.
/// - `seed`: Base RNG seed.
/// - `ind_all`: Stacked individual-count slice.
/// - `sperm_all`: Stacked sperm-storage slice.
/// - `n_demes`: Number of demes.
/// - `tick`: Current tick.
///
/// ## Returns
/// ``Ok(())`` or an error string if a deme hook stops or shapes mismatch.
pub fn run_spatial_tick(
    cfg: &SimConfig,
    hooks: &HookProgram,
    seed: u64,
    ind_all: &mut [f64],
    sperm_all: &mut [f64],
    n_demes: usize,
    tick: i64,
) -> Result<(), String> {
    // Split stacked state into per-deme chunks and run each deme tick in
    // parallel with an independent RNG derived from seed + deme id.
    let n_ages = cfg.n_ages;
    let n_ztypes = cfg.n_ztypes;
    let ind_stride = 2 * n_ages * n_ztypes;
    let sperm_stride = n_ages * n_ztypes * n_ztypes;

    let mut ind_chunks: Vec<&mut [f64]> = ind_all.chunks_mut(ind_stride).collect();
    let mut sperm_chunks: Vec<&mut [f64]> = sperm_all.chunks_mut(sperm_stride).collect();
    if ind_chunks.len() != n_demes || sperm_chunks.len() != n_demes {
        return Err(format!(
            "stacked state length mismatch: expected {n_demes} demes, got ind={} sperm={}",
            ind_chunks.len(),
            sperm_chunks.len()
        ));
    }

    let results: Vec<Result<i32, String>> = ind_chunks
        .par_iter_mut()
        .zip(sperm_chunks.par_iter_mut())
        .enumerate()
        .map(|(deme_id, (ind, sperm))| {
            let mut rng: SmallRng = new_rng(seed.wrapping_add(deme_id as u64));
            lifecycle::run_tick(&mut rng, cfg, hooks, ind, sperm, tick, deme_id as i64)
        })
        .collect::<Vec<Result<i32, String>>>();

    for result in results {
        let code = result?;
        if code != 0 {
            return Err(format!("deme hook requested stop with code {code}"));
        }
    }
    Ok(())
}

/// Run one tick for every deme with per-deme configs from a config bank.
///
/// ## Parameters
/// - `configs`: Config bank.
/// - `hooks`: CSR hook program.
/// - `deme_config_ids`: Per-deme config index.
/// - `seed`: Base RNG seed.
/// - `ind_all`: Stacked individual-count slice.
/// - `sperm_all`: Stacked sperm-storage slice.
/// - `tick`: Current tick.
///
/// ## Returns
/// ``Ok(())`` or an error string.
pub fn run_spatial_tick_heterogeneous(
    configs: &[SimConfig],
    hooks: &HookProgram,
    deme_config_ids: &[usize],
    seed: u64,
    ind_all: &mut [f64],
    sperm_all: &mut [f64],
    tick: i64,
) -> Result<(), String> {
    // Like run_spatial_tick, but each deme looks up its own config from
    // the config bank via deme_config_ids.
    if configs.is_empty() || deme_config_ids.is_empty() {
        return Err(
            "heterogeneous spatial run requires at least one config and one deme".to_string(),
        );
    }
    let n_demes = deme_config_ids.len();
    let first = &configs[0];
    let n_ages = first.n_ages;
    let n_ztypes = first.n_ztypes;
    let ind_stride = 2 * n_ages * n_ztypes;
    let sperm_stride = n_ages * n_ztypes * n_ztypes;

    let mut ind_chunks: Vec<&mut [f64]> = ind_all.chunks_mut(ind_stride).collect();
    let mut sperm_chunks: Vec<&mut [f64]> = sperm_all.chunks_mut(sperm_stride).collect();
    if ind_chunks.len() != n_demes || sperm_chunks.len() != n_demes {
        return Err(format!(
            "stacked state length mismatch: expected {n_demes} demes, got ind={} sperm={}",
            ind_chunks.len(),
            sperm_chunks.len()
        ));
    }

    let results: Vec<Result<i32, String>> = ind_chunks
        .par_iter_mut()
        .zip(sperm_chunks.par_iter_mut())
        .enumerate()
        .map(|(deme_id, (ind, sperm))| {
            let cfg_idx = *deme_config_ids.get(deme_id).ok_or("missing config id")?;
            let cfg = configs.get(cfg_idx).ok_or("config id out of range")?;
            if cfg.n_ages != n_ages || cfg.n_ztypes != n_ztypes {
                return Err(format!(
                    "config {cfg_idx} dimensions do not match the stacked state"
                ));
            }
            let mut rng: SmallRng = new_rng(seed.wrapping_add(deme_id as u64));
            lifecycle::run_tick(&mut rng, cfg, hooks, ind, sperm, tick, deme_id as i64)
        })
        .collect::<Vec<Result<i32, String>>>();

    for result in results {
        let code = result?;
        if code != 0 {
            return Err(format!("deme hook requested stop with code {code}"));
        }
    }
    Ok(())
}

/// Deterministically move individuals and stored sperm across an adjacency matrix.
///
/// Outbound counts are ``value * rate`` and are distributed by adjacency
/// weights.  Female virgins and stored sperm are migrated separately from males.
///
/// ## Returns
/// ``(out_ind, out_sperm)`` new stacked arrays.
pub fn migrate_adjacency_deterministic(
    ind_all: &[f64],
    sperm_all: &[f64],
    adjacency: &[f64],
    rate: &[f64],
    n_demes: usize,
    n_ages: usize,
    n_ztypes: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    // For each source deme, compute outbound individuals/sperm using the
    // per-age migration rate, then distribute them by adjacency weights.
    // Males and stored sperm are handled separately from female virgins.
    let ind_stride = 2 * n_ages * n_ztypes;
    if adjacency.len() != n_demes * n_demes {
        return Err("adjacency must be a dense (n_demes, n_demes) matrix".to_string());
    }
    let mut out_ind = vec![0.0; ind_all.len()];
    let mut out_sperm = vec![0.0; sperm_all.len()];

    for src in 0..n_demes {
        for age in 0..n_ages {
            let migration_rate = if rate.len() == 1 { rate[0] } else { rate[age] };
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

                let outbound = virgin_count * migration_rate;
                let stay = virgin_count - outbound;
                let src_ind_idx = src * ind_stride + age * n_ztypes + female_ztype;
                out_ind[src_ind_idx] += stay;
                for dst in 0..n_demes {
                    let prob = adjacency[src * n_demes + dst];
                    if prob > 0.0 {
                        out_ind[dst * ind_stride + age * n_ztypes + female_ztype] +=
                            outbound * prob;
                    }
                }

                for male_ztype in 0..n_ztypes {
                    let sperm_idx = (src * n_ages + age) * n_ztypes * n_ztypes
                        + female_ztype * n_ztypes
                        + male_ztype;
                    let value = sperm_all[sperm_idx];
                    let outbound_sperm = value * migration_rate;
                    let stay_sperm = value - outbound_sperm;
                    out_sperm[sperm_idx] += stay_sperm;
                    out_ind[src_ind_idx] += stay_sperm;
                    for dst in 0..n_demes {
                        let prob = adjacency[src * n_demes + dst];
                        if prob > 0.0 {
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

        for sex in 1..2 {
            for age in 0..n_ages {
                let migration_rate = if rate.len() == 1 { rate[0] } else { rate[age] };
                for ztype in 0..n_ztypes {
                    let src_idx = src * ind_stride + (sex * n_ages + age) * n_ztypes + ztype;
                    let value = ind_all[src_idx];
                    let outbound = value * migration_rate;
                    let stay = value - outbound;
                    out_ind[src_idx] += stay;
                    for dst in 0..n_demes {
                        let prob = adjacency[src * n_demes + dst];
                        if prob > 0.0 {
                            out_ind[dst * ind_stride + (sex * n_ages + age) * n_ztypes + ztype] +=
                                outbound * prob;
                        }
                    }
                }
            }
        }
    }
    Ok((out_ind, out_sperm))
}

/// Stochastically move individuals and stored sperm across an adjacency matrix.
///
/// Outbound counts are sampled with binomial/continuous-binomial and then
/// multinomially distributed among destinations.  Each source deme uses its own
/// RNG stream derived from ``seed + deme_id``, matching the per-deme RNG policy
/// used by spatial lifecycle ticks.
///
/// ## Returns
/// ``(out_ind, out_sperm)`` new stacked arrays.
pub fn migrate_adjacency_stochastic(
    ind_all: &[f64],
    sperm_all: &[f64],
    adjacency: &[f64],
    rate: &[f64],
    seed: u64,
    continuous_sampling: bool,
    n_demes: usize,
    n_ages: usize,
    n_ztypes: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    // Stochastic variant of adjacency migration: sample outbound counts
    // and multinomially distribute them among destinations.
    let ind_stride = 2 * n_ages * n_ztypes;
    if adjacency.len() != n_demes * n_demes {
        return Err("adjacency must be a dense (n_demes, n_demes) matrix".to_string());
    }
    let mut out_ind = vec![0.0; ind_all.len()];
    let mut out_sperm = vec![0.0; sperm_all.len()];
    let mut distributed = vec![0.0; n_demes];
    let mut probs = vec![0.0; n_demes];

    // Each source deme gets its own RNG stream derived from the base seed.
    // This keeps per-deme stochastic migration reproducible and independent.
    for src in 0..n_demes {
        let mut rng = new_rng(seed.wrapping_add(src as u64));
        for age in 0..n_ages {
            let migration_rate = if rate.len() == 1 { rate[0] } else { rate[age] };
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

                let outbound =
                    sample_outbound(&mut rng, virgin_count, migration_rate, continuous_sampling);
                distribute_outbound(
                    &mut rng,
                    outbound,
                    adjacency,
                    src,
                    n_demes,
                    continuous_sampling,
                    &mut distributed,
                    &mut probs,
                );
                let mut moved_total = 0.0;
                for dst_pos in 0..n_demes {
                    let prob = adjacency[src * n_demes + dst_pos];
                    if prob <= 0.0 {
                        continue;
                    }
                    let moved = distributed[dst_pos];
                    moved_total += moved;
                    out_ind[dst_pos * ind_stride + age * n_ztypes + female_ztype] += moved;
                }
                out_ind[src * ind_stride + age * n_ztypes + female_ztype] +=
                    virgin_count - moved_total;

                for male_ztype in 0..n_ztypes {
                    let sperm_idx = (src * n_ages + age) * n_ztypes * n_ztypes
                        + female_ztype * n_ztypes
                        + male_ztype;
                    let value = sperm_all[sperm_idx];
                    let outbound_sperm =
                        sample_outbound(&mut rng, value, migration_rate, continuous_sampling);
                    distribute_outbound(
                        &mut rng,
                        outbound_sperm,
                        adjacency,
                        src,
                        n_demes,
                        continuous_sampling,
                        &mut distributed,
                        &mut probs,
                    );
                    let mut moved_total = 0.0;
                    for dst in 0..n_demes {
                        let prob = adjacency[src * n_demes + dst];
                        if prob <= 0.0 {
                            continue;
                        }
                        let moved = distributed[dst];
                        moved_total += moved;
                        let dst_sperm_idx = (dst * n_ages + age) * n_ztypes * n_ztypes
                            + female_ztype * n_ztypes
                            + male_ztype;
                        out_sperm[dst_sperm_idx] += moved;
                        out_ind[dst * ind_stride + age * n_ztypes + female_ztype] += moved;
                    }
                    out_sperm[sperm_idx] += value - moved_total;
                    out_ind[src * ind_stride + age * n_ztypes + female_ztype] +=
                        value - moved_total;
                }
            }
        }

        for sex in 1..2 {
            for age in 0..n_ages {
                let migration_rate = if rate.len() == 1 { rate[0] } else { rate[age] };
                for ztype in 0..n_ztypes {
                    let src_idx = src * ind_stride + (sex * n_ages + age) * n_ztypes + ztype;
                    let value = ind_all[src_idx];
                    let outbound =
                        sample_outbound(&mut rng, value, migration_rate, continuous_sampling);
                    distribute_outbound(
                        &mut rng,
                        outbound,
                        adjacency,
                        src,
                        n_demes,
                        continuous_sampling,
                        &mut distributed,
                        &mut probs,
                    );
                    let mut moved_total = 0.0;
                    for dst in 0..n_demes {
                        let prob = adjacency[src * n_demes + dst];
                        if prob <= 0.0 {
                            continue;
                        }
                        let moved = distributed[dst];
                        moved_total += moved;
                        out_ind[dst * ind_stride + (sex * n_ages + age) * n_ztypes + ztype] +=
                            moved;
                    }
                    out_ind[src_idx] += value - moved_total;
                }
            }
        }
    }
    Ok((out_ind, out_sperm))
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
fn sample_outbound(rng: &mut SmallRng, value: f64, rate: f64, continuous_sampling: bool) -> f64 {
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

/// Distribute outbound migrants among destination demes according to adjacency.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `outbound`: Total migrants leaving.
/// - `adjacency`: Dense adjacency matrix.
/// - `src`: Source deme index.
/// - `n_demes`: Number of demes.
/// - `continuous_sampling`: Use continuous sampling.
/// - `distributed`: Output per-destination counts.
/// - `probs`: Scratch probability buffer.
fn distribute_outbound(
    rng: &mut SmallRng,
    outbound: f64,
    adjacency: &[f64],
    src: usize,
    n_demes: usize,
    continuous_sampling: bool,
    distributed: &mut [f64],
    probs: &mut [f64],
) {
    // Collect positive adjacency probabilities and multinomially distribute
    // the outbound count among them.
    for slot in distributed.iter_mut() {
        *slot = 0.0;
    }
    if outbound <= 0.0 {
        return;
    }
    let mut total = 0.0;
    let mut count = 0;
    for dst in 0..n_demes {
        let prob = adjacency[src * n_demes + dst];
        if prob > 0.0 {
            probs[count] = prob;
            count += 1;
            total += prob;
        }
    }
    if total <= 0.0 {
        return;
    }
    if continuous_sampling {
        crate::rng::continuous_multinomial(rng, outbound, &probs[..count], distributed);
    } else {
        crate::rng::multinomial(rng, outbound.round() as i64, &probs[..count], distributed);
    }
}

/// Build a dense adjacency matrix from a migration kernel and topology.
///
/// Kernel offsets are applied to each source cell with optional wrapping;
/// each source row is normalized to sum to one.
///
/// ## Returns
/// A dense ``n_demes x n_demes`` adjacency matrix.
fn build_kernel_adjacency(
    migration_kernel: &[f64],
    topology_rows: usize,
    topology_cols: usize,
    topology_wrap: bool,
    kernel_include_center: bool,
) -> Result<Vec<f64>, String> {
    // Convert a topology migration kernel into a dense adjacency matrix by
    // wrapping or clipping kernel offsets and normalizing each source row.
    let n_demes = topology_rows * topology_cols;
    if n_demes == 0 {
        return Err("topology must have at least one deme".to_string());
    }
    let kernel_rows = topology_rows.max(1);
    let kernel_cols = topology_cols.max(1);
    if migration_kernel.len() != kernel_rows * kernel_cols {
        return Err(format!(
            "migration_kernel length {} does not match topology shape {}x{}",
            migration_kernel.len(),
            topology_rows,
            topology_cols
        ));
    }

    let mut adjacency = vec![0.0; n_demes * n_demes];
    let rows_i = topology_rows as isize;
    let cols_i = topology_cols as isize;
    let center_row = topology_rows / 2;
    let center_col = topology_cols / 2;
    for src_row in 0..topology_rows {
        for src_col in 0..topology_cols {
            let src = src_row * topology_cols + src_col;
            let mut total = 0.0;
            for kernel_row in 0..topology_rows {
                for kernel_col in 0..topology_cols {
                    if !kernel_include_center
                        && kernel_row == center_row
                        && kernel_col == center_col
                    {
                        continue;
                    }
                    let weight = migration_kernel[kernel_row * topology_cols + kernel_col];
                    if weight <= 0.0 {
                        continue;
                    }
                    let dst_row = src_row as isize + kernel_row as isize - center_row as isize;
                    let dst_col = src_col as isize + kernel_col as isize - center_col as isize;
                    let (dst_row, dst_col) = if topology_wrap {
                        (dst_row.rem_euclid(rows_i), dst_col.rem_euclid(cols_i))
                    } else if dst_row < 0 || dst_row >= rows_i || dst_col < 0 || dst_col >= cols_i {
                        continue;
                    } else {
                        (dst_row, dst_col)
                    };
                    let dst = dst_row as usize * topology_cols + dst_col as usize;
                    adjacency[src * n_demes + dst] += weight;
                    total += weight;
                }
            }
            if total > 0.0 {
                for dst in 0..n_demes {
                    adjacency[src * n_demes + dst] /= total;
                }
            }
        }
    }
    Ok(adjacency)
}

/// Deterministically migrate using a topology migration kernel.
///
/// Builds the kernel adjacency matrix and delegates to deterministic
/// adjacency migration.
///
/// ## Returns
/// ``(out_ind, out_sperm)``.
pub fn migrate_kernel_deterministic(
    ind_all: &[f64],
    sperm_all: &[f64],
    migration_kernel: &[f64],
    topology_rows: usize,
    topology_cols: usize,
    topology_wrap: bool,
    kernel_include_center: bool,
    rate: &[f64],
    n_ages: usize,
    n_ztypes: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    // Build the kernel adjacency once, then reuse deterministic adjacency
    // migration with the generated matrix.
    let adjacency = build_kernel_adjacency(
        migration_kernel,
        topology_rows,
        topology_cols,
        topology_wrap,
        kernel_include_center,
    )?;
    migrate_adjacency_deterministic(
        ind_all,
        sperm_all,
        &adjacency,
        rate,
        topology_rows * topology_cols,
        n_ages,
        n_ztypes,
    )
}

/// Stochastically migrate using a topology migration kernel.
///
/// Builds the kernel adjacency matrix and delegates to stochastic
/// adjacency migration, which uses a per-source-deme RNG stream.
///
/// ## Returns
/// ``(out_ind, out_sperm)``.
pub fn migrate_kernel_stochastic(
    ind_all: &[f64],
    sperm_all: &[f64],
    migration_kernel: &[f64],
    topology_rows: usize,
    topology_cols: usize,
    topology_wrap: bool,
    kernel_include_center: bool,
    rate: &[f64],
    seed: u64,
    continuous_sampling: bool,
    n_ages: usize,
    n_ztypes: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    // Build the kernel adjacency once, then reuse stochastic adjacency
    // migration with the generated matrix.
    let adjacency = build_kernel_adjacency(
        migration_kernel,
        topology_rows,
        topology_cols,
        topology_wrap,
        kernel_include_center,
    )?;
    migrate_adjacency_stochastic(
        ind_all,
        sperm_all,
        &adjacency,
        rate,
        seed,
        continuous_sampling,
        topology_rows * topology_cols,
        n_ages,
        n_ztypes,
    )
}
