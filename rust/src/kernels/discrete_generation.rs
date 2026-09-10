//! Discrete-generation and Wright-Fisher lifecycle kernels.
//!
//! The stage kernels read the frozen [`Blueprint`] (dimensions and sampling
//! flags), the live [`EcologyParams`] columns at the requested deme, and the
//! shared [`GeneticsTensors`] directly — the per-deme scalars live in cells
//! of the unified ``(2, n_ages)`` vectors (adult column 1, juvenile column 0,
//! for the normalized two-age discrete draft).

#![allow(clippy::needless_range_loop)] // Index loops mirror the Python reference for parity review.
#![allow(clippy::too_many_arguments)] // Stage boundaries pass parallel state/config channels.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::hooks::interpreter::HookProgram;
use crate::kernels::age_structured::EcoCtx;
use crate::kernels::density_regulation;
use crate::kernels::equilibrium::equilibrium_metrics;
use crate::kernels::rng::{
    binomial, clamp01, continuous_binomial, continuous_multinomial, continuous_poisson,
    multinomial, poisson, SessionRng, EPS,
};
use crate::model::blueprint::Blueprint;
use crate::model::ecology::EcologyParams;
use crate::model::genetics::GeneticsTensors;

/// Juvenile growth mode: no density regulation.
const NO_COMPETITION: i64 = 0;
/// Juvenile growth mode: fixed carrying-capacity ceiling.
const FIXED: i64 = 1;
/// Wright-Fisher mode: multinomial offspring sampling.
const WF_MULTINOMIAL: i64 = 1;
/// Wright-Fisher mode: Poisson offspring sampling.
const WF_POISSON: i64 = 2;
/// Wright-Fisher mode: deterministic expected offspring counts.
const WF_DETERMINISTIC: i64 = 3;

/// Validate the discrete normalization (two sexes, two ages).
///
/// Discrete simulations always use two sexes and two ages; the session
/// constructors call this once so a non-discrete blueprint is rejected at
/// the boundary instead of inside a tick.
///
/// ## Errors
/// Returns ``PyValueError`` when the blueprint is not discrete-shaped.
pub fn validate_discrete_shape(bp: &Blueprint) -> PyResult<()> {
    if bp.n_ages != 2 {
        return Err(PyValueError::new_err(format!(
            "discrete blueprint must have n_ages == 2, got {}",
            bp.n_ages
        )));
    }
    Ok(())
}

/// Flat index for discrete state with exactly two sexes and two ages.
///
/// ## Parameters
/// - `sex`: Sex index (0/1).
/// - `age`: Age index (0/1).
/// - `ztype`: Zygote type index.
/// - `n_ztypes`: Number of zygote types.
///
/// ## Returns
/// Flat index into the discrete individual-count slice.
#[inline]
fn idx(sex: usize, age: usize, ztype: usize, n_ztypes: usize) -> usize {
    (sex * 2 + age) * n_ztypes + ztype
}

/// Normalize male mating weights into a female x male probability matrix.
///
/// ## Parameters
/// - `n_ztypes`: Number of zygote types.
/// - `sexual_selection_fitness`: Flat female x male fitness table.
/// - `male_counts`: Effective adult male counts.
/// - `out`: Output matrix, overwritten.
fn compute_mating_probability(
    n_ztypes: usize,
    sexual_selection_fitness: &[f64],
    male_counts: &[f64],
    out: &mut [f64],
) {
    // Same row-normalized mating probabilities as the age-structured path,
    // but for the discrete two-age adult class only.
    for gf in 0..n_ztypes {
        let mut row_sum = 0.0;
        for gm in 0..n_ztypes {
            let value = sexual_selection_fitness[gf * n_ztypes + gm] * male_counts[gm];
            out[gf * n_ztypes + gm] = value;
            row_sum += value;
        }
        if row_sum.is_finite() && row_sum > EPS {
            for gm in 0..n_ztypes {
                out[gf * n_ztypes + gm] /= row_sum;
            }
        } else {
            for gm in 0..n_ztypes {
                out[gf * n_ztypes + gm] = 0.0;
            }
        }
    }
}

/// Sample the number of matings per female genotype and distribute male partners.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `bp`: The frozen blueprint (dimensions and sampling flags).
/// - `p_mating`: Clamped adult female mating rate for this tick.
/// - `females`: Adult female counts per zygote type.
/// - `mating_prob`: Precomputed mating probability matrix.
/// - `pair_counts`: Output mated pair counts, accumulated in place.
fn mate_discrete(
    rng: &mut SessionRng,
    bp: &Blueprint,
    p_mating: f64,
    females: &[f64],
    mating_prob: &[f64],
    pair_counts: &mut [f64],
) {
    // For each adult female genotype, sample the number of matings and
    // distribute male partners using the mating probability row.
    let g = bp.n_ztypes;
    let mut tmp = vec![0.0; g];
    for gf in 0..g {
        let n_female = females[gf];
        if n_female <= 0.0 {
            continue;
        }
        let n_mating = if bp.stochastic {
            if bp.continuous_sampling {
                continuous_binomial(rng, n_female, p_mating)
            } else {
                let n_int = (n_female.round() as i64).max(0);
                if n_int > 0 {
                    binomial(rng, n_int, p_mating)
                } else {
                    0.0
                }
            }
        } else {
            n_female * p_mating
        };
        if n_mating <= EPS {
            continue;
        }
        let row = &mating_prob[gf * g..(gf + 1) * g];
        if bp.stochastic {
            if bp.continuous_sampling {
                continuous_multinomial(rng, n_mating, row, &mut tmp);
                for gm in 0..g {
                    pair_counts[gf * g + gm] += tmp[gm];
                }
            } else {
                let n_int = (n_mating.round() as i64).max(0);
                if n_int > 0 {
                    multinomial(rng, n_int, row, &mut tmp);
                    for gm in 0..g {
                        pair_counts[gf * g + gm] += tmp[gm];
                    }
                }
            }
        } else {
            for gm in 0..g {
                pair_counts[gf * g + gm] += n_mating * row[gm];
            }
        }
    }
}

/// Convert mated pairs into age-0 female/male offspring counts.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `bp`: The frozen blueprint (dimensions and sampling flags).
/// - `eco`: Current ecology columns; the deme's segment feeds every rate.
/// - `genetics`: Shared genetics tables.
/// - `deme`: Deme column the rates are read from.
/// - `pair_counts`: Mated pair counts.
/// - `n_f`: Output female age-0 counts.
/// - `n_m`: Output male age-0 counts.
fn fertilize_discrete(
    rng: &mut SessionRng,
    bp: &Blueprint,
    eco: &EcologyParams,
    genetics: &GeneticsTensors,
    deme: usize,
    pair_counts: &[f64],
    n_f: &mut [f64],
    n_m: &mut [f64],
) {
    // Convert mated pairs into age-0 offspring counts:
    // - Egg production uses fecundity and reproduction rate.
    // - Offspring genotype comes from the offspring tensor.
    // - Sex is assigned by sex ratio or sex-chromosome rules.
    let g = bp.n_ztypes;
    let a = bp.n_ages;
    let p_reproduce = clamp01(eco.reproduction_rates[deme * a + 1]);
    let sex_ratio = clamp01(eco.sex_ratio[deme]);
    let eggs_per_female = eco.eggs_per_female[deme];
    let fecundity_f = &genetics.fecundity_fitness[..g];
    let fecundity_m = &genetics.fecundity_fitness[g..(2 * g)];
    let mut offspring = vec![0.0; g];
    let mut p_norm = vec![0.0; g];
    let mut tmp = vec![0.0; g];
    let mut has_any = false;

    for gf in 0..g {
        let ff = fecundity_f[gf];
        for gm in 0..g {
            let n_pairs = pair_counts[gf * g + gm];
            if n_pairs <= 0.0 {
                continue;
            }
            has_any = true;
            let eggs_per_pair = eggs_per_female * ff * fecundity_m[gm];
            let n_total = if bp.stochastic {
                let n_pairs_eff = if bp.continuous_sampling {
                    n_pairs
                } else {
                    n_pairs.round()
                };
                if n_pairs_eff <= 0.0 {
                    continue;
                }
                let n_reproducing = if p_reproduce < 1.0 - EPS {
                    if bp.continuous_sampling {
                        continuous_binomial(rng, n_pairs_eff, p_reproduce)
                    } else {
                        binomial(rng, n_pairs_eff as i64, p_reproduce)
                    }
                } else {
                    n_pairs_eff
                };
                let total_lambda = (n_reproducing * eggs_per_pair).max(0.0);
                if bp.continuous_sampling {
                    continuous_poisson(rng, total_lambda)
                } else {
                    poisson(rng, total_lambda)
                }
            } else {
                n_pairs * p_reproduce * eggs_per_pair
            };
            if n_total <= EPS {
                continue;
            }

            let mut p_surv = 0.0;
            for go in 0..g {
                p_surv += genetics.offspring_tensor[(gf * g + gm) * g + go];
            }
            if bp.stochastic {
                if p_surv <= EPS {
                    continue;
                }
                let n_viable = if p_surv >= 1.0 - EPS {
                    n_total
                } else if bp.continuous_sampling {
                    continuous_binomial(rng, n_total, p_surv)
                } else {
                    binomial(rng, n_total.round() as i64, p_surv)
                };
                if n_viable <= EPS {
                    continue;
                }
                let inv = 1.0 / p_surv;
                for go in 0..g {
                    p_norm[go] = genetics.offspring_tensor[(gf * g + gm) * g + go] * inv;
                }
                if bp.continuous_sampling {
                    continuous_multinomial(rng, n_viable, &p_norm, &mut tmp);
                    for go in 0..g {
                        offspring[go] += tmp[go];
                    }
                } else {
                    multinomial(rng, n_viable.round() as i64, &p_norm, &mut tmp);
                    for go in 0..g {
                        offspring[go] += tmp[go];
                    }
                }
            } else {
                for go in 0..g {
                    offspring[go] += n_total * genetics.offspring_tensor[(gf * g + gm) * g + go];
                }
            }
        }
    }

    if !has_any || offspring.iter().sum::<f64>() <= EPS {
        return;
    }
    for go in 0..g {
        let n_g = offspring[go];
        if n_g <= EPS {
            continue;
        }
        if bp.has_sex_chromosomes && bp.female_only_by_sex_chrom[go] {
            n_f[go] = n_g;
        } else if bp.has_sex_chromosomes && bp.male_only_by_sex_chrom[go] {
            n_m[go] = n_g;
        } else {
            let p_f = if bp.has_sex_chromosomes {
                let denom =
                    genetics.female_ztype_compatibility[go] + genetics.male_ztype_compatibility[go];
                if denom > EPS {
                    clamp01(genetics.female_ztype_compatibility[go] / denom)
                } else {
                    0.5
                }
            } else {
                sex_ratio
            };
            let n_fem = if bp.stochastic {
                if bp.continuous_sampling {
                    continuous_binomial(rng, n_g, p_f)
                } else {
                    binomial(rng, n_g.round() as i64, p_f)
                }
            } else {
                n_g * p_f
            };
            n_f[go] = n_fem;
            n_m[go] = n_g - n_fem;
        }
    }
}

/// Run the discrete reproduction stage in place.
///
/// Builds effective males, samples mating pairs, and fertilizes them into
/// age-0 offspring.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `bp`: The frozen blueprint (dimensions and sampling flags).
/// - `eco`: Current ecology columns; the deme's segment feeds every rate.
/// - `genetics`: Shared genetics tables.
/// - `deme`: Deme column the rates are read from.
/// - `ind`: Mutable discrete individual-count slice.
pub fn reproduction(
    rng: &mut SessionRng,
    bp: &Blueprint,
    eco: &EcologyParams,
    genetics: &GeneticsTensors,
    deme: usize,
    ind: &mut [f64],
) {
    // Discrete reproduction pipeline:
    // 1. Build effective adult males.
    // 2. Build mating probabilities and sample pair counts.
    // 3. Fertilize pairs into age-0 female/male offspring.
    let g = bp.n_ztypes;
    let a = bp.n_ages;
    let mating = &eco.mating_rates[deme * 2 * a..(deme + 1) * 2 * a];
    let female_adult_mating_rate = mating[1];
    let male_adult_mating_rate = mating[2 + 1];
    let mut effective_males = vec![0.0; g];
    for z in 0..g {
        effective_males[z] = ind[idx(1, 1, z, g)] * male_adult_mating_rate;
    }
    let females_total: f64 = (0..g).map(|z| ind[idx(0, 1, z, g)]).sum();
    let males_total: f64 = effective_males.iter().sum();
    if males_total == 0.0 || females_total == 0.0 {
        return;
    }

    let mut mating_prob = vec![0.0; g * g];
    compute_mating_probability(
        g,
        &genetics.sexual_selection_fitness,
        &effective_males,
        &mut mating_prob,
    );
    let mut pair_counts = vec![0.0; g * g];
    let females: Vec<f64> = (0..g).map(|z| ind[idx(0, 1, z, g)]).collect();
    let p_mating = clamp01(female_adult_mating_rate);
    mate_discrete(rng, bp, p_mating, &females, &mating_prob, &mut pair_counts);

    let mut n_f = vec![0.0; g];
    let mut n_m = vec![0.0; g];
    fertilize_discrete(
        rng,
        bp,
        eco,
        genetics,
        deme,
        &pair_counts,
        &mut n_f,
        &mut n_m,
    );
    for z in 0..g {
        ind[idx(0, 0, z, g)] = n_f[z];
        ind[idx(1, 0, z, g)] = n_m[z];
    }
}

/// Compute juvenile density-regulation scaling for discrete populations.
///
/// ## Parameters
/// - `bp`: The frozen blueprint (dimensions).
/// - `eco`: Current ecology columns; the deme's segment feeds every rate.
/// - `deme`: Deme column the rates are read from.
/// - `ind`: Current discrete individual-count slice.
///
/// ## Returns
/// A non-negative scaling factor.
fn scaling_factor(bp: &Blueprint, eco: &EcologyParams, deme: usize, ind: &[f64]) -> f64 {
    // Juvenile density regulation for discrete populations;
    // identical growth-mode semantics to the age-structured engine.
    let g = bp.n_ztypes;
    let total_age_0: f64 = (0..g)
        .map(|z| ind[idx(0, 0, z, g)] + ind[idx(1, 0, z, g)])
        .sum();
    let juvenile_growth_mode = eco.growth_mode[deme];
    if juvenile_growth_mode == NO_COMPETITION {
        return 1.0;
    }
    if juvenile_growth_mode == FIXED {
        // The equilibrium metrics are derived from the current deme column
        // on demand (the same computation the retired config assembly
        // performed per tick); the fixed ceiling itself does not use the
        // competition strength.
        let (_, expected_survival_rate) = equilibrium_metrics(bp, eco, deme);
        return density_regulation::regulation_scaling(
            juvenile_growth_mode,
            total_age_0,
            eco.carrying_capacity[deme],
            eco.low_density_growth_rate[deme],
            expected_survival_rate,
        )
        .unwrap_or(1.0);
    }
    let (expected_competition_strength, expected_survival_rate) =
        equilibrium_metrics(bp, eco, deme);
    density_regulation::regulation_scaling(
        juvenile_growth_mode,
        total_age_0,
        expected_competition_strength,
        eco.low_density_growth_rate[deme],
        expected_survival_rate,
    )
    .unwrap_or(1.0)
}

/// Apply juvenile scaling by resampling age-0 counts.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `bp`: The frozen blueprint (dimensions and sampling flags).
/// - `ind`: Mutable discrete individual-count slice.
/// - `scaling`: Scaling factor.
fn recruit_juveniles(rng: &mut SessionRng, bp: &Blueprint, ind: &mut [f64], scaling: f64) {
    // Resample age-0 counts to the scaled total;
    // stochastic uses multinomial, deterministic scales proportionally.
    let g = bp.n_ztypes;
    let stochastic = bp.stochastic;
    let continuous = bp.continuous_sampling;
    let mut combined = Vec::with_capacity(2 * g);
    let mut total = 0.0;
    for sex in 0..2 {
        for z in 0..g {
            let mut value = ind[idx(sex, 0, z, g)];
            if stochastic && !continuous {
                value = value.round();
            }
            combined.push(value);
            total += value;
        }
    }
    if total <= 0.0 {
        for sex in 0..2 {
            for z in 0..g {
                ind[idx(sex, 0, z, g)] = 0.0;
            }
        }
        return;
    }
    let desired = if stochastic && !continuous {
        (total * scaling).round()
    } else {
        total * scaling
    };
    if desired <= 0.0 {
        for sex in 0..2 {
            for z in 0..g {
                ind[idx(sex, 0, z, g)] = 0.0;
            }
        }
        return;
    }
    let mut probs = vec![0.0; combined.len()];
    for (prob, &count) in probs.iter_mut().zip(combined.iter()) {
        *prob = count / total;
    }
    let mut draws = vec![0.0; combined.len()];
    if stochastic {
        if continuous {
            continuous_multinomial(rng, desired, &probs, &mut draws);
        } else {
            multinomial(rng, desired.round() as i64, &probs, &mut draws);
        }
    } else {
        for (draw, &count) in draws.iter_mut().zip(combined.iter()) {
            *draw = count * (desired / total);
        }
    }
    for sex in 0..2 {
        for z in 0..g {
            ind[idx(sex, 0, z, g)] = draws[sex * g + z];
        }
    }
}

/// Run the discrete survival stage in place (density regulation plus viability).
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `bp`: The frozen blueprint (dimensions and sampling flags).
/// - `eco`: Current ecology columns; the deme's segment feeds every rate.
/// - `genetics`: Shared genetics tables.
/// - `deme`: Deme column the rates are read from.
/// - `ind`: Mutable discrete individual-count slice.
pub fn survival(
    rng: &mut SessionRng,
    bp: &Blueprint,
    eco: &EcologyParams,
    genetics: &GeneticsTensors,
    deme: usize,
    ind: &mut [f64],
) {
    // Discrete survival applies density scaling first, then viability
    // survival separately for female and male age-0 individuals.
    let g = bp.n_ztypes;
    let a = bp.n_ages;
    let scaling = scaling_factor(bp, eco, deme, ind);
    recruit_juveniles(rng, bp, ind, scaling);

    let survival = &eco.survival_rates[deme * 2 * a..(deme + 1) * 2 * a];
    let s_f = survival[0];
    let s_m = survival[2];
    let viability_f = &genetics.viability_fitness[..g];
    let viability_m = &genetics.viability_fitness[(2 * g)..(3 * g)];
    for z in 0..g {
        let f = ind[idx(0, 0, z, g)];
        let m = ind[idx(1, 0, z, g)];
        let rate_f = s_f * viability_f[z];
        let rate_m = s_m * viability_m[z];
        if bp.stochastic {
            if bp.continuous_sampling {
                ind[idx(0, 0, z, g)] = continuous_binomial(rng, f, rate_f);
                ind[idx(1, 0, z, g)] = continuous_binomial(rng, m, rate_m);
            } else {
                let nf = f.round() as i64;
                let nm = m.round() as i64;
                ind[idx(0, 0, z, g)] = if nf > 0 {
                    binomial(rng, nf, rate_f)
                } else {
                    0.0
                };
                ind[idx(1, 0, z, g)] = if nm > 0 {
                    binomial(rng, nm, rate_m)
                } else {
                    0.0
                };
            }
        } else {
            ind[idx(0, 0, z, g)] = f * rate_f;
            ind[idx(1, 0, z, g)] = m * rate_m;
        }
    }
}

/// Move age-0 juveniles to age 1 and clear age 0.
///
/// ## Parameters
/// - `bp`: The frozen blueprint (dimensions).
/// - `ind`: Mutable discrete individual-count slice.
pub fn aging(bp: &Blueprint, ind: &mut [f64]) {
    // Move age-0 juveniles into age 1 and clear age 0.
    let g = bp.n_ztypes;
    for z in 0..g {
        ind[idx(0, 1, z, g)] = ind[idx(0, 0, z, g)];
        ind[idx(0, 0, z, g)] = 0.0;
        ind[idx(1, 1, z, g)] = ind[idx(1, 0, z, g)];
        ind[idx(1, 0, z, g)] = 0.0;
    }
}

/// Run one discrete-generation tick with hooks in the reference stage order.
///
/// Stage order: first hook -> reproduction -> early hook -> survival -> late
/// hook -> aging.  The stage kernels read the live ecology columns and
/// genetics at each stage boundary, so set_param writes and callback
/// genetics candidates committed at a boundary are visible to the later
/// stages of the same tick without any snapshot rebuild.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `bp`: The frozen blueprint.
/// - `hooks`: CSR hook program.
/// - `ind`: Mutable discrete individual-count slice.
/// - `tick`: Current tick.
/// - `deme_id`: Current deme id (hook selectors and journaling).
/// - `eco_values`: Live ECO scratch for ``OP_SET_PARAM``.
/// - `eco_ctx`: Optional write-back context; when present, ECO writes are
///   committed after each event boundary and the stage kernels read the
///   committed columns through the context (same-tick visibility, Python
///   parity).  When absent, *columns* supplies the stage sources.
/// - `columns`: Stage sources ``(ecology, genetics, deme)`` used only when
///   *eco_ctx* is ``None`` — hook-free spatial demes read their session
///   column segment directly.  Panmictic sessions always lend a context and
///   pass ``None``.
///
/// ## Returns
/// ``Ok(0)`` for continue, ``Ok(1)`` if a hook requested stop, or an error string.
pub fn run_tick(
    rng: &mut SessionRng,
    bp: &Blueprint,
    hooks: &HookProgram,
    ind: &mut [f64],
    tick: i64,
    deme_id: i64,
    eco_values: &mut [f64],
    eco_ctx: &mut Option<EcoCtx<'_>>,
    columns: Option<(&EcologyParams, &GeneticsTensors, usize)>,
) -> Result<i32, String> {
    // One discrete tick follows: first hook -> reproduction -> early hook
    // -> survival -> late hook -> aging.
    // Discrete tick order mirrors the age-structured engine; each event
    // executes CSR plan slots and Python callback slots in one cross-type
    // priority order.  With an EcoCtx, set_param writes are committed at
    // each boundary so the stage kernels observe them through the live
    // columns (Python parity).  The ctx tick is re-stamped per tick so
    // batch loops journal under the correct tick value.
    if let Some(ctx) = eco_ctx.as_mut() {
        ctx.tick = tick;
        ctx.phase = 0;
    }

    let mut result = hooks.execute_event(
        rng,
        0,
        ind,
        &mut [],
        2,
        2,
        bp.n_ztypes,
        tick,
        bp.stochastic,
        bp.continuous_sampling,
        deme_id,
        eco_values,
        eco_ctx,
    )?;
    if let Some(ctx) = eco_ctx.as_mut() {
        ctx.commit(eco_values)?;
    }
    if result != 0 {
        return Ok(result);
    }
    if let Some(ctx) = eco_ctx.as_mut() {
        ctx.phase = 1;
    }
    let (eco, genetics, deme) = super::age_structured::stage_sources(eco_ctx.as_ref(), columns);
    reproduction(rng, bp, eco, genetics, deme, ind);
    if let Some(ctx) = eco_ctx.as_mut() {
        ctx.phase = 2;
    }
    result = hooks.execute_event(
        rng,
        1,
        ind,
        &mut [],
        2,
        2,
        bp.n_ztypes,
        tick,
        bp.stochastic,
        bp.continuous_sampling,
        deme_id,
        eco_values,
        eco_ctx,
    )?;
    if let Some(ctx) = eco_ctx.as_mut() {
        ctx.commit(eco_values)?;
    }
    if result != 0 {
        return Ok(result);
    }
    if let Some(ctx) = eco_ctx.as_mut() {
        ctx.phase = 3;
    }
    let (eco, genetics, deme) = super::age_structured::stage_sources(eco_ctx.as_ref(), columns);
    survival(rng, bp, eco, genetics, deme, ind);
    if let Some(ctx) = eco_ctx.as_mut() {
        ctx.phase = 4;
    }
    result = hooks.execute_event(
        rng,
        2,
        ind,
        &mut [],
        2,
        2,
        bp.n_ztypes,
        tick,
        bp.stochastic,
        bp.continuous_sampling,
        deme_id,
        eco_values,
        eco_ctx,
    )?;
    if let Some(ctx) = eco_ctx.as_mut() {
        ctx.commit(eco_values)?;
    }
    if result != 0 {
        return Ok(result);
    }
    if let Some(ctx) = eco_ctx.as_mut() {
        ctx.phase = 5;
    }
    aging(bp, ind);
    Ok(0)
}

/// Run one fused Wright-Fisher tick (first hook plus full WF update).
///
/// Only the first hook runs; the next generation is computed directly from
/// adult allele frequencies using the configured extreme-speed mode.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `bp`: The frozen blueprint (dimensions and sampling flags).
/// - `eco`: Current ecology columns; the deme's segment feeds every rate.
/// - `genetics`: Shared genetics tables.
/// - `deme`: Deme column the rates are read from.
/// - `ind`: Mutable discrete individual-count slice.
///
/// ## Returns
/// ``Ok(())`` or an error string for an unknown WF mode.
pub fn run_wf_tick(
    rng: &mut SessionRng,
    bp: &Blueprint,
    eco: &EcologyParams,
    genetics: &GeneticsTensors,
    deme: usize,
    ind: &mut [f64],
) -> Result<(), String> {
    // Fused Wright-Fisher tick: only the first hook runs, then the entire
    // next generation is computed from adult allele frequencies in one step.
    let mode = bp.extreme_speed_mode;
    if !matches!(mode, WF_MULTINOMIAL | WF_POISSON | WF_DETERMINISTIC) {
        return Err(format!("unrecognised extreme_speed_mode={mode}"));
    }
    let g = bp.n_ztypes;
    let a = bp.n_ages;
    let mating = &eco.mating_rates[deme * 2 * a..(deme + 1) * 2 * a];
    let female_adult_mating_rate = mating[1];
    let male_adult_mating_rate = mating[2 + 1];
    let reproduction_rate = eco.reproduction_rates[deme * a + 1];
    let eggs_per_female = eco.eggs_per_female[deme];
    let sex_ratio = eco.sex_ratio[deme];
    let fecundity_f = &genetics.fecundity_fitness[..g];
    let fecundity_m = &genetics.fecundity_fitness[g..(2 * g)];
    let viability_f = &genetics.viability_fitness[..g];
    let viability_m = &genetics.viability_fitness[(2 * g)..(3 * g)];
    let mut expected_f = vec![0.0; g];
    let mut expected_m = vec![0.0; g];
    let adult_f: Vec<f64> = (0..g).map(|z| ind[idx(0, 1, z, g)]).collect();
    let adult_m: Vec<f64> = (0..g).map(|z| ind[idx(1, 1, z, g)]).collect();
    let effective_m: Vec<f64> = adult_m
        .iter()
        .map(|&v| v * male_adult_mating_rate)
        .collect();

    for gf in 0..g {
        let nf = adult_f[gf] * fecundity_f[gf] * female_adult_mating_rate;
        if nf <= 0.0 {
            continue;
        }
        let row_sum: f64 = (0..g)
            .map(|gm| genetics.sexual_selection_fitness[gf * g + gm] * effective_m[gm])
            .sum();
        if row_sum <= 0.0 {
            continue;
        }
        for gm in 0..g {
            let nm_eff = effective_m[gm];
            if nm_eff <= 0.0 {
                continue;
            }
            let pair_weight =
                nf * (nm_eff / row_sum) * genetics.sexual_selection_fitness[gf * g + gm];
            if pair_weight <= 0.0 {
                continue;
            }
            for go in 0..g {
                let prob = genetics.offspring_tensor[(gf * g + gm) * g + go];
                if prob <= 0.0 {
                    continue;
                }
                let offspring =
                    pair_weight * prob * eggs_per_female * reproduction_rate * fecundity_m[gm];
                if bp.has_sex_chromosomes {
                    if bp.female_only_by_sex_chrom[go] {
                        expected_f[go] += offspring;
                    } else if bp.male_only_by_sex_chrom[go] {
                        expected_m[go] += offspring;
                    } else {
                        expected_f[go] += offspring * genetics.female_ztype_compatibility[go];
                        expected_m[go] += offspring * genetics.male_ztype_compatibility[go];
                    }
                } else {
                    expected_f[go] += offspring * sex_ratio;
                    expected_m[go] += offspring * (1.0 - sex_ratio);
                }
            }
        }
    }
    for go in 0..g {
        expected_f[go] *= viability_f[go];
        expected_m[go] *= viability_m[go];
    }
    let juvenile_growth_mode = eco.growth_mode[deme];
    if juvenile_growth_mode > 0 {
        let total: f64 = expected_f.iter().chain(expected_m.iter()).sum();
        // Same curve dispatch as the staged tick; for the fused Wright-Fisher
        // update the "actual competition strength" is the full offspring total.
        let (expected_competition_strength, expected_survival_rate) =
            equilibrium_metrics(bp, eco, deme);
        // Same curve dispatch as the staged tick; the fixed ceiling caps at
        // the carrying capacity, the compensatory family at the competition
        // strength.
        let sf = density_regulation::regulation_scaling(
            juvenile_growth_mode,
            total,
            if juvenile_growth_mode == FIXED {
                eco.carrying_capacity[deme]
            } else {
                expected_competition_strength
            },
            eco.low_density_growth_rate[deme],
            expected_survival_rate,
        )
        .unwrap_or(1.0);
        for value in expected_f.iter_mut().chain(expected_m.iter_mut()) {
            *value *= sf;
        }
    }
    let mut new_f = vec![0.0; g];
    let mut new_m = vec![0.0; g];
    if mode == WF_DETERMINISTIC || !bp.stochastic {
        new_f.copy_from_slice(&expected_f);
        new_m.copy_from_slice(&expected_m);
    } else if mode == WF_MULTINOMIAL {
        let total: f64 = expected_f.iter().chain(expected_m.iter()).sum();
        if total > 0.0 {
            let mut probs = Vec::with_capacity(2 * g);
            for &v in expected_f.iter().chain(expected_m.iter()) {
                probs.push(v / total);
            }
            let n_total = total.round() as i64;
            if n_total > 0 {
                let mut draws = vec![0.0; 2 * g];
                multinomial(rng, n_total, &probs, &mut draws);
                new_f[..g].copy_from_slice(&draws[..g]);
                new_m[..g].copy_from_slice(&draws[g..(g + g)]);
            }
        }
    } else if mode == WF_POISSON {
        for go in 0..g {
            new_f[go] = poisson(rng, expected_f[go].max(0.0));
            new_m[go] = poisson(rng, expected_m[go].max(0.0));
        }
    }
    for z in 0..g {
        ind[idx(0, 0, z, g)] = 0.0;
        ind[idx(1, 0, z, g)] = 0.0;
        ind[idx(0, 1, z, g)] = new_f[z];
        ind[idx(1, 1, z, g)] = new_m[z];
    }
    Ok(())
}
