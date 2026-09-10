// Numeric loops deliberately mirror the Python reference index-by-index;
// iterator rewrites would obscure parity review.
#![allow(clippy::needless_range_loop)]
// Lifecycle stage functions accept several parallel state/config channels.
#![allow(clippy::too_many_arguments)]
//! Age-structured lifecycle kernels: reproduction, survival, aging, and the
//! unified tick orchestration.
//!
//! The stage order mirrors ``natal.engine.lifecycle.run_structured_tick``:
//! first hook → reproduction → early hook → survival → late hook → aging.

use crate::kernels::rng::SessionRng;

use crate::hooks::interpreter::HookProgram;
use crate::kernels::density_regulation;
use crate::kernels::equilibrium::equilibrium_metrics;
use crate::kernels::rng::{
    binomial, clamp01, continuous_binomial, continuous_multinomial, continuous_poisson,
    multinomial, poisson, EPS,
};
use crate::model::blueprint::Blueprint;
use crate::model::ecology::EcologyParams;
use crate::model::genetics::GeneticsTensors;

/// Juvenile growth mode: no density regulation.
const NO_COMPETITION: i64 = 0;
/// Juvenile growth mode: fixed carrying-capacity ceiling.
const FIXED: i64 = 1;

/// Row-major flat index for ``(sex, age, ztype)``.
///
/// ## Parameters
/// - `sex`: Sex index (0 female, 1 male).
/// - `age`: Age class index.
/// - `ztype`: Zygote type index.
/// - `n_ages`: Number of age classes.
/// - `n_ztypes`: Number of zygote types.
///
/// ## Returns
/// Flat index into the individual-count slice.
#[inline]
fn ind_idx(sex: usize, age: usize, ztype: usize, n_ages: usize, n_ztypes: usize) -> usize {
    (sex * n_ages + age) * n_ztypes + ztype
}

/// Row-major flat index for ``(age, female_ztype, male_ztype)``.
///
/// ## Parameters
/// - `age`: Female age class.
/// - `female_ztype`: Female zygote type.
/// - `male_ztype`: Male zygote type.
/// - `n_ztypes`: Number of zygote types.
///
/// ## Returns
/// Flat index into the sperm-storage slice.
#[inline]
fn sperm_idx(age: usize, female_ztype: usize, male_ztype: usize, n_ztypes: usize) -> usize {
    (age * n_ztypes + female_ztype) * n_ztypes + male_ztype
}

/// First adult age class (mirrors the assembled `adult_start_age`).
///
/// ``Blueprint::validate`` guarantees ``adult_ages`` is exactly the adult
/// range, so this equals ``new_adult_age``; the fallback keeps the exact
/// arithmetic the retired config assembly used.
#[inline]
fn adult_start_age(bp: &Blueprint) -> usize {
    bp.adult_ages.first().copied().unwrap_or(0) as usize
}

/// Normalize male mating weights into a female x male probability matrix.
///
/// Each female row is proportional to ``sexual_selection_fitness * male_count``.
/// Rows with zero or non-finite totals become all-zero so no matings are produced.
///
/// ## Parameters
/// - `n_ztypes`: Number of zygote types.
/// - `sexual_selection_fitness`: Flat female x male fitness table.
/// - `male_counts`: Effective adult male counts per zygote type.
/// - `out`: Output matrix of shape ``(n_ztypes, n_ztypes)``, overwritten.
fn compute_mating_probability_matrix(
    n_ztypes: usize,
    sexual_selection_fitness: &[f64],
    male_counts: &[f64],
    out: &mut [f64],
) {
    // Each female row is proportional to sexual_selection_fitness * male_count.
    // Rows are normalized; zero/non-finite rows become all-zero so no matings occur.
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

/// Sample matings, displace stored sperm, and add new sperm for adult females.
///
/// For each adult female age/genotype the function computes virgin matings,
/// remating events that displace old sperm, and multinomial allocation of new
/// sperm among male zygote types.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `bp`: The frozen blueprint (dimensions and sampling flags).
/// - `eco`: Current ecology columns; the deme's segment feeds every rate.
/// - `deme`: Deme column the rates are read from.
/// - `female_counts`: Female counts per age/genotype (flat).
/// - `sperm`: Mutable sperm-storage slice.
/// - `mating_prob`: Precomputed female x male mating probabilities.
fn sample_mating(
    rng: &mut SessionRng,
    bp: &Blueprint,
    eco: &EcologyParams,
    deme: usize,
    female_counts: &[f64],
    sperm: &mut [f64],
    mating_prob: &[f64],
) {
    // For each adult female age/genotype:
    // 1. Compute virgins = females not already carrying sperm.
    // 2. Sample new virgin matings (binomial/continuous).
    // 3. Sample remating events that displace old sperm.
    // 4. Distribute new sperm using the mating probability row.
    let n_ages = bp.n_ages;
    let n_ztypes = bp.n_ztypes;
    let adult_start = adult_start_age(bp);
    let stochastic = bp.stochastic;
    let continuous = bp.continuous_sampling;
    let mating_rates = &eco.mating_rates[deme * 2 * n_ages..(deme + 1) * 2 * n_ages];
    let p_displace = clamp01(eco.sperm_displacement_rate[deme]);

    let mut tmp = vec![0.0; n_ztypes];
    for age in adult_start..n_ages {
        let p_mating = clamp01(mating_rates[age]);
        for gf in 0..n_ztypes {
            let n_female = female_counts[age * n_ztypes + gf];
            let mut mated_count = 0.0;
            for gm in 0..n_ztypes {
                mated_count += sperm[sperm_idx(age, gf, gm, n_ztypes)];
            }
            let virgins = (n_female - mated_count).max(0.0);

            let n_mating_virgins = if stochastic {
                if continuous {
                    continuous_binomial(rng, virgins, p_mating)
                } else {
                    binomial(rng, virgins.round() as i64, p_mating)
                }
            } else {
                virgins * p_mating
            };

            let p_remating = p_displace * p_mating;
            let n_remating = if stochastic {
                if mated_count > EPS && p_remating > EPS {
                    if continuous {
                        let removed_frac = p_remating.min(1.0);
                        for gm in 0..n_ztypes {
                            let idx = sperm_idx(age, gf, gm, n_ztypes);
                            sperm[idx] -= sperm[idx] * removed_frac;
                        }
                        mated_count * removed_frac
                    } else {
                        let mut total_removed = 0.0;
                        for gm in 0..n_ztypes {
                            let idx = sperm_idx(age, gf, gm, n_ztypes);
                            let count = sperm[idx];
                            if count > EPS {
                                let n_remove = binomial(rng, count.round() as i64, p_remating);
                                sperm[idx] = (sperm[idx] - n_remove).max(0.0);
                                total_removed += n_remove;
                            }
                        }
                        total_removed
                    }
                } else {
                    0.0
                }
            } else {
                let removed = mated_count * p_remating;
                if removed > EPS && mated_count > EPS {
                    let frac = (removed / mated_count).min(1.0);
                    for gm in 0..n_ztypes {
                        let idx = sperm_idx(age, gf, gm, n_ztypes);
                        sperm[idx] -= sperm[idx] * frac;
                    }
                }
                removed
            };

            let n_new = n_mating_virgins + n_remating;
            if n_new > EPS {
                if stochastic {
                    if continuous {
                        continuous_multinomial(
                            rng,
                            n_new,
                            &mating_prob[gf * n_ztypes..(gf + 1) * n_ztypes],
                            &mut tmp,
                        );
                        for gm in 0..n_ztypes {
                            sperm[sperm_idx(age, gf, gm, n_ztypes)] += tmp[gm];
                        }
                    } else {
                        let n_int = n_new.round() as i64;
                        if n_int > 0 {
                            multinomial(
                                rng,
                                n_int,
                                &mating_prob[gf * n_ztypes..(gf + 1) * n_ztypes],
                                &mut tmp,
                            );
                            for gm in 0..n_ztypes {
                                sperm[sperm_idx(age, gf, gm, n_ztypes)] += tmp[gm];
                            }
                        }
                    }
                } else {
                    for gm in 0..n_ztypes {
                        sperm[sperm_idx(age, gf, gm, n_ztypes)] +=
                            n_new * mating_prob[gf * n_ztypes + gm];
                    }
                }
            }
        }
    }
}

/// Turn stored sperm pairs into age-0 offspring counts.
///
/// Offspring production accounts for fecundity, female fertility, reproduction
/// rate, offspring genotype probabilities, and sex assignment.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `bp`: The frozen blueprint (dimensions and sampling flags).
/// - `eco`: Current ecology columns; the deme's segment feeds every rate.
/// - `genetics`: Shared genetics tables.
/// - `deme`: Deme column the rates are read from.
/// - `sperm`: Stored sperm slice.
/// - `n_f`: Output female age-0 counts per zygote type.
/// - `n_m`: Output male age-0 counts per zygote type.
fn fertilize(
    rng: &mut SessionRng,
    bp: &Blueprint,
    eco: &EcologyParams,
    genetics: &GeneticsTensors,
    deme: usize,
    sperm: &[f64],
    n_f: &mut [f64],
    n_m: &mut [f64],
) {
    // Convert stored sperm pairs into age-0 offspring:
    // - Egg count depends on female/male fecundity and female fertility.
    // - Stochastic reproduction uses binomial/poisson counts.
    // - Offspring genotypes are drawn from the offspring tensor.
    // - Sex is assigned according to sex ratio or sex-chromosome rules.
    let n_ages = bp.n_ages;
    let n_ztypes = bp.n_ztypes;
    let adult_start = adult_start_age(bp);
    let stochastic = bp.stochastic;
    let continuous = bp.continuous_sampling;
    let reproduction_rates = &eco.reproduction_rates[deme * n_ages..(deme + 1) * n_ages];
    let fertility = &eco.fertility[deme * n_ages..(deme + 1) * n_ages];
    let eggs_per_female = eco.eggs_per_female[deme].max(0.0);
    let mut offspring_acc = vec![0.0; n_ztypes];
    let mut prob_norm = vec![0.0; n_ztypes];
    let mut tmp = vec![0.0; n_ztypes];
    let mut has_any = false;

    for age in adult_start..n_ages {
        let p_reproduce = clamp01(reproduction_rates[age]);
        let fertility_factor = clamp01(fertility[age]);
        for gf in 0..n_ztypes {
            let ff = genetics.fecundity_fitness[gf];
            for gm in 0..n_ztypes {
                let n_pairs = sperm[sperm_idx(age, gf, gm, n_ztypes)];
                if n_pairs <= 0.0 {
                    continue;
                }
                has_any = true;
                let eggs_per_pair = eggs_per_female
                    * ff
                    * genetics.fecundity_fitness[n_ztypes + gm]
                    * fertility_factor;

                let n_total = if stochastic {
                    let n_pairs_eff = if continuous { n_pairs } else { n_pairs.round() };
                    if n_pairs_eff <= 0.0 {
                        continue;
                    }
                    let n_reproducing = if p_reproduce < 1.0 - EPS {
                        if continuous {
                            continuous_binomial(rng, n_pairs_eff, p_reproduce)
                        } else {
                            binomial(rng, n_pairs_eff as i64, p_reproduce)
                        }
                    } else {
                        n_pairs_eff
                    };
                    let total_lambda = n_reproducing * eggs_per_pair;
                    if bp.fixed_egg_count {
                        if continuous {
                            total_lambda
                        } else {
                            total_lambda.round()
                        }
                    } else if continuous {
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
                for go in 0..n_ztypes {
                    p_surv += genetics.offspring_tensor[(gf * n_ztypes + gm) * n_ztypes + go];
                }

                if stochastic {
                    if p_surv <= EPS {
                        continue;
                    }
                    let n_viable = if p_surv >= 1.0 - EPS {
                        n_total
                    } else if continuous {
                        continuous_binomial(rng, n_total, p_surv)
                    } else {
                        binomial(rng, n_total.round() as i64, p_surv)
                    };
                    if n_viable <= EPS {
                        continue;
                    }
                    let inv = 1.0 / p_surv;
                    for go in 0..n_ztypes {
                        prob_norm[go] =
                            genetics.offspring_tensor[(gf * n_ztypes + gm) * n_ztypes + go] * inv;
                    }
                    if continuous {
                        continuous_multinomial(rng, n_viable, &prob_norm, &mut tmp);
                        for go in 0..n_ztypes {
                            offspring_acc[go] += tmp[go];
                        }
                    } else {
                        multinomial(rng, n_viable.round() as i64, &prob_norm, &mut tmp);
                        for go in 0..n_ztypes {
                            offspring_acc[go] += tmp[go];
                        }
                    }
                } else {
                    for go in 0..n_ztypes {
                        offspring_acc[go] += n_total
                            * genetics.offspring_tensor[(gf * n_ztypes + gm) * n_ztypes + go];
                    }
                }
            }
        }
    }

    if !has_any {
        return;
    }
    let total: f64 = offspring_acc.iter().sum();
    if total <= EPS {
        return;
    }

    let sex_ratio = clamp01(eco.sex_ratio[deme]);
    for go in 0..n_ztypes {
        let n_g = offspring_acc[go];
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
            let n_fem = if stochastic {
                if continuous {
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

/// Run the age-structured reproduction stage in place.
///
/// The stage aggregates effective adult males, builds mating probabilities,
/// samples matings into stored sperm, fertilizes to age-0 offspring, and
/// applies zygote viability to newborns.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `bp`: The frozen blueprint (dimensions and sampling flags).
/// - `eco`: Current ecology columns; the deme's segment feeds every rate.
/// - `genetics`: Shared genetics tables.
/// - `deme`: Deme column the rates are read from.
/// - `ind`: Mutable individual-count flat slice.
/// - `sperm`: Mutable sperm-storage flat slice.
///
/// ## Returns
/// ``Ok(())`` on success, or a descriptive error string for invalid states.
pub fn reproduction(
    rng: &mut SessionRng,
    bp: &Blueprint,
    eco: &EcologyParams,
    genetics: &GeneticsTensors,
    deme: usize,
    ind: &mut [f64],
    sperm: &mut [f64],
) -> Result<(), String> {
    // Age-structured reproduction pipeline:
    // 1. Aggregate effective adult males (mating-rate weighted).
    // 2. Build female x male mating probabilities.
    // 3. Sample matings and update stored sperm.
    // 4. Fertilize stored sperm into age-0 offspring.
    // 5. Apply zygote viability to newborns.
    let n_ages = bp.n_ages;
    let n_ztypes = bp.n_ztypes;
    let mating_rates = &eco.mating_rates[deme * 2 * n_ages..(deme + 1) * 2 * n_ages];
    let mut effective_male_counts = vec![0.0; n_ztypes];
    for &age in &bp.adult_ages {
        let age = age as usize;
        if age < n_ages {
            let male_rate = mating_rates[n_ages + age];
            for ztype in 0..n_ztypes {
                effective_male_counts[ztype] +=
                    ind[ind_idx(1, age, ztype, n_ages, n_ztypes)] * male_rate;
            }
        }
    }
    if effective_male_counts.iter().sum::<f64>() == 0.0 {
        return Ok(());
    }

    let mut mating_prob = vec![0.0; n_ztypes * n_ztypes];
    compute_mating_probability_matrix(
        n_ztypes,
        &genetics.sexual_selection_fitness,
        &effective_male_counts,
        &mut mating_prob,
    );

    sample_mating(
        rng,
        bp,
        eco,
        deme,
        &ind[0..n_ages * n_ztypes],
        sperm,
        &mating_prob,
    );

    let mut n_female = vec![0.0; n_ztypes];
    let mut n_male = vec![0.0; n_ztypes];
    fertilize(
        rng,
        bp,
        eco,
        genetics,
        deme,
        sperm,
        &mut n_female,
        &mut n_male,
    );
    for ztype in 0..n_ztypes {
        ind[ind_idx(0, 0, ztype, n_ages, n_ztypes)] = n_female[ztype];
        ind[ind_idx(1, 0, ztype, n_ages, n_ztypes)] = n_male[ztype];
    }

    if bp.stochastic {
        for ztype in 0..n_ztypes {
            let f = ind[ind_idx(0, 0, ztype, n_ages, n_ztypes)];
            let m = ind[ind_idx(1, 0, ztype, n_ages, n_ztypes)];
            if bp.continuous_sampling {
                ind[ind_idx(0, 0, ztype, n_ages, n_ztypes)] = if f > 0.0 {
                    continuous_binomial(rng, f, genetics.zygote_viability_fitness[ztype])
                } else {
                    0.0
                };
                ind[ind_idx(1, 0, ztype, n_ages, n_ztypes)] = if m > 0.0 {
                    continuous_binomial(rng, m, genetics.zygote_viability_fitness[n_ztypes + ztype])
                } else {
                    0.0
                };
            } else {
                let n_f = f.round() as i64;
                ind[ind_idx(0, 0, ztype, n_ages, n_ztypes)] = if n_f > 0 {
                    binomial(rng, n_f, genetics.zygote_viability_fitness[ztype])
                } else {
                    0.0
                };
                let n_m = m.round() as i64;
                ind[ind_idx(1, 0, ztype, n_ages, n_ztypes)] = if n_m > 0 {
                    binomial(
                        rng,
                        n_m,
                        genetics.zygote_viability_fitness[n_ztypes + ztype],
                    )
                } else {
                    0.0
                };
            }
        }
    } else {
        for ztype in 0..n_ztypes {
            ind[ind_idx(0, 0, ztype, n_ages, n_ztypes)] *= genetics.zygote_viability_fitness[ztype];
            ind[ind_idx(1, 0, ztype, n_ages, n_ztypes)] *=
                genetics.zygote_viability_fitness[n_ztypes + ztype];
        }
    }
    Ok(())
}

/// Compute juvenile density-regulation scaling for the current tick.
///
/// Supported modes are no competition, fixed carrying capacity, logistic,
/// and Beverton-Holt.  The returned factor is multiplied into age-0 totals.
///
/// ## Parameters
/// - `bp`: The frozen blueprint (dimensions).
/// - `eco`: Current ecology columns; the deme's segment feeds every rate.
/// - `deme`: Deme column the rates are read from.
/// - `ind`: Current individual-count flat slice.
///
/// ## Returns
/// A non-negative scaling factor.
fn scaling_factor(bp: &Blueprint, eco: &EcologyParams, deme: usize, ind: &[f64]) -> f64 {
    // Juvenile density regulation by growth mode, dispatched through the
    // shared curve library with the Python reference operation order:
    // - NO_COMPETITION: 1.0 (no regulation).
    // - FIXED: cap total age-0 at carrying capacity.
    // - LINEAR / BEVERTON_HOLT / RICKER: curve at the competition ratio,
    //   scaled by the equilibrium survival rate.
    // Totals group by sex first (female row sum + male row sum), matching
    // the Python reference's `f_row.sum() + m_row.sum()` — an interleaved
    // accumulation rounds differently in the last ulp.
    let n_ages = bp.n_ages;
    let n_ztypes = bp.n_ztypes;
    let juvenile_growth_mode = eco.growth_mode[deme];
    if juvenile_growth_mode == NO_COMPETITION {
        return 1.0;
    }
    if juvenile_growth_mode == FIXED {
        let mut female_sum = 0.0;
        let mut male_sum = 0.0;
        for ztype in 0..n_ztypes {
            female_sum += ind[ind_idx(0, 0, ztype, n_ages, n_ztypes)];
            male_sum += ind[ind_idx(1, 0, ztype, n_ages, n_ztypes)];
        }
        let total_age_0 = female_sum + male_sum;
        return density_regulation::regulation_scaling(
            FIXED,
            total_age_0,
            eco.carrying_capacity[deme],
            0.0,
            0.0,
        )
        .unwrap_or(1.0);
    }
    // Compensatory family: blend juvenile counts below adulthood by the
    // per-age competition weights, then evaluate the curve.  The equilibrium
    // metrics are derived from the current deme column on demand (the same
    // computation the retired config assembly performed per tick).
    let new_adult_age = bp.new_adult_age;
    let competition_weights = &eco.competition_weights[deme * n_ages..(deme + 1) * n_ages];
    let mut juvenile_counts = vec![0.0; new_adult_age];
    for age in 0..new_adult_age {
        let mut female_sum = 0.0;
        let mut male_sum = 0.0;
        for ztype in 0..n_ztypes {
            female_sum += ind[ind_idx(0, age, ztype, n_ages, n_ztypes)];
            male_sum += ind[ind_idx(1, age, ztype, n_ages, n_ztypes)];
        }
        juvenile_counts[age] = female_sum + male_sum;
    }
    let mut actual_comp = 0.0;
    for age in 0..new_adult_age {
        actual_comp += juvenile_counts[age] * competition_weights[age];
    }
    let (expected_competition_strength, expected_survival_rate) =
        equilibrium_metrics(bp, eco, deme);
    density_regulation::regulation_scaling(
        juvenile_growth_mode,
        actual_comp,
        expected_competition_strength,
        eco.low_density_growth_rate[deme],
        expected_survival_rate,
    )
    .unwrap_or(1.0)
}

/// Apply the juvenile scaling factor by resampling age-0 counts.
///
/// Stochastic mode uses multinomial/continuous multinomial sampling;
/// deterministic mode scales each category proportionally.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `bp`: The frozen blueprint (dimensions and sampling flags).
/// - `ind`: Mutable individual-count flat slice.
/// - `scaling`: Scaling factor from [`scaling_factor`].
fn recruit_juveniles(rng: &mut SessionRng, bp: &Blueprint, ind: &mut [f64], scaling: f64) {
    // Resample age-0 counts so the total equals total * scaling.
    // Stochastic mode uses multinomial/continuous multinomial;
    // deterministic mode scales each category proportionally.
    let n_ages = bp.n_ages;
    let n_ztypes = bp.n_ztypes;
    let stochastic = bp.stochastic;
    let continuous = bp.continuous_sampling;

    let mut combined = Vec::with_capacity(2 * n_ztypes);
    // Python parity: the reference computes `total` as the *grouped* sum
    // `female_arr.sum() + male_arr.sum()` (used for `desired` and the
    // deterministic scaling), while the stochastic probability normalizer
    // divides by the sequential sum over the concatenation.  The two
    // round differently in the last ulp, so both accumulations are kept.
    let mut female_sum = 0.0;
    let mut male_sum = 0.0;
    for sex in 0..2 {
        for ztype in 0..n_ztypes {
            let mut value = ind[ind_idx(sex, 0, ztype, n_ages, n_ztypes)];
            if stochastic && !continuous {
                value = value.round();
            }
            combined.push(value);
            if sex == 0 {
                female_sum += value;
            } else {
                male_sum += value;
            }
        }
    }
    let total = female_sum + male_sum;
    let total_counts: f64 = combined.iter().sum();
    if total <= 0.0 {
        for sex in 0..2 {
            for ztype in 0..n_ztypes {
                ind[ind_idx(sex, 0, ztype, n_ages, n_ztypes)] = 0.0;
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
            for ztype in 0..n_ztypes {
                ind[ind_idx(sex, 0, ztype, n_ages, n_ztypes)] = 0.0;
            }
        }
        return;
    }

    let mut probs = vec![0.0; combined.len()];
    for (prob, &count) in probs.iter_mut().zip(combined.iter()) {
        *prob = count / total_counts;
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
        for ztype in 0..n_ztypes {
            ind[ind_idx(sex, 0, ztype, n_ages, n_ztypes)] = draws[sex * n_ztypes + ztype];
        }
    }
}

/// Stochastic survival that also scales stored sperm for surviving females.
///
/// Female survival must keep the relationship between counts and stored sperm
/// consistent: each sperm category is scaled independently and virgins survive
/// with the same probability.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `bp`: The frozen blueprint (dimensions and sampling flags).
/// - `ind`: Mutable individual-count flat slice.
/// - `sperm`: Mutable sperm-storage flat slice.
/// - `s_combined_f`: Combined female survival rates per age/genotype.
/// - `s_combined_m`: Combined male survival rates per age/genotype.
///
/// ## Returns
/// ``Ok(())`` or an error if the state is inconsistent.
fn sample_survival_with_sperm(
    rng: &mut SessionRng,
    bp: &Blueprint,
    ind: &mut [f64],
    sperm: &mut [f64],
    s_combined_f: &[f64],
    s_combined_m: &[f64],
) -> Result<(), String> {
    // Stochastic female survival must keep stored sperm consistent:
    // surviving sperm categories are scaled, and virgins survive independently.
    // Males are sampled with a simple binomial.
    let n_ages = bp.n_ages;
    let n_ztypes = bp.n_ztypes;
    let continuous = bp.continuous_sampling;
    for age in 0..n_ages {
        for g in 0..n_ztypes {
            let n_f_raw = ind[ind_idx(0, age, g, n_ages, n_ztypes)];
            let p_f = clamp01(s_combined_f[age * n_ztypes + g]);
            let mut total_sperm = 0.0;
            for gm in 0..n_ztypes {
                total_sperm += sperm[sperm_idx(age, g, gm, n_ztypes)];
            }
            let mut n_virgins_raw = n_f_raw - total_sperm;
            if n_virgins_raw < -EPS {
                return Err(format!(
                    "Invalid state: n_virgins < 0 in sample_survival_with_sperm_storage \
                     (age={age}, g={g}, n_f_raw={n_f_raw}, total_sperm={total_sperm})"
                ));
            }
            n_virgins_raw = n_virgins_raw.max(0.0);
            let n_virgins = if continuous {
                n_virgins_raw
            } else {
                n_virgins_raw.round()
            };

            let mut new_sperm_sum = 0.0;
            for gm in 0..n_ztypes {
                let idx = sperm_idx(age, g, gm, n_ztypes);
                let n_sperm = if continuous {
                    sperm[idx]
                } else {
                    sperm[idx].round()
                };
                sperm[idx] = if n_sperm > EPS {
                    if continuous {
                        continuous_binomial(rng, n_sperm, p_f)
                    } else {
                        binomial(rng, n_sperm as i64, p_f)
                    }
                } else {
                    0.0
                };
                new_sperm_sum += sperm[idx];
            }
            let surv_virgins = if n_virgins > EPS {
                if continuous {
                    continuous_binomial(rng, n_virgins, p_f)
                } else {
                    binomial(rng, n_virgins as i64, p_f)
                }
            } else {
                0.0
            };
            ind[ind_idx(0, age, g, n_ages, n_ztypes)] = new_sperm_sum + surv_virgins;

            let n_m = if continuous {
                ind[ind_idx(1, age, g, n_ages, n_ztypes)]
            } else {
                ind[ind_idx(1, age, g, n_ages, n_ztypes)].round()
            };
            let p_m = clamp01(s_combined_m[age * n_ztypes + g]);
            ind[ind_idx(1, age, g, n_ages, n_ztypes)] = if n_m > EPS {
                if continuous {
                    continuous_binomial(rng, n_m, p_m)
                } else {
                    binomial(rng, n_m as i64, p_m)
                }
            } else {
                0.0
            };
        }
    }
    Ok(())
}

/// Deterministic survival: multiply counts and stored sperm by survival rates.
///
/// ## Parameters
/// - `bp`: The frozen blueprint (dimensions).
/// - `ind`: Mutable individual-count flat slice.
/// - `sperm`: Mutable sperm-storage flat slice.
/// - `s_combined_f`: Combined female survival rates.
/// - `s_combined_m`: Combined male survival rates.
fn apply_survival_deterministic(
    bp: &Blueprint,
    ind: &mut [f64],
    sperm: &mut [f64],
    s_combined_f: &[f64],
    s_combined_m: &[f64],
) {
    // Deterministic survival multiplies every count and sperm category
    // by the combined age/viability survival probability.
    let n_ages = bp.n_ages;
    let n_ztypes = bp.n_ztypes;
    for age in 0..n_ages {
        for g in 0..n_ztypes {
            let f_rate = s_combined_f[age * n_ztypes + g];
            ind[ind_idx(0, age, g, n_ages, n_ztypes)] *= f_rate;
            for gm in 0..n_ztypes {
                sperm[sperm_idx(age, g, gm, n_ztypes)] *= f_rate;
            }
            ind[ind_idx(1, age, g, n_ages, n_ztypes)] *= s_combined_m[age * n_ztypes + g];
        }
    }
}

/// Run the age-structured survival stage in place.
///
/// The stage first applies juvenile density regulation, builds combined
/// age/viability survival rates, and then applies stochastic or deterministic
/// survival.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `bp`: The frozen blueprint (dimensions and sampling flags).
/// - `eco`: Current ecology columns; the deme's segment feeds every rate.
/// - `genetics`: Shared genetics tables.
/// - `deme`: Deme column the rates are read from.
/// - `ind`: Mutable individual-count flat slice.
/// - `sperm`: Mutable sperm-storage flat slice.
///
/// ## Returns
/// ``Ok(())`` on success, or an error string for invalid states.
pub fn survival(
    rng: &mut SessionRng,
    bp: &Blueprint,
    eco: &EcologyParams,
    genetics: &GeneticsTensors,
    deme: usize,
    ind: &mut [f64],
    sperm: &mut [f64],
) -> Result<(), String> {
    // Survival pipeline:
    // 1. Apply juvenile density regulation (scaling).
    // 2. Build combined age x viability survival rates.
    // 3. Apply stochastic or deterministic survival.
    let n_ages = bp.n_ages;
    let n_ztypes = bp.n_ztypes;
    let scaling = scaling_factor(bp, eco, deme, ind);
    recruit_juveniles(rng, bp, ind, scaling);

    let survival_rates = &eco.survival_rates[deme * 2 * n_ages..(deme + 1) * 2 * n_ages];
    let mut s_combined_f = vec![1.0; n_ages * n_ztypes];
    let mut s_combined_m = vec![1.0; n_ages * n_ztypes];
    let target_viability_age = bp.new_adult_age - 1;
    for age in 0..n_ages {
        let age_survival_f = survival_rates[age];
        let age_survival_m = survival_rates[n_ages + age];
        for ztype in 0..n_ztypes {
            let viability_f = if age == target_viability_age {
                genetics.viability_fitness[age * n_ztypes + ztype]
            } else {
                1.0
            };
            let viability_m = if age == target_viability_age {
                genetics.viability_fitness[(n_ages + age) * n_ztypes + ztype]
            } else {
                1.0
            };
            s_combined_f[age * n_ztypes + ztype] = age_survival_f * viability_f;
            s_combined_m[age * n_ztypes + ztype] = age_survival_m * viability_m;
        }
    }

    if bp.stochastic {
        sample_survival_with_sperm(rng, bp, ind, sperm, &s_combined_f, &s_combined_m)
    } else {
        apply_survival_deterministic(bp, ind, sperm, &s_combined_f, &s_combined_m);
        Ok(())
    }
}

/// Advance ages by one tick and clear the newborn age class.
///
/// Every age class shifts down one slot (oldest is dropped), then age 0 is
/// zeroed for both individual counts and stored sperm.
///
/// ## Parameters
/// - `bp`: The frozen blueprint (dimensions).
/// - `ind`: Mutable individual-count flat slice.
/// - `sperm`: Mutable sperm-storage flat slice.
pub fn aging(bp: &Blueprint, ind: &mut [f64], sperm: &mut [f64]) {
    // Shift every age class down by one, dropping the oldest class.
    // Then zero the newborn age class (age 0) for counts and sperm.
    let n_ages = bp.n_ages;
    let n_ztypes = bp.n_ztypes;
    for age in (1..n_ages).rev() {
        let older = age - 1;
        for sex in 0..2 {
            for ztype in 0..n_ztypes {
                ind[ind_idx(sex, age, ztype, n_ages, n_ztypes)] =
                    ind[ind_idx(sex, older, ztype, n_ages, n_ztypes)];
            }
        }
        for female_ztype in 0..n_ztypes {
            for male_ztype in 0..n_ztypes {
                sperm[sperm_idx(age, female_ztype, male_ztype, n_ztypes)] =
                    sperm[sperm_idx(older, female_ztype, male_ztype, n_ztypes)];
            }
        }
    }
    for sex in 0..2 {
        for ztype in 0..n_ztypes {
            ind[ind_idx(sex, 0, ztype, n_ages, n_ztypes)] = 0.0;
        }
    }
    for female_ztype in 0..n_ztypes {
        for male_ztype in 0..n_ztypes {
            sperm[sperm_idx(0, female_ztype, male_ztype, n_ztypes)] = 0.0;
        }
    }
}

/// Per-deme ``OP_SET_PARAM`` write-back context.
///
/// Owns a mutable borrow of the session's (or a spatial deme's local copy of
/// the) [`EcologyParams`] plus the immutable contracts the lifecycle stages
/// read.  After each event boundary the tick commits the ECO scratch into the
/// deme's ecology column, so the stage kernels — which read the columns
/// through this context — observe the writes on the **same tick's** later
/// lifecycle stages, matching the Python executor, where the early-hook write
/// is visible to survival within one tick.  The same holds for a candidate
/// ``updated_genetics`` committed by a Python callback.
pub struct EcoCtx<'a> {
    /// Blueprint providing dimensions and sampling flags.
    pub bp: &'a Blueprint,
    /// Session-owned ecology columns (written via ``set_eco_value``); for
    /// spatial ticks this is the deme's private single-deme local copy.
    pub params: &'a mut EcologyParams,
    /// Shared genetics tables for config assembly.
    pub genetics: &'a GeneticsTensors,
    /// Candidate genetics committed by Python callbacks at this event.
    pub updated_genetics: Option<GeneticsTensors>,
    /// Current within-tick stage, retained when a callback stops or fails.
    pub phase: usize,
    /// Deme column the writes target (0 for panmictic sessions and for
    /// the per-deme local copies of spatial ticks).
    pub deme: usize,
    /// Current simulation tick, stamped onto journal rows.
    pub tick: i64,
    /// Audited committed transitions ``(tick, param_id, old, new)`` — one
    /// row per event-boundary commit whose value actually changed.  The
    /// owning session drains this at batch end and hands it to the Python
    /// adapter so ``params_log`` stays complete on the Rust run path.
    pub journal: Vec<crate::hooks::interpreter::EcoJournalRow>,
}

impl EcoCtx<'_> {
    /// Commit the ECO scratch into the deme's ecology column.
    ///
    /// Every value passes the wire bounds table first: a non-finite or
    /// out-of-bounds RPN result (e.g. ``"K / 0"``) is rejected here with a
    /// message naming the parameter, tick, and value instead of silently
    /// entering the columns.  Actual value changes are journaled for the
    /// session audit trail.
    ///
    /// ## Parameters
    /// - `values`: Values written by the CSR interpreter.
    ///
    /// ## Returns
    /// ``Ok(())``, or an error string for an invalid value.
    pub fn commit(&mut self, values: &[f64]) -> Result<(), String> {
        for (id, value) in values.iter().enumerate() {
            if id >= crate::hooks::interpreter::N_ECO_PARAMS {
                break;
            }
            if let Err(reason) = crate::hooks::interpreter::validate_eco_param(id, *value) {
                return Err(format!(
                    "set_param value out of bounds: {reason} (tick {})",
                    self.tick
                ));
            }
            let old = self.params.eco_value(id, self.deme);
            if old != *value {
                self.journal.push((self.tick, id, old, *value, self.phase));
            }
            self.params.set_eco_value(id, self.deme, *value);
        }
        Ok(())
    }
}

/// Resolve the lifecycle-stage sources `(ecology, genetics, deme)`.
///
/// With a live [`EcoCtx`] the stages read the context's committed columns at
/// its deme (using a callback's candidate genetics when present); otherwise
/// the caller-provided session columns are used — the hook-free spatial path.
pub(crate) fn stage_sources<'a>(
    ctx: Option<&'a EcoCtx<'_>>,
    columns: Option<(&'a EcologyParams, &'a GeneticsTensors, usize)>,
) -> (&'a EcologyParams, &'a GeneticsTensors, usize) {
    match ctx {
        Some(ctx) => (
            &*ctx.params,
            ctx.updated_genetics.as_ref().unwrap_or(ctx.genetics),
            ctx.deme,
        ),
        None => columns.expect("hook-free spatial ticks carry their ecology columns"),
    }
}

/// Run one full age-structured tick with hooks in the reference stage order.
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
/// - `ind`: Mutable individual-count flat slice.
/// - `sperm`: Mutable sperm-storage flat slice.
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
    sperm: &mut [f64],
    tick: i64,
    deme_id: i64,
    eco_values: &mut [f64],
    eco_ctx: &mut Option<EcoCtx<'_>>,
    columns: Option<(&EcologyParams, &GeneticsTensors, usize)>,
) -> Result<i32, String> {
    // One structured tick follows the Python reference order:
    // first hook -> reproduction -> early hook -> survival -> late hook -> aging.
    // Each event executes its CSR plan slots and Python callback slots in
    // one cross-type priority order; a nonzero result stops the run.  With
    // an EcoCtx, set_param writes are committed at each boundary so the
    // stage kernels observe them through the live columns.  The ctx tick is
    // re-stamped here so batch loops journal every tick under its own tick
    // value (the ctx outlives one batch, not one tick).
    if let Some(ctx) = eco_ctx.as_mut() {
        ctx.tick = tick;
        ctx.phase = 0;
    }

    let mut result = hooks.execute_event(
        rng,
        0,
        ind,
        sperm,
        2,
        bp.n_ages,
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
    let (eco, genetics, deme) = stage_sources(eco_ctx.as_ref(), columns);
    reproduction(rng, bp, eco, genetics, deme, ind, sperm)?;

    if let Some(ctx) = eco_ctx.as_mut() {
        ctx.phase = 2;
    }
    result = hooks.execute_event(
        rng,
        1,
        ind,
        sperm,
        2,
        bp.n_ages,
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
    let (eco, genetics, deme) = stage_sources(eco_ctx.as_ref(), columns);
    survival(rng, bp, eco, genetics, deme, ind, sperm)?;

    if let Some(ctx) = eco_ctx.as_mut() {
        ctx.phase = 4;
    }
    result = hooks.execute_event(
        rng,
        2,
        ind,
        sperm,
        2,
        bp.n_ages,
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
    aging(bp, ind, sperm);
    Ok(0)
}

/// One full in-memory checkpoint captured at a record-aligned tick.
///
/// The S2 record-point store: alongside every raw history
/// row the session keeps a complete save — state arrays, the RNG words
/// (continuation, not a reseed), and the ecology section — so the public
/// ``restore_checkpoint`` rolls back everything, not just counts.
#[derive(Clone)]
pub struct TickCheckpoint {
    /// Lifecycle status and cursor distinguish partial snapshots.
    pub execution: crate::sessions::status::ExecutionStatus,
    pub phase: usize,
    /// Tick the checkpoint was captured at.
    pub tick: i64,
    /// Flattened individual counts at capture time.
    pub ind: Vec<f64>,
    /// Flattened sperm storage at capture time (empty for discrete).
    pub sperm: Vec<f64>,
    /// Four Xoshiro256++ state words at capture time.
    pub rng_words: [u64; 4],
    /// Ecology scalars in ``ECOLOGY_SCALARS`` wire order.
    pub eco_scalars: Vec<f64>,
    /// Ecology vectors in ``ECOLOGY_VECTORS`` order.
    pub eco_vectors: Vec<Vec<f64>>,
    /// User-defined ecology belongs to the same atomic checkpoint.
    pub custom_slots: std::collections::HashMap<String, crate::model::custom_fields::CustomSlot>,
}

/// Capture one checkpoint into *store* from the current batch state.
///
/// Ecology comes from the borrowed EcoCtx params (the session always lends
/// one on the batch path); without an EcoCtx the ecology section is left
/// empty and only state + RNG are captured.
pub(crate) fn capture_checkpoint(
    rng: &SessionRng,
    ind: &[f64],
    sperm: &[f64],
    tick: i64,
    eco_ctx: &Option<EcoCtx<'_>>,
    store: &mut Vec<TickCheckpoint>,
) -> Result<(), String> {
    if store.iter().any(|checkpoint| checkpoint.tick == tick) {
        return Ok(());
    }
    let (eco_scalars, eco_vectors) = match eco_ctx
        .as_ref()
        .map(|ctx| ctx.params.ecology_snapshot_words())
    {
        Some(Ok(words)) => words,
        Some(Err(err)) => return Err(err.to_string()),
        None => (Vec::new(), Vec::new()),
    };
    store.push(TickCheckpoint {
        execution: crate::sessions::status::ExecutionStatus::Ready,
        phase: 0,
        tick,
        ind: ind.to_vec(),
        sperm: sperm.to_vec(),
        rng_words: rng.state_words(),
        eco_scalars,
        eco_vectors,
        custom_slots: eco_ctx
            .as_ref()
            .map(|ctx| ctx.params.custom_slots[ctx.deme].clone())
            .unwrap_or_default(),
    });
    Ok(())
}
