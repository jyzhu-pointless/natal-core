// Numeric loops deliberately mirror the Python reference index-by-index;
// iterator rewrites would obscure parity review.
#![allow(clippy::needless_range_loop)]
// Batch signature mirrors the Numba kernel tuple layout.
#![allow(clippy::too_many_arguments)]
//! Age-structured lifecycle kernels: reproduction, survival, aging, and the
//! unified tick orchestration.
//!
//! The stage order mirrors ``natal.engine.lifecycle.run_structured_tick``:
//! first hook → reproduction → early hook → survival → late hook → aging.

use rand::rngs::SmallRng;

use crate::config::SimConfig;
use crate::hooks::HookProgram;
use crate::rng::{
    binomial, clamp01, continuous_binomial, continuous_multinomial, continuous_poisson,
    multinomial, poisson, EPS,
};

/// Juvenile growth mode: no density regulation.
const NO_COMPETITION: i64 = 0;
/// Juvenile growth mode: fixed carrying-capacity ceiling.
const FIXED: i64 = 1;
/// Juvenile growth mode: logistic density regulation.
const LOGISTIC: i64 = 2;
/// Juvenile growth mode: Beverton-Holt density regulation.
const BEVERTON_HOLT: i64 = 3;

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

/// Normalize male mating weights into a female x male probability matrix.
///
/// Each female row is proportional to ``sexual_selection_fitness * male_count``.
/// Rows with zero or non-finite totals become all-zero so no matings are produced.
///
/// ## Parameters
/// - `cfg`: Simulation config.
/// - `male_counts`: Effective adult male counts per zygote type.
/// - `out`: Output matrix of shape ``(n_ztypes, n_ztypes)``, overwritten.
fn compute_mating_probability_matrix(cfg: &SimConfig, male_counts: &[f64], out: &mut [f64]) {
    // Each female row is proportional to sexual_selection_fitness * male_count.
    // Rows are normalized; zero/non-finite rows become all-zero so no matings occur.
    let n_ztypes = cfg.n_ztypes;
    for gf in 0..n_ztypes {
        let mut row_sum = 0.0;
        for gm in 0..n_ztypes {
            let value = cfg.sexual_selection_fitness[gf * n_ztypes + gm] * male_counts[gm];
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
/// - `cfg`: Simulation config.
/// - `female_counts`: Female counts per age/genotype (flat).
/// - `sperm`: Mutable sperm-storage slice.
/// - `mating_prob`: Precomputed female x male mating probabilities.
fn sample_mating(
    rng: &mut SmallRng,
    cfg: &SimConfig,
    female_counts: &[f64],
    sperm: &mut [f64],
    mating_prob: &[f64],
) {
    // For each adult female age/genotype:
    // 1. Compute virgins = females not already carrying sperm.
    // 2. Sample new virgin matings (binomial/continuous).
    // 3. Sample remating events that displace old sperm.
    // 4. Distribute new sperm using the mating probability row.
    let n_ages = cfg.n_ages;
    let n_ztypes = cfg.n_ztypes;
    let adult_start = cfg.adult_start_age;
    let stochastic = cfg.stochastic;
    let continuous = cfg.continuous_sampling;

    let mut tmp = vec![0.0; n_ztypes];
    for age in adult_start..n_ages {
        let p_mating = clamp01(cfg.age_based_mating_rates[age]);
        let p_displace = clamp01(cfg.sperm_displacement_rate);
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
/// - `cfg`: Simulation config.
/// - `sperm`: Stored sperm slice.
/// - `n_f`: Output female age-0 counts per zygote type.
/// - `n_m`: Output male age-0 counts per zygote type.
fn fertilize(rng: &mut SmallRng, cfg: &SimConfig, sperm: &[f64], n_f: &mut [f64], n_m: &mut [f64]) {
    // Convert stored sperm pairs into age-0 offspring:
    // - Egg count depends on female/male fecundity and female fertility.
    // - Stochastic reproduction uses binomial/poisson counts.
    // - Offspring genotypes are drawn from the offspring tensor.
    // - Sex is assigned according to sex ratio or sex-chromosome rules.
    let n_ages = cfg.n_ages;
    let n_ztypes = cfg.n_ztypes;
    let adult_start = cfg.adult_start_age;
    let stochastic = cfg.stochastic;
    let continuous = cfg.continuous_sampling;
    let mut offspring_acc = vec![0.0; n_ztypes];
    let mut prob_norm = vec![0.0; n_ztypes];
    let mut tmp = vec![0.0; n_ztypes];
    let mut has_any = false;

    for age in adult_start..n_ages {
        let p_reproduce = clamp01(cfg.age_based_reproduction_rates[age]);
        let fertility_factor = clamp01(cfg.female_age_based_fertility[age]);
        for gf in 0..n_ztypes {
            let ff = cfg.fecundity_fitness[gf];
            for gm in 0..n_ztypes {
                let n_pairs = sperm[sperm_idx(age, gf, gm, n_ztypes)];
                if n_pairs <= 0.0 {
                    continue;
                }
                has_any = true;
                let eggs_per_pair = cfg.eggs_per_female
                    * ff
                    * cfg.fecundity_fitness[n_ztypes + gm]
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
                    if cfg.fixed_egg_count {
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
                    p_surv += cfg.offspring_tensor[(gf * n_ztypes + gm) * n_ztypes + go];
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
                            cfg.offspring_tensor[(gf * n_ztypes + gm) * n_ztypes + go] * inv;
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
                        offspring_acc[go] +=
                            n_total * cfg.offspring_tensor[(gf * n_ztypes + gm) * n_ztypes + go];
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

    let sex_ratio = clamp01(cfg.sex_ratio);
    for go in 0..n_ztypes {
        let n_g = offspring_acc[go];
        if n_g <= EPS {
            continue;
        }
        if cfg.has_sex_chromosomes && cfg.female_only_by_sex_chrom[go] {
            n_f[go] = n_g;
        } else if cfg.has_sex_chromosomes && cfg.male_only_by_sex_chrom[go] {
            n_m[go] = n_g;
        } else {
            let p_f = if cfg.has_sex_chromosomes {
                let denom = cfg.female_ztype_compatibility[go] + cfg.male_ztype_compatibility[go];
                if denom > EPS {
                    clamp01(cfg.female_ztype_compatibility[go] / denom)
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
/// - `cfg`: Simulation config.
/// - `ind`: Mutable individual-count flat slice.
/// - `sperm`: Mutable sperm-storage flat slice.
///
/// ## Returns
/// ``Ok(())`` on success, or a descriptive error string for invalid states.
pub fn reproduction(
    rng: &mut SmallRng,
    cfg: &SimConfig,
    ind: &mut [f64],
    sperm: &mut [f64],
) -> Result<(), String> {
    // Age-structured reproduction pipeline:
    // 1. Aggregate effective adult males (mating-rate weighted).
    // 2. Build female x male mating probabilities.
    // 3. Sample matings and update stored sperm.
    // 4. Fertilize stored sperm into age-0 offspring.
    // 5. Apply zygote viability to newborns.
    let n_ages = cfg.n_ages;
    let n_ztypes = cfg.n_ztypes;
    let mut effective_male_counts = vec![0.0; n_ztypes];
    for &age in &cfg.adult_ages {
        if age < n_ages {
            let male_rate = cfg.age_based_mating_rates[n_ages + age];
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
    compute_mating_probability_matrix(cfg, &effective_male_counts, &mut mating_prob);

    sample_mating(rng, cfg, &ind[0..n_ages * n_ztypes], sperm, &mating_prob);

    let mut n_female = vec![0.0; n_ztypes];
    let mut n_male = vec![0.0; n_ztypes];
    fertilize(rng, cfg, sperm, &mut n_female, &mut n_male);
    for ztype in 0..n_ztypes {
        ind[ind_idx(0, 0, ztype, n_ages, n_ztypes)] = n_female[ztype];
        ind[ind_idx(1, 0, ztype, n_ages, n_ztypes)] = n_male[ztype];
    }

    if cfg.stochastic {
        for ztype in 0..n_ztypes {
            let f = ind[ind_idx(0, 0, ztype, n_ages, n_ztypes)];
            let m = ind[ind_idx(1, 0, ztype, n_ages, n_ztypes)];
            if cfg.continuous_sampling {
                ind[ind_idx(0, 0, ztype, n_ages, n_ztypes)] = if f > 0.0 {
                    continuous_binomial(rng, f, cfg.zygote_viability_fitness[ztype])
                } else {
                    0.0
                };
                ind[ind_idx(1, 0, ztype, n_ages, n_ztypes)] = if m > 0.0 {
                    continuous_binomial(rng, m, cfg.zygote_viability_fitness[n_ztypes + ztype])
                } else {
                    0.0
                };
            } else {
                let n_f = f.round() as i64;
                ind[ind_idx(0, 0, ztype, n_ages, n_ztypes)] = if n_f > 0 {
                    binomial(rng, n_f, cfg.zygote_viability_fitness[ztype])
                } else {
                    0.0
                };
                let n_m = m.round() as i64;
                ind[ind_idx(1, 0, ztype, n_ages, n_ztypes)] = if n_m > 0 {
                    binomial(rng, n_m, cfg.zygote_viability_fitness[n_ztypes + ztype])
                } else {
                    0.0
                };
            }
        }
    } else {
        for ztype in 0..n_ztypes {
            ind[ind_idx(0, 0, ztype, n_ages, n_ztypes)] *= cfg.zygote_viability_fitness[ztype];
            ind[ind_idx(1, 0, ztype, n_ages, n_ztypes)] *=
                cfg.zygote_viability_fitness[n_ztypes + ztype];
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
/// - `cfg`: Simulation config.
/// - `ind`: Current individual-count flat slice.
///
/// ## Returns
/// A non-negative scaling factor.
fn scaling_factor(cfg: &SimConfig, ind: &[f64]) -> f64 {
    // Juvenile density regulation by growth mode:
    // - NO_COMPETITION: 1.0 (no regulation).
    // - FIXED: cap total age-0 at carrying capacity.
    // - LOGISTIC / BEVERTON_HOLT: use competition ratio and growth rate.
    let n_ages = cfg.n_ages;
    let n_ztypes = cfg.n_ztypes;
    let mut total_age_0 = 0.0;
    for ztype in 0..n_ztypes {
        total_age_0 += ind[ind_idx(0, 0, ztype, n_ages, n_ztypes)];
        total_age_0 += ind[ind_idx(1, 0, ztype, n_ages, n_ztypes)];
    }

    match cfg.juvenile_growth_mode {
        NO_COMPETITION => 1.0,
        FIXED => {
            if total_age_0 > 0.0 {
                (cfg.carrying_capacity / total_age_0).min(1.0)
            } else {
                1.0
            }
        }
        LOGISTIC | BEVERTON_HOLT => {
            let mut juvenile_counts = vec![0.0; cfg.new_adult_age];
            for age in 0..cfg.new_adult_age {
                for ztype in 0..n_ztypes {
                    juvenile_counts[age] += ind[ind_idx(0, age, ztype, n_ages, n_ztypes)];
                    juvenile_counts[age] += ind[ind_idx(1, age, ztype, n_ages, n_ztypes)];
                }
            }
            let mut actual_comp = 0.0;
            for age in 0..cfg.new_adult_age {
                actual_comp +=
                    juvenile_counts[age] * cfg.age_based_relative_competition_strength[age];
            }
            let competition_ratio = if cfg.expected_competition_strength > 0.0 {
                actual_comp / cfg.expected_competition_strength
            } else {
                1.0
            };
            let r = cfg.low_density_growth_rate;
            let actual_growth_rate = if cfg.juvenile_growth_mode == LOGISTIC {
                (-competition_ratio * (r - 1.0) + r).max(0.0)
            } else {
                r / (competition_ratio * (r - 1.0) + 1.0)
            };
            actual_growth_rate * cfg.expected_survival_rate
        }
        _ => 1.0,
    }
}

/// Apply the juvenile scaling factor by resampling age-0 counts.
///
/// Stochastic mode uses multinomial/continuous multinomial sampling;
/// deterministic mode scales each category proportionally.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `cfg`: Simulation config.
/// - `ind`: Mutable individual-count flat slice.
/// - `scaling`: Scaling factor from [`scaling_factor`].
fn recruit_juveniles(rng: &mut SmallRng, cfg: &SimConfig, ind: &mut [f64], scaling: f64) {
    // Resample age-0 counts so the total equals total * scaling.
    // Stochastic mode uses multinomial/continuous multinomial;
    // deterministic mode scales each category proportionally.
    let n_ages = cfg.n_ages;
    let n_ztypes = cfg.n_ztypes;
    let stochastic = cfg.stochastic;
    let continuous = cfg.continuous_sampling;

    let mut combined = Vec::with_capacity(2 * n_ztypes);
    let mut total = 0.0;
    for sex in 0..2 {
        for ztype in 0..n_ztypes {
            let mut value = ind[ind_idx(sex, 0, ztype, n_ages, n_ztypes)];
            if stochastic && !continuous {
                value = value.round();
            }
            combined.push(value);
            total += value;
        }
    }
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
/// - `cfg`: Simulation config.
/// - `ind`: Mutable individual-count flat slice.
/// - `sperm`: Mutable sperm-storage flat slice.
/// - `s_combined_f`: Combined female survival rates per age/genotype.
/// - `s_combined_m`: Combined male survival rates per age/genotype.
///
/// ## Returns
/// ``Ok(())`` or an error if the state is inconsistent.
fn sample_survival_with_sperm(
    rng: &mut SmallRng,
    cfg: &SimConfig,
    ind: &mut [f64],
    sperm: &mut [f64],
    s_combined_f: &[f64],
    s_combined_m: &[f64],
) -> Result<(), String> {
    // Stochastic female survival must keep stored sperm consistent:
    // surviving sperm categories are scaled, and virgins survive independently.
    // Males are sampled with a simple binomial.
    let n_ages = cfg.n_ages;
    let n_ztypes = cfg.n_ztypes;
    let continuous = cfg.continuous_sampling;
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
/// - `cfg`: Simulation config.
/// - `ind`: Mutable individual-count flat slice.
/// - `sperm`: Mutable sperm-storage flat slice.
/// - `s_combined_f`: Combined female survival rates.
/// - `s_combined_m`: Combined male survival rates.
fn apply_survival_deterministic(
    cfg: &SimConfig,
    ind: &mut [f64],
    sperm: &mut [f64],
    s_combined_f: &[f64],
    s_combined_m: &[f64],
) {
    // Deterministic survival multiplies every count and sperm category
    // by the combined age/viability survival probability.
    let n_ages = cfg.n_ages;
    let n_ztypes = cfg.n_ztypes;
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
/// - `cfg`: Simulation config.
/// - `ind`: Mutable individual-count flat slice.
/// - `sperm`: Mutable sperm-storage flat slice.
///
/// ## Returns
/// ``Ok(())`` on success, or an error string for invalid states.
pub fn survival(
    rng: &mut SmallRng,
    cfg: &SimConfig,
    ind: &mut [f64],
    sperm: &mut [f64],
) -> Result<(), String> {
    // Survival pipeline:
    // 1. Apply juvenile density regulation (scaling).
    // 2. Build combined age x viability survival rates.
    // 3. Apply stochastic or deterministic survival.
    let n_ages = cfg.n_ages;
    let n_ztypes = cfg.n_ztypes;
    let scaling = scaling_factor(cfg, ind);
    recruit_juveniles(rng, cfg, ind, scaling);

    let mut s_combined_f = vec![1.0; n_ages * n_ztypes];
    let mut s_combined_m = vec![1.0; n_ages * n_ztypes];
    let target_viability_age = cfg.new_adult_age - 1;
    for age in 0..n_ages {
        let age_survival_f = cfg.age_based_survival_rates[age];
        let age_survival_m = cfg.age_based_survival_rates[n_ages + age];
        for ztype in 0..n_ztypes {
            let viability_f = if age == target_viability_age {
                cfg.viability_fitness[age * n_ztypes + ztype]
            } else {
                1.0
            };
            let viability_m = if age == target_viability_age {
                cfg.viability_fitness[(n_ages + age) * n_ztypes + ztype]
            } else {
                1.0
            };
            s_combined_f[age * n_ztypes + ztype] = age_survival_f * viability_f;
            s_combined_m[age * n_ztypes + ztype] = age_survival_m * viability_m;
        }
    }

    if cfg.stochastic {
        sample_survival_with_sperm(rng, cfg, ind, sperm, &s_combined_f, &s_combined_m)
    } else {
        apply_survival_deterministic(cfg, ind, sperm, &s_combined_f, &s_combined_m);
        Ok(())
    }
}

/// Advance ages by one tick and clear the newborn age class.
///
/// Every age class shifts down one slot (oldest is dropped), then age 0 is
/// zeroed for both individual counts and stored sperm.
///
/// ## Parameters
/// - `cfg`: Simulation config.
/// - `ind`: Mutable individual-count flat slice.
/// - `sperm`: Mutable sperm-storage flat slice.
pub fn aging(cfg: &SimConfig, ind: &mut [f64], sperm: &mut [f64]) {
    // Shift every age class down by one, dropping the oldest class.
    // Then zero the newborn age class (age 0) for counts and sperm.
    let n_ages = cfg.n_ages;
    let n_ztypes = cfg.n_ztypes;
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

/// Run one full age-structured tick with hooks in the reference stage order.
///
/// Stage order: first hook -> reproduction -> early hook -> survival -> late
/// hook -> aging.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `cfg`: Simulation config.
/// - `hooks`: CSR hook program.
/// - `ind`: Mutable individual-count flat slice.
/// - `sperm`: Mutable sperm-storage flat slice.
/// - `tick`: Current tick.
/// - `deme_id`: Current deme id.
///
/// ## Returns
/// ``Ok(0)`` for continue, ``Ok(1)`` if a hook requested stop, or an error string.
pub fn run_tick(
    rng: &mut SmallRng,
    cfg: &SimConfig,
    hooks: &HookProgram,
    ind: &mut [f64],
    sperm: &mut [f64],
    tick: i64,
    deme_id: i64,
) -> Result<i32, String> {
    // One structured tick follows the Python reference order:
    // first hook -> reproduction -> early hook -> survival -> late hook -> aging.
    let mut result = hooks.execute_event(
        rng,
        0,
        ind,
        sperm,
        2,
        cfg.n_ages,
        cfg.n_ztypes,
        tick,
        cfg.stochastic,
        cfg.continuous_sampling,
        deme_id,
    );
    if result != 0 {
        return Ok(result);
    }

    reproduction(rng, cfg, ind, sperm)?;

    result = hooks.execute_event(
        rng,
        1,
        ind,
        sperm,
        2,
        cfg.n_ages,
        cfg.n_ztypes,
        tick,
        cfg.stochastic,
        cfg.continuous_sampling,
        deme_id,
    );
    if result != 0 {
        return Ok(result);
    }

    survival(rng, cfg, ind, sperm)?;

    result = hooks.execute_event(
        rng,
        2,
        ind,
        sperm,
        2,
        cfg.n_ages,
        cfg.n_ztypes,
        tick,
        cfg.stochastic,
        cfg.continuous_sampling,
        deme_id,
    );
    if result != 0 {
        return Ok(result);
    }

    aging(cfg, ind, sperm);
    Ok(0)
}

/// Run up to ``n_ticks`` complete ticks inside Rust and optionally record flattened history rows.
///
/// Recording mirrors the Numba ``_run_loop_structured`` layout:
/// ``[tick, individual_count..., sperm_storage...]`` in raw mode, or
/// ``[tick, observed...]`` where each observation group is summed over the
/// ztype axis before flattening.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `cfg`: Simulation config.
/// - `hooks`: CSR hook program.
/// - `ind`: Mutable individual-count flat slice.
/// - `sperm`: Mutable sperm-storage flat slice.
/// - `tick`: Starting tick.
/// - `n_ticks`: Number of ticks to run.
/// - `record_interval`: Record every N ticks; 0 disables recording.
/// - `observation_mask`: Optional observation mask.
///
/// ## Returns
/// ``(final_tick, flat_history, n_rows, was_stopped)``.
pub fn run_batch(
    rng: &mut SmallRng,
    cfg: &SimConfig,
    hooks: &HookProgram,
    ind: &mut [f64],
    sperm: &mut [f64],
    tick: i64,
    n_ticks: i64,
    record_interval: i64,
    observation_mask: Option<&[f64]>,
) -> Result<(i64, Vec<f64>, usize, bool), String> {
    // Loop n_ticks entirely in Rust.
    // When recording is enabled, append a history row at the requested interval.
    // Stops requested by hooks terminate the loop early.
    let n_sexes = 2;
    let n_ages = cfg.n_ages;
    let n_ztypes = cfg.n_ztypes;
    let n_obs_values = n_sexes * n_ages * n_ztypes;
    let observation_groups = match observation_mask {
        Some(mask) => {
            if n_obs_values == 0 || mask.len() % n_obs_values != 0 {
                return Err(format!(
                    "observation_mask length {} is not a multiple of {}",
                    mask.len(),
                    n_obs_values
                ));
            }
            mask.len() / n_obs_values
        }
        None => 0,
    };

    let mut current_tick = tick;
    let mut history: Vec<f64> = Vec::new();
    let mut n_rows: usize = 0;

    let record_row = |history: &mut Vec<f64>,
                      ind: &[f64],
                      sperm: &[f64],
                      observation_mask: Option<&[f64]>,
                      observation_groups: usize,
                      current_tick: i64| {
        history.push(current_tick as f64);
        match observation_mask {
            Some(mask) => {
                for group in 0..observation_groups {
                    for sex in 0..n_sexes {
                        for age in 0..n_ages {
                            let mut total = 0.0;
                            for ztype in 0..n_ztypes {
                                let state_idx = (sex * n_ages + age) * n_ztypes + ztype;
                                let mask_idx =
                                    ((group * n_sexes + sex) * n_ages + age) * n_ztypes + ztype;
                                total += ind[state_idx] * mask[mask_idx];
                            }
                            history.push(total);
                        }
                    }
                }
            }
            None => {
                history.extend_from_slice(ind);
                history.extend_from_slice(sperm);
            }
        }
    };

    if record_interval > 0 && current_tick % record_interval == 0 {
        record_row(
            &mut history,
            ind,
            sperm,
            observation_mask,
            observation_groups,
            current_tick,
        );
        n_rows += 1;
    }

    for _ in 0..n_ticks {
        let result = run_tick(rng, cfg, hooks, ind, sperm, current_tick, -1)?;
        if result != 0 {
            return Ok((current_tick, history, n_rows, true));
        }
        current_tick += 1;
        if record_interval > 0 && current_tick % record_interval == 0 {
            record_row(
                &mut history,
                ind,
                sperm,
                observation_mask,
                observation_groups,
                current_tick,
            );
            n_rows += 1;
        }
    }

    Ok((current_tick, history, n_rows, false))
}
