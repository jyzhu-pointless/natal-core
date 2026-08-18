//! Discrete-generation and Wright-Fisher lifecycle kernels.

#![allow(clippy::needless_range_loop)] // Index loops mirror the Python reference for parity review.
#![allow(clippy::too_many_arguments)] // Batch signature mirrors the Numba kernel tuple layout.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::SmallRng;

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
/// Wright-Fisher mode: multinomial offspring sampling.
const WF_MULTINOMIAL: i64 = 1;
/// Wright-Fisher mode: Poisson offspring sampling.
const WF_POISSON: i64 = 2;
/// Wright-Fisher mode: deterministic expected offspring counts.
const WF_DETERMINISTIC: i64 = 3;

/// Plain Rust snapshot of the discrete-generation ``DiscretePopulationConfig``.
///
/// The Python config is copied once into plain Rust fields so discrete and
/// Wright-Fisher kernels never touch Python objects.
///
/// ## Notes
/// - Discrete simulations always use two sexes and two ages.
/// - Wright-Fisher fields such as ``extreme_speed_mode`` are stored here.
#[derive(Clone)]
pub struct DiscreteConfig {
    // --- Dimensions and sampling flags ---
    pub n_ztypes: usize,
    pub stochastic: bool,
    pub continuous_sampling: bool,

    // --- Mating / reproduction scalars ---
    pub female_adult_mating_rate: f64,
    pub male_adult_mating_rate: f64,
    pub reproduction_rate: f64,
    pub eggs_per_female: f64,
    pub sex_ratio: f64,
    pub female_age0_survival: f64,
    pub male_age0_survival: f64,

    // --- Mode switches ---
    pub has_sex_chromosomes: bool,
    pub extreme_speed_mode: i64,
    pub juvenile_growth_mode: i64,

    // --- Density regulation ---
    pub carrying_capacity: f64,
    pub expected_competition_strength: f64,
    pub expected_survival_rate: f64,
    pub low_density_growth_rate: f64,

    // --- Fitness / inheritance arrays ---
    pub sexual_selection_fitness: Vec<f64>,
    pub offspring_tensor: Vec<f64>,
    pub fecundity_f: Vec<f64>,
    pub fecundity_m: Vec<f64>,
    pub viability_f: Vec<f64>,
    pub viability_m: Vec<f64>,
    pub female_ztype_compatibility: Vec<f64>,
    pub male_ztype_compatibility: Vec<f64>,
    pub female_only_by_sex_chrom: Vec<bool>,
    pub male_only_by_sex_chrom: Vec<bool>,
}

/// Build a ``PyValueError`` for an array shape mismatch.
///
/// ## Parameters
/// - `name`: Attribute name.
/// - `expected`: Expected shape description.
/// - `got`: Actual shape.
///
/// ## Returns
/// A ``PyValueError`` with a descriptive message.
fn shape_error(name: &str, expected: &str, got: &[usize]) -> PyErr {
    PyValueError::new_err(format!("{name} must have shape {expected}, got {got:?}"))
}

/// Extract a float64 config scalar, accepting 0-d NumPy arrays.
///
/// ## Parameters
/// - `obj`: Python config object.
/// - `name`: Attribute name.
///
/// ## Returns
/// The scalar as ``f64``.
fn extract_f64(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<f64> {
    let value = obj.getattr(name)?;
    if let Ok(scalar) = value.extract::<f64>() {
        return Ok(scalar);
    }
    value.call_method0("item")?.extract::<f64>()
}

/// Extract an int64 config scalar, accepting 0-d NumPy arrays.
///
/// ## Parameters
/// - `obj`: Python config object.
/// - `name`: Attribute name.
///
/// ## Returns
/// The scalar as ``i64``.
fn extract_i64(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<i64> {
    let value = obj.getattr(name)?;
    if let Ok(scalar) = value.extract::<i64>() {
        return Ok(scalar);
    }
    value.call_method0("item")?.extract::<i64>()
}

/// Extract a 1-D float64 config array with an exact length check.
///
/// ## Parameters
/// - `obj`: Python config object.
/// - `name`: Attribute name.
/// - `expected`: Required length.
///
/// ## Returns
/// A ``Vec<f64>`` copy.
fn extract_f64_1d(obj: &Bound<'_, PyAny>, name: &str, expected: usize) -> PyResult<Vec<f64>> {
    let array = obj.getattr(name)?.extract::<PyReadonlyArray1<'_, f64>>()?;
    if array.len() != expected {
        return Err(shape_error(name, &format!("({expected},)"), array.shape()));
    }
    Ok(array.as_slice()?.to_vec())
}

/// Extract a 2-D float64 config array with an exact shape check.
///
/// ## Parameters
/// - `obj`: Python config object.
/// - `name`: Attribute name.
/// - `rows`: Required rows.
/// - `cols`: Required columns.
///
/// ## Returns
/// A ``Vec<f64>`` copy in row-major order.
fn extract_f64_2d(
    obj: &Bound<'_, PyAny>,
    name: &str,
    rows: usize,
    cols: usize,
) -> PyResult<Vec<f64>> {
    let array = obj.getattr(name)?.extract::<PyReadonlyArray2<'_, f64>>()?;
    let shape = array.shape();
    if shape.len() != 2 || shape[0] != rows || shape[1] != cols {
        return Err(shape_error(name, &format!("({rows}, {cols})"), shape));
    }
    Ok(array.as_slice()?.to_vec())
}

/// Extract a 3-D float64 config array with an exact shape check.
///
/// ## Parameters
/// - `obj`: Python config object.
/// - `name`: Attribute name.
/// - `d0`, `d1`, `d2`: Required dimensions.
///
/// ## Returns
/// A ``Vec<f64>`` copy in row-major order.
fn extract_f64_3d(
    obj: &Bound<'_, PyAny>,
    name: &str,
    d0: usize,
    d1: usize,
    d2: usize,
) -> PyResult<Vec<f64>> {
    let array = obj.getattr(name)?.extract::<PyReadonlyArray3<'_, f64>>()?;
    let shape = array.shape();
    if shape.len() != 3 || shape[0] != d0 || shape[1] != d1 || shape[2] != d2 {
        return Err(shape_error(name, &format!("({d0}, {d1}, {d2})"), shape));
    }
    Ok(array.as_slice()?.to_vec())
}

/// Extract a 1-D boolean config array with an exact length check.
///
/// ## Parameters
/// - `obj`: Python config object.
/// - `name`: Attribute name.
/// - `expected`: Required length.
///
/// ## Returns
/// A ``Vec<bool>`` copy.
fn extract_bool_1d(obj: &Bound<'_, PyAny>, name: &str, expected: usize) -> PyResult<Vec<bool>> {
    let array = obj.getattr(name)?.extract::<PyReadonlyArray1<'_, bool>>()?;
    if array.len() != expected {
        return Err(shape_error(name, &format!("({expected},)"), array.shape()));
    }
    Ok(array.as_slice()?.to_vec())
}

impl DiscreteConfig {
    /// Build ``DiscreteConfig`` from a Python ``DiscretePopulationConfig`` object.
    ///
    /// ## Parameters
    /// - `config`: Python discrete config.
    ///
    /// ## Returns
    /// A ``DiscreteConfig`` owning plain Rust copies.
    ///
    /// ## Errors
    /// Returns ``PyValueError`` if the config is not a valid discrete config.
    pub fn from_python(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        let n_ztypes = extract_i64(config, "n_ztypes")? as usize;
        if config.getattr("n_ages")?.extract::<i64>()? != 2 {
            return Err(PyValueError::new_err(
                "discrete config must have n_ages == 2",
            ));
        }
        let cfg = Self {
            n_ztypes,
            stochastic: config.getattr("stochastic")?.extract::<bool>()?,
            continuous_sampling: config.getattr("continuous_sampling")?.extract::<bool>()?,
            female_adult_mating_rate: extract_f64(config, "female_adult_mating_rate")?,
            male_adult_mating_rate: extract_f64(config, "male_adult_mating_rate")?,
            reproduction_rate: extract_f64(config, "reproduction_rate")?,
            eggs_per_female: extract_f64(config, "eggs_per_female")?,
            sex_ratio: extract_f64(config, "sex_ratio")?,
            female_age0_survival: extract_f64(config, "female_age0_survival")?,
            male_age0_survival: extract_f64(config, "male_age0_survival")?,
            has_sex_chromosomes: config.getattr("has_sex_chromosomes")?.extract::<bool>()?,
            extreme_speed_mode: extract_i64(config, "extreme_speed_mode")?,
            juvenile_growth_mode: extract_i64(config, "juvenile_growth_mode")?,
            carrying_capacity: extract_f64(config, "carrying_capacity")?,
            expected_competition_strength: extract_f64(config, "expected_competition_strength")?,
            expected_survival_rate: extract_f64(config, "expected_survival_rate")?,
            low_density_growth_rate: extract_f64(config, "low_density_growth_rate")?,
            sexual_selection_fitness: extract_f64_2d(
                config,
                "sexual_selection_fitness",
                n_ztypes,
                n_ztypes,
            )?,
            offspring_tensor: extract_f64_3d(
                config,
                "offspring_tensor",
                n_ztypes,
                n_ztypes,
                n_ztypes,
            )?,
            fecundity_f: extract_f64_1d(config, "fecundity_f", n_ztypes)?,
            fecundity_m: extract_f64_1d(config, "fecundity_m", n_ztypes)?,
            viability_f: extract_f64_1d(config, "viability_f", n_ztypes)?,
            viability_m: extract_f64_1d(config, "viability_m", n_ztypes)?,
            female_ztype_compatibility: extract_f64_1d(
                config,
                "female_ztype_compatibility",
                n_ztypes,
            )?,
            male_ztype_compatibility: extract_f64_1d(config, "male_ztype_compatibility", n_ztypes)?,
            female_only_by_sex_chrom: extract_bool_1d(
                config,
                "female_only_by_sex_chrom",
                n_ztypes,
            )?,
            male_only_by_sex_chrom: extract_bool_1d(config, "male_only_by_sex_chrom", n_ztypes)?,
        };
        Ok(cfg)
    }
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
/// - `cfg`: Discrete config.
/// - `male_counts`: Effective adult male counts.
/// - `out`: Output matrix, overwritten.
fn compute_mating_probability(cfg: &DiscreteConfig, male_counts: &[f64], out: &mut [f64]) {
    // Same row-normalized mating probabilities as the age-structured path,
    // but for the discrete two-age adult class only.
    let g = cfg.n_ztypes;
    for gf in 0..g {
        let mut row_sum = 0.0;
        for gm in 0..g {
            let value = cfg.sexual_selection_fitness[gf * g + gm] * male_counts[gm];
            out[gf * g + gm] = value;
            row_sum += value;
        }
        if row_sum.is_finite() && row_sum > EPS {
            for gm in 0..g {
                out[gf * g + gm] /= row_sum;
            }
        } else {
            for gm in 0..g {
                out[gf * g + gm] = 0.0;
            }
        }
    }
}

/// Sample the number of matings per female genotype and distribute male partners.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `cfg`: Discrete config.
/// - `females`: Adult female counts per zygote type.
/// - `mating_prob`: Precomputed mating probability matrix.
/// - `pair_counts`: Output mated pair counts, accumulated in place.
fn mate_discrete(
    rng: &mut SmallRng,
    cfg: &DiscreteConfig,
    females: &[f64],
    mating_prob: &[f64],
    pair_counts: &mut [f64],
) {
    // For each adult female genotype, sample the number of matings and
    // distribute male partners using the mating probability row.
    let g = cfg.n_ztypes;
    let p_mating = clamp01(cfg.female_adult_mating_rate);
    let mut tmp = vec![0.0; g];
    for gf in 0..g {
        let n_female = females[gf];
        if n_female <= 0.0 {
            continue;
        }
        let n_mating = if cfg.stochastic {
            if cfg.continuous_sampling {
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
        if cfg.stochastic {
            if cfg.continuous_sampling {
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
/// - `cfg`: Discrete config.
/// - `pair_counts`: Mated pair counts.
/// - `n_f`: Output female age-0 counts.
/// - `n_m`: Output male age-0 counts.
fn fertilize_discrete(
    rng: &mut SmallRng,
    cfg: &DiscreteConfig,
    pair_counts: &[f64],
    n_f: &mut [f64],
    n_m: &mut [f64],
) {
    // Convert mated pairs into age-0 offspring counts:
    // - Egg production uses fecundity and reproduction rate.
    // - Offspring genotype comes from the offspring tensor.
    // - Sex is assigned by sex ratio or sex-chromosome rules.
    let g = cfg.n_ztypes;
    let p_reproduce = clamp01(cfg.reproduction_rate);
    let sex_ratio = clamp01(cfg.sex_ratio);
    let mut offspring = vec![0.0; g];
    let mut p_norm = vec![0.0; g];
    let mut tmp = vec![0.0; g];
    let mut has_any = false;

    for gf in 0..g {
        let ff = cfg.fecundity_f[gf];
        for gm in 0..g {
            let n_pairs = pair_counts[gf * g + gm];
            if n_pairs <= 0.0 {
                continue;
            }
            has_any = true;
            let eggs_per_pair = cfg.eggs_per_female * ff * cfg.fecundity_m[gm];
            let n_total = if cfg.stochastic {
                let n_pairs_eff = if cfg.continuous_sampling {
                    n_pairs
                } else {
                    n_pairs.round()
                };
                if n_pairs_eff <= 0.0 {
                    continue;
                }
                let n_reproducing = if p_reproduce < 1.0 - EPS {
                    if cfg.continuous_sampling {
                        continuous_binomial(rng, n_pairs_eff, p_reproduce)
                    } else {
                        binomial(rng, n_pairs_eff as i64, p_reproduce)
                    }
                } else {
                    n_pairs_eff
                };
                let total_lambda = (n_reproducing * eggs_per_pair).max(0.0);
                if cfg.continuous_sampling {
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
                p_surv += cfg.offspring_tensor[(gf * g + gm) * g + go];
            }
            if cfg.stochastic {
                if p_surv <= EPS {
                    continue;
                }
                let n_viable = if p_surv >= 1.0 - EPS {
                    n_total
                } else if cfg.continuous_sampling {
                    continuous_binomial(rng, n_total, p_surv)
                } else {
                    binomial(rng, n_total.round() as i64, p_surv)
                };
                if n_viable <= EPS {
                    continue;
                }
                let inv = 1.0 / p_surv;
                for go in 0..g {
                    p_norm[go] = cfg.offspring_tensor[(gf * g + gm) * g + go] * inv;
                }
                if cfg.continuous_sampling {
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
                    offspring[go] += n_total * cfg.offspring_tensor[(gf * g + gm) * g + go];
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
            let n_fem = if cfg.stochastic {
                if cfg.continuous_sampling {
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
/// - `cfg`: Discrete config.
/// - `ind`: Mutable discrete individual-count slice.
pub fn reproduction(rng: &mut SmallRng, cfg: &DiscreteConfig, ind: &mut [f64]) {
    // Discrete reproduction pipeline:
    // 1. Build effective adult males.
    // 2. Build mating probabilities and sample pair counts.
    // 3. Fertilize pairs into age-0 female/male offspring.
    let g = cfg.n_ztypes;
    let mut effective_males = vec![0.0; g];
    for z in 0..g {
        effective_males[z] = ind[idx(1, 1, z, g)] * cfg.male_adult_mating_rate;
    }
    let females_total: f64 = (0..g).map(|z| ind[idx(0, 1, z, g)]).sum();
    let males_total: f64 = effective_males.iter().sum();
    if males_total == 0.0 || females_total == 0.0 {
        return;
    }

    let mut mating_prob = vec![0.0; g * g];
    compute_mating_probability(cfg, &effective_males, &mut mating_prob);
    let mut pair_counts = vec![0.0; g * g];
    let females: Vec<f64> = (0..g).map(|z| ind[idx(0, 1, z, g)]).collect();
    mate_discrete(rng, cfg, &females, &mating_prob, &mut pair_counts);

    let mut n_f = vec![0.0; g];
    let mut n_m = vec![0.0; g];
    fertilize_discrete(rng, cfg, &pair_counts, &mut n_f, &mut n_m);
    for z in 0..g {
        ind[idx(0, 0, z, g)] = n_f[z];
        ind[idx(1, 0, z, g)] = n_m[z];
    }
}

/// Compute juvenile density-regulation scaling for discrete populations.
///
/// ## Parameters
/// - `cfg`: Discrete config.
/// - `ind`: Current discrete individual-count slice.
///
/// ## Returns
/// A non-negative scaling factor.
fn scaling_factor(cfg: &DiscreteConfig, ind: &[f64]) -> f64 {
    // Juvenile density regulation for discrete populations;
    // identical growth-mode semantics to the age-structured engine.
    let g = cfg.n_ztypes;
    let total_age_0: f64 = (0..g)
        .map(|z| ind[idx(0, 0, z, g)] + ind[idx(1, 0, z, g)])
        .sum();
    match cfg.juvenile_growth_mode {
        NO_COMPETITION => 1.0,
        FIXED => {
            if total_age_0 > 0.0 {
                (cfg.carrying_capacity / total_age_0).min(1.0)
            } else {
                1.0
            }
        }
        LOGISTIC => {
            let ratio = if cfg.expected_competition_strength > 0.0 {
                total_age_0 / cfg.expected_competition_strength
            } else {
                1.0
            };
            ((-ratio * (cfg.low_density_growth_rate - 1.0)) + cfg.low_density_growth_rate).max(0.0)
                * cfg.expected_survival_rate
        }
        BEVERTON_HOLT => {
            let ratio = if cfg.expected_competition_strength > 0.0 {
                total_age_0 / cfg.expected_competition_strength
            } else {
                1.0
            };
            (cfg.low_density_growth_rate / (ratio * (cfg.low_density_growth_rate - 1.0) + 1.0))
                * cfg.expected_survival_rate
        }
        _ => 1.0,
    }
}

/// Apply juvenile scaling by resampling age-0 counts.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `cfg`: Discrete config.
/// - `ind`: Mutable discrete individual-count slice.
/// - `scaling`: Scaling factor.
fn recruit_juveniles(rng: &mut SmallRng, cfg: &DiscreteConfig, ind: &mut [f64], scaling: f64) {
    // Resample age-0 counts to the scaled total;
    // stochastic uses multinomial, deterministic scales proportionally.
    let g = cfg.n_ztypes;
    let stochastic = cfg.stochastic;
    let continuous = cfg.continuous_sampling;
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
/// - `cfg`: Discrete config.
/// - `ind`: Mutable discrete individual-count slice.
pub fn survival(rng: &mut SmallRng, cfg: &DiscreteConfig, ind: &mut [f64]) {
    // Discrete survival applies density scaling first, then viability
    // survival separately for female and male age-0 individuals.
    let g = cfg.n_ztypes;
    let scaling = scaling_factor(cfg, ind);
    recruit_juveniles(rng, cfg, ind, scaling);

    let s_f = cfg.female_age0_survival;
    let s_m = cfg.male_age0_survival;
    for z in 0..g {
        let f = ind[idx(0, 0, z, g)];
        let m = ind[idx(1, 0, z, g)];
        let rate_f = s_f * cfg.viability_f[z];
        let rate_m = s_m * cfg.viability_m[z];
        if cfg.stochastic {
            if cfg.continuous_sampling {
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
/// - `_cfg`: Discrete config (used for n_ztypes).
/// - `ind`: Mutable discrete individual-count slice.
pub fn aging(_cfg: &DiscreteConfig, ind: &mut [f64]) {
    // Move age-0 juveniles into age 1 and clear age 0.
    let g = _cfg.n_ztypes;
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
/// hook -> aging.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `cfg`: Discrete config.
/// - `hooks`: CSR hook program.
/// - `ind`: Mutable discrete individual-count slice.
/// - `tick`: Current tick.
///
/// ## Returns
/// ``Ok(0)`` for continue, ``Ok(1)`` if a hook requested stop, or an error string.
pub fn run_tick(
    rng: &mut SmallRng,
    cfg: &DiscreteConfig,
    hooks: &HookProgram,
    ind: &mut [f64],
    tick: i64,
) -> Result<i32, String> {
    // One discrete tick follows: first hook -> reproduction -> early hook
    // -> survival -> late hook -> aging.
    let mut result = hooks.execute_event(
        rng,
        0,
        ind,
        &mut [],
        2,
        2,
        cfg.n_ztypes,
        tick,
        cfg.stochastic,
        cfg.continuous_sampling,
        -1,
    );
    if result != 0 {
        return Ok(result);
    }
    reproduction(rng, cfg, ind);
    result = hooks.execute_event(
        rng,
        1,
        ind,
        &mut [],
        2,
        2,
        cfg.n_ztypes,
        tick,
        cfg.stochastic,
        cfg.continuous_sampling,
        -1,
    );
    if result != 0 {
        return Ok(result);
    }
    survival(rng, cfg, ind);
    result = hooks.execute_event(
        rng,
        2,
        ind,
        &mut [],
        2,
        2,
        cfg.n_ztypes,
        tick,
        cfg.stochastic,
        cfg.continuous_sampling,
        -1,
    );
    if result != 0 {
        return Ok(result);
    }
    aging(cfg, ind);
    Ok(0)
}

/// Run one fused Wright-Fisher tick (first hook plus full WF update).
///
/// Only the first hook runs; the next generation is computed directly from
/// adult allele frequencies using the configured extreme-speed mode.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `cfg`: Discrete config.
/// - `ind`: Mutable discrete individual-count slice.
///
/// ## Returns
/// ``Ok(())`` or an error string for an unknown WF mode.
pub fn run_wf_tick(
    rng: &mut SmallRng,
    cfg: &DiscreteConfig,
    ind: &mut [f64],
) -> Result<(), String> {
    // Fused Wright-Fisher tick: only the first hook runs, then the entire
    // next generation is computed from adult allele frequencies in one step.
    let mode = cfg.extreme_speed_mode;
    if !matches!(mode, WF_MULTINOMIAL | WF_POISSON | WF_DETERMINISTIC) {
        return Err(format!("unrecognised extreme_speed_mode={mode}"));
    }
    let g = cfg.n_ztypes;
    let mut expected_f = vec![0.0; g];
    let mut expected_m = vec![0.0; g];
    let adult_f: Vec<f64> = (0..g).map(|z| ind[idx(0, 1, z, g)]).collect();
    let adult_m: Vec<f64> = (0..g).map(|z| ind[idx(1, 1, z, g)]).collect();
    let effective_m: Vec<f64> = adult_m
        .iter()
        .map(|&v| v * cfg.male_adult_mating_rate)
        .collect();

    for gf in 0..g {
        let nf = adult_f[gf] * cfg.fecundity_f[gf] * cfg.female_adult_mating_rate;
        if nf <= 0.0 {
            continue;
        }
        let row_sum: f64 = (0..g)
            .map(|gm| cfg.sexual_selection_fitness[gf * g + gm] * effective_m[gm])
            .sum();
        if row_sum <= 0.0 {
            continue;
        }
        for gm in 0..g {
            let nm_eff = effective_m[gm];
            if nm_eff <= 0.0 {
                continue;
            }
            let pair_weight = nf * (nm_eff / row_sum) * cfg.sexual_selection_fitness[gf * g + gm];
            if pair_weight <= 0.0 {
                continue;
            }
            for go in 0..g {
                let prob = cfg.offspring_tensor[(gf * g + gm) * g + go];
                if prob <= 0.0 {
                    continue;
                }
                let offspring = pair_weight
                    * prob
                    * cfg.eggs_per_female
                    * cfg.reproduction_rate
                    * cfg.fecundity_m[gm];
                if cfg.has_sex_chromosomes {
                    if cfg.female_only_by_sex_chrom[go] {
                        expected_f[go] += offspring;
                    } else if cfg.male_only_by_sex_chrom[go] {
                        expected_m[go] += offspring;
                    } else {
                        expected_f[go] += offspring * cfg.female_ztype_compatibility[go];
                        expected_m[go] += offspring * cfg.male_ztype_compatibility[go];
                    }
                } else {
                    expected_f[go] += offspring * cfg.sex_ratio;
                    expected_m[go] += offspring * (1.0 - cfg.sex_ratio);
                }
            }
        }
    }
    for go in 0..g {
        expected_f[go] *= cfg.viability_f[go];
        expected_m[go] *= cfg.viability_m[go];
    }
    if cfg.juvenile_growth_mode > 0 {
        let total: f64 = expected_f.iter().chain(expected_m.iter()).sum();
        let sf = match cfg.juvenile_growth_mode {
            FIXED => {
                if total > 0.0 {
                    (cfg.carrying_capacity / total).min(1.0)
                } else {
                    1.0
                }
            }
            LOGISTIC => {
                let ratio = if cfg.expected_competition_strength > 0.0 {
                    total / cfg.expected_competition_strength
                } else {
                    1.0
                };
                ((-ratio * (cfg.low_density_growth_rate - 1.0)) + cfg.low_density_growth_rate)
                    .max(0.0)
                    * cfg.expected_survival_rate
            }
            BEVERTON_HOLT => {
                let ratio = if cfg.expected_competition_strength > 0.0 {
                    total / cfg.expected_competition_strength
                } else {
                    1.0
                };
                (cfg.low_density_growth_rate / (ratio * (cfg.low_density_growth_rate - 1.0) + 1.0))
                    * cfg.expected_survival_rate
            }
            _ => 1.0,
        };
        for value in expected_f.iter_mut().chain(expected_m.iter_mut()) {
            *value *= sf;
        }
    }
    let mut new_f = vec![0.0; g];
    let mut new_m = vec![0.0; g];
    if mode == WF_DETERMINISTIC || !cfg.stochastic {
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

/// Run up to ``n_ticks`` discrete or Wright-Fisher ticks inside Rust with optional recording.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `cfg`: Discrete config.
/// - `hooks`: CSR hook program.
/// - `ind`: Mutable discrete individual-count slice.
/// - `tick`: Starting tick.
/// - `n_ticks`: Number of ticks.
/// - `record_interval`: Record interval; 0 disables recording.
/// - `observation_mask`: Optional observation mask.
/// - `wf`: Whether to use the fused Wright-Fisher tick.
///
/// ## Returns
/// ``(final_tick, flat_history, n_rows, was_stopped)``.
pub fn run_batch(
    rng: &mut SmallRng,
    cfg: &DiscreteConfig,
    hooks: &HookProgram,
    ind: &mut [f64],
    tick: i64,
    n_ticks: i64,
    record_interval: i64,
    observation_mask: Option<&[f64]>,
    wf: bool,
) -> Result<(i64, Vec<f64>, usize, bool), String> {
    // Loop discrete or WF ticks in Rust with optional history recording.
    let g = cfg.n_ztypes;
    let n_obs_values = 2 * 2 * g;
    let groups = match observation_mask {
        Some(mask) => {
            if mask.len() % n_obs_values != 0 {
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
    let record = |history: &mut Vec<f64>,
                  ind: &[f64],
                  mask: Option<&[f64]>,
                  groups: usize,
                  current_tick: i64| {
        history.push(current_tick as f64);
        match mask {
            Some(mask) => {
                for group in 0..groups {
                    for sex in 0..2 {
                        for age in 0..2 {
                            let mut total = 0.0;
                            for z in 0..g {
                                let state_idx = (sex * 2 + age) * g + z;
                                let mask_idx = ((group * 2 + sex) * 2 + age) * g + z;
                                total += ind[state_idx] * mask[mask_idx];
                            }
                            history.push(total);
                        }
                    }
                }
            }
            None => history.extend_from_slice(ind),
        }
    };

    if record_interval > 0 && current_tick % record_interval == 0 {
        record(&mut history, ind, observation_mask, groups, current_tick);
        n_rows += 1;
    }
    for _ in 0..n_ticks {
        let result = if wf {
            let result = hooks.execute_event(
                rng,
                0,
                ind,
                &mut [],
                2,
                2,
                g,
                current_tick,
                cfg.stochastic,
                cfg.continuous_sampling,
                -1,
            );
            if result != 0 {
                return Ok((current_tick, history, n_rows, true));
            }
            run_wf_tick(rng, cfg, ind)?;
            0
        } else {
            run_tick(rng, cfg, hooks, ind, current_tick)?
        };
        if result != 0 {
            return Ok((current_tick, history, n_rows, true));
        }
        current_tick += 1;
        if record_interval > 0 && current_tick % record_interval == 0 {
            record(&mut history, ind, observation_mask, groups, current_tick);
            n_rows += 1;
        }
    }
    Ok((current_tick, history, n_rows, false))
}
