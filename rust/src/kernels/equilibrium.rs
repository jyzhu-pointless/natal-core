//! Equilibrium calibration.
//!
//! [`equilibrium_metrics`] computes the equilibrium metrics operation by
//! operation with fixed multiplication order, branches, and guards (the
//! numerics were originally ported bit-for-bit from the retired
//! pure-Python reference engine).  It is evaluated on demand from the owned contract
//! ([`crate::model::blueprint::Blueprint`] + [`crate::model::ecology::EcologyParams`])
//! whenever a lifecycle stage needs the derived values (juvenile density
//! regulation and the fused Wright-Fisher update), so parameter changes take
//! effect without any stored derived state.  Columnized params are read at
//! the requested deme's column entry / vector segment.

use crate::kernels::rng::clamp01;
use crate::model::blueprint::Blueprint;
use crate::model::ecology::EcologyParams;

/// Offspring female fraction of a balanced sex-chromosome system.
///
/// A heterogametic parent (XY male, ZW female) segregates its sex chromosomes
/// 1:1 under ordinary Mendelian meiosis, so a species whose sex is determined
/// by sex chromosomes produces an even split regardless of the ``sex_ratio``
/// parameter — the documented contract calls that parameter "ignored" there.
const BALANCED_SEX_CHROMOSOME_FEMALE_FRACTION: f64 = 0.5;

/// Resolve the offspring female fraction the owning engine actually uses.
///
/// Models without sex chromosomes split offspring by the ``sex_ratio``
/// parameter.  Models with sex chromosomes ignore it and follow the genetic
/// split above; feeding the parameter into the calibration moved the
/// calibrated equilibrium off ``K``, because both the age-1 reference split
/// and ``s_0_avg`` consumed it.
///
/// This is a wild-type property: the calibration reads the species structure
/// and the ecology columns only, never the genetics tables.  A modifier or
/// preset that rewrites the transmission tables (a sex-ratio distorter
/// included) is an overlay, exactly like fitness: it changes the realized
/// composition without changing what ``K`` anchors.  A model whose anchor
/// should be something other than the wild-type equilibrium declares it
/// explicitly through ``equilibrium_distribution``.
///
/// ## Parameters
/// - `has_sex_chromosomes`: Whether the species determines sex from sex chromosomes.
/// - `sex_ratio`: The stored reproduction parameter.
///
/// ## Returns
/// The offspring female fraction the tick's sex allocation produces.
#[must_use]
pub(crate) fn effective_offspring_sex_ratio(has_sex_chromosomes: bool, sex_ratio: f64) -> f64 {
    if has_sex_chromosomes {
        BALANCED_SEX_CHROMOSOME_FEMALE_FRACTION
    } else {
        sex_ratio
    }
}

/// Compute the equilibrium competition strength C* and survival rate s*
/// for one deme's ecology column.
///
/// The two-branch structure mirrors the Python function exactly:
/// a user-declared equilibrium distribution (non-empty
/// ``EcologyParams::equilibrium_distribution``) is used as-is, otherwise the
/// distribution is derived from the carrying capacity (age 1 total = K,
/// females split by the *surviving* sex ratio — the offspring sex ratio
/// filtered by each sex's own age-0 survival, which reduces to the offspring
/// sex ratio when both sexes survive equally — and later ages decayed by
/// survival).
/// ``external_expected_eggs`` overrides the egg production used for the
/// survival rate only, never for the competition strength.  A negative
/// ``external_expected_eggs`` means "unused" (the materialization
/// convention), matching Python's ``None``.
///
/// Two inputs are read through the owning engine's semantics rather than
/// verbatim, so the reference state is the one the tick actually reaches:
/// the offspring sex ratio follows the genetic split for sex-chromosome
/// species (see [`effective_offspring_sex_ratio`]), and the per-age fertility
/// weight is the implicit 1.0 of the discrete-generation tick or the
/// age-structured tick's ``clamp01``.
///
/// ## Parameters
/// - `bp`: The frozen blueprint (dimensions and execution flags).
/// - `params`: Current runtime parameters (columnized).
/// - `deme`: Deme whose ecology column feeds the computation.
///
/// ## Returns
/// ``(expected_competition_strength, expected_survival_rate)``.
#[must_use]
pub fn equilibrium_metrics(bp: &Blueprint, params: &EcologyParams, deme: usize) -> (f64, f64) {
    let n_ages = bp.n_ages;
    let new_adult_age = bp.new_adult_age;
    // Panmictic single-deme params (length-1 columns) and per-deme segments
    // read through the same accessors; entry 0 == the pre-columnization
    // scalar, so single-population arithmetic is bit-identical.
    // Defensive clamp: an out-of-range deme reads the last column instead of
    // panicking.
    let deme_idx = deme.min(params.n_demes.saturating_sub(1));
    let carrying_capacity = params.carrying_capacity[deme_idx];
    let eggs_per_female = params.eggs_per_female[deme_idx];
    // Sex-chromosome species determine offspring sex from the chromosomes, so
    // the calibration must follow the genetic split instead of the parameter
    // the engine ignores there.
    let sex_ratio =
        effective_offspring_sex_ratio(bp.has_sex_chromosomes, params.sex_ratio[deme_idx]);

    // Python falls back to the female mating-rate row when the reproduction
    // vector is not supplied; the contract always carries one, so the
    // fallback branch is unreachable here by construction.
    // Flat row-major per-deme segments: `a` entries for the (A,) vectors and
    // `2*a` for the two-sex matrices; the base offset is `deme * extent`.
    let a = n_ages;
    let reproduce_rates: &[f64] = &params.reproduction_rates[deme_idx * a..(deme_idx + 1) * a];
    let fertility: &[f64] = &params.fertility[deme_idx * a..(deme_idx + 1) * a];
    let competition_weights: &[f64] = &params.competition_weights[deme_idx * a..(deme_idx + 1) * a];
    let survival_rates: &[f64] = &params.survival_rates[deme_idx * 2 * a..(deme_idx + 1) * 2 * a];
    // Derive-mode sentinel: an undeclared deme passes an empty slice so the
    // core falls back to the carrying-capacity derivation.
    let declared: &[f64] = if !params.equilibrium_declared[deme_idx] {
        &[]
    } else {
        &params.equilibrium_distribution[deme_idx * 2 * a..(deme_idx + 1) * 2 * a]
    };
    let external = params.external_expected_eggs[deme_idx];

    equilibrium_metrics_core(
        carrying_capacity,
        eggs_per_female,
        sex_ratio,
        survival_rates,
        reproduce_rates,
        fertility,
        competition_weights,
        declared,
        external,
        new_adult_age,
        n_ages,
        bp.discrete_generation,
    )
}

/// The pure equilibrium computation shared by every caller (the
/// Rust owns the numeric algorithm; both the contract-column wrapper
/// above and the flat PyO3 entry below funnel through this core).
///
/// Mirrors the Python reference statement by statement.  An empty
/// ``declared`` slice derives the distribution; a negative
/// ``external_expected_eggs`` means "unused" (Python's ``None``).
///
/// ## Parameters
/// - `survival_rates`: Flat ``(2 * n_ages)`` row-major survival matrix.
/// - `reproduce_rates`: Resolved ``(n_ages,)`` reproduction participation.
/// - `fertility`: ``(n_ages,)`` stored relative female fertility.
/// - `competition_weights`: ``(n_ages,)`` juvenile competition weights.
/// - `declared`: Flat ``(2 * n_ages)`` declared distribution, or empty.
/// - `external_expected_eggs`: Champer egg override (negative = unused).
/// - `new_adult_age` / `n_ages`: Age-structure bounds.
/// - `discrete_generation`: True when the owning tick is the discrete-
///   generation engine, whose reproduction reads no per-age fertility.
///
/// ## Returns
/// ``(expected_competition_strength, expected_survival_rate)``.
// The flat parameter table mirrors the Python reference signature
// one-to-one; a params struct would decouple the
// two spellings the single-source rule keeps aligned.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn equilibrium_metrics_core(
    carrying_capacity: f64,
    eggs_per_female: f64,
    sex_ratio: f64,
    survival_rates: &[f64],
    reproduce_rates: &[f64],
    fertility: &[f64],
    competition_weights: &[f64],
    declared: &[f64],
    external_expected_eggs: f64,
    new_adult_age: usize,
    n_ages: usize,
    discrete_generation: bool,
) -> (f64, f64) {
    // Per-age fertility weight as the owning engine consumes it.  Discrete
    // generations have no age-dependent fertility (their tick uses an implicit
    // 1.0); the age-structured tick clamps the stored weight to [0, 1].
    // Reading the raw stored value here made the calibration disagree with the
    // tick whenever a raw tensor write left the builder's [0, 1] domain, which
    // moved the calibrated equilibrium by exactly the written factor.
    let age_fertility = |age: usize| -> f64 {
        if discrete_generation {
            1.0
        } else {
            clamp01(fertility[age])
        }
    };
    // Reproduction participation is a probability; clamp to [0, 1].  Only ages
    // >= new_adult_age can reproduce, so juvenile entries stay exactly 0.
    let mut p_reproducing = vec![0.0_f64; n_ages];
    for age in new_adult_age..n_ages {
        p_reproducing[age] = clamp01(reproduce_rates[age]);
    }

    // produced_age_0 accumulates the egg production
    // Σ N_f[age] * P_reproducing[age] * fertility[age] * eggs_per_female;
    // total_age_1 is the age-1 head count that the survival-rate ratio solves for.
    let expected_distribution: Vec<f64>;
    let total_age_1: f64;
    let mut produced_age_0 = 0.0_f64;

    if !declared.is_empty() {
        // 1. Use the user-provided equilibrium distribution (flat (2, A)
        //    segment of the deme's column).
        expected_distribution = declared.to_vec();
        for age in new_adult_age..n_ages {
            let n_f = expected_distribution[age]; // row 0 = female
                                                  // Contribution of this age to age-0 production:
                                                  // n_f * P(reproducing_this_tick) * relative_fertility * eggs_per_female
            produced_age_0 += n_f * p_reproducing[age] * age_fertility(age) * eggs_per_female;
        }
        // Python: expected_distribution[0, 1] + expected_distribution[1, 1].
        total_age_1 = expected_distribution[1] + expected_distribution[n_ages + 1];
    } else {
        // 2. Derive the equilibrium distribution with age-1 total = K.
        total_age_1 = carrying_capacity;
        let mut dist = vec![0.0_f64; 2 * n_ages];
        // Age 1 baseline allocation.  The reference composition has to be the
        // one the model actually reaches: offspring are produced at the
        // engine's offspring sex ratio (``sex_ratio``, or the balanced genetic
        // split for sex-chromosome species) and each sex then survives its own
        // age-0 rate, so the
        // age-1 female share is
        //     sex_ratio * s_f0 / (sex_ratio * s_f0 + (1 - sex_ratio) * s_m0).
        // Splitting by the raw sex ratio instead misstated the reference
        // female count whenever the sexes survive differently, which moved the
        // calibrated equilibrium off K by up to tens of percent (the survival
        // rate the calibration solves for follows the same reference).
        // Equal age-0 survival keeps the historical expression bit-for-bit, so
        // no model that survives equally moves.
        let female_mass = sex_ratio * survival_rates[0];
        let male_mass = (1.0 - sex_ratio) * survival_rates[n_ages];
        if survival_rates[0] == survival_rates[n_ages] || female_mass + male_mass <= 0.0 {
            // Historical split; also the degenerate case where no juvenile
            // survives at all, which has no composition to derive (the
            // survival-rate guard below already collapses the calibration).
            dist[1] = total_age_1 * sex_ratio;
            dist[n_ages + 1] = total_age_1 * (1.0 - sex_ratio);
        } else {
            let surviving = female_mass + male_mass;
            dist[1] = total_age_1 * (female_mass / surviving);
            dist[n_ages + 1] = total_age_1 * (male_mass / surviving);
        }
        // Later ages decay by the previous age's survival rate.
        for age in 2..n_ages {
            dist[age] = dist[age - 1] * survival_rates[age - 1];
            dist[n_ages + age] = dist[n_ages + age - 1] * survival_rates[n_ages + age - 1];
        }
        // Same egg-production sum as the declared branch, now over the derived
        // female distribution.
        for age in new_adult_age..n_ages {
            let n_f = dist[age];
            produced_age_0 += n_f * p_reproducing[age] * age_fertility(age) * eggs_per_female;
        }
        expected_distribution = dist;
    }

    // Expected competition strength: weighted sum over the juvenile ages
    // only.  Age 0 contributes the produced eggs; ages 1..new_adult_age
    // contribute the surviving distribution counts.
    let mut expected_competition_strength = produced_age_0 * competition_weights[0];
    for age in 1..new_adult_age {
        let n_total = expected_distribution[age] + expected_distribution[n_ages + age];
        expected_competition_strength += n_total * competition_weights[age];
    }

    // Expected survival rate: scaling factor from egg production to age-1
    // entrants.  s_0_avg blends the age-0 survival by the sex ratio.
    let s_0_avg = sex_ratio * survival_rates[0] + (1.0 - sex_ratio) * survival_rates[n_ages];

    // external_expected_eggs replaces produced_age_0 in the survival-rate
    // formula only; negative means "unused" (Python's None).
    let survival_eggs = if external_expected_eggs < 0.0 {
        produced_age_0
    } else {
        external_expected_eggs
    };

    // Zero (or negligible) egg production, or negligible age-0 survival, has no
    // scale to solve for; fall back to unit survival instead of dividing by zero.
    let expected_survival_rate = if survival_eggs > 0.0 && s_0_avg > 1e-10 {
        total_age_1 / (survival_eggs * s_0_avg)
    } else {
        1.0
    };

    (expected_competition_strength, expected_survival_rate)
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/equilibrium.rs"]
mod tests;
