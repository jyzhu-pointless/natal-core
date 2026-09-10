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

/// Compute the equilibrium competition strength C* and survival rate s*
/// for one deme's ecology column.
///
/// The two-branch structure mirrors the Python function exactly:
/// a user-declared equilibrium distribution (non-empty
/// ``EcologyParams::equilibrium_distribution``) is used as-is, otherwise the
/// distribution is derived from the carrying capacity (age 1 total = K,
/// females split by the sex ratio, later ages decayed by survival).
/// ``external_expected_eggs`` overrides the egg production used for the
/// survival rate only, never for the competition strength.  A negative
/// ``external_expected_eggs`` means "unused" (the materialization
/// convention), matching Python's ``None``.
///
/// ## Parameters
/// - `bp`: The frozen blueprint (dimensions).
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
    let deme_idx = deme.min(params.n_demes.saturating_sub(1));
    let carrying_capacity = params.carrying_capacity[deme_idx];
    let eggs_per_female = params.eggs_per_female[deme_idx];
    let sex_ratio = params.sex_ratio[deme_idx];

    // Python falls back to the female mating-rate row when the reproduction
    // vector is not supplied; the contract always carries one, so the
    // fallback branch is unreachable here by construction.
    let a = n_ages;
    let reproduce_rates: &[f64] = &params.reproduction_rates[deme_idx * a..(deme_idx + 1) * a];
    let fertility: &[f64] = &params.fertility[deme_idx * a..(deme_idx + 1) * a];
    let competition_weights: &[f64] = &params.competition_weights[deme_idx * a..(deme_idx + 1) * a];
    let survival_rates: &[f64] = &params.survival_rates[deme_idx * 2 * a..(deme_idx + 1) * 2 * a];
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
/// - `fertility`: ``(n_ages,)`` relative female fertility.
/// - `competition_weights`: ``(n_ages,)`` juvenile competition weights.
/// - `declared`: Flat ``(2 * n_ages)`` declared distribution, or empty.
/// - `external_expected_eggs`: Champer egg override (negative = unused).
/// - `new_adult_age` / `n_ages`: Age-structure bounds.
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
) -> (f64, f64) {
    let mut p_reproducing = vec![0.0_f64; n_ages];
    for age in new_adult_age..n_ages {
        p_reproducing[age] = clamp01(reproduce_rates[age]);
    }

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
            produced_age_0 += n_f * p_reproducing[age] * fertility[age] * eggs_per_female;
        }
        // Python: expected_distribution[0, 1] + expected_distribution[1, 1].
        total_age_1 = expected_distribution[1] + expected_distribution[n_ages + 1];
    } else {
        // 2. Derive the equilibrium distribution with age-1 total = K.
        total_age_1 = carrying_capacity;
        let mut dist = vec![0.0_f64; 2 * n_ages];
        // Age 1 baseline allocation: females by sex ratio, males by the rest.
        // Python: dist[0, 1] = K * sex_ratio; dist[1, 1] = K * (1 - sex_ratio).
        dist[1] = total_age_1 * sex_ratio;
        dist[n_ages + 1] = total_age_1 * (1.0 - sex_ratio);
        // Later ages decay by the previous age's survival rate.
        for age in 2..n_ages {
            dist[age] = dist[age - 1] * survival_rates[age - 1];
            dist[n_ages + age] = dist[n_ages + age - 1] * survival_rates[n_ages + age - 1];
        }
        for age in new_adult_age..n_ages {
            let n_f = dist[age];
            produced_age_0 += n_f * p_reproducing[age] * fertility[age] * eggs_per_female;
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
