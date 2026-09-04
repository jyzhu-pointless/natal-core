//! Equilibrium calibration ported bit-for-bit from the Python reference.
//!
//! [`equilibrium_metrics`] mirrors
//! ``natal.backends.reference.simulation.age_structured.compute_equilibrium_metrics``
//! operation by operation: same multiplication order, same branches, same
//! guards.  It is evaluated on demand from the owned contract
//! ([`crate::contract::Blueprint`] + [`crate::contract::Params`]) whenever a
//! tick batch assembles its [`crate::config::SimConfig`], so parameter
//! changes take effect without any stored derived state.  Columnized params
//! are read at the requested deme's column entry / vector segment.

use crate::contract::{Blueprint, Params};
use crate::rng::clamp01;

/// Compute the equilibrium competition strength C* and survival rate s*
/// for one deme's ecology column.
///
/// The two-branch structure mirrors the Python function exactly:
/// a user-declared equilibrium distribution (non-empty
/// ``Params::equilibrium_distribution``) is used as-is, otherwise the
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
pub fn equilibrium_metrics(bp: &Blueprint, params: &Params, deme: usize) -> (f64, f64) {
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

    let mut p_reproducing = vec![0.0_f64; n_ages];
    for age in new_adult_age..n_ages {
        p_reproducing[age] = clamp01(reproduce_rates[age]);
    }

    let has_declared = !params.equilibrium_distribution.is_empty();
    let expected_distribution: Vec<f64>;
    let total_age_1: f64;
    let mut produced_age_0 = 0.0_f64;

    if has_declared {
        // 1. Use the user-provided equilibrium distribution (flat (2, A)
        //    segment of the deme's column).
        expected_distribution =
            params.equilibrium_distribution[deme_idx * 2 * a..(deme_idx + 1) * 2 * a].to_vec();
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
    let external = params.external_expected_eggs[deme_idx];
    let survival_eggs = if external < 0.0 {
        produced_age_0
    } else {
        external
    };

    let expected_survival_rate = if survival_eggs > 0.0 && s_0_avg > 1e-10 {
        total_age_1 / (survival_eggs * s_0_avg)
    } else {
        1.0
    };

    (expected_competition_strength, expected_survival_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Build a small blueprint/params pair (2 sexes, 4 ages, 2 ztypes).
    fn fixture() -> (Blueprint, Params, crate::contract::TensorSet) {
        let n_ages = 4;
        let z = 2;
        let bp = Blueprint {
            n_sexes: 2,
            n_ages,
            n_ztypes: z,
            n_gtypes: z,
            n_glabs: 1,
            new_adult_age: 1,
            adult_ages: vec![1, 2, 3],
            stochastic: false,
            continuous_sampling: false,
            fixed_egg_count: false,
            has_sex_chromosomes: false,
            extreme_speed_mode: 0,
            ztype_names: vec!["r0|r0".into(), "r0|r1".into()],
            gtype_names: vec!["r0".into(), "r1".into()],
            female_only_by_sex_chrom: vec![false, false],
            male_only_by_sex_chrom: vec![false, false],
            initial_individual_count: vec![0.0; 2 * n_ages * z],
            initial_sperm_storage: vec![],
            n_demes: 1,
            migration_indptr: vec![0; n_demes_ptrs(1)],
            migration_dest_idx: vec![],
            migration_weights: vec![],
        };
        let params = Params {
            n_demes: 1,
            carrying_capacity: vec![400.0],
            eggs_per_female: vec![30.0],
            sex_ratio: vec![0.5],
            sperm_displacement_rate: vec![0.1],
            low_density_growth_rate: vec![2.0],
            growth_mode: vec![2],
            external_expected_eggs: vec![-1.0],
            survival_rates: vec![0.9, 0.8, 0.7, 0.6, 0.85, 0.75, 0.65, 0.55],
            mating_rates: vec![0.0, 0.9, 0.8, 0.7, 0.0, 0.9, 0.8, 0.7],
            reproduction_rates: vec![0.0, 0.8, 0.7, 0.6],
            fertility: vec![0.0, 1.0, 0.9, 0.8],
            competition_weights: vec![1.0, 0.8, 0.7, 0.6],
            equilibrium_distribution: vec![],
            migration_rate: vec![],
            custom_slots: HashMap::new(),
        };
        let genetics = crate::contract::TensorSet {
            viability_fitness: vec![1.0; 2 * n_ages * z],
            fecundity_fitness: vec![1.0; 2 * z],
            sexual_selection_fitness: vec![1.0; z * z],
            zygote_viability_fitness: vec![1.0; 2 * z],
            offspring_tensor: vec![0.0; z * z * z],
            meiosis_map: vec![0.0; 2 * z * z],
            female_ztype_compatibility: vec![0.5, 0.5],
            male_ztype_compatibility: vec![0.5, 0.5],
        };
        (bp, params, genetics)
    }

    fn n_demes_ptrs(n: usize) -> usize {
        n + 1
    }

    /// Derived-distribution fixture must match the Python reference
    /// bit-for-bit (values computed from
    /// ``compute_equilibrium_metrics`` on the identical inputs).
    #[test]
    fn derived_distribution_matches_python_reference() {
        let (bp, params, _) = fixture();
        let (comp, surv) = equilibrium_metrics(&bp, &params, 0);
        assert_eq!(comp, 9436.8);
        assert_eq!(surv, 0.04844257133168629);
    }

    /// Declared distribution + external egg override must match the
    /// Python reference bit-for-bit.
    #[test]
    fn declared_distribution_and_external_eggs_match_python_reference() {
        let (bp, mut params, _) = fixture();
        params.equilibrium_distribution = vec![0.0, 200.0, 150.0, 100.0, 0.0, 200.0, 150.0, 100.0];
        params.external_expected_eggs = vec![5000.0];
        let (comp, surv) = equilibrium_metrics(&bp, &params, 0);
        assert_eq!(comp, 9075.0);
        assert_eq!(surv, 0.09142857142857143);
    }

    /// With no reproduction and no external override, the survival rate
    /// degenerates to 1.0 (guard branch).
    #[test]
    fn zero_egg_production_yields_unit_survival() {
        let (bp, mut params, _) = fixture();
        params.reproduction_rates = vec![0.0; 4];
        params.external_expected_eggs = vec![-1.0];
        let (_, surv) = equilibrium_metrics(&bp, &params, 0);
        assert_eq!(surv, 1.0);
    }

    /// The declared distribution path reproduces the exact multiply-add
    /// order of the Python reference for the competition strength:
    /// eggs (age-0 production) times weight[0], then each juvenile age's
    /// total times its weight.
    #[test]
    fn competition_strength_sums_juvenile_mass() {
        let (bp, mut params, _) = fixture();
        params.equilibrium_distribution = vec![0.0, 200.0, 150.0, 100.0, 0.0, 200.0, 150.0, 100.0];
        // produced_age_0 = 200*0.8*1.0*30 + 150*0.7*0.9*30 + 100*0.6*0.8*30.
        let produced = 200.0_f64 * 0.8 * 1.0 * 30.0
            + 150.0_f64 * 0.7 * 0.9 * 30.0
            + 100.0_f64 * 0.6 * 0.8 * 30.0;
        // new_adult_age = 1: no juvenile ages beyond age 0 contribute.
        let expected = produced * 1.0;
        let (comp, _) = equilibrium_metrics(&bp, &params, 0);
        assert_eq!(comp, expected);
    }

    /// A multi-deme column set yields per-deme metrics: rewriting deme 1's
    /// K changes only deme 1's competition strength.
    #[test]
    fn deme_columns_feed_per_deme_metrics() {
        let (bp, mut params, _) = fixture();
        params.n_demes = 2;
        // Tile every column to two demes (identical demes first).
        params.carrying_capacity = vec![400.0, 400.0];
        params.eggs_per_female = vec![30.0, 30.0];
        params.sex_ratio = vec![0.5, 0.5];
        params.external_expected_eggs = vec![-1.0, -1.0];
        params.reproduction_rates = Params::tile_vec(&params.reproduction_rates, 2);
        params.fertility = Params::tile_vec(&params.fertility, 2);
        params.competition_weights = Params::tile_vec(&params.competition_weights, 2);
        params.survival_rates = Params::tile_vec(&params.survival_rates, 2);
        let (comp0, _) = equilibrium_metrics(&bp, &params, 0);
        let (comp1, _) = equilibrium_metrics(&bp, &params, 1);
        assert_eq!(comp0, comp1);
        // Deme 1 gets a different K; deme 0 must be unaffected.
        params.carrying_capacity[1] = 50.0;
        let (comp0_after, _) = equilibrium_metrics(&bp, &params, 0);
        let (comp1_after, _) = equilibrium_metrics(&bp, &params, 1);
        assert_eq!(comp0_after, comp0);
        assert_ne!(comp1_after, comp1);
    }
}
