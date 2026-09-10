use super::*;
use std::collections::HashMap;

/// Build a small blueprint/params pair (2 sexes, 4 ages, 2 ztypes).
fn fixture() -> (
    Blueprint,
    EcologyParams,
    crate::model::genetics::GeneticsTensors,
) {
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
    let params = EcologyParams {
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
        equilibrium_declared: vec![false],
        migration_rate: vec![],
        custom_slots: vec![HashMap::new()],
    };
    let genetics = crate::model::genetics::GeneticsTensors {
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
    params.equilibrium_declared = vec![true];
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
    params.equilibrium_declared = vec![true];
    // produced_age_0 = 200*0.8*1.0*30 + 150*0.7*0.9*30 + 100*0.6*0.8*30.
    let produced =
        200.0_f64 * 0.8 * 1.0 * 30.0 + 150.0_f64 * 0.7 * 0.9 * 30.0 + 100.0_f64 * 0.6 * 0.8 * 30.0;
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
    params.equilibrium_declared = vec![false; 2];
    // Tile every column to two demes (identical demes first).
    params.carrying_capacity = vec![400.0, 400.0];
    params.eggs_per_female = vec![30.0, 30.0];
    params.sex_ratio = vec![0.5, 0.5];
    params.external_expected_eggs = vec![-1.0, -1.0];
    params.reproduction_rates = EcologyParams::tile_vec(&params.reproduction_rates, 2);
    params.fertility = EcologyParams::tile_vec(&params.fertility, 2);
    params.competition_weights = EcologyParams::tile_vec(&params.competition_weights, 2);
    params.survival_rates = EcologyParams::tile_vec(&params.survival_rates, 2);
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
