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
        discrete_generation: false,
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

/// Derived distribution with sex-specific age-0 survival: the age-1 split
/// follows the *surviving* sex ratio, so the calibrated equilibrium stays at
/// K.  The fixture's female age-0 survival (0.9) differs from the male's
/// (0.85), so these constants are the corrected reference rather than the
/// retired Python reference (that parity claim now holds only for
/// equal-survival inputs — see the sibling test).  Values replicated
/// independently from the documented operation order:
///   female share = 0.5 * 0.9 / (0.5 * 0.9 + 0.5 * 0.85) -> 205.714...
///   produced_age_0 = 205.714...*0.8*1.0*30 + 164.571...*0.7*0.9*30
///                  + 115.2*0.6*0.8*30
///   s* = 400 / (produced_age_0 * (0.5 * 0.9 + 0.5 * 0.85))
#[test]
fn derived_distribution_uses_the_surviving_sex_ratio() {
    let (bp, params, _) = fixture();
    let (comp, surv) = equilibrium_metrics(&bp, &params, 0);
    assert_eq!(comp, 9706.42285714286);
    assert_eq!(surv, 0.04709694435025054);
}

/// Equal age-0 survival keeps the historical age-1 split bit-for-bit: the
/// calibration must not move for any model whose sexes survive equally.
/// sex_ratio = 0.1 with both survivals at 0.3 is chosen because the
/// survival-weighted form alone rounds the female entry to 39.99999999999999
/// instead of 40.0, shifting both constants in the last ulp — so this test
/// fails if the equal-survival branch is dropped.
#[test]
fn equal_sex_survival_keeps_the_historical_age1_split() {
    let (bp, mut params, _) = fixture();
    params.sex_ratio = vec![0.1];
    params.survival_rates = vec![0.3; 8];
    let (comp, surv) = equilibrium_metrics(&bp, &params, 0);
    assert_eq!(comp, 1238.6399999999999);
    assert_eq!(surv, 1.076449439169842);
}

/// With `new_adult_age = 2` the age-1 juveniles of *both* sexes enter the
/// competition strength, so the male half of the split is load-bearing: the
/// same fixture with `sex_ratio = 0.3` is checked here, where the surviving
/// male share (0.6878612716763005 of K) is 4.86 head below the raw offspring
/// share (0.7 of K) and C* would move to 3218.5341040462435 — off the
/// composition the model reaches, which is what moves the calibrated
/// equilibrium away from K.  Values replicated independently from the
/// documented operation order:
///   female share = 0.3 * 0.9 / (0.3 * 0.9 + 0.7 * 0.85) -> 124.855...
///   male share -> 275.144... (age-1 row sums to K = 400)
///   female age-2/3 = 124.855...*0.8 and *0.8*0.7
///   produced_age_0 = 99.884...*0.7*0.9*30 + 69.919...*0.6*0.8*30
///                  -> 2894.649710982659
///   C* = produced_age_0*1.0 + (124.855... + 275.144...)*0.8
///   s* = 400 / (produced_age_0 * (0.3 * 0.9 + 0.7 * 0.85))
#[test]
fn derived_distribution_puts_the_surviving_male_share_into_c_star() {
    let (mut bp, mut params, _) = fixture();
    bp.new_adult_age = 2;
    params.sex_ratio = vec![0.3];
    let (comp, surv) = equilibrium_metrics(&bp, &params, 0);
    assert_eq!(comp, 3214.6497109826596);
    assert_eq!(surv, 0.15975257521151237);
}

/// Degenerate reference: with no female offspring at all (sex_ratio = 0) the
/// surviving mass is empty, so the split falls back to the offspring ratio
/// and the survival-rate guard yields 1.0.
#[test]
fn degenerate_zero_surviving_mass_falls_back_to_the_offspring_split() {
    let (bp, mut params, _) = fixture();
    params.sex_ratio = vec![0.0];
    params.survival_rates = vec![0.5, 0.8, 0.7, 0.6, 0.0, 0.75, 0.65, 0.55];
    let (comp, surv) = equilibrium_metrics(&bp, &params, 0);
    assert_eq!(comp, 0.0);
    assert_eq!(surv, 1.0);
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

/// Sex-chromosome species determine offspring sex from the chromosomes, and
/// the documented contract calls `sex_ratio` ignored there.  The calibration
/// must follow the genetic split, so every stored ratio reproduces the
/// balanced 0.5 metrics; feeding the parameter through moved the calibrated
/// equilibrium by tens of percent (R4-05).
#[test]
fn sex_chromosome_models_ignore_the_sex_ratio_parameter() {
    let (mut bp, mut params, _) = fixture();
    bp.has_sex_chromosomes = true;
    bp.female_only_by_sex_chrom = vec![true, false];
    bp.male_only_by_sex_chrom = vec![false, true];
    params.sex_ratio = vec![0.5];
    let (comp_balanced, surv_balanced) = equilibrium_metrics(&bp, &params, 0);
    // 0.0 and 1.0 also exercise the degenerate surviving-mass guard: with the
    // parameter ignored, neither is degenerate.
    for sex_ratio in [0.0, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0] {
        params.sex_ratio = vec![sex_ratio];
        let (comp, surv) = equilibrium_metrics(&bp, &params, 0);
        assert_eq!(
            comp, comp_balanced,
            "derived comp moved at sex_ratio={sex_ratio}"
        );
        assert_eq!(
            surv, surv_balanced,
            "derived s* moved at sex_ratio={sex_ratio}"
        );
    }
    // The declared branch consumes the same ratio through `s_0_avg`; the
    // declared composition itself is untouched, so C* stays put while s*
    // must too.
    params.equilibrium_distribution = vec![
        0.0,
        205.71428571428572,
        150.0,
        100.0,
        0.0,
        194.28571428571428,
        150.0,
        100.0,
    ];
    params.equilibrium_declared = vec![true];
    params.sex_ratio = vec![0.5];
    let (comp_declared, surv_declared) = equilibrium_metrics(&bp, &params, 0);
    for sex_ratio in [0.2, 0.3, 0.8] {
        params.sex_ratio = vec![sex_ratio];
        let (comp, surv) = equilibrium_metrics(&bp, &params, 0);
        assert_eq!(
            comp, comp_declared,
            "declared comp moved at sex_ratio={sex_ratio}"
        );
        assert_eq!(
            surv, surv_declared,
            "declared s* moved at sex_ratio={sex_ratio}"
        );
    }
}

/// The per-age fertility weight must be read the way the owning tick reads it:
/// discrete generations have no age-dependent fertility (implicit 1.0), while
/// the age-structured tick clamps the stored weight to [0, 1].  Reading the
/// stored value verbatim let a raw tensor write move the calibrated
/// equilibrium by exactly the written factor (R4-11).
#[test]
fn fertility_weight_follows_the_owning_engine() {
    let (mut bp, mut params, _) = fixture();
    // Discrete: the stored vector is inert, including out-of-domain values.
    bp.discrete_generation = true;
    params.fertility = vec![0.0, 1.0, 1.0, 1.0];
    let (comp_unit, surv_unit) = equilibrium_metrics(&bp, &params, 0);
    params.fertility = vec![0.0, 0.5, 2.0, 0.0];
    let (comp_inert, surv_inert) = equilibrium_metrics(&bp, &params, 0);
    assert_eq!(comp_inert, comp_unit);
    assert_eq!(surv_inert, surv_unit);
    // Age-structured with unit weights: clamping leaves them untouched, so the
    // two engines agree — the discrete rule is exactly "unit fertility".
    bp.discrete_generation = false;
    params.fertility = vec![0.0, 1.0, 1.0, 1.0];
    let (comp_age_unit, surv_age_unit) = equilibrium_metrics(&bp, &params, 0);
    assert_eq!(comp_age_unit, comp_unit);
    assert_eq!(surv_age_unit, surv_unit);
    // Age-structured clamps out-of-domain weights to 1 rather than reading
    // them raw, so they match the unit run and differ from in-domain weights.
    params.fertility = vec![0.0, 1.0, 2.0, 3.0];
    let (comp_clamped, surv_clamped) = equilibrium_metrics(&bp, &params, 0);
    assert_eq!(comp_clamped, comp_age_unit);
    assert_eq!(surv_clamped, surv_age_unit);
    params.fertility = vec![0.0, 1.0, 0.9, 0.8];
    let (comp_default, surv_default) = equilibrium_metrics(&bp, &params, 0);
    assert_ne!(comp_default, comp_clamped);
    assert_ne!(surv_default, surv_clamped);
}
