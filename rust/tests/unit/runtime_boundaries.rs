//! Boundary rejection must happen before state or persistent RNG changes.
use std::collections::HashMap;

use crate::hooks::interpreter::HookProgram;
use crate::kernels::rng::new_rng;
use crate::kernels::spatial;
use crate::model::blueprint::Blueprint;
use crate::model::ecology::EcologyParams;
use crate::model::genetics::GeneticsTensors;

/// Minimal consistent pair (2 sexes, 2 ages, 2 ztypes, 2 gtypes).
fn fixture() -> (Blueprint, EcologyParams, GeneticsTensors) {
    let bp = Blueprint {
        n_sexes: 2,
        n_ages: 2,
        n_ztypes: 2,
        n_gtypes: 2,
        n_glabs: 1,
        new_adult_age: 1,
        adult_ages: vec![1],
        stochastic: false,
        continuous_sampling: false,
        fixed_egg_count: false,
        has_sex_chromosomes: false,
        extreme_speed_mode: 0,
        ztype_names: vec!["A|A".into(), "A|B".into()],
        gtype_names: vec!["A".into(), "B".into()],
        female_only_by_sex_chrom: vec![false, false],
        male_only_by_sex_chrom: vec![false, false],
        initial_individual_count: vec![0.0; 2 * 2 * 2],
        initial_sperm_storage: vec![],
        n_demes: 1,
        migration_indptr: vec![0, 0],
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
        survival_rates: vec![0.9, 0.8, 0.85, 0.75],
        mating_rates: vec![0.9, 0.9, 0.8, 0.8],
        reproduction_rates: vec![0.0, 0.8],
        fertility: vec![0.0, 1.0],
        competition_weights: vec![1.0, 0.8],
        equilibrium_distribution: vec![],
        equilibrium_declared: vec![false],
        migration_rate: vec![],
        custom_slots: vec![HashMap::new()],
    };
    let genetics = GeneticsTensors {
        viability_fitness: vec![1.0; 8],
        fecundity_fitness: vec![1.0; 4],
        sexual_selection_fitness: vec![1.0; 4],
        zygote_viability_fitness: vec![1.0; 4],
        offspring_tensor: vec![0.0; 8],
        meiosis_map: vec![0.0; 8],
        female_ztype_compatibility: vec![0.5, 0.5],
        male_ztype_compatibility: vec![0.5, 0.5],
    };
    (bp, params, genetics)
}

#[test]
fn scheduler_rejects_mismatched_stacked_state_before_calling_any_deme() {
    let hooks = HookProgram::default();
    let mut rngs = vec![new_rng(7)];
    let words = rngs[0].state_words();
    let mut ind = vec![11.0];
    let mut sperm = vec![13.0];
    let result = spatial::schedule_deme_ticks(
        &hooks,
        &mut rngs,
        &mut ind,
        &mut sperm,
        &mut [0.0; crate::hooks::interpreter::N_ECO_PARAMS],
        2,
        1,
        1,
        &mut vec![],
        |_, _, _, _, _| panic!("invalid shape must never reach the numerical kernel"),
    );
    assert!(result
        .unwrap_err()
        .contains("stacked state length mismatch"));
    assert_eq!(ind, vec![11.0]);
    assert_eq!(sperm, vec![13.0]);
    assert_eq!(rngs[0].state_words(), words);
}

#[test]
fn migration_rejects_missing_rng_stream_without_consuming_existing_stream() {
    let mut rngs = vec![new_rng(7)];
    let words = rngs[0].state_words();
    let individuals = vec![10.0, 20.0, 30.0, 40.0];
    let result = spatial::migrate_csr_stochastic_rngs(
        &mut rngs,
        &individuals,
        &[0.0; 2],
        &[0, 1, 2],
        &[1, 0],
        &[1.0; 2],
        &[0.5; 4],
        false,
        2,
        1,
        1,
    );
    assert!(result.unwrap_err().contains("per-deme RNG streams"));
    assert_eq!(rngs[0].state_words(), words);
    assert_eq!(individuals, vec![10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn spatial_kernels_reject_inconsistent_deme_metadata_before_mutation() {
    let (bp, mut params, genetics) = fixture();
    let mut rngs = vec![new_rng(7)];
    let words = rngs[0].state_words();
    let mut ind = vec![11.0; 8];
    let mut sperm = vec![13.0; 8];
    let mut eco = [0.0; crate::hooks::interpreter::N_ECO_PARAMS];
    let hooks = HookProgram::default();
    let mut journal = vec![];
    let variants = [genetics];
    params.n_demes = 2;
    let age_result = spatial::run_spatial_tick_heterogeneous(
        &hooks,
        &mut rngs,
        &mut ind,
        &mut sperm,
        0,
        &mut eco,
        &bp,
        &params,
        &variants,
        &[0],
        &mut journal,
    );
    assert!(age_result
        .unwrap_err()
        .contains("ecology columns, variant ids, and RNG streams"));
    let discrete_result = spatial::run_spatial_tick_discrete(
        &hooks,
        &mut rngs,
        &mut ind,
        0,
        &mut eco,
        &bp,
        &params,
        &variants,
        &[0],
        &mut journal,
    );
    assert!(discrete_result
        .unwrap_err()
        .contains("ecology columns, variant ids, and RNG streams"));
    let empty_result = spatial::run_spatial_tick_discrete(
        &hooks,
        &mut rngs,
        &mut ind,
        0,
        &mut eco,
        &bp,
        &params,
        &variants,
        &[],
        &mut journal,
    );
    assert!(empty_result.unwrap_err().contains("at least one deme"));
    assert_eq!(ind, vec![11.0; 8]);
    assert_eq!(sperm, vec![13.0; 8]);
    assert!(journal.is_empty());
    assert_eq!(rngs[0].state_words(), words);
}
