use super::*;

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

/// Invalid internal custom-column cardinality is rejected before assembly.
#[test]
fn malformed_custom_column_is_rejected() {
    let (bp, mut params, _) = fixture();
    params.custom_slots.clear();
    assert!(params.validate(&bp).is_err());
}

/// An absent migration column means zero migration for untouched demes.
#[test]
fn adding_deme_migration_preserves_zero_migration_for_its_neighbor() {
    let (_, source, _) = fixture();
    let mut column = source.clone();
    column.n_demes = 2;
    column.equilibrium_declared = vec![false; 2];
    for values in [
        &mut column.carrying_capacity,
        &mut column.eggs_per_female,
        &mut column.sex_ratio,
        &mut column.sperm_displacement_rate,
        &mut column.low_density_growth_rate,
        &mut column.external_expected_eggs,
        &mut column.survival_rates,
        &mut column.mating_rates,
        &mut column.reproduction_rates,
        &mut column.fertility,
        &mut column.competition_weights,
    ] {
        values.extend_from_within(..);
    }
    column.growth_mode = vec![2, 2];
    column.custom_slots = vec![HashMap::new(), HashMap::new()];
    let mut candidate = source;
    candidate.migration_rate = vec![0.25];
    column.replace_deme(1, &candidate);
    assert_eq!(column.migration_rate, vec![0.0, 0.25]);
    assert_eq!(column.single_deme(0).carrying_capacity, vec![400.0]);
    assert_eq!(
        column.single_deme(0).survival_rates,
        vec![0.9, 0.8, 0.85, 0.75]
    );
}

/// An unknown field must fail in the validation pass: the legal entry in
/// the same batch must not be committed (method-level atomicity).
#[test]
fn apply_unknown_field_writes_nothing() {
    let (_, mut params, _) = fixture();
    let writes = HashMap::from([
        ("carrying_capacity".to_string(), 111.0),
        ("no_such_field".to_string(), 1.0),
    ]);
    assert!(params.apply(writes).is_err());
    assert_eq!(params.carrying_capacity[0], 400.0);
}

/// A tensor field routed through the scalar channel must be rejected, and
/// with a single entry there is nothing else to leak a partial write.
#[test]
fn apply_rejects_tensor_field_without_writing() {
    let (_, mut params, _) = fixture();
    let writes = HashMap::from([("survival_rates".to_string(), 0.5)]);
    assert!(params.apply(writes).is_err());
    assert_eq!(params.survival_rates, vec![0.9, 0.8, 0.85, 0.75]);
}

/// Scalars go through their native channels: f64 fields keep their float
/// value, the growth-mode selector goes through the i64 channel.
#[test]
fn apply_writes_scalars_through_native_channels() {
    let (_, mut params, _) = fixture();
    let writes = HashMap::from([
        ("carrying_capacity".to_string(), 42.5),
        ("growth_mode".to_string(), 4.0),
    ]);
    assert!(params.apply(writes).is_ok());
    assert_eq!(params.carrying_capacity[0], 42.5);
    assert_eq!(params.growth_mode[0], 4);
    assert_eq!(params.get_scalar("carrying_capacity").unwrap(), 42.5);
    assert_eq!(params.get_scalar("growth_mode").unwrap(), 4.0);
}

/// Size must be validated before commit: a wrong-size write changes
/// nothing, the correct size commits verbatim.
#[test]
fn tensor_write_rejects_wrong_size_and_preserves_contents() {
    let (bp, mut params, _) = fixture();
    assert!(params
        .tensor_write(&bp, "survival_rates", vec![0.1; 3])
        .is_err());
    assert_eq!(params.survival_rates, vec![0.9, 0.8, 0.85, 0.75]);
    assert!(params
        .tensor_write(&bp, "survival_rates", vec![0.1; 4])
        .is_ok());
    assert_eq!(params.survival_rates, vec![0.1, 0.1, 0.1, 0.1]);
}

/// Scalar fields are unreachable through the tensor channel, and
/// genetics tensors are rejected by the ecology channel.
#[test]
fn tensor_write_rejects_scalar_and_genetics_fields() {
    let (bp, mut params, _) = fixture();
    assert!(params
        .tensor_write(&bp, "carrying_capacity", vec![1.0])
        .is_err());
    assert_eq!(params.carrying_capacity[0], 400.0);
    assert!(params
        .tensor_write(&bp, "viability_fitness", vec![1.0; 8])
        .is_err());
}

/// The derive-mode sentinel of ``equilibrium_distribution``: empty ->
/// empty keeps deriving, empty -> full widens to declared contents, and
/// full -> empty explicitly resumes deriving. Other lengths fail atomically.
#[test]
fn tensor_write_equilibrium_empty_sentinel_semantics() {
    let (bp, mut params, _) = fixture();
    assert!(params
        .tensor_write(&bp, "equilibrium_distribution", vec![])
        .is_ok());
    assert!(params.equilibrium_distribution.is_empty());
    assert!(params
        .tensor_write(&bp, "equilibrium_distribution", vec![1.0, 2.0, 3.0, 4.0])
        .is_ok());
    assert_eq!(params.equilibrium_distribution, vec![1.0, 2.0, 3.0, 4.0]);
    assert!(params
        .tensor_write(&bp, "equilibrium_distribution", vec![1.0; 3])
        .is_err());
    assert_eq!(params.equilibrium_distribution, vec![1.0, 2.0, 3.0, 4.0]);
    assert!(params
        .tensor_write(&bp, "equilibrium_distribution", vec![])
        .is_ok());
    assert!(params.equilibrium_distribution.is_empty());
    assert_eq!(params.equilibrium_declared, vec![false]);
}

/// ``validate`` checks every column against the blueprint-derived size
/// while accepting the empty equilibrium sentinel (derive mode).
#[test]
fn validate_enforces_blueprint_derived_sizes() {
    let (bp, mut params, genetics) = fixture();
    assert!(params.validate(&bp).is_ok());
    assert!(genetics.validate(&bp).is_ok());
    params.survival_rates = vec![0.1; 3]; // expected 1 deme * 2 * a = 4
    assert!(params.validate(&bp).is_err());
    let (bp2, fresh, _) = fixture();
    assert!(fresh.equilibrium_distribution.is_empty());
    assert!(fresh.validate(&bp2).is_ok());
}

/// Read channels reject cross-kind and unknown names.
#[test]
fn get_scalar_rejects_tensor_and_unknown_names() {
    let (_, params, _) = fixture();
    assert!(params.get_scalar("survival_rates").is_err());
    assert!(params.get_scalar("bogus").is_err());
}

/// Multi-deme columns: a per-deme scalar write lands only in the
/// targeted deme's column entry.
#[test]
fn deme_scoped_scalar_writes_are_independent() {
    let (_, mut params, _) = fixture();
    params.n_demes = 3;
    params.equilibrium_declared = vec![false; 3];
    params.custom_slots.resize(3, HashMap::new());
    params.carrying_capacity = vec![400.0, 400.0, 400.0];
    params.growth_mode = vec![2, 2, 2];
    let writes = HashMap::from([("carrying_capacity".to_string(), 50.0)]);
    assert!(params.apply(writes).is_ok());
    assert_eq!(params.carrying_capacity, vec![50.0, 400.0, 400.0]);
    // Direct deme-scoped scalar_ref write.
    match params.scalar_ref("carrying_capacity", 2).unwrap() {
        ScalarRef::F64(target) => *target = 7.0,
        ScalarRef::I64(_) => panic!("expected f64 slot"),
    }
    assert_eq!(params.carrying_capacity, vec![50.0, 400.0, 7.0]);
}

/// Columnized tensor writes at a deme segment leave other demes intact.
#[test]
fn deme_scoped_tensor_segments_are_isolated() {
    let (bp, mut params, _) = fixture();
    params.n_demes = 2;
    params.equilibrium_declared = vec![false; 2];
    params.custom_slots.resize(2, HashMap::new());
    params.survival_rates = vec![0.9, 0.8, 0.85, 0.75, 0.9, 0.8, 0.85, 0.75];
    // Simulate the per-deme pull channel at deme 1: expected segment is
    // 2 * a = 4 elements, committed at offset 4.
    let segment = vec![0.2, 0.3, 0.4, 0.5];
    params
        .write_deme_tensor(&bp, "survival_rates", 1, &segment)
        .unwrap();
    assert_eq!(params.survival_rates[..4], vec![0.9, 0.8, 0.85, 0.75]);
    assert_eq!(params.survival_rates[4..], segment[..]);
    // Whole-column write still validates against n_demes * per_deme.
    assert!(params
        .tensor_write(&bp, "survival_rates", vec![0.1; 4])
        .is_err());
}

/// `from_python` tiles per-deme-0 values into n_demes columns while the
/// migration-rate column is taken as-is.
#[test]
fn tile_semantics_keep_panmictic_bit_identical() {
    // Length-1 columns reproduce the flat pre-columnization layout.
    let (_, params, _) = fixture();
    assert_eq!(params.n_demes, 1);
    assert_eq!(params.survival_rates.len(), 4);
    // Tiling to 3 demes triples the vector; scalars repeat.
    let mut tiled = params.clone();
    tiled.n_demes = 3;
    tiled.equilibrium_declared = vec![false; 3];
    tiled.custom_slots.resize(3, HashMap::new());
    tiled.carrying_capacity = EcologyParams::tile(400.0, 3);
    tiled.survival_rates = EcologyParams::tile_vec(&params.survival_rates, 3);
    assert_eq!(tiled.carrying_capacity, vec![400.0, 400.0, 400.0]);
    assert_eq!(tiled.survival_rates.len(), 12);
    assert_eq!(&tiled.survival_rates[..4], &params.survival_rates[..]);
}

/// Genetics writes through the dedicated channel are size-validated and
/// atomic.
#[test]
fn genetics_tensor_write_validates_sizes() {
    let (bp, _, mut genetics) = fixture();
    assert!(genetics
        .tensor_write(&bp, "viability_fitness", vec![1.0; 7])
        .is_err());
    assert_eq!(genetics.viability_fitness, vec![1.0; 8]);
    assert!(genetics
        .tensor_write(&bp, "viability_fitness", vec![0.5; 8])
        .is_ok());
    assert_eq!(genetics.viability_fitness, vec![0.5; 8]);
    assert!(genetics
        .tensor_write(&bp, "survival_rates", vec![1.0])
        .is_err());
    assert!(GeneticsTensors::expected_len(&bp, "nope").is_err());
}

/// ``single_deme`` is the HB-1 correctness core: the local single-deme copy
/// must be numerically identical to the session's per-deme view — scalar
/// entries, vector segments, and the derived equilibrium metrics — otherwise
/// the local-EcoCtx spatial tick would change demography without any
/// set_param write.  Deme 1 carries deliberately distinct ecology so a
/// mis-cut segment cannot pass.
#[test]
fn single_deme_local_assembly_matches_per_deme_config() {
    let (bp, mut params, _genetics) = fixture();
    // Widen to 3 demes; give every deme distinct ecology columns.
    params.n_demes = 3;
    params.equilibrium_declared = vec![false; 3];
    params.custom_slots.resize(3, HashMap::new());
    params.carrying_capacity = vec![400.0, 650.0, 310.0];
    params.eggs_per_female = vec![30.0, 22.5, 41.0];
    params.sex_ratio = vec![0.5, 0.6, 0.45];
    params.sperm_displacement_rate = vec![0.1, 0.2, 0.05];
    params.low_density_growth_rate = vec![2.0, 3.5, 1.25];
    params.growth_mode = vec![2, 1, 2];
    params.external_expected_eggs = vec![-1.0, 5000.0, -1.0];
    params.survival_rates = EcologyParams::tile_vec(&params.survival_rates, 3);
    params.mating_rates = EcologyParams::tile_vec(&params.mating_rates, 3);
    params.reproduction_rates = EcologyParams::tile_vec(&params.reproduction_rates, 3);
    params.fertility = EcologyParams::tile_vec(&params.fertility, 3);
    params.competition_weights = EcologyParams::tile_vec(&params.competition_weights, 3);
    // Skew deme 1's vector segments so a wrong cut is detectable.
    for i in 4..8 {
        params.survival_rates[i] *= 0.5;
        params.mating_rates[i] *= 0.5;
    }
    for i in 2..4 {
        params.reproduction_rates[i] *= 0.5;
        params.fertility[i] *= 0.5;
        params.competition_weights[i] *= 0.5;
    }

    for deme in 0..3 {
        let local = params.single_deme(deme);
        assert_eq!(local.n_demes, 1, "deme {deme}");
        // Key scalars: the local column entry must equal the session's
        // per-deme entry (the lifecycle kernels read the local copy at
        // deme 0, or the session columns at the deme itself).
        assert_eq!(local.carrying_capacity[0], params.carrying_capacity[deme]);
        assert_eq!(local.eggs_per_female[0], params.eggs_per_female[deme]);
        assert_eq!(local.sex_ratio[0], params.sex_ratio[deme]);
        assert_eq!(
            local.sperm_displacement_rate[0],
            params.sperm_displacement_rate[deme]
        );
        assert_eq!(
            local.low_density_growth_rate[0],
            params.low_density_growth_rate[deme]
        );
        assert_eq!(local.growth_mode[0], params.growth_mode[deme]);
        assert_eq!(
            local.external_expected_eggs[0],
            params.external_expected_eggs[deme]
        );
        assert_eq!(
            local.equilibrium_declared[0],
            params.equilibrium_declared[deme]
        );
        // The derived equilibrium metrics — which re-read every ecology
        // input — must be identical whether computed from the local column
        // or the session's per-deme segment.
        let from_local = crate::kernels::equilibrium::equilibrium_metrics(&bp, &local, 0);
        let from_session = crate::kernels::equilibrium::equilibrium_metrics(&bp, &params, deme);
        assert_eq!(
            from_local.0, from_session.0,
            "deme {deme}: equilibrium metrics re-read the local column"
        );
        assert_eq!(from_local.1, from_session.1, "deme {deme}");
        // Vector segments: each deme's (2, A) / (A,) extent is cut exactly.
        let a = bp.n_ages;
        assert_eq!(
            local.survival_rates,
            params.survival_rates[deme * 2 * a..(deme + 1) * 2 * a]
        );
        assert_eq!(
            local.mating_rates,
            params.mating_rates[deme * 2 * a..(deme + 1) * 2 * a]
        );
        assert_eq!(
            local.reproduction_rates,
            params.reproduction_rates[deme * a..(deme + 1) * a]
        );
        assert_eq!(local.fertility, params.fertility[deme * a..(deme + 1) * a]);
        assert_eq!(
            local.competition_weights,
            params.competition_weights[deme * a..(deme + 1) * a]
        );
    }

    // Sentinel columns survive the cut: derive-mode equilibrium and an
    // undeclared migration column stay empty; a declared migration
    // column yields exactly the deme's (2, A) segment.
    assert!(params.single_deme(1).equilibrium_distribution.is_empty());
    assert!(params.single_deme(1).migration_rate.is_empty());
    params.migration_rate = EcologyParams::tile_vec(&[0.01, 0.02, 0.01, 0.02], 3);
    assert_eq!(
        params.single_deme(2).migration_rate,
        vec![0.01, 0.02, 0.01, 0.02]
    );
}
