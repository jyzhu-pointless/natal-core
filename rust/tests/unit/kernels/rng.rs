use super::*;

/// Reference values produced by rand 0.10 SmallRng (Xoshiro256PlusPlus)
/// seeded with seed_from_u64(42) prove stream parity with the old type.
#[test]
fn stream_matches_rand_smallrng() {
    let mut rng = new_rng(42);
    let words: Vec<u64> = (0..8).map(|_| rng.try_next_u64().unwrap()).collect();
    // Generated with: SmallRng::seed_from_u64(42) in rand 0.10.2.
    assert_eq!(
        words,
        vec![
            15_021_278_609_987_233_951,
            5_881_210_131_331_364_753,
            18_149_643_915_985_481_100,
            12_933_668_939_759_105_464,
            14_637_574_242_682_825_331,
            10_848_501_901_068_131_965,
            2_312_344_417_745_909_078,
            11_162_538_943_635_311_430,
        ]
    );
}

#[test]
fn state_words_round_trip_continues_stream() {
    let mut rng = new_rng(7);
    let _ = rng.try_next_u64().unwrap();
    let checkpoint = rng.state_words();
    let expected: Vec<u64> = (0..4).map(|_| rng.try_next_u64().unwrap()).collect();
    let mut restored = SessionRng::from_state_words(checkpoint);
    let resumed: Vec<u64> = (0..4).map(|_| restored.try_next_u64().unwrap()).collect();
    assert_eq!(resumed, expected);
}

#[test]
fn stream_seed_xor_differs_per_deme() {
    assert_eq!(stream_seed(100, 0), 100);
    assert_eq!(stream_seed(100, 3), 103);
    assert_ne!(stream_seed(100, 1), stream_seed(100, 2));
}

// ════════════════════════════════════════════════════════════════════
// Deterministic boundary guards of the sampling helpers.
// ════════════════════════════════════════════════════════════════════

/// Boundary fast paths must return exact values and never consume the
/// stream (they short-circuit before any sampling).
#[test]
fn binomial_boundary_fast_paths() {
    let mut rng = new_rng(1);
    assert_eq!(binomial(&mut rng, 0, 0.5), 0.0);
    assert_eq!(binomial(&mut rng, -3, 0.5), 0.0);
    assert_eq!(binomial(&mut rng, 10, 0.0), 0.0);
    assert_eq!(binomial(&mut rng, 10, 1.0), 10.0);
}

/// The sequential-conditional-binomial scheme must conserve the trial
/// total exactly: the last category receives whatever trials remain.
#[test]
fn multinomial_preserves_total_exactly() {
    let mut rng = new_rng(11);
    let p = [0.2, 0.3, 0.5];
    let mut out = [0.0_f64; 3];
    for _ in 0..20 {
        multinomial(&mut rng, 17, &p, &mut out);
        let total: f64 = out.iter().sum();
        assert_eq!(total, 17.0, "multinomial must conserve the trial total");
        assert!(out.iter().all(|&v| (0.0..=17.0).contains(&v)));
    }
    // A single fully-concentrated category receives everything.
    let mut two = [7.0_f64; 2];
    multinomial(&mut rng, 9, &[1.0], &mut two);
    assert_eq!(two[0], 9.0);
    assert_eq!(two[1], 0.0);
    // Degenerate inputs return all zeros.
    let mut zeros = [7.0_f64; 2];
    multinomial(&mut rng, 0, &p, &mut zeros);
    assert_eq!(zeros, [0.0, 0.0]);
}

/// Extreme distribution parameters must hit the deterministic guards
/// instead of the samplers: zero -> 0, beyond the resolution limit ->
/// the mean/shape itself.
#[test]
fn poisson_and_gamma_extreme_parameter_guards() {
    let mut rng = new_rng(3);
    assert_eq!(poisson(&mut rng, 0.0), 0.0);
    assert_eq!(poisson(&mut rng, -1.0), 0.0);
    assert_eq!(poisson(&mut rng, 1e300), 1e300);
    assert_eq!(gamma(&mut rng, 0.0), 0.0);
    assert_eq!(gamma(&mut rng, 1e300), 1e300);
}

/// Lambdas between rand_distr's own ceiling and the resolution limit
/// must return the mean, not trip ``Poisson::new``'s ShapeTooLarge
/// rejection: the sampler rejects lambdas above
/// ``Poisson::MAX_LAMBDA``.
#[test]
fn poisson_lambda_above_distr_ceiling_returns_mean() {
    let mut rng = new_rng(9);
    assert_eq!(poisson(&mut rng, 2.0e19), 2.0e19);
    assert_eq!(
        poisson(&mut rng, Poisson::<f64>::MAX_LAMBDA),
        Poisson::<f64>::MAX_LAMBDA
    );
    // Just below the ceiling the sampler must still draw from the
    // distribution: identical samples would mean a guard was lowered
    // into the window and the huge-lambda mean shortcut moved down.
    let samples: [f64; 8] = std::array::from_fn(|_| poisson(&mut rng, 1.8e19));
    assert!(samples.iter().all(|sample| sample.is_finite()));
    assert!(
        samples.iter().any(|&sample| sample != 1.8e19),
        "1.8e19 is below MAX_LAMBDA and must be sampled, not returned as the mean"
    );
}

/// Below the small-n threshold the continuous multinomial is the exact
/// scaled probability vector (no sampling, no normalization drift).
#[test]
fn continuous_multinomial_small_n_uses_exact_proportions() {
    let mut rng = new_rng(5);
    let p = [0.25, 0.75];
    let mut out = [0.0_f64; 2];
    continuous_multinomial(&mut rng, 0.5, &p, &mut out);
    assert_eq!(out[0], 0.125);
    assert_eq!(out[1], 0.375);
}
