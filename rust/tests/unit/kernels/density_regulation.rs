use super::*;
use crate::kernels::rng::clamp01;

/// r used across the property tests (exercises all regimes: BH stable,
/// linear marginally oscillating, ricker > e oscillating).
const R: f64 = 3.0;

#[test]
fn fixed_point_exact_at_one() {
    assert_eq!(g_fixed(1.0), 1.0);
    assert_eq!(g_linear(1.0, R), 1.0);
    assert_eq!(g_beverton_holt(1.0, R), 1.0);
    assert_eq!(g_ricker(1.0, R), 1.0);
}

#[test]
fn compensatory_family_has_g0_equals_r() {
    assert_eq!(g_fixed(0.0), 1.0); // non-compensatory: capped at 1
    assert_eq!(g_linear(0.0, R), R);
    assert_eq!(g_beverton_holt(0.0, R), R);
    assert_eq!(g_ricker(0.0, R), R);
}

#[test]
fn monotone_non_increasing() {
    let xs: Vec<f64> = (0..=300).map(|i| i as f64 * 0.01).collect();
    type CurvePair<'a> = (&'a str, Box<dyn Fn(f64) -> f64>);
    let curves: [CurvePair; 4] = [
        ("fixed", Box::new(g_fixed)),
        ("linear", Box::new(|x| g_linear(x, R))),
        ("bh", Box::new(|x| g_beverton_holt(x, R))),
        ("ricker", Box::new(|x| g_ricker(x, R))),
    ];
    for (name, g) in &curves {
        for w in xs.windows(2) {
            assert!(
                g(w[1]) <= g(w[0]) + 1e-12,
                "{name} increased between {} and {}",
                w[0],
                w[1]
            );
        }
    }
}

#[test]
fn non_negative_everywhere() {
    for i in 0..=3_000 {
        let x = i as f64 * 0.005;
        assert!(g_fixed(x) >= 0.0);
        assert!(g_linear(x, R) >= 0.0);
        assert!(g_beverton_holt(x, R) > 0.0);
        assert!(g_ricker(x, R) > 0.0);
    }
}

#[test]
fn linear_clamps_to_zero_past_break_even() {
    assert_eq!(g_linear(10.0, R), 0.0);
    assert_eq!(g_linear(2.5, R), 0.0); // break-even at r/(r-1) = 1.5
    assert!(g_linear(1.4, R) > 0.0);
}

#[test]
fn ricker_overcompensates_relative_to_bh_past_equilibrium() {
    // For x > 1, Ricker decays faster than BH when r > 1.
    for i in 11..100 {
        let x = i as f64 / 10.0;
        assert!(g_ricker(x, R) < g_beverton_holt(x, R));
    }
}

#[test]
fn contract_checker_accepts_builtins_and_rejects_broken() {
    check_curve_contract(&|x| g_fixed(x)).unwrap();
    check_curve_contract(&|x| g_linear(x, R)).unwrap();
    check_curve_contract(&|x| g_beverton_holt(x, R)).unwrap();
    check_curve_contract(&|x| g_ricker(x, R)).unwrap();
    // Broken curve: shifted fixed point.
    assert!(check_curve_contract(&|x| g_beverton_holt(x, R) * 1.01).is_err());
    // Broken curve: increasing.
    assert!(check_curve_contract(&|x| x).is_err());
    // Broken curve: negative arm.
    assert!(check_curve_contract(&|x| -x).is_err());
}

#[test]
fn dispatch_matches_direct_calls_and_rejects_unknown() {
    let x = 1.7;
    assert_eq!(scaling_factor(1, x, R).unwrap(), g_fixed(x));
    assert_eq!(scaling_factor(2, x, R).unwrap(), g_linear(x, R));
    assert_eq!(scaling_factor(3, x, R).unwrap(), g_beverton_holt(x, R));
    assert_eq!(scaling_factor(4, x, R).unwrap(), g_ricker(x, R));
    assert_eq!(scaling_factor(0, x, R).unwrap(), 1.0);
    assert!(scaling_factor(9, x, R).is_err());
}

#[test]
fn competition_ratio_guards_zero_denominator() {
    assert_eq!(competition_ratio(5.0, 2.0), 2.5);
    assert_eq!(competition_ratio(5.0, 0.0), 1.0);
}

/// regulation_scaling must reproduce the Python reference expressions
/// bit-for-bit (identical operation order, no double rounding).
#[test]
fn regulation_scaling_matches_python_reference_exactly() {
    // Python fixed: min(1.0, K / total) — single division, not 1/(t/K).
    let (k, total) = (1234.5678_f64, 987.654_321_f64);
    let python_fixed = 1.0_f64.min(k / total);
    assert_eq!(
        regulation_scaling(1, total, k, 3.0, 0.8).unwrap(),
        python_fixed
    );
    // Python logistic: max(0.0, -ratio * (r - 1) + r) * s_star.
    let (actual, expected_c, r, s) = (77.7_f64, 210.3_f64, std::f64::consts::E, 0.7311_f64);
    let ratio = if expected_c > 0.0 {
        actual / expected_c
    } else {
        1.0
    };
    let python_logistic = 0.0_f64.max(-ratio * (r - 1.0) + r) * s;
    assert_eq!(
        regulation_scaling(2, actual, expected_c, r, s).unwrap(),
        python_logistic
    );
    // Python beverton-holt: r / (ratio * (r - 1) + 1) * s_star.
    let python_bh = r / (ratio * (r - 1.0) + 1.0) * s;
    assert_eq!(
        regulation_scaling(3, actual, expected_c, r, s).unwrap(),
        python_bh
    );
    // Zero-equilibrium guards fall back to ratio 1.0 like Python.
    assert_eq!(
        regulation_scaling(2, actual, 0.0, r, s).unwrap(),
        0.0_f64.max(-1.0 * (r - 1.0) + r) * s
    );
    // Ricker has no Python precedent; check the closed form and dispatch.
    assert_eq!(
        regulation_scaling(4, actual, expected_c, r, s).unwrap(),
        g_ricker(ratio, r) * s
    );
    assert!(regulation_scaling(9, 1.0, 1.0, 1.0, 1.0).is_err());
}

#[test]
fn fixed_is_flat_below_equilibrium_then_concave_up() {
    assert_eq!(g_fixed(0.5), 1.0);
    assert_eq!(g_fixed(2.0), 0.5);
}

// ════════════════════════════════════════════════════════════════════
// Recruitment dynamics induced by each curve.
//
// Applying the curve at the current competition ratio induces the
// discrete-time recruitment map x_{t+1} = x_t * g(x_t) (for Ricker this
// is the classic map x*exp(a(1-x)) with a = ln r; for Beverton-Holt the
// classic r*x/(1+(r-1)x)).  The two maps must be dynamical opposites:
// Ricker overcompensates (oscillates across x=1), Beverton-Holt never
// crosses equilibrium.
// ════════════════════════════════════════════════════════════════════

/// One step of the Ricker recruitment map.
fn ricker_map(x: f64, r: f64) -> f64 {
    x * g_ricker(x, r)
}

/// One step of the Beverton-Holt recruitment map.
fn beverton_holt_map(x: f64, r: f64) -> f64 {
    x * g_beverton_holt(x, r)
}

/// e < r < e^2: f'(1) = 1 - ln(r) is negative but |f'(1)| < 1, so the
/// fixed point is approached by a *damped* oscillation: every step after
/// the first crossing alternates sides of x = 1 with strictly decaying
/// amplitude, and the long run lands on the fixed point.
#[test]
fn ricker_damped_oscillation_between_e_and_e_squared() {
    let r = 4.0_f64;
    let mut x = 0.25_f64;
    let mut seq = Vec::new();
    for _ in 0..12 {
        seq.push(x);
        x = ricker_map(x, r);
    }
    // Both first two iterates stay below 1 (the map increases on
    // (0, 1/ln r)), so alternation is asserted from the first crossing.
    for w in seq[1..10].windows(2) {
        assert!(
            (w[1] - 1.0) * (w[0] - 1.0) < 0.0,
            "expected oscillation across equilibrium at {} -> {}",
            w[0],
            w[1]
        );
        assert!(
            (w[1] - 1.0).abs() < (w[0] - 1.0).abs(),
            "amplitude must decay at {} -> {}",
            w[0],
            w[1]
        );
    }
    // Damped, not sustained: the long run converges to the fixed point.
    let mut x = 0.25_f64;
    for _ in 0..4_000 {
        x = ricker_map(x, r);
    }
    assert!(
        (x - 1.0).abs() < 1e-15,
        "damped regime must converge to the fixed point, got {x}"
    );
}

/// r > e^2: the fixed point loses stability (|1 - ln r| > 1) and the map
/// settles into a stable period-2 cycle.  For r = 8 the cycle is exactly
/// (2/3, 4/3) because 8^(1/3) = 2: (2/3)*2 = 4/3 and (4/3)*(1/2) = 2/3.
#[test]
fn ricker_sustained_two_cycle_above_e_squared() {
    let r = 8.0_f64;
    let mut x = 0.5_f64;
    for _ in 0..5_000 {
        x = ricker_map(x, r);
    }
    let a = x;
    let b = ricker_map(a, r);
    let c = ricker_map(b, r);
    assert!((a - 2.0 / 3.0).abs() < 1e-9, "cycle low point {a} != 2/3");
    assert!((b - 4.0 / 3.0).abs() < 1e-9, "cycle high point {b} != 4/3");
    assert!(
        (c - a).abs() < 1e-12,
        "period-2 closure broken: {a} -> {b} -> {c}"
    );
    assert!(
        (a - 1.0) * (b - 1.0) < 0.0,
        "cycle points must straddle x=1"
    );
}

/// Beverton-Holt at the same r = 8 must never overshoot: its recruitment
/// map is strictly increasing below equilibrium and approaches x = 1
/// from one side only — the dynamical opposite of Ricker overcompensation.
#[test]
fn beverton_holt_recruitment_never_overcompensates() {
    let r = 8.0_f64;
    let mut x = 0.5_f64;
    for step in 0..500 {
        let next = beverton_holt_map(x, r);
        assert!(
            next <= 1.0,
            "BH map crossed equilibrium at step {step}: {next}"
        );
        assert!(
            next >= x,
            "BH map moved away from equilibrium at step {step}"
        );
        if step < 10 {
            assert!(next > x, "BH map must grow strictly below equilibrium");
        }
        x = next;
    }
    assert!(
        (x - 1.0).abs() < 1e-9,
        "BH map must converge to x=1, got {x}"
    );
}

/// The contract checker must reject non-finite curve values at sampled
/// points (NaN and infinite arms), in addition to the shifted, increasing,
/// and negative breakers covered above.
#[test]
fn contract_checker_rejects_non_finite_values() {
    // NaN arm exactly at a sampled grid point (step 3000 * 3/3000 = 3).
    let nan_at_three = |x: f64| if x == 3.0 { f64::NAN } else { 1.0 };
    assert!(check_curve_contract(&nan_at_three).is_err());
    // Infinite spike at a sampled grid point (step 2000 * 3/3000 = 2).
    let inf_at_two = |x: f64| if x == 2.0 { f64::INFINITY } else { 1.0 };
    assert!(check_curve_contract(&inf_at_two).is_err());
}

/// Guard branches of the kernel entry: mode 0 ignores every input and
/// mode 1 returns 1.0 for zero actual strength and the uncapped single
/// division below equilibrium.
#[test]
fn regulation_scaling_mode_guards() {
    assert_eq!(regulation_scaling(0, 9_999.0, 1.0, 3.0, 0.9).unwrap(), 1.0);
    assert_eq!(regulation_scaling(1, 0.0, 500.0, 3.0, 0.8).unwrap(), 1.0);
    assert_eq!(
        regulation_scaling(1, 1_250.0, 500.0, 3.0, 0.8).unwrap(),
        1.0_f64.min(500.0 / 1_250.0)
    );
}

#[test]
fn ensure_nonzero_guards_zero_denominator() {
    assert!(ensure_nonzero(1.0, "test denominator").is_ok());
    assert!(ensure_nonzero(0.0, "test denominator").is_err());
}

/// clamp01 must preserve the Python helper's semantics, including NaN
/// pass-through.
#[test]
fn clamp01_preserves_python_nan_semantics() {
    assert_eq!(clamp01(-0.5), 0.0);
    assert_eq!(clamp01(1.5), 1.0);
    assert_eq!(clamp01(0.25), 0.25);
    assert!(clamp01(f64::NAN).is_nan());
}
