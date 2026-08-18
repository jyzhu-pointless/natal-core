//! RNG helpers for the Rust lifecycle backend.
//!
//! The stochastic path intentionally does not reproduce NumPy's legacy random
//! stream bit-for-bit.  It only has to be distributionally equivalent, so the
//! Rust side uses modern rand distributions.  Deterministic paths do not call
//! any function in this module.

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Binomial, Distribution, Gamma, Normal, Poisson};

/// Small positive tolerance used for boundary decisions.
///
/// Values at or below ``EPS`` are treated as zero in most stochastic kernels.
pub const EPS: f64 = 1e-10;
const RESOLUTION_LIMIT: f64 = 2.028_240_960_365_167e31;
const GAMMA_NORMAL_APPROXIMATION_THRESHOLD: f64 = 1e8;

/// Clamp a value into ``[0, 1]`` while preserving Python NaN semantics.
///
/// ## Parameters
/// - `x`: Input probability or rate.
///
/// ## Returns
/// ``0.0`` if `x` is non-positive, ``1.0`` if `x` is at least one, otherwise `x`.
///
/// ## Notes
/// NaN is returned unchanged, matching the Python helper used by the reference engine.
#[allow(clippy::manual_clamp)] // if-chain preserves Python _clamp01 NaN semantics
pub fn clamp01(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else if x >= 1.0 {
        1.0
    } else {
        x
    }
}

/// Create a new ``SmallRng`` from a u64 seed.
///
/// ## Parameters
/// - `seed`: Seed value.
///
/// ## Returns
/// A deterministic ``SmallRng`` instance.
pub fn new_rng(seed: u64) -> SmallRng {
    SmallRng::seed_from_u64(seed)
}

/// Sample Binomial(n, p) and return the count as ``f64``.
///
/// Boundary cases mirror the Python fast path: ``p <= 0`` or ``n <= 0`` return
/// zero, and ``p >= 1`` returns ``n``.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `n`: Number of trials.
/// - `p`: Success probability.
///
/// ## Returns
/// The sampled count as ``f64``.
pub fn binomial(rng: &mut SmallRng, n: i64, p: f64) -> f64 {
    // Fast-path boundary cases before constructing the distribution.
    // This matches the Python helper exactly for n <= 0, p <= 0, p >= 1.
    if n <= 0 || p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return n as f64;
    }
    let dist = Binomial::new(n as u64, p).expect("clamped binomial parameters must be valid");
    dist.sample(rng) as f64
}

/// Sample a Poisson count, with extreme and large-lambda guards.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `lambda`: Mean of the Poisson distribution.
///
/// ## Returns
/// The sampled count as ``f64``.  Very large lambdas return the mean directly
/// to avoid numerical overflow.
pub fn poisson(rng: &mut SmallRng, lambda: f64) -> f64 {
    // Guard tiny and huge lambdas to avoid numerical issues in rand_distr.
    if lambda <= EPS {
        return 0.0;
    }
    if lambda >= RESOLUTION_LIMIT {
        return lambda;
    }
    Poisson::new(lambda)
        .expect("positive lambda must be valid")
        .sample(rng)
}

/// Sample a unit-scale Gamma(shape, 1) variate.
///
/// Python falls back to the mean for extreme or degenerate shapes; the same
/// guards are kept here.  The intermediate sampling algorithm may differ from
/// NumPy's while preserving the Gamma distribution.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `shape`: Gamma shape parameter.
///
/// ## Returns
/// A Gamma-distributed sample as ``f64``.
pub fn gamma(rng: &mut SmallRng, shape: f64) -> f64 {
    // Use mean for degenerate/extreme shapes; approximate with Normal
    // for large shapes; otherwise sample a standard Gamma.
    if shape >= RESOLUTION_LIMIT {
        return shape;
    }
    if shape >= GAMMA_NORMAL_APPROXIMATION_THRESHOLD {
        let sample = Normal::new(shape, shape.sqrt())
            .expect("finite gamma parameters must be valid")
            .sample(rng);
        return sample.max(0.0);
    }
    if shape <= EPS {
        return 0.0;
    }
    Gamma::new(shape, 1.0)
        .expect("positive gamma shape must be valid")
        .sample(rng)
}

/// Continuous analogue of Poisson(lambda): Gamma(lambda, 1).
///
/// This matches Python's moment-matching semantics for continuous sampling.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `lambda`: Mean.
///
/// ## Returns
/// A continuous non-negative sample as ``f64``.
///
/// ## Panics
/// Panics if `lambda` is not finite.
pub fn continuous_poisson(rng: &mut SmallRng, lambda: f64) -> f64 {
    if !lambda.is_finite() {
        panic!("continuous_poisson(): lambda must be finite");
    }
    if lambda <= EPS {
        return 0.0;
    }
    gamma(rng, lambda)
}

/// Continuous analogue of Binomial(n, p) via a Beta-distributed proportion.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `n`: Continuous population size.
/// - `p`: Success probability.
///
/// ## Returns
/// A continuous sample in ``[0, n]``.
///
/// ## Panics
/// Panics if `n` or `p` is not finite.
pub fn continuous_binomial(rng: &mut SmallRng, n: f64, p: f64) -> f64 {
    // Represent the continuous binomial as a Beta proportion times n.
    // For small n the proportion is replaced by the mean to avoid instability.
    if !n.is_finite() || !p.is_finite() {
        panic!("continuous_binomial(): n and p must be finite");
    }
    if p <= EPS {
        return 0.0;
    }
    if p >= 1.0 - EPS {
        return n;
    }
    if n <= 1.0 + EPS {
        return n * p;
    }

    let concentration = n - 1.0;
    let alpha = (p * concentration).max(EPS);
    let beta = ((1.0 - p) * concentration).max(EPS);
    let numerator = gamma(rng, alpha);
    let denominator_component = gamma(rng, beta);
    let proportion = if numerator == 0.0 {
        0.0
    } else {
        numerator / (numerator + denominator_component)
    };
    proportion * n
}

/// Discrete multinomial implemented with conditional binomial draws.
///
/// This is the same algorithm used by NumPy and by ``nbc.multinomial``; the
/// last category receives the remaining trials.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `n`: Number of trials.
/// - `p`: Probability vector.
/// - `out`: Output slice, overwritten with category counts.
pub fn multinomial(rng: &mut SmallRng, n: i64, p: &[f64], out: &mut [f64]) {
    // Sequential conditional binomial draws; the final category receives
    // whatever trials remain, preserving the total exactly.
    for slot in out.iter_mut() {
        *slot = 0.0;
    }
    if n <= 0 || p.is_empty() {
        return;
    }
    let mut n_remaining = n;
    let mut p_sum = 1.0;
    for (idx, &p_j) in p.iter().enumerate().take(p.len().saturating_sub(1)) {
        if n_remaining <= 0 {
            break;
        }
        if p_sum > 0.0 && p_j > 0.0 {
            let p_cond = clamp01(p_j / p_sum);
            let n_j = binomial(rng, n_remaining, p_cond) as i64;
            out[idx] = n_j as f64;
            n_remaining -= n_j;
        }
        p_sum -= p_j;
    }
    if n_remaining > 0 {
        let last = p.len() - 1;
        out[last] = n_remaining as f64;
    }
}

/// Continuous analogue of Multinomial(n, p) via normalized Gamma draws.
///
/// ## Parameters
/// - `rng`: Random number generator.
/// - `n`: Continuous total.
/// - `p`: Probability vector.
/// - `out`: Output slice, overwritten with category counts.
///
/// ## Notes
/// A final correction pass keeps the sum close to `n` when floating-point
/// rounding would otherwise drift.
pub fn continuous_multinomial(rng: &mut SmallRng, n: f64, p: &[f64], out: &mut [f64]) {
    // Draw independent Gamma variates and normalize to the target total.
    // A final correction handles floating-point drift in the sum.
    if n <= 1.0 + EPS {
        for (slot, &prob) in out.iter_mut().zip(p.iter()) {
            *slot = n * prob;
        }
        return;
    }

    let concentration = n - 1.0;
    let mut sum_gamma = 0.0;
    for (slot, &prob) in out.iter_mut().zip(p.iter()) {
        let alpha = prob * concentration;
        let value = if alpha <= EPS { 0.0 } else { gamma(rng, alpha) };
        *slot = value;
        sum_gamma += value;
    }

    if sum_gamma > EPS {
        let factor = n / sum_gamma;
        for slot in out.iter_mut() {
            *slot *= factor;
        }
    } else {
        for (slot, &prob) in out.iter_mut().zip(p.iter()) {
            *slot = n * prob;
        }
    }

    let total: f64 = out.iter().sum();
    let tolerance = 1e-6 * n.max(1.0);
    if total > EPS && (total - n).abs() > tolerance {
        let correction = n / total;
        for slot in out.iter_mut() {
            *slot *= correction;
        }
    }
}
