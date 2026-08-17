//! RNG helpers for the Rust lifecycle backend.
//!
//! The stochastic path intentionally does not reproduce NumPy's legacy random
//! stream bit-for-bit.  It only has to be distributionally equivalent, so the
//! Rust side uses modern rand distributions.  Deterministic paths do not call
//! any function in this module.

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Binomial, Distribution, Gamma, Normal, Poisson};

pub const EPS: f64 = 1e-10;
const RESOLUTION_LIMIT: f64 = 2.028_240_960_365_167e31;
const GAMMA_NORMAL_APPROXIMATION_THRESHOLD: f64 = 1e8;

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

pub fn new_rng(seed: u64) -> SmallRng {
    SmallRng::seed_from_u64(seed)
}

/// Sample Binomial(n, p) and return the count as ``f64``.
///
/// Boundary cases mirror the Python fast path: p <= 0 or n <= 0 return 0,
/// p >= 1 returns n.
pub fn binomial(rng: &mut SmallRng, n: i64, p: f64) -> f64 {
    if n <= 0 || p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return n as f64;
    }
    let dist = Binomial::new(n as u64, p).expect("clamped binomial parameters must be valid");
    dist.sample(rng) as f64
}

pub fn poisson(rng: &mut SmallRng, lambda: f64) -> f64 {
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
pub fn gamma(rng: &mut SmallRng, shape: f64) -> f64 {
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

/// Continuous analogue of Poisson(λ): Gamma(λ, 1), matching Python's
/// moment-matching semantics.
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
pub fn continuous_binomial(rng: &mut SmallRng, n: f64, p: f64) -> f64 {
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
pub fn multinomial(rng: &mut SmallRng, n: i64, p: &[f64], out: &mut [f64]) {
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
pub fn continuous_multinomial(rng: &mut SmallRng, n: f64, p: &[f64], out: &mut [f64]) {
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
