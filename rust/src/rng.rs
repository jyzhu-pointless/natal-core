//! RNG helpers for the Rust lifecycle backend.
//!
//! The stochastic path intentionally does not reproduce NumPy's legacy random
//! stream bit-for-bit.  It only has to be distributionally equivalent, so the
//! Rust side uses modern rand distributions.  Deterministic paths do not call
//! any function in this module.
//!
//! The session generator is [`SessionRng`], a byte-for-byte replica of the
//! ``Xoshiro256PlusPlus`` algorithm that ``rand 0.10`` uses for ``SmallRng``
//! on 64-bit platforms.  The replica exists because rand keeps the internal
//! state private: memory checkpoints must read and restore the full 4-word
//! state so ``restore -> run`` continues the exact stream of the original run.

use rand::rand_core::utils;
use rand::rand_core::{SeedableRng, TryRng};
use rand_distr::{Binomial, Distribution, Gamma, Normal, Poisson};

/// Small positive tolerance used for boundary decisions.
///
/// Values at or below ``EPS`` are treated as zero in most stochastic kernels.
pub const EPS: f64 = 1e-10;
const RESOLUTION_LIMIT: f64 = 2.028_240_960_365_167e31;
const GAMMA_NORMAL_APPROXIMATION_THRESHOLD: f64 = 1e8;

/// Derive a per-deme stream seed from the base seed: ``seed ^ deme_id``.
///
/// XOR keeps every bit of the base seed present in each stream while mixing
/// in the deme identity, so adjacent demes no longer share high-order seed
/// bits the way ``seed + deme`` streams do.
///
/// ## Parameters
/// - `seed`: Base RNG seed shared by all demes.
/// - `deme_id`: Deme index.
///
/// ## Returns
/// The derived stream seed.
#[must_use]
pub fn stream_seed(seed: u64, deme_id: i64) -> u64 {
    seed ^ (deme_id as u64)
}

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

/// Create a new session generator from a u64 seed.
///
/// ## Parameters
/// - `seed`: Seed value.
///
/// ## Returns
/// A deterministic [`SessionRng`] instance.
pub fn new_rng(seed: u64) -> SessionRng {
    SessionRng::seed_from_u64(seed)
}

/// Snapshot-capable replica of rand 0.10's ``SmallRng`` (Xoshiro256PlusPlus).
///
/// The next-word update, ``seed_from_u64`` (SplitMix64 expansion), and
/// ``from_seed`` zero handling mirror ``rand::rngs::xoshiro256plusplus``
/// exactly, so streams produced through this type are identical to the
/// streams previously produced by ``SmallRng``.  The four state words are
/// exposed for memory checkpoints: capturing them and rebuilding a
/// generator via [`SessionRng::from_state_words`] resumes the same stream
/// (continuation semantics, not a reseed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionRng {
    s: [u64; 4],
}

impl SessionRng {
    /// Rebuild a generator from four captured state words.
    ///
    /// An all-zero state is remapped exactly like ``from_seed`` does, which
    /// cannot occur for states produced by ``seed_from_u64`` but keeps the
    /// constructor total.
    ///
    /// ## Parameters
    /// - `words`: The ``[s0, s1, s2, s3]`` state words from
    ///   [`SessionRng::state_words`].
    ///
    /// ## Returns
    /// A generator that continues the captured stream.
    #[must_use]
    pub fn from_state_words(words: [u64; 4]) -> Self {
        if words == [0; 4] {
            return Self::seed_from_u64(0);
        }
        Self { s: words }
    }

    /// Copy the full internal state for checkpointing.
    ///
    /// ## Returns
    /// The ``[s0, s1, s2, s3]`` state words.
    #[must_use]
    pub fn state_words(&self) -> [u64; 4] {
        self.s
    }
}

impl SeedableRng for SessionRng {
    type Seed = [u8; 32];

    /// Create a generator from a 32-byte seed.  If `seed` is entirely 0, it
    /// will be mapped to a different seed (matching rand 0.10).
    fn from_seed(seed: [u8; 32]) -> Self {
        let state = utils::read_words(&seed);
        // An all-zero state is illegal for xoshiro; remap it like rand does.
        if state.iter().all(|&x| x == 0) {
            return Self::seed_from_u64(0);
        }
        Self { s: state }
    }

    /// Create a generator from a u64 seed via SplitMix64 expansion,
    /// matching ``rand_core``'s default and rand's ``SmallRng`` streams.
    fn seed_from_u64(mut state: u64) -> Self {
        const PHI: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut s = [0_u64; 4];
        for slot in &mut s {
            state = state.wrapping_add(PHI);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            *slot = z;
        }
        Self { s }
    }
}

impl TryRng for SessionRng {
    type Error = core::convert::Infallible;

    #[inline]
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        // The lowest bits have some linear dependencies, so use the upper bits.
        Ok((self.try_next_u64()? >> 32) as u32)
    }

    #[inline]
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        let res = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);

        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;

        self.s[3] = self.s[3].rotate_left(45);

        Ok(res)
    }

    #[inline]
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        utils::fill_bytes_via_next_word(dst, || self.try_next_u64())
    }
}

// `RngCore` is a blanket wrapper over an infallible `TryRng` in rand_core
// 0.10, so `SessionRng` satisfies the bounds used by `rand_distr`.

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
pub fn binomial(rng: &mut SessionRng, n: i64, p: f64) -> f64 {
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
pub fn poisson(rng: &mut SessionRng, lambda: f64) -> f64 {
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
pub fn gamma(rng: &mut SessionRng, shape: f64) -> f64 {
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
pub fn continuous_poisson(rng: &mut SessionRng, lambda: f64) -> f64 {
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
pub fn continuous_binomial(rng: &mut SessionRng, n: f64, p: f64) -> f64 {
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
pub fn multinomial(rng: &mut SessionRng, n: i64, p: &[f64], out: &mut [f64]) {
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
pub fn continuous_multinomial(rng: &mut SessionRng, n: f64, p: &[f64], out: &mut [f64]) {
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

#[cfg(test)]
mod tests {
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
    // Slice-2: deterministic boundary guards of the sampling helpers.
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
}
