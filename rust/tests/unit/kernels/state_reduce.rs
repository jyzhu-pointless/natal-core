//! Bit-exactness contract of [`numpy_pairwise_sum`] against NumPy's
//! pairwise summation.

use super::numpy_pairwise_sum;

/// Reference implementation of NumPy's `pairwise_sum` (scalar path).
fn numpy_reference(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 8 {
        return values.iter().fold(0.0, |acc, v| acc + *v);
    }
    if n <= 128 {
        let mut r = [
            values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
        ];
        let aligned_end = n - (n % 8);
        let mut i = 8;
        while i < aligned_end {
            for slot in r.iter_mut() {
                *slot += values[i];
                i += 1;
            }
        }
        let mut sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            sum += values[i];
            i += 1;
        }
        return sum;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    numpy_reference(&values[..n2]) + numpy_reference(&values[n2..])
}

#[test]
fn empty_slice_sums_to_zero() {
    let values: Vec<f64> = Vec::new();
    assert_eq!(numpy_pairwise_sum(&values), 0.0);
}

#[test]
fn matches_reference_across_block_boundaries() {
    // Sizes straddle every algorithm branch: sequential (< 8), the
    // 8-accumulator block (8..=128, including the non-multiple-of-8
    // tail), and the recursive split (> 128, including halves that are
    // not multiples of eight).
    let sizes = [
        0, 1, 7, 8, 9, 15, 16, 127, 128, 129, 136, 255, 256, 257, 1000, 4096,
    ];
    for size in sizes {
        let values: Vec<f64> = (0..size).map(|i| (i % 17) as f64 * 0.375 - 3.0).collect();
        assert_eq!(
            numpy_pairwise_sum(&values),
            numpy_reference(&values),
            "size {size}"
        );
    }
}

#[test]
fn integer_counts_are_order_independent() {
    // Integer-valued float counts below 2^53 sum exactly under any
    // order; the pairwise result must equal the sequential one.
    let values: Vec<f64> = (1..=2000).map(|i| (i % 97) as f64).collect();
    let sequential = values.iter().fold(0.0, |acc, v| acc + *v);
    assert_eq!(numpy_pairwise_sum(&values), sequential);
}

#[test]
fn sex_planes_reduce_independently() {
    // The per-sex counts must equal the pairwise sum of each contiguous
    // plane, matching Python's ``individual_count[sex].sum()``.
    let plane = 45_usize;
    let state: Vec<f64> = (0..2 * plane).map(|i| (i % 13) as f64 * 0.5).collect();
    assert_eq!(numpy_pairwise_sum(&state), numpy_reference(&state));
    assert_eq!(
        numpy_pairwise_sum(&state[..plane]),
        numpy_reference(&state[..plane])
    );
    assert_eq!(
        numpy_pairwise_sum(&state[plane..2 * plane]),
        numpy_reference(&state[plane..2 * plane])
    );
}
