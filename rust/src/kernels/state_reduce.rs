//! State reduction primitives shared by the session query surface.
//!
//! The count queries replace Python-side ``ndarray.sum()`` calls; the
//! reduction below therefore replicates NumPy's pairwise summation
//! exactly (same accumulator layout, same block splitting, same
//! recursion) so native counts stay bitwise identical to the previous
//! Python sums, including fractional count states.

/// NumPy's pairwise-sum block size (`PW_BLOCKSIZE` in `loops.c.src`).
const PAIRWISE_BLOCKSIZE: usize = 128;

/// Sum a float slice in NumPy's pairwise summation order.
///
/// The result is bitwise identical to ``numpy.ndarray.sum()`` over the
/// same contiguous values (scalar fallback path, verified against the
/// shipped NumPy on this platform): blocks up to eight elements sum
/// sequentially, blocks up to [`PAIRWISE_BLOCKSIZE`] use eight strided
/// accumulators combined two-by-two, and larger inputs recurse on both
/// halves with the first half rounded down to a multiple of eight.
pub(crate) fn numpy_pairwise_sum(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 8 {
        let mut sum = 0.0;
        for value in values {
            sum += *value;
        }
        return sum;
    }
    if n <= PAIRWISE_BLOCKSIZE {
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
    numpy_pairwise_sum(&values[..n2]) + numpy_pairwise_sum(&values[n2..])
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/state_reduce.rs"]
mod tests;
