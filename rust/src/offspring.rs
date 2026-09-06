//! The offspring-probability-tensor numeric kernel (plan 5.2: Rust owns
//! the probability-tensor computation).
//!
//! The loop nest and the skip-zero short-circuit mirror the reference
//! Python kernel *statement for statement* so both produce bit-identical
//! f64 results (Rust's default FP semantics do not fuse multiply-adds).
//! The Python side funnels every caller through this kernel when the
//! extension is available; the pure-Python spelling stays only as the
//! extension-less fallback until the Rust-only stage retires it.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Compute the offspring probability tensor on flattened row-major tables.
///
/// `P[gf, gm, go] = Σ_{hf,hm} meiosis_f[gf,hf] · meiosis_m[gm,hm] ·
/// fusion[hf,hm,go]` with the same statement order and zero-skips as the
/// reference kernel, so results are bit-identical.
///
/// ## Parameters
/// - `meiosis`: `(2 * z * g)` row-major meiosis table `(2, z, g)`.
/// - `fusion`: `(g * g * z)` row-major fusion table `(g, g, z)`.
/// - `z`: Number of ztype rows.
/// - `g`: Number of gtype columns.
///
/// ## Returns
/// The `(z * z * z)` row-major offspring tensor.
pub fn compute_offspring_tensor_flat(
    meiosis: &[f64],
    fusion: &[f64],
    z: usize,
    g: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; z * z * z];
    for gf in 0..z {
        for gm in 0..z {
            for go in 0..z {
                let mut s = 0.0f64;
                for hf in 0..g {
                    let mf = meiosis[gf * g + hf];
                    if mf == 0.0 {
                        continue;
                    }
                    for hm in 0..g {
                        let mm = meiosis[(z + gm) * g + hm];
                        if mm == 0.0 {
                            continue;
                        }
                        s += mf * mm * fusion[(hf * g + hm) * z + go];
                    }
                }
                out[(gf * z + gm) * z + go] = s;
            }
        }
    }
    out
}

/// PyO3 wrapper: compute the offspring tensor from 3-D meiosis/fusion tables.
///
/// ## Parameters
/// - `meiosis`: Read-only `(2, n_ztypes, n_gtypes)` meiosis table.
/// - `fusion`: Read-only `(n_gtypes, n_gtypes, n_ztypes)` fusion table.
///
/// ## Returns
/// A fresh `(n_ztypes^3,)` row-major flat tensor as a NumPy array.
///
/// ## Errors
/// Returns `PyValueError` on shape mismatches.
#[pyfunction]
pub fn compute_offspring_tensor<'py>(
    py: Python<'py>,
    meiosis: PyReadonlyArray3<'py, f64>,
    fusion: PyReadonlyArray3<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let m_shape = meiosis.shape();
    let f_shape = fusion.shape();
    if m_shape[0] != 2 {
        return Err(PyValueError::new_err(format!(
            "meiosis must have a leading sex axis of 2, got shape {m_shape:?}"
        )));
    }
    let (z, g) = (m_shape[1], m_shape[2]);
    if f_shape != [g, g, z] {
        return Err(PyValueError::new_err(format!(
            "fusion shape must be ({g}, {g}, {z}), got {f_shape:?}"
        )));
    }
    // Both inputs are C-contiguous by construction on the Python side;
    // as_slice errors on a non-contiguous view, which we surface as-is.
    let m_view = meiosis.as_array();
    let f_view = fusion.as_array();
    let m_slice = m_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("meiosis must be C-contiguous"))?;
    let f_slice = f_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("fusion must be C-contiguous"))?;
    let out = compute_offspring_tensor_flat(m_slice, f_slice, z, g);
    Ok(out.into_pyarray(py))
}

#[cfg(test)]
mod tests {
    use super::compute_offspring_tensor_flat;

    /// Hand-computed Mendelian case: every homozygote cross yields only
    /// its own genotype; mixed crosses yield nothing (the fusion map
    /// only joins matching haplotypes here); skip-zero must not skip
    /// nonzero contributions.
    #[test]
    fn offspring_kernel_matches_hand_computed_mendelian_cells() {
        // z = 2 (A|A, a|a), g = 2 (A, a); both sexes segregate purely.
        let meiosis = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]; // (2, 2, 2)
        let fusion = [
            1.0, 0.0, // A,A -> (A|A, a|a)
            0.0, 0.0, // A,a
            0.0, 0.0, // a,A
            0.0, 1.0, // a,a
        ];
        let out = compute_offspring_tensor_flat(&meiosis, &fusion, 2, 2);
        // Cell order ((gf,gm) pairs x 2 offspring columns):
        // (A|A,A|A)=[1,0], (A|A,a|a)=[0,0], (a|a,A|A)=[0,0], (a|a,a|a)=[0,1].
        assert_eq!(out, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    }

    /// Statement-order parity: the accumulated sum must follow
    /// (hf, hm) lexicographic order with zero-terms skipped — pinned by
    /// a table where different orderings give different last-bits.
    #[test]
    fn offspring_kernel_accumulates_in_pinned_order() {
        // One (gf, gm, go) cell fed by two nonzero (hf, hm) pairs whose
        // contributions do not commute exactly in f64.
        let z = 1;
        let g = 2;
        let meiosis = [1.0, 1.0, 1.0, 1.0]; // both sexes use both gtypes
        let fusion = [
            0.1, // (0,0) -> go 0
            0.0,
            0.0,       // (0,1), (1,0) unused at go 1 (z=1 → fusion is (g,g,1))
            1.0 / 3.0, // (1,1) -> go 0
        ];
        // fusion flat (2,2,1): [f(0,0), f(0,1), f(1,0), f(1,1)] = [0.1, 0, 0, 1/3]
        let expect: f64 = 1.0 * 0.1 + 1.0 * (1.0f64 / 3.0);
        let out = compute_offspring_tensor_flat(&meiosis, &fusion, z, g);
        assert_eq!(out[0].to_bits(), expect.to_bits());
    }
}
