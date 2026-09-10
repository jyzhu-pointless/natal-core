//! Density-regulation curve library.
//!
//! Every curve shares one signature: input ``x`` = current juvenile
//! competition strength divided by the equilibrium competition strength,
//! output ``g(x)`` = the survival/growth factor applied on top of the
//! equilibrium survival rate.  All built-in curves pass the contract
//! tests below: ``g(1) == 1`` exactly, non-increasing in ``x``, finite
//! and non-negative; the compensatory sub-family additionally satisfies
//! ``g(0) == r``.
//!
//! Growth-mode ids: 0 = no regulation, 1 = ``fixed``, 2 = ``linear``
//! (alias ``logistic``), 3 = ``beverton_holt``, 4 = ``ricker``,
//! ``>= 5`` = user-registered custom slot.

use pyo3::exceptions::{PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;

/// The low-density growth rate ``r`` carried by every compensatory curve.
pub type GrowthRate = f64;

/// Compute the scaling factor for growth mode 1 (``fixed``): hard cap.
///
/// ``g(x) = min(1, 1/x)`` — no compensation below equilibrium, exact
/// proportional culling above it.
///
/// ## Parameters
/// - `x`: Competition ratio (actual / equilibrium).
///
/// ## Returns
/// The survival factor.
#[must_use]
pub fn g_fixed(x: f64) -> f64 {
    if x <= 0.0 {
        1.0
    } else {
        1.0_f64.min(1.0 / x)
    }
}

/// Compute the scaling factor for growth mode 2 (``linear``/``logistic``).
///
/// ``g(x) = max(0, r - (r - 1) x)`` — linear compensation reaching zero.
///
/// ## Parameters
/// - `x`: Competition ratio.
/// - `r`: Low-density growth rate.
///
/// ## Returns
/// The survival factor.
#[must_use]
pub fn g_linear(x: f64, r: f64) -> f64 {
    (r - (r - 1.0) * x).max(0.0)
}

/// Compute the scaling factor for growth mode 3 (``beverton_holt``).
///
/// ``g(x) = r / (1 + (r - 1) x)`` — hyperbolic compensation, never zero,
/// unconditionally stable at equilibrium (|g'(1)| < 1 for r > 1).
///
/// ## Parameters
/// - `x`: Competition ratio.
/// - `r`: Low-density growth rate.
///
/// ## Returns
/// The survival factor.
#[must_use]
pub fn g_beverton_holt(x: f64, r: f64) -> f64 {
    r / (1.0 + (r - 1.0) * x)
}

/// Compute the scaling factor for growth mode 4 (``ricker``).
///
/// ``g(x) = r^(1 - x)`` — exponential overcompensation; oscillates around
/// equilibrium for ``r > e``.
///
/// ## Parameters
/// - `x`: Competition ratio.
/// - `r`: Low-density growth rate.
///
/// ## Returns
/// The survival factor.
#[must_use]
pub fn g_ricker(x: f64, r: f64) -> f64 {
    r.powf(1.0 - x)
}

/// The three shared contract properties every curve must satisfy.
///
/// - equilibrium fixed point: ``g(1) == 1`` exactly;
/// - monotone non-increasing;
/// - non-negative and bounded by ``g(0)``.
///
/// ## Parameters
/// - `g`: The curve under test.
///
/// ## Errors
/// Returns ``PyValueError`` with the failing sample on any violation.
pub fn check_curve_contract(g: &impl Fn(f64) -> f64) -> PyResult<()> {
    // 1. Equilibrium fixed point (exact).
    if g(1.0) != 1.0 {
        return Err(PyValueError::new_err(format!(
            "contract violated: g(1) = {} != 1",
            g(1.0)
        )));
    }
    // 2. Monotone non-increasing + 3. non-negative and bounded, sampled.
    let mut prev = g(0.0);
    if !(prev.is_finite() && prev >= 0.0) {
        return Err(PyValueError::new_err(format!(
            "contract violated: g(0) = {prev} must be finite and non-negative"
        )));
    }
    let bound = prev;
    for step in 1..=3_000 {
        let x = step as f64 * (3.0 / 3_000.0);
        let v = g(x);
        if !v.is_finite() || v < 0.0 || v > bound + 1e-12 {
            return Err(PyValueError::new_err(format!(
                "contract violated at x={x}: g={v} (bound {bound})"
            )));
        }
        if v > prev + 1e-12 {
            return Err(PyValueError::new_err(format!(
                "contract violated at x={x}: g increased ({prev} -> {v})"
            )));
        }
        prev = v;
    }
    Ok(())
}

/// Dispatch a built-in growth mode to its curve and evaluate it.
///
/// ## Parameters
/// - `mode`: Growth-mode id (see the module docs).
/// - `x`: Competition ratio.
/// - `r`: Low-density growth rate.
///
/// ## Returns
/// The survival factor.
///
/// ## Errors
/// Returns ``PyValueError`` for unknown mode ids.
pub fn scaling_factor(mode: i64, x: f64, r: f64) -> PyResult<f64> {
    Ok(match mode {
        1 => g_fixed(x),
        2 => g_linear(x, r),
        3 => g_beverton_holt(x, r),
        4 => g_ricker(x, r),
        0 => 1.0,
        other => {
            return Err(PyValueError::new_err(format!(
                "unrecognized growth mode {other} (built-ins: 0-4; custom slots via the curve registry)"
            )))
        }
    })
}

/// Compute the density-regulation scaling from raw strength inputs.
///
/// This is the kernel-facing entry point.  It preserves the Python
/// reference operation order exactly
/// (``compute_scaling_factor_{fixed,logistic,beverton_holt}`` in
/// ``backends/reference/simulation/age_structured.py``, now retired):
///
/// - fixed: ``min(1, equilibrium / actual)`` — evaluated as a single
///   division.  Going through [`g_fixed`] would compute ``1 / (a / e)``,
///   a double rounding that is *not* bit-identical to the reference, so
///   mode 1 bypasses the ratio form deliberately.
/// - linear/logistic and beverton_holt: evaluate the curve at
///   ``x = competition_ratio(actual, equilibrium)`` and multiply by the
///   equilibrium survival rate afterwards, matching
///   ``actual_growth_rate * expected_survival_rate``.
/// - ricker (mode 4): same shape, no Python precedent.
///
/// ## Parameters
/// - `mode`: Growth-mode id (0 none, 1 fixed, 2 linear, 3 beverton_holt,
///   4 ricker).
/// - `actual`: Current juvenile competition strength (age-structured:
///   weight-blended juvenile counts; discrete: total age-0 count).
/// - `equilibrium`: Equilibrium competition strength C* (or K for fixed).
/// - `r`: Low-density growth rate.
/// - `survival_rate`: Equilibrium survival rate s* (unused for fixed).
///
/// ## Returns
/// The scaling factor applied to newborn/juvenile counts.
///
/// ## Errors
/// Returns ``PyValueError`` for unknown mode ids.
pub fn regulation_scaling(
    mode: i64,
    actual: f64,
    equilibrium: f64,
    r: f64,
    survival_rate: f64,
) -> PyResult<f64> {
    match mode {
        0 => Ok(1.0),
        1 => Ok(if actual > 0.0 {
            (equilibrium / actual).min(1.0)
        } else {
            1.0
        }),
        2..=4 => {
            let ratio = competition_ratio(actual, equilibrium);
            Ok(scaling_factor(mode, ratio, r)? * survival_rate)
        }
        other => Err(PyValueError::new_err(format!(
            "unrecognized growth mode {other} (built-ins: 0-4; custom slots via the curve registry)"
        ))),
    }
}

/// Compute the competition ratio guarding against a zero denominator.
///
/// ## Parameters
/// - `actual`: Current competition strength.
/// - `expected`: Equilibrium competition strength.
///
/// ## Returns
/// ``actual / expected``, or 1.0 when the equilibrium is zero.
#[must_use]
pub fn competition_ratio(actual: f64, expected: f64) -> f64 {
    if expected > 0.0 {
        actual / expected
    } else {
        1.0
    }
}

/// Guard a division denominator.
///
/// ## Errors
/// Returns ``ZeroDivisionError`` when the denominator is zero.
pub fn ensure_nonzero(denominator: f64, what: &str) -> PyResult<()> {
    if denominator == 0.0 {
        return Err(PyZeroDivisionError::new_err(format!(
            "{what} must be nonzero"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/density_regulation.rs"]
mod tests;
