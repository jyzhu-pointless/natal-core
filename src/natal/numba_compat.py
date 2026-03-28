"""
Numba-Compatible Helper Functions (Dual Implementation)
========================================================

This module provides helper functions that work in both Numba-compiled and pure Python contexts,
with **optimized implementations for each mode**:

- Numba mode: Use explicit loops (fast when JIT-compiled)
- Non-Numba mode: Use vectorized NumPy operations (fast in pure Python)

The correct implementation is selected at **import time** based on the global Numba configuration.

Usage:
    from natal.numba_compat import binomial_2d, fancy_index_3d_to_2d

    # These work in both @njit functions and regular Python with optimal performance
    result = binomial_2d(n_array, p_array, n_rows, n_cols)
"""

import numpy as np

from .numba_utils import is_numba_enabled, njit_switch

__all__ = [
    "binomial_2d", "multinomial_rows", "multinomial", "set_numba_seed"
]


# The original binomial implementation in Numba (numba.cpython.randomimpl) has a performance issue
# for large n*p due to a fallback to a slower algorithm (BTPE not implemented).
# This is a custom, efficient implementation of the BTPE algorithm (adapted from numba.np.random)
# for binomial sampling.
@njit_switch(cache=True)
def binomial_btpe(n: int, p: float) -> int:
    # C source code initialization
    r = min(p, 1.0 - p)
    q = 1.0 - r
    fm = n * r + r
    m = int(np.floor(fm))
    p1 = np.floor(2.195 * np.sqrt(n * r * q) - 4.6 * q) + 0.5
    xm = m + 0.5
    xl = xm - p1
    xr = xm + p1
    c = 0.134 + 20.5 / (15.3 + m)
    a = (fm - xl) / (fm - xl * r)
    laml = a * (1.0 + a / 2.0)
    a = (xr - fm) / (xr * q)
    lamr = a * (1.0 + a / 2.0)
    p2 = p1 * (1.0 + 2.0 * c)
    p3 = p2 + c / laml
    p4 = p3 + c / lamr

    nrq = n * r * q

    # Start sampling loop
    while True:
        # Corresponds to Step10 header
        u = np.random.random() * p4
        v = np.random.random()

        # ---------- Region division start ----------
        if u <= p1:
            # Corresponds to Step10 lower half: when u <= p1, jump directly to Step60
            y = int(np.floor(xm - p1 * v + u))
            break # goto Step60

        elif u <= p2:
            # Corresponds to Step20
            x = xl + (u - p1) / c
            v = v * c + 1.0 - np.fabs(m - x + 0.5) / p1
            if v > 1.0:
                continue # goto Step10
            y = int(np.floor(x))
            # Continue to Step50 verification below

        elif u <= p3:
            # Corresponds to Step30
            y = int(np.floor(xl + np.log(v) / laml))
            if y < 0 or v == 0.0:
                continue # goto Step10
            v = v * (u - p2) * laml
            # Continue to Step50 verification below

        else:
            # Corresponds to Step40
            y = int(np.floor(xr - np.log(v) / lamr))
            if y > n or v == 0.0:
                continue # goto Step10
            v = v * (u - p3) * lamr
            # Continue to Step50 verification below
        # ---------- Region division end ----------

        # ---------- Verification phase (corresponds to Step50 & Step52) ----------
        k = abs(y - m)
        if (k > 20) and (k < (nrq / 2.0 - 1)):
            # Corresponds to Step52 (Squeeze verification and precise approximation)
            rho = (k / nrq) * ((k * (k / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5)
            t = -k * k / (2.0 * nrq)
            A = np.log(v)

            if A < (t - rho):
                break # goto Step60
            if A > (t + rho):
                continue # goto Step10

            # Corresponds to precise approximation formula in Step52
            x1 = float(y + 1)
            f1 = float(m + 1)
            z = float(n + 1 - m)
            w = float(n - y + 1)
            x2 = x1 * x1
            f2 = f1 * f1
            z2 = z * z
            w2 = w * w

            # All plus signs in C source code, and all denominators are 166320
            if A > (xm * np.log(f1 / x1) + (n - m + 0.5) * np.log(z / w) +
                    (y - m) * np.log(w * r / (x1 * q)) +
                    (13680. - (462. - (132. - (99. - 140. / f2) / f2) / f2) / f2) / f1 / 166320. +
                    (13680. - (462. - (132. - (99. - 140. / z2) / z2) / z2) / z2) / z / 166320. +
                    (13680. - (462. - (132. - (99. - 140. / x2) / x2) / x2) / x2) / x1 / 166320. +
                    (13680. - (462. - (132. - (99. - 140. / w2) / w2) / w2) / w2) / w / 166320.):
                continue # goto Step10

            break # Fallback acceptance, goto Step60

        else:
            # Corresponds to precise factorial multiplication f in lower half of Step50
            s = r / q
            a = s * (n + 1)
            f = 1.0
            if m < y:
                for i in range(m + 1, y + 1):
                    f *= (a / i - s)
            elif m > y:
                for i in range(y + 1, m + 1):
                    f /= (a / i - s)

            if v > f:
                continue # goto Step10
            break # goto Step60

    # Corresponds to Step60
    if p > 0.5:
        y = n - y
    return y

# Usage Example
@njit_switch(cache=True)
def fast_binomial(n: int, p: float) -> int:
    # 1. Exception handling: validate p range
    if not (0.0 <= p <= 1.0):
        raise ValueError("fast_binomial(): p outside of [0, 1]")

    # 2. Exception handling: validate n range
    if n < 0:
        raise ValueError("fast_binomial(): n <= 0")

    # 3. Extreme boundary cases (O(1) immediate return)
    if p == 0.0 or n == 0:
        return 0
    if p == 1.0:
        return n

    # 4. Core routing logic
    # NumPy default uses inversion method (Inversion/BINV) when n * min(p, 1-p) <= 30
    # Because when n is small, BINV has fewer loop iterations, and the overhead is smaller than preparing BTPE constants
    if p <= 0.5:
        if n * p <= 30.0:
            return np.random.binomial(n, p)  # Numba's built-in BINV is fast here
        else:
            return binomial_btpe(n, p)
    else:
        q = 1.0 - p
        if n * q <= 30.0:
            return n - np.random.binomial(n, q)
        else:
            return n - binomial_btpe(n, q)

# ============================================================================
# Dual Implementation Pattern
# ============================================================================
# Each function has two implementations:
# - _xxx_numba: Loop-based, Numba-compatible (fast when JIT-compiled)
# - _xxx_numpy: Vectorized NumPy (fast in pure Python)
#
# The exported function is selected at import time based on config.


# ============================================================================
# binomial_2d: Element-wise binomial sampling for 2D arrays
# ============================================================================

@njit_switch(cache=True)
def _binomial_2d_numba(
    n: np.ndarray,  # shape (A, G), int64
    p: np.ndarray,  # shape (A,) or (A, G), float64
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    """Numba-compatible loop implementation."""
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    p_is_1d = (p.ndim == 1)

    for i in range(n_rows):
        for j in range(n_cols):
            n_val = int(n[i, j])
            if p_is_1d:
                p_val = float(p[i])
            else:
                p_val = float(p[i, j])

            if n_val > 0 and 0.0 <= p_val <= 1.0:
                result[i, j] = fast_binomial(n_val, p_val)
            else:
                result[i, j] = 0.0

    return result


def _binomial_2d_numpy(
    n: np.ndarray,  # shape (A, G), int64
    p: np.ndarray,  # shape (A,) or (A, G), float64
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    """NumPy vectorized implementation for pure Python mode."""
    # Broadcast p to match n's shape if needed
    if p.ndim == 1:
        p_broadcast = p[:, None]  # (A, 1) -> broadcasts to (A, G)
    else:
        p_broadcast = p

    # NumPy's binomial supports array inputs natively
    return np.random.binomial(n.astype(np.int64), p_broadcast).astype(np.float64)


# ============================================================================
# fancy_index_3d_to_2d: Extract 2D slices from 3D array
# ============================================================================

@njit_switch(cache=True)
def _fancy_index_3d_to_2d_numba(
    arr_3d: np.ndarray,     # shape (D0, D1, D2)
    idx0: np.ndarray,       # shape (N,)
    idx1: np.ndarray,       # shape (N,)
    n_indices: int,
    last_dim: int,
) -> np.ndarray:
    """Numba-compatible loop implementation."""
    result = np.empty((n_indices, last_dim), dtype=arr_3d.dtype)

    for i in range(n_indices):
        i0 = int(idx0[i])
        i1 = int(idx1[i])
        for k in range(last_dim):
            result[i, k] = arr_3d[i0, i1, k]

    return result


def _fancy_index_3d_to_2d_numpy(
    arr_3d: np.ndarray,
    idx0: np.ndarray,
    idx1: np.ndarray,
    n_indices: int,
    last_dim: int,
) -> np.ndarray:
    """NumPy vectorized implementation for pure Python mode."""
    # NumPy supports fancy indexing natively
    return arr_3d[idx0, idx1, :]


# ============================================================================
# fancy_index_3d_flat: Extract elements from 3D array using triple indices
# ============================================================================

@njit_switch(cache=True)
def _fancy_index_3d_flat_numba(
    arr_3d: np.ndarray,
    idx0: np.ndarray,
    idx1: np.ndarray,
    idx2: np.ndarray,
    n_indices: int,
) -> np.ndarray:
    """Numba-compatible loop implementation using flat indexing."""
    shape = arr_3d.shape
    d1, d2 = shape[1], shape[2]

    arr_flat = arr_3d.ravel()
    result = np.empty(n_indices, dtype=arr_3d.dtype)

    for i in range(n_indices):
        flat_idx = int(idx0[i]) * d1 * d2 + int(idx1[i]) * d2 + int(idx2[i])
        result[i] = arr_flat[flat_idx]

    return result


def _fancy_index_3d_flat_numpy(
    arr_3d: np.ndarray,
    idx0: np.ndarray,
    idx1: np.ndarray,
    idx2: np.ndarray,
    n_indices: int,
) -> np.ndarray:
    """NumPy vectorized implementation for pure Python mode."""
    return arr_3d[idx0, idx1, idx2]


# ============================================================================
# multinomial_rows: Row-wise multinomial sampling
# ============================================================================

@njit_switch(cache=True)
def _multinomial_rows_numba(
    n_per_row: np.ndarray,
    p_matrix: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    """Numba-compatible loop implementation."""
    result = np.empty((n_rows, n_cols), dtype=np.int64)

    for i in range(n_rows):
        n_trials = int(n_per_row[i])
        if n_trials > 0:
            result[i, :] = _multinomial_numba(n_trials, p_matrix[i, :])
        else:
            for j in range(n_cols):
                result[i, j] = 0

    return result


def _multinomial_rows_numpy(
    n_per_row: np.ndarray,
    p_matrix: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    """NumPy implementation (still needs loop, but avoids Numba overhead)."""
    result = np.empty((n_rows, n_cols), dtype=np.int64)

    for i in range(n_rows):
        n_trials = int(n_per_row[i])
        if n_trials > 0:
            result[i, :] = np.random.multinomial(n_trials, p_matrix[i, :])
        else:
            result[i, :] = 0

    return result


# ============================================================================
# multinomial: Single multinomial sampling (workaround for Numba bug)
# ============================================================================
# Numba's np.random.multinomial has a type inference bug when called with
# dynamically computed probability arrays in nested JIT functions.
# This implementation uses explicit binomial sampling (like NumPy's algorithm).

@njit_switch(cache=True)
def _multinomial_numba(
    n: int,
    pvals: np.ndarray,  # shape (k,), float64
) -> np.ndarray:
    """Hand-written multinomial using binomial sampling.

    This is the same algorithm used by NumPy/Numba internally, but written
    explicitly to avoid the type inference bug in nested JIT functions.

    Algorithm: For each category j (except the last), sample from Binomial
    with conditional probability p_j / (1 - sum(p_0..p_{j-1})).

    Args:
        n: Total number of trials
        pvals: Probability vector with shape (k,) that sums to 1.0

    Returns:
        Array of sampled counts with shape (k,) that sums to n
    """
    k = len(pvals)
    result = np.zeros(k, dtype=np.int64)

    if n <= 0:
        return result

    # Remaining trials and probability sum
    n_remaining = int(n)
    p_sum = 1.0

    for j in range(k - 1):
        if n_remaining <= 0:
            break

        p_j = float(pvals[j])

        if p_sum > 0.0 and p_j > 0.0:
            # Conditional probability
            p_cond = p_j / p_sum
            # Clamp to valid range
            if p_cond > 1.0:
                p_cond = 1.0
            elif p_cond < 0.0:
                p_cond = 0.0

            # Sample from binomial
            n_j = fast_binomial(n_remaining, p_cond)
            result[j] = n_j
            n_remaining -= n_j

        p_sum -= p_j

    # Last category gets all remaining
    if n_remaining > 0:
        result[k - 1] = n_remaining

    return result


def _multinomial_numpy(
    n: int,
    pvals: np.ndarray,
) -> np.ndarray:
    """Use NumPy's native multinomial in non-Numba mode."""
    return np.random.multinomial(n, pvals)


@njit_switch(cache=True)
def _set_numba_seed_numba(seed: int) -> None:
    """Set RNG seed inside Numba/JIT random state."""
    np.random.seed(seed)


def _set_numba_seed_numpy(seed: int) -> None:
    """Set RNG seed for NumPy random state in non-Numba mode."""
    np.random.seed(seed)


# ============================================================================
# Export the correct implementation based on Numba configuration
# ============================================================================

if is_numba_enabled():
    # Use Numba-compatible loop implementations
    binomial_2d = _binomial_2d_numba
    fancy_index_3d_to_2d = _fancy_index_3d_to_2d_numba
    fancy_index_3d_flat = _fancy_index_3d_flat_numba
    multinomial_rows = _multinomial_rows_numba
    multinomial = _multinomial_numba
    binomial = fast_binomial
    set_numba_seed = _set_numba_seed_numba
else:
    # Use NumPy vectorized implementations
    binomial_2d = _binomial_2d_numpy
    fancy_index_3d_to_2d = _fancy_index_3d_to_2d_numpy
    fancy_index_3d_flat = _fancy_index_3d_flat_numpy
    multinomial_rows = _multinomial_rows_numpy
    multinomial = _multinomial_numpy
    binomial = np.random.binomial
    set_numba_seed = _set_numba_seed_numpy
