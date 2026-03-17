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
from .numba_utils import njit_switch, is_numba_enabled

__all__ = [
    "binomial_2d", "multinomial_rows", "multinomial"
]

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

@njit_switch(cache=False)
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
                result[i, j] = np.random.binomial(n_val, p_val)
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

@njit_switch(cache=False)
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

@njit_switch(cache=False)
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

@njit_switch(cache=False)
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
            result[i, :] = np.random.multinomial(n_trials, p_matrix[i, :])
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

@njit_switch(cache=False)
def _multinomial_numba(
    n: int,
    pvals: np.ndarray,  # shape (k,), float64
) -> np.ndarray:
    """Hand-written multinomial using binomial sampling.
    
    This is the same algorithm used by NumPy/Numba internally, but written
    explicitly to avoid the type inference bug in nested JIT functions.
    
    Algorithm: For each category j (except the last), sample from Binomial
    with conditional probability p_j / (1 - sum(p_0..p_{j-1})).
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
            n_j = np.random.binomial(n_remaining, p_cond)
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
else:
    # Use NumPy vectorized implementations
    binomial_2d = _binomial_2d_numpy
    fancy_index_3d_to_2d = _fancy_index_3d_to_2d_numpy
    fancy_index_3d_flat = _fancy_index_3d_flat_numpy
    multinomial_rows = _multinomial_rows_numpy
    multinomial = _multinomial_numpy
