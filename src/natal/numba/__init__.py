"""Numba acceleration subpackage.

Provides configurable JIT compilation decorators (:mod:`natal.numba.utils`)
and Numba-compatible helper functions with dual implementations
(:mod:`natal.numba.compat`).
"""

from .compat import (
    binomial,
    binomial_2d,
    binomial_btpe,
    clamp01,
    continuous_binomial,
    continuous_multinomial,
    continuous_poisson,
    fancy_index_3d_flat,
    fancy_index_3d_to_2d,
    multinomial,
    multinomial_rows,
    set_numba_seed,
)
from .utils import (
    NUMBA_ENABLED,
    disable_numba,
    disable_numba_log,
    disable_numba_signature_trace,
    enable_numba,
    enable_numba_log,
    enable_numba_signature_trace,
    get_numba_cache_dir,
    is_numba_enabled,
    is_numba_log_enabled,
    is_numba_signature_trace_enabled,
    njit_switch,
    numba_disabled,
    numba_enabled,
    with_numba_disabled,
    with_numba_enabled,
)

__all__ = [
    # utils
    "NUMBA_ENABLED",
    "disable_numba",
    "disable_numba_log",
    "disable_numba_signature_trace",
    "enable_numba",
    "enable_numba_log",
    "enable_numba_signature_trace",
    "get_numba_cache_dir",
    "is_numba_enabled",
    "is_numba_log_enabled",
    "is_numba_signature_trace_enabled",
    "njit_switch",
    "numba_disabled",
    "numba_enabled",
    "with_numba_disabled",
    "with_numba_enabled",
    # compat
    "binomial",
    "binomial_2d",
    "binomial_btpe",
    "clamp01",
    "continuous_binomial",
    "continuous_multinomial",
    "continuous_poisson",
    "fancy_index_3d_flat",
    "fancy_index_3d_to_2d",
    "multinomial",
    "multinomial_rows",
    "set_numba_seed",
]
