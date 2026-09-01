"""Forwarding shim: the ``numba`` package now lives at
``natal.backends.numba``.

This shim preserves the legacy import path (and the top-level lazy-export
table) during the Phase-0 directory reorganisation; it will be removed once
the migration completes.
"""

import sys as _sys

import natal.backends.numba.compat as _m0
import natal.backends.numba.utils as _m1
from natal.backends.numba import (
    NUMBA_ENABLED,
    binomial,
    binomial_2d,
    binomial_btpe,
    clamp01,
    continuous_binomial,
    continuous_multinomial,
    continuous_poisson,
    disable_numba,
    disable_numba_log,
    disable_numba_signature_trace,
    enable_numba,
    enable_numba_log,
    enable_numba_signature_trace,
    fancy_index_3d_flat,
    fancy_index_3d_to_2d,
    get_numba_cache_dir,
    is_numba_enabled,
    is_numba_log_enabled,
    is_numba_signature_trace_enabled,
    multinomial,
    multinomial_rows,
    njit_switch,
    numba_disabled,
    numba_enabled,
    set_numba_seed,
    with_numba_disabled,
    with_numba_enabled,
)

_sys.modules["natal.numba.compat"] = _m0
_sys.modules["natal.numba.utils"] = _m1

# Legacy parity: real packages expose imported children as attributes; the
# sys.modules aliases above do not.  Re-bind every aliased submodule onto its
# (aliased) parent so ``package.submodule`` attribute access keeps working.
for _alias in [a for a in _sys.modules if a.startswith(__name__ + ".")]:
    _parent, _, _leaf = _alias.rpartition(".")
    setattr(_sys.modules[_parent], _leaf, _sys.modules[_alias])

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
