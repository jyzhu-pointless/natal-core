"""Forwarding shim: the ``fitness`` package now lives at
``natal.frontend.fitness``.

This module preserves the legacy import path during the Phase-0
directory reorganization; it will be removed once the migration
completes.
"""
import sys as _sys

# Alias imports for submodule forwarding (registered below).
import natal.frontend.fitness._patch as _m0
import natal.frontend.fitness._types as _m1
import natal.frontend.fitness._writer as _m2
from natal.frontend.fitness import (
    apply_preset_fitness_patch,
    write_fitness_field,
)

# Register legacy submodule paths -> relocated modules.
_sys.modules["natal.fitness._patch"] = _m0
_sys.modules["natal.fitness._types"] = _m1
_sys.modules["natal.fitness._writer"] = _m2

# Legacy parity: real packages expose imported children as attributes; the
# sys.modules aliases above do not.  Re-bind every aliased submodule onto its
# (aliased) parent so ``package.submodule`` attribute access keeps working.
for _alias in [a for a in _sys.modules if a.startswith(__name__ + ".")]:
    _parent, _, _leaf = _alias.rpartition(".")
    setattr(_sys.modules[_parent], _leaf, _sys.modules[_alias])

__all__ = [
    "apply_preset_fitness_patch",
    "write_fitness_field",
]
