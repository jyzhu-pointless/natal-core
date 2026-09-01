"""Forwarding shim: the ``population`` package now lives at
``natal.frontend.population``.

This module preserves the legacy import path during the Phase-0
directory reorganisation; it will be removed once the migration
completes.
"""
import sys as _sys

# Alias imports for submodule forwarding (registered below).
import natal.frontend.population._mixins as _m0
import natal.frontend.population._mixins._hooks as _m1
import natal.frontend.population._mixins._modifiers as _m2
import natal.frontend.population._mixins._observation as _m3
import natal.frontend.population._mixins._output as _m4
import natal.frontend.population.age_structured as _m5
import natal.frontend.population.base as _m6
import natal.frontend.population.discrete_generation as _m7
from natal.frontend.population import (
    AgeStructuredPopulation,
    BasePopulation,
    DiscreteGenerationPopulation,
)

# Register legacy submodule paths -> relocated modules.
_sys.modules["natal.population._mixins"] = _m0
_sys.modules["natal.population._mixins._hooks"] = _m1
_sys.modules["natal.population._mixins._modifiers"] = _m2
_sys.modules["natal.population._mixins._observation"] = _m3
_sys.modules["natal.population._mixins._output"] = _m4
_sys.modules["natal.population.age_structured"] = _m5
_sys.modules["natal.population.base"] = _m6
_sys.modules["natal.population.discrete_generation"] = _m7

# Legacy parity: real packages expose imported children as attributes; the
# sys.modules aliases above do not.  Re-bind every aliased submodule onto its
# (aliased) parent so ``package.submodule`` attribute access keeps working.
for _alias in [a for a in _sys.modules if a.startswith(__name__ + ".")]:
    _parent, _, _leaf = _alias.rpartition(".")
    setattr(_sys.modules[_parent], _leaf, _sys.modules[_alias])

__all__ = [
    "AgeStructuredPopulation",
    "BasePopulation",
    "DiscreteGenerationPopulation",
]
