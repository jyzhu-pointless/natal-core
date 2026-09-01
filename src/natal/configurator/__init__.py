"""Forwarding shim: the ``configurator`` package now lives at
``natal.frontend.configurator``.

This module preserves the legacy import path during the Phase-0
directory reorganization; it will be removed once the migration
completes.
"""
import sys as _sys

# Alias imports for submodule forwarding (registered below).
import natal.frontend.configurator._base as _m0
import natal.frontend.configurator._factory as _m1
import natal.frontend.configurator._fitness as _m2
import natal.frontend.configurator._params as _m3
import natal.frontend.configurator._registry_builder as _m4
import natal.frontend.configurator.age_structured as _m5
import natal.frontend.configurator.discrete as _m6
from natal.frontend.configurator import (
    AgeStructuredConfigurator,
    Configurator,
    DiscreteConfigurator,
    PopulationConfigBuilder,
    hook_set_param,
    merge_hooks,
    set_param,
)

# Register legacy submodule paths -> relocated modules.
_sys.modules["natal.configurator._base"] = _m0
_sys.modules["natal.configurator._factory"] = _m1
_sys.modules["natal.configurator._fitness"] = _m2
_sys.modules["natal.configurator._params"] = _m3
_sys.modules["natal.configurator._registry_builder"] = _m4
_sys.modules["natal.configurator.age_structured"] = _m5
_sys.modules["natal.configurator.discrete"] = _m6

# Legacy parity: real packages expose imported children as attributes; the
# sys.modules aliases above do not.  Re-bind every aliased submodule onto its
# (aliased) parent so ``package.submodule`` attribute access keeps working.
for _alias in [a for a in _sys.modules if a.startswith(__name__ + ".")]:
    _parent, _, _leaf = _alias.rpartition(".")
    setattr(_sys.modules[_parent], _leaf, _sys.modules[_alias])

__all__ = [
    "AgeStructuredConfigurator",
    "Configurator",
    "DiscreteConfigurator",
    "PopulationConfigBuilder",
    "hook_set_param",
    "merge_hooks",
    "set_param",
]
