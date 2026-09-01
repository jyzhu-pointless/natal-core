"""Forwarding shim: the ``engine`` package has been split across backends.

Phase-0 reorganisation relocated every engine module:

- ``natal.engine.simulation``        -> ``natal.backends.reference.simulation``
- ``natal.engine.migration``         -> ``natal.backends.reference.migration``
- ``natal.engine.<*_simulator>``     -> ``natal.backends.reference.<...>``
- ``natal.engine.spatial_migrator``  -> ``natal.backends.reference.spatial_migrator``
- ``natal.engine.lifecycle``         -> ``natal.backends.numba.lifecycle``
- ``natal.engine.lifecycle_wrappers``-> ``natal.backends.numba.lifecycle_wrappers``
- ``natal.engine.backends.rust_backend`` -> ``natal.backends.rust.rust_backend``

This shim preserves all legacy import paths during the migration; it will be
removed once external references are updated (Phase 6).
"""

import importlib as _importlib
import pkgutil as _pkgutil
import sys as _sys
import types as _types

import natal.backends.numba.lifecycle as _numba_lifecycle
import natal.backends.numba.lifecycle_wrappers as _numba_lw
import natal.backends.reference.age_structured_simulator as _ref_aas
import natal.backends.reference.discrete_generation_simulator as _ref_dgs
import natal.backends.reference.migration as _ref_migration
import natal.backends.reference.simulation as _ref_simulation
import natal.backends.reference.spatial_migrator as _ref_spm
import natal.backends.reference.spatial_simulator as _ref_sps
import natal.backends.rust.rust_backend as _rust_backend

# A bare parent module for the legacy ``natal.engine.backends`` path so that
# ``from natal.engine.backends.rust_backend import X`` resolves end-to-end.
_engine_backends = _types.ModuleType("natal.engine.backends")
_engine_backends.__path__ = []  # type: ignore[attr-defined]  # marker: virtual pkg
_sys.modules["natal.engine.backends"] = _engine_backends

_sys.modules["natal.engine.backends.rust_backend"] = _rust_backend
_sys.modules["natal.engine.simulation"] = _ref_simulation
_sys.modules["natal.engine.migration"] = _ref_migration
_sys.modules["natal.engine.age_structured_simulator"] = _ref_aas
_sys.modules["natal.engine.discrete_generation_simulator"] = _ref_dgs
_sys.modules["natal.engine.spatial_simulator"] = _ref_sps
_sys.modules["natal.engine.spatial_migrator"] = _ref_spm
_sys.modules["natal.engine.lifecycle"] = _numba_lifecycle
_sys.modules["natal.engine.lifecycle_wrappers"] = _numba_lw

# Register the nested submodules (simulation.age_structured, migration.kernel, ...)
# Eagerly import every leaf first: only modules present in sys.modules at
# shim-execution time would otherwise be aliased, and a legacy-path import of
# a not-yet-loaded leaf (e.g. simulation.mgdrive1_compatible) would re-execute
# the file under the old name, producing a second module object.
for _pkg_new, _pkg_old in (
    ("natal.backends.reference.simulation", "natal.engine.simulation"),
    ("natal.backends.reference.migration", "natal.engine.migration"),
):
    _pkg = _sys.modules[_pkg_new]
    for _info in _pkgutil.iter_modules(_pkg.__path__):
        _leaf_new = f"{_pkg_new}.{_info.name}"
        _sys.modules[f"{_pkg_old}.{_info.name}"] = _importlib.import_module(_leaf_new)

# Legacy parity: real packages expose imported children as attributes; the
# sys.modules aliases above do not.  Re-bind every aliased submodule onto its
# (aliased) parent so ``package.submodule`` attribute access keeps working.
for _alias in [a for a in _sys.modules if a.startswith(__name__ + ".")]:
    _parent, _, _leaf = _alias.rpartition(".")
    setattr(_sys.modules[_parent], _leaf, _sys.modules[_alias])

__all__: list[str] = []
