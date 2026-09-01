"""Forwarding shim: the ``patterns`` package now lives at
``natal.frontend.patterns``.

This module preserves the legacy import path during the Phase-0 directory
reorganisation.  New code should import from ``natal.frontend.patterns``;
this shim (and the legacy path) will be removed once the migration completes.
"""
import sys as _sys

# Alias imports for submodule forwarding (registered below).
import natal.frontend.patterns.elements as _m0
import natal.frontend.patterns.individual_selector as _m1
import natal.frontend.patterns.parser as _m2
import natal.frontend.patterns.selector as _m3
from natal.frontend.patterns import (
    GameteTypePattern,
    GenotypePatternParser,
    GenotypeSelector,
    IndividualSelector,
    LabPattern,
    PatternParseError,
    ZygoteTypePattern,
    resolve_zygote_type,
)

# Register legacy submodule paths -> relocated modules.
_sys.modules["natal.patterns.elements"] = _m0
_sys.modules["natal.patterns.individual_selector"] = _m1
_sys.modules["natal.patterns.parser"] = _m2
_sys.modules["natal.patterns.selector"] = _m3

# Legacy parity: real packages expose imported children as attributes; the
# sys.modules aliases above do not.  Re-bind every aliased submodule onto its
# (aliased) parent so ``package.submodule`` attribute access keeps working.
for _alias in [a for a in _sys.modules if a.startswith(__name__ + ".")]:
    _parent, _, _leaf = _alias.rpartition(".")
    setattr(_sys.modules[_parent], _leaf, _sys.modules[_alias])

__all__ = [
    "GameteTypePattern",
    "GenotypePatternParser",
    "GenotypeSelector",
    "IndividualSelector",
    "LabPattern",
    "PatternParseError",
    "ZygoteTypePattern",
    "resolve_zygote_type",
]
