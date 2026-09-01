"""Forwarding shim: the ``modifiers`` package now lives at
``natal.frontend.modifiers``.

This module preserves the legacy import path during the Phase-0
directory reorganisation; it will be removed once the migration
completes.
"""
import sys as _sys

# Alias imports for submodule forwarding (registered below).
import natal.frontend.modifiers.conditions as _m0
import natal.frontend.modifiers.gamete_conversion as _m1
import natal.frontend.modifiers.module as _m2
import natal.frontend.modifiers.zygote_conversion as _m3
from natal.frontend.modifiers import (
    GameteAlleleConversionRule,
    GameteConversionRuleSet,
    GameteGlabConversionRule,
    GameteGtypeConversionRule,
    GameteHaploidGenomeConversionRule,
    GameteModifier,
    GenotypeFilter,
    GlabSelector,
    ZygoteAlleleConversionRule,
    ZygoteConversionRuleSet,
    ZygoteGenotypeConversionRule,
    ZygoteGlabRedirectRule,
    ZygoteModifier,
    ZygoteZtypeConversionRule,
    build_modifier_wrappers,
    evaluate_genotype_filter,
    wrap_gamete_modifier,
    wrap_zygote_modifier,
)

# Register legacy submodule paths -> relocated modules.
_sys.modules["natal.modifiers.conditions"] = _m0
_sys.modules["natal.modifiers.gamete_conversion"] = _m1
_sys.modules["natal.modifiers.module"] = _m2
_sys.modules["natal.modifiers.zygote_conversion"] = _m3

# Legacy parity: real packages expose imported children as attributes; the
# sys.modules aliases above do not.  Re-bind every aliased submodule onto its
# (aliased) parent so ``package.submodule`` attribute access keeps working.
for _alias in [a for a in _sys.modules if a.startswith(__name__ + ".")]:
    _parent, _, _leaf = _alias.rpartition(".")
    setattr(_sys.modules[_parent], _leaf, _sys.modules[_alias])

__all__ = [
    "GameteAlleleConversionRule",
    "GameteConversionRuleSet",
    "GameteGlabConversionRule",
    "GameteGtypeConversionRule",
    "GameteHaploidGenomeConversionRule",
    "GameteModifier",
    "GenotypeFilter",
    "GlabSelector",
    "ZygoteAlleleConversionRule",
    "ZygoteConversionRuleSet",
    "ZygoteGenotypeConversionRule",
    "ZygoteGlabRedirectRule",
    "ZygoteModifier",
    "ZygoteZtypeConversionRule",
    "build_modifier_wrappers",
    "evaluate_genotype_filter",
    "wrap_gamete_modifier",
    "wrap_zygote_modifier",
]
