"""Forwarding shim: the ``presets`` package now lives at
``natal.frontend.presets``.

This module preserves the legacy import path during the Phase-0
directory reorganisation; it will be removed once the migration
completes.
"""
import sys as _sys

# Alias imports for submodule forwarding (registered below).
import natal.frontend.presets._base as _m0
import natal.frontend.presets._fitness as _m1
import natal.frontend.presets._types as _m2
import natal.frontend.presets.cytoplasmic as _m3
import natal.frontend.presets.homing as _m4
import natal.frontend.presets.toxin_antidote as _m5
from natal.frontend.presets import (
    CytoplasmicPreset,
    GameteAlleleConversionRule,
    GameteConversionRuleSet,
    GameteGlabConversionRule,
    GameteGtypeConversionRule,
    GameteHaploidGenomeConversionRule,
    GeneticPreset,
    HomingDrive,
    PresetFitnessPatch,
    ToxinAntidoteDrive,
    TransgenicBackground,
    Wolbachia,
    ZygoteAlleleConversionRule,
    ZygoteConversionRuleSet,
    ZygoteGenotypeConversionRule,
    ZygoteGlabRedirectRule,
    ZygoteZtypeConversionRule,
    apply_preset_to_population,
    count_allele_copies,
)

# Imported from the cycle-free ``_fitness`` module directly: its canonical
# re-export through ``frontend.presets`` is deferred (PEP 562), which static
# stub resolution cannot see through.
from natal.frontend.presets._fitness import apply_preset_fitness_patch

# Register legacy submodule paths -> relocated modules.
_sys.modules["natal.presets._base"] = _m0
_sys.modules["natal.presets._fitness"] = _m1
_sys.modules["natal.presets._types"] = _m2
_sys.modules["natal.presets.cytoplasmic"] = _m3
_sys.modules["natal.presets.homing"] = _m4
_sys.modules["natal.presets.toxin_antidote"] = _m5

# Legacy parity: real packages expose imported children as attributes; the
# sys.modules aliases above do not.  Re-bind every aliased submodule onto its
# (aliased) parent so ``package.submodule`` attribute access keeps working.
for _alias in [a for a in _sys.modules if a.startswith(__name__ + ".")]:
    _parent, _, _leaf = _alias.rpartition(".")
    setattr(_sys.modules[_parent], _leaf, _sys.modules[_alias])

__all__ = [
    "CytoplasmicPreset",
    "GameteAlleleConversionRule",
    "GameteConversionRuleSet",
    "GameteGlabConversionRule",
    "GameteGtypeConversionRule",
    "GameteHaploidGenomeConversionRule",
    "GeneticPreset",
    "HomingDrive",
    "PresetFitnessPatch",
    "ToxinAntidoteDrive",
    "TransgenicBackground",
    "Wolbachia",
    "ZygoteAlleleConversionRule",
    "ZygoteConversionRuleSet",
    "ZygoteGenotypeConversionRule",
    "ZygoteGlabRedirectRule",
    "ZygoteZtypeConversionRule",
    "apply_preset_fitness_patch",
    "apply_preset_to_population",
    "count_allele_copies",
]
