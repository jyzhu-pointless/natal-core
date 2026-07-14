"""Genetic presets subpackage.

Provides genetic modification presets including gene drives (HomingDrive,
ToxinAntidoteDrive), cytoplasmic inheritance (Wolbachia), allele conversion
systems, and the GeneticPreset base class.
"""

from natal.modifiers.gamete_conversion import (  # noqa: F401 (re-export, canonical location)
    GameteAlleleConversionRule,
    GameteConversionRuleSet,
    GameteGlabConversionRule,
    GameteGtypeConversionRule,
    GameteHaploidGenomeConversionRule,
)
from natal.modifiers.zygote_conversion import (  # noqa: F401 (re-export, canonical location)
    ZygoteAlleleConversionRule,
    ZygoteConversionRuleSet,
    ZygoteGenotypeConversionRule,
    ZygoteGlabRedirectRule,
    ZygoteZtypeConversionRule,
)

from ._base import GeneticPreset, apply_preset_to_population
from ._fitness import apply_preset_fitness_patch
from ._types import PresetFitnessPatch, count_allele_copies
from .cytoplasmic import CytoplasmicPreset, TransgenicBackground, Wolbachia
from .homing import HomingDrive
from .toxin_antidote import ToxinAntidoteDrive

__all__ = [
    "GeneticPreset",
    "HomingDrive",
    "ToxinAntidoteDrive",
    "CytoplasmicPreset",
    "Wolbachia",
    "TransgenicBackground",
    "apply_preset_to_population",
    "apply_preset_fitness_patch",
    "PresetFitnessPatch",
    "count_allele_copies",
    "GameteAlleleConversionRule",
    "GameteConversionRuleSet",
    "GameteGlabConversionRule",
    "GameteGtypeConversionRule",
    "GameteHaploidGenomeConversionRule",
    "ZygoteAlleleConversionRule",
    "ZygoteConversionRuleSet",
    "ZygoteGenotypeConversionRule",
    "ZygoteGlabRedirectRule",
    "ZygoteZtypeConversionRule",
]
