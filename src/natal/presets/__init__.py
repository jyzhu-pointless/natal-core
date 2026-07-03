"""Genetic presets subpackage.

Provides genetic modification presets including gene drives (HomingDrive,
ToxinAntidoteDrive), cytoplasmic inheritance (Wolbachia), allele conversion
systems, and the GeneticPreset base class.
"""

from ._base import GeneticPreset, apply_preset_to_population
from ._fitness import apply_preset_fitness_patch
from ._types import PresetFitnessPatch, count_allele_copies
from .cytoplasmic import CytoplasmicPreset, TransgenicBackground, Wolbachia
from .gamete_conversion import (
    GameteAlleleConversionRule,
    GameteConversionRuleSet,
    GameteHaploidGenomeConversionRule,
)
from .homing import HomingDrive
from .toxin_antidote import ToxinAntidoteDrive
from .zygote_conversion import (
    ZygoteAlleleConversionRule,
    ZygoteConversionRuleSet,
    ZygoteGenotypeConversionRule,
)

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
    "GameteHaploidGenomeConversionRule",
    "ZygoteAlleleConversionRule",
    "ZygoteConversionRuleSet",
    "ZygoteGenotypeConversionRule",
]
