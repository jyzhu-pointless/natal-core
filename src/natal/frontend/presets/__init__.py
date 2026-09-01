"""Genetic presets subpackage.

Provides genetic modification presets including gene drives (HomingDrive,
ToxinAntidoteDrive), cytoplasmic inheritance (Wolbachia), allele conversion
systems, and the GeneticPreset base class.
"""

from typing import TYPE_CHECKING

from natal.frontend.modifiers.gamete_conversion import (  # noqa: F401 (re-export, canonical location)
    GameteAlleleConversionRule,
    GameteConversionRuleSet,
    GameteGlabConversionRule,
    GameteGtypeConversionRule,
    GameteHaploidGenomeConversionRule,
)
from natal.frontend.modifiers.zygote_conversion import (  # noqa: F401 (re-export, canonical location)
    ZygoteAlleleConversionRule,
    ZygoteConversionRuleSet,
    ZygoteGenotypeConversionRule,
    ZygoteGlabRedirectRule,
    ZygoteZtypeConversionRule,
)

from ._base import GeneticPreset, apply_preset_to_population
from ._types import PresetFitnessPatch, count_allele_copies
from .cytoplasmic import CytoplasmicPreset, TransgenicBackground, Wolbachia
from .homing import HomingDrive
from .toxin_antidote import ToxinAntidoteDrive

if TYPE_CHECKING:
    # Static-only import: gives type checkers the precise signature while the
    # runtime export stays deferred through ``__getattr__`` below.
    from ._fitness import apply_preset_fitness_patch


def __getattr__(name: str) -> object:
    """Lazily re-export ``apply_preset_fitness_patch``.

    ``_fitness`` imports from ``natal.frontend.fitness._patch`` at module
    level, while ``_patch`` imports type helpers from ``presets._types``.
    Re-exporting eagerly makes the package-initialisation order sensitive
    (importing fitness first used to leave ``_patch`` partially initialised
    when ``presets.__init__`` reached ``_fitness``).  PEP 562 deferral
    breaks that cycle for every import order.
    """
    if name == "apply_preset_fitness_patch":
        from ._fitness import apply_preset_fitness_patch

        return apply_preset_fitness_patch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
